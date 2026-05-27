"""
OPTNet: Improved Design (Option 2)
- Added relative position encoding in attention
- Increased model capacity (256 dim, deeper)
- Added sparse convolution for early layers
- Fixed attention mechanism with 3D position awareness
- Removed differentiable sorting (hard mode only)
- Increased dropout and FFN ratio to match PTv3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.amp import autocast
from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule
from pointcept.models.utils.structure import Point
from pointcept.models.default import DefaultSegmentorV2
import pointops
from pointcept.models.losses import build_criteria

# ==============================================================================
# Utils
# ==============================================================================

def batch_to_padded(feat, lengths):
    """Converts packed batch (N_total, C) to padded batch (B, N_max, C)."""
    B = len(lengths)
    N_max = lengths.max().item()
    device = feat.device
    is_1d = (feat.dim() == 1)
    if is_1d: 
        feat = feat.unsqueeze(-1)
    C = feat.shape[-1]
    
    padded = torch.zeros(B, N_max, C, device=device, dtype=feat.dtype)
    valid_mask = torch.zeros(B, N_max, device=device, dtype=torch.bool)
    
    start = 0
    for i, length in enumerate(lengths):
        l = length.item()
        if l > 0:
            padded[i, :l] = feat[start:start+l]
            valid_mask[i, :l] = True
            start += l
    
    if is_1d: 
        padded = padded.squeeze(-1)
    return padded, valid_mask

def padded_to_batch(padded, valid_mask):
    """Converts padded batch (B, N_max, C) to packed batch (N_total, C)."""
    is_1d = (padded.dim() == 2)
    if is_1d: 
        padded = padded.unsqueeze(-1)
    packed = padded[valid_mask]
    if is_1d: 
        packed = packed.squeeze(-1)
    return packed

# ==============================================================================
# Permutation (Hard Mode Only - Simpler and Faster)
# ==============================================================================

class Permutation:
    def __init__(self, hard_order, hard_inverse):
        self.hard_order = hard_order
        self.hard_inverse = hard_inverse
    
    def apply(self, x):
        """Apply permutation to tensor."""
        return x[self.hard_order]
    
    def inverse(self, x):
        """Inverse permutation to restore original order."""
        out = torch.zeros_like(x)
        out[self.hard_inverse] = x
        return out

# ==============================================================================
# Improved Layers
# ==============================================================================

class PointSorter(nn.Module):
    """Learns to predict ordering scores for points."""
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    
    def _init_weights(self):
        # Initialize to bias toward spatial coordinates
        nn.init.constant_(self.mlp[0].weight[:, 3:], 0.0)
        nn.init.normal_(self.mlp[0].weight[:, :3], std=0.1)
        with torch.no_grad():
            self.mlp[0].weight[0, :3] = 1.0
        nn.init.constant_(self.mlp[-1].bias, 0.0)
    
    def forward(self, point: Point):
        """Returns ordering scores for each point."""
        inp = torch.cat([point.coord, point.feat.detach()], 1) if point.feat is not None else point.coord
        return self.sigmoid(self.mlp(inp)).squeeze(-1)

class SparseConvBlock(nn.Module):
    """Sparse convolution block for early feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        # MLP for feature aggregation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),  # +3 for relative coords
            norm_layer(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            norm_layer(out_channels),
            nn.GELU()
        )
    
    def forward(self, point: Point):
        """Apply sparse convolution using KNN neighbors."""
        # Find neighbors
        idx = pointops.knn_query(self.kernel_size, point.coord, point.offset)[0]
        
        # Gather neighbor features
        neighbor_coord = point.coord[idx.long()]  # (N, K, 3)
        neighbor_feat = point.feat[idx.long()]    # (N, K, C)
        
        # Compute relative positions
        rel_pos = neighbor_coord - point.coord.unsqueeze(1)  # (N, K, 3)
        
        # Concatenate features with relative positions
        feat_with_pos = torch.cat([neighbor_feat, rel_pos], dim=-1)  # (N, K, C+3)
        
        # Aggregate
        feat_agg = feat_with_pos.mean(dim=1)  # (N, C+3)
        point.feat = self.mlp(feat_agg)
        
        return point

class GatedPosEmbedding(nn.Module):
    """Gated position embedding for displacement features."""
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_channels, bias=False)
        )
        self.enc_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )
    
    def forward(self, rel_coord):
        with autocast("cuda", enabled=False):
            gate = self.mlp(rel_coord.float())
            return self.enc_mlp(rel_coord.float() * gate + gate)

class MLPCPE(nn.Module):
    """MLP-based conditional position encoding."""
    def __init__(self, channels, hidden_ratio=2.0, drop=0.1):
        super().__init__()
        hidden = int(channels * hidden_ratio)
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, channels),
            nn.Dropout(drop)
        )
    
    def forward(self, point: Point):
        point.feat = point.feat + self.net(self.norm(point.feat))
        return point

class ImprovedWindowAttention1D(nn.Module):
    """Window attention with relative position encoding."""
    def __init__(self, channels, num_heads=8, qkv_bias=True, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position encoding MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, num_heads)
        )
    
    def forward(self, x, coords):
        """
        Args:
            x: (num_windows, window_size, C) - features
            coords: (num_windows, window_size, 3) - 3D coordinates
        """
        Wn, K, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(Wn, K, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, Wn, H, K, C//H)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute relative positions within each window
        # (Wn, K, 1, 3) - (Wn, 1, K, 3) = (Wn, K, K, 3)
        rel_pos = coords.unsqueeze(2) - coords.unsqueeze(1)
        
        # Flatten for MLP
        Wn_orig, K_orig = rel_pos.shape[0], rel_pos.shape[1]
        rel_pos_flat = rel_pos.reshape(-1, 3)  # (Wn*K*K, 3)
        pos_bias_flat = self.pos_mlp(rel_pos_flat)  # (Wn*K*K, H)
        pos_bias = pos_bias_flat.reshape(Wn_orig, K_orig, K_orig, self.num_heads)
        pos_bias = pos_bias.permute(0, 3, 1, 2)  # (Wn, H, K, K)
        
        # Attention with position bias
        attn = (q @ k.transpose(-2, -1)) * self.scale + pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(Wn, K, C)
        return self.proj_drop(self.proj(out))

# ==============================================================================
# Blocks
# ==============================================================================

class OPTBlock(nn.Module):
    """Improved OPT block with position-aware attention."""
    def __init__(self, channels, num_heads=8, win_size=32, shift_size=0, 
                 dropout=0.1, ffn_ratio=4.0, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.win_size = int(win_size)
        self.shift_size = int(shift_size)
        
        self.cpe = MLPCPE(channels, hidden_ratio=2.0, drop=dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = ImprovedWindowAttention1D(
            channels, 
            num_heads=num_heads, 
            qkv_bias=True, 
            attn_drop=dropout, 
            proj_drop=dropout
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, int(ffn_ratio * channels)),
            nn.LayerNorm(int(ffn_ratio * channels)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(ffn_ratio * channels), channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, point: Point, lengths, perm: Permutation):
        point = self.cpe(point)
        shortcut = point.feat
        x = self.norm1(point.feat)
        
        # Apply permutation to features AND coordinates
        x_sorted = perm.apply(x)
        coord_sorted = perm.apply(point.coord)
        
        Total_N, C = x_sorted.shape
        
        # Apply window shift
        if self.shift_size > 0:
            x_sorted = torch.roll(x_sorted, shifts=-self.shift_size, dims=0)
            coord_sorted = torch.roll(coord_sorted, shifts=-self.shift_size, dims=0)
        
        # Pad to make divisible by window size
        pad_len = (-Total_N) % self.win_size
        if pad_len > 0:
            x_sorted = x_sorted.unsqueeze(0)
            x_sorted = F.pad(x_sorted, (0, 0, 0, pad_len), mode='replicate')
            x_sorted = x_sorted.squeeze(0)
            
            coord_sorted = coord_sorted.unsqueeze(0)
            coord_sorted = F.pad(coord_sorted, (0, 0, 0, pad_len), mode='replicate')
            coord_sorted = coord_sorted.squeeze(0)
        
        # Reshape into windows
        x_win = x_sorted.view(-1, self.win_size, C)
        coord_win = coord_sorted.view(-1, self.win_size, 3)
        
        # Apply attention with position encoding
        out_win = self.attn(x_win, coord_win)
        
        # Reshape back
        out_sorted = out_win.view(-1, C)
        
        # Remove padding
        if pad_len > 0:
            out_sorted = out_sorted[:Total_N]
        
        # Reverse shift
        if self.shift_size > 0:
            out_sorted = torch.roll(out_sorted, shifts=self.shift_size, dims=0)
        
        # Inverse permutation
        out_final = perm.inverse(out_sorted)
        
        # Residual connection
        x = shortcut + out_final
        x = x + self.ffn(self.norm2(x))
        
        point.feat = x
        return point

class DownsamplingBlock(nn.Module):
    """Downsampling with proper stride handling."""
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=None, act_layer=None):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.act = act_layer() if act_layer else None
    
    def forward(self, point: Point, lengths, perm: Permutation):
        # Apply permutation
        feat_sorted = perm.apply(point.feat)
        coord_sorted = perm.apply(point.coord)
        
        # Use padded strategy for consistent per-batch stride
        feat_padded, valid_mask = batch_to_padded(feat_sorted, lengths)
        feat_down_padded = feat_padded[:, ::self.stride, :]
        
        coord_padded, _ = batch_to_padded(coord_sorted, lengths)
        coord_down_padded = coord_padded[:, ::self.stride, :]
        
        lengths_down = (lengths + self.stride - 1) // self.stride
        
        # Create valid mask for downsampled data
        B, N_down_max, _ = feat_down_padded.shape
        valid_mask_down = torch.zeros(B, N_down_max, device=point.feat.device, dtype=torch.bool)
        for i in range(B):
            valid_mask_down[i, :lengths_down[i]] = True
        
        # Convert back to packed
        feat_down = padded_to_batch(feat_down_padded, valid_mask_down)
        coord_down = padded_to_batch(coord_down_padded, valid_mask_down)
        
        # Project features
        feat_down = self.proj(feat_down)
        if self.norm:
            feat_down = self.norm(feat_down)
        if self.act:
            feat_down = self.act(feat_down)
        
        # Create downsampled point
        point_down = Point({
            'feat': feat_down,
            'coord': coord_down,
            'offset': torch.cumsum(lengths_down, dim=0)
        })
        point_down['down_stride'] = self.stride
        point_down['down_parent_lengths'] = lengths
        point_down['down_parent_point'] = point
        
        return point_down, lengths_down

class UpsamplingBlock(nn.Module):
    """Upsampling with skip connections."""
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.proj_skip = nn.Linear(skip_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.norm_skip = norm_layer(out_channels) if norm_layer else None
        self.act = act_layer() if act_layer else None
        self.act_skip = act_layer() if act_layer else None
    
    def forward(self, point: Point, lengths, perm: Permutation):
        stride = point['down_stride']
        lengths_up = point['down_parent_lengths']
        parent_point = point.pop('down_parent_point')
        
        # Process skip connection
        feat_skip = self.proj_skip(parent_point.feat)
        if self.norm_skip:
            feat_skip = self.norm_skip(feat_skip)
        if self.act_skip:
            feat_skip = self.act_skip(feat_skip)
        
        # Upsample low-res features
        feat_low = self.proj(point.feat)
        if self.norm:
            feat_low = self.norm(feat_low)
        if self.act:
            feat_low = self.act(feat_low)
        
        # Repeat features according to stride
        feat_up_list = []
        start = 0
        for i, l_high in enumerate(lengths_up):
            l_low = lengths[i].item()
            if l_low == 0:
                feat_up_list.append(feat_low[start:start])
                continue
            
            curr = feat_low[start:start + l_low]
            idx = torch.arange(l_high.item(), device=curr.device) // stride
            idx = idx.clamp(max=l_low - 1)
            feat_up_list.append(curr[idx])
            start += l_low
        
        feat_up_sorted = torch.cat(feat_up_list, 0)
        feat_up_physical = perm.inverse(feat_up_sorted)
        
        # Add skip connection
        parent_point.feat = feat_up_physical + feat_skip
        
        return parent_point, lengths_up

class EncodingBlock(nn.Module):
    """Encoding or decoding block with multiple OPT blocks."""
    def __init__(self, channels, depth, num_heads=8, win_size=32, dropout=0.1, 
                 ffn_ratio=4.0, is_encoder=True, downsample_stride=2, 
                 norm_layer=nn.BatchNorm1d, act_layer=nn.GELU):
        super().__init__()
        self.is_encoder = is_encoder
        
        # Alternate shift sizes for cross-window interaction
        self.blocks = nn.ModuleList([
            OPTBlock(
                channels, 
                num_heads, 
                win_size, 
                shift_size=(0 if i % 2 == 0 else win_size // 2),
                dropout=dropout, 
                ffn_ratio=ffn_ratio, 
                norm_layer=norm_layer
            ) 
            for i in range(depth)
        ])
        
        if is_encoder:
            self.downsample = DownsamplingBlock(channels, channels, downsample_stride, norm_layer, act_layer)
        else:
            self.upsample = UpsamplingBlock(channels, channels, channels, norm_layer, act_layer)
    
    def forward(self, point, lengths, perm, upsample_perm=None):
        # Apply blocks
        for block in self.blocks:
            point = block(point, lengths, perm)
        
        # Downsample or upsample
        if self.is_encoder:
            if self.downsample:
                point, lengths = self.downsample(point, lengths, perm)
            return point, lengths
        else:
            perm_for_up = upsample_perm if upsample_perm is not None else perm
            if self.upsample:
                point, lengths = self.upsample(point, lengths, perm_for_up)
            return point, lengths

# ==============================================================================
# Main Model
# ==============================================================================

@MODELS.register_module("OPTNet")
class OPTNet(PointModule):
    """Improved OPTNet with PTv3-inspired design."""
    def __init__(
        self,
        in_channels=6,
        embed_dim=256,
        enc_depths=(2, 6, 18, 6),
        dec_depths=(2, 2, 2, 2),
        num_heads=8,
        win_sizes=(32, 32, 32, 32),
        pool_factors=(2, 2, 2, 2),
        dropout=0.1,
        ffn_ratio=4.0,
        ordering_loss_weight=0.5,
        ordering_k=16,
        **kwargs
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.pool_factors = pool_factors
        self.ordering_loss_weight = ordering_loss_weight
        self.ordering_k = ordering_k
        
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # Sparse convolution for initial feature extraction
        self.sparse_conv = SparseConvBlock(3, embed_dim // 2, kernel_size=16, norm_layer=bn_layer)
        self.patch_embed = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim),
            bn_layer(embed_dim),
            nn.GELU()
        )
        
        # Feature processing
        self.feat_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Ordering module
        self.sorter = PointSorter(embed_dim, hidden_dim=128)
        
        # Position encoding
        self.gated_pos = GatedPosEmbedding(3, embed_dim)
        
        # Encoder stages
        self.encoders = nn.ModuleList([
            EncodingBlock(
                embed_dim,
                enc_depths[si],
                num_heads,
                win_sizes[si],
                dropout,
                ffn_ratio,
                is_encoder=True,
                downsample_stride=pool_factors[si] if si < self.num_stages - 1 else 1,
                norm_layer=bn_layer,
                act_layer=nn.GELU
            )
            for si in range(self.num_stages)
        ])
        
        # Decoder stages
        self.decoders = nn.ModuleList([
            EncodingBlock(
                embed_dim,
                dec_depths[si],
                num_heads,
                win_sizes[si],
                dropout,
                ffn_ratio,
                is_encoder=False,
                norm_layer=bn_layer,
                act_layer=nn.GELU
            )
            for si in range(self.num_stages - 1)
        ])
        
        self.head_norm = nn.LayerNorm(embed_dim)
    
    def compute_ordering_loss(self, point: Point, scores: torch.Tensor):
        """Compute locality and distribution losses for ordering."""
        scores = scores.view(-1)
        offset = point.offset if (hasattr(point, "offset") and point.offset is not None) \
                 else torch.tensor([scores.numel()], device=scores.device)
        
        # Locality loss: neighbors should have similar scores
        idx = pointops.knn_query(self.ordering_k, point.coord, offset)[0]
        neighbor_scores = scores[idx.long()]
        
        # Compute mean squared difference (proper scalar reduction)
        diff = scores.unsqueeze(1) - neighbor_scores  # (N, K)
        loss_locality = (diff ** 2).mean()  # Scalar
        
        # Distribution loss: scores should span [0, 1] uniformly
        sorted_scores, _ = torch.sort(scores)
        target = torch.linspace(0, 1, steps=len(sorted_scores), device=scores.device)
        loss_dist = ((sorted_scores - target) ** 2).mean()  # Scalar
        
        total_loss = loss_locality + loss_dist
        
        return total_loss
    
    def forward(self, data_dict):
        point = Point(data_dict)
        feat_raw = point.feat
        
        # Extract color and displacement features
        color_feat = feat_raw[:, :3] if feat_raw.shape[1] >= 6 else feat_raw
        disp_feat = feat_raw[:, -3:] if feat_raw.shape[1] >= 6 else torch.zeros_like(point.coord)
        
        # Initial feature extraction with sparse convolution
        point.feat = color_feat
        point = self.sparse_conv(point)
        point.feat = self.patch_embed(point.feat)
        
        # Process features
        x = self.feat_mlp(point.feat)
        point.feat = x
        
        # Learn ordering
        scores = self.sorter(point)
        
        # Compute ordering loss (ensure scalar)
        if self.training and self.ordering_loss_weight > 0:
            ordering_loss = self.compute_ordering_loss(point, scores)
            total_aux_loss = ordering_loss * self.ordering_loss_weight
        else:
            total_aux_loss = torch.tensor(0.0, device=point.feat.device, dtype=point.feat.dtype)
        
        # Add position encoding
        point.feat = x + self.gated_pos(disp_feat)
        
        # Prepare lengths and permutation
        offset = point.offset
        lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long() \
                  if offset is not None else torch.tensor([point.feat.shape[0]], device=point.feat.device)
        
        # Create hard permutation based on learned scores
        batch = point.batch.long()
        global_scores = scores + batch.float() * (scores.max().detach() + 2.0)
        order = torch.argsort(global_scores)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        
        initial_perm = Permutation(hard_order=order, hard_inverse=inverse)
        current_perm = initial_perm
        
        # Encoder
        skip_points = []
        for si, encoder in enumerate(self.encoders):
            if si < self.num_stages - 1:
                skip_points.append(Point(point))
                skip_points[-1].feat = point.feat.clone()
            
            point, lengths = encoder(point, lengths, current_perm)
            
            # Switch to identity permutation for subsequent stages
            N_curr = point.feat.shape[0]
            current_perm = Permutation(
                hard_order=torch.arange(N_curr, device=point.feat.device),
                hard_inverse=torch.arange(N_curr, device=point.feat.device)
            )
        
        # Decoder
        for si in reversed(range(self.num_stages - 1)):
            point['down_parent_point'] = skip_points[si]
            point['down_parent_lengths'] = torch.cat([
                skip_points[si].offset[0].unsqueeze(0),
                skip_points[si].offset[1:] - skip_points[si].offset[:-1]
            ])
            point['down_stride'] = self.pool_factors[si]
            
            # Block permutation (identity for low-res)
            current_N = point.feat.shape[0]
            block_perm = Permutation(
                hard_order=torch.arange(current_N, device=point.feat.device),
                hard_inverse=torch.arange(current_N, device=point.feat.device)
            )
            
            # Upsample permutation (initial ordering for first stage, identity otherwise)
            if si == 0:
                upsample_perm = initial_perm
            else:
                target_N = skip_points[si].feat.shape[0]
                upsample_perm = Permutation(
                    hard_order=torch.arange(target_N, device=point.feat.device),
                    hard_inverse=torch.arange(target_N, device=point.feat.device)
                )
            
            point, lengths = self.decoders[si](point, lengths, block_perm, upsample_perm=upsample_perm)
        
        # Final normalization
        point.feat = self.head_norm(point.feat)
        
        # Store auxiliary loss (must be scalar)
        if self.training:
            point.aux_loss = total_aux_loss
        
        return point


@MODELS.register_module("OPTNetSegmentor")
class OPTNetSegmentor(nn.Module):
    """Segmentor wrapper for OPTNet."""
    def __init__(self, backbone, criteria, num_classes, backbone_out_channels):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        
        # Manual loss construction to ensure proper reduction
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        
        self.num_classes = num_classes
        
        # Segmentation head
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
    
    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        
        # Get auxiliary loss
        aux_loss = getattr(point, "aux_loss", None)
        
        # Get segmentation logits
        seg_logits = self.seg_head(point.feat)
        
        # Initialize return dict
        return_dict = {}
        
        if self.training:
            # TRAINING MODE: Only return loss (scalar)
            seg_loss = self.criterion(seg_logits, input_dict["segment"])
            
            # Add auxiliary loss if exists
            total_loss = seg_loss
            if aux_loss is not None:
                total_loss = total_loss + aux_loss
            
            # CRITICAL: Only return loss during training
            # Do NOT return seg_logits as it will be logged
            return_dict["loss"] = total_loss
            
        else:
            # EVALUATION MODE: Return both loss and logits
            if "segment" in input_dict:
                loss = self.criterion(seg_logits, input_dict["segment"])
                return_dict["loss"] = loss
            
            # Return seg_logits for evaluation metrics
            return_dict["seg_logits"] = seg_logits
        
        # Optionally return point
        if return_point:
            return_dict['point'] = point
            
        return return_dict

