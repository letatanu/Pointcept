"""
OPTNet: Optimized Point Transformer with Cross-Window Attention
Memory-efficient version with gradient checkpointing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule
from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.utils.structure import Point
from pointcept.models.default import DefaultSegmentorV2
import pointops


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
# Permutation (Hard mode only for memory efficiency)
# ==============================================================================
class Permutation:
    def __init__(self, mode='hard', hard_order=None, hard_inverse=None):
        self.mode = mode
        self.hard_order = hard_order
        self.hard_inverse = hard_inverse

    def apply(self, x):
        return x[self.hard_order]

    def inverse(self, x):
        out = torch.zeros_like(x)
        out[self.hard_inverse] = x
        return out


# ==============================================================================
# Basic Layers
# ==============================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_channels, norm_layer):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            norm_layer(embed_channels),
            nn.ReLU(True)
        )

    def forward(self, point: Point):
        point.feat = self.embed(point.feat)
        return point


class PointSorter(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.mlp[0].weight[:, 3:], 0.0)
        nn.init.normal_(self.mlp[0].weight[:, :3], std=0.1)
        with torch.no_grad():
            self.mlp[0].weight[0, :3] = 1.0
        nn.init.constant_(self.mlp[6].bias, 0.0)

    def forward(self, point: Point):
        inp = torch.cat([point.coord, point.feat.detach()], 1) if point.feat is not None else point.coord
        return self.sigmoid(self.mlp(inp)).squeeze(-1)


class GatedPosEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
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
    def __init__(self, channels, hidden_ratio=2.0, drop=0.0):
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


class WindowAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        Wn, K, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(Wn, K, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        out = attn @ v
        out = out.transpose(1, 2).reshape(Wn, K, C)
        
        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class CrossWindowAttention(nn.Module):
    """Lightweight attention between window representatives"""
    def __init__(self, channels, num_heads=4, window_group_size=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_group_size = window_group_size
        
        assert channels % num_heads == 0
        
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x_win):
        """
        Args:
            x_win: (num_windows, window_size, channels)
        Returns:
            x_win: (num_windows, window_size, channels)
        """
        Wn, K, C = x_win.shape
        
        if Wn < 2:
            return x_win
        
        # Aggregate windows to tokens
        window_tokens = x_win.mean(dim=1)  # (num_windows, channels)
        
        # Group windows
        if Wn > self.window_group_size:
            num_groups = Wn // self.window_group_size
            pad_windows = num_groups * self.window_group_size
            window_tokens_grouped = window_tokens[:pad_windows].view(
                num_groups, self.window_group_size, C
            )
            G, Wg = num_groups, self.window_group_size
        else:
            window_tokens_grouped = window_tokens.unsqueeze(0)
            G, Wg = 1, Wn
        
        # QKV projection
        qkv = self.qkv(window_tokens_grouped).reshape(
            G, Wg, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(G, Wg, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Broadcast to all points in each window
        out_flat = out.view(-1, 1, C)
        x_win[:out_flat.shape[0]] = x_win[:out_flat.shape[0]] + out_flat
        
        return x_win


# ==============================================================================
# Blocks
# ==============================================================================
class OPTBlock(nn.Module):
    def __init__(self, channels, num_heads=4, win_size=32, shift_size=0, 
                 dropout=0.0, ffn_ratio=2.0, norm_layer=nn.BatchNorm1d,
                 use_cross_window=False):
        super().__init__()
        self.win_size = int(win_size)
        self.shift_size = int(shift_size)
        self.use_cross_window = use_cross_window
        
        self.cpe = MLPCPE(channels, hidden_ratio=2.0, drop=dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention1D(channels, num_heads=num_heads, 
                                      qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, int(ffn_ratio*channels)),
            nn.LayerNorm(int(ffn_ratio*channels)),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(int(ffn_ratio*channels), channels),
            nn.Dropout(dropout)
        )
        
        if use_cross_window:
            self.cross_attn = CrossWindowAttention(
                channels, num_heads=max(2, num_heads//2), dropout=dropout
            )

    def forward(self, point: Point, lengths, perm: Permutation):
        point = self.cpe(point)
        shortcut = point.feat
        x = self.norm1(point.feat)
        
        # Apply permutation and create windows
        x_sorted = perm.apply(x)
        Total_N, C = x_sorted.shape
        
        if self.shift_size > 0:
            x_sorted = torch.roll(x_sorted, shifts=-self.shift_size, dims=0)
        
        # Pad to window size
        pad_len = (-Total_N) % self.win_size
        if pad_len > 0:
            x_sorted = x_sorted.unsqueeze(0)
            x_sorted = F.pad(x_sorted, (0, 0, 0, pad_len), mode='replicate')
            x_sorted = x_sorted.squeeze(0)
        
        # Create windows
        x_win = x_sorted.view(-1, self.win_size, C)
        
        # Intra-window attention
        out_win = self.attn(x_win)
        
        # Cross-window attention
        if self.use_cross_window:
            out_win = self.cross_attn(out_win)
        
        # Reshape back
        out_sorted = out_win.view(-1, C)
        if pad_len > 0:
            out_sorted = out_sorted[:Total_N]
        if self.shift_size > 0:
            out_sorted = torch.roll(out_sorted, shifts=self.shift_size, dims=0)
        
        out_final = perm.inverse(out_sorted)
        
        # Residual and FFN
        x = shortcut + out_final
        x = x + self.ffn(self.norm2(x))
        point.feat = x
        
        return point


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=None, act_layer=None):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.act = act_layer() if act_layer else None

    def forward(self, point: Point, lengths, perm: Permutation):
        # Apply permutation for consistent downsampling
        feat_sorted = perm.apply(point.feat)
        coord_sorted = perm.apply(point.coord)
        
        # Use padded strategy for per-batch consistency
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
        
        feat_down = padded_to_batch(feat_down_padded, valid_mask_down)
        coord_down = padded_to_batch(coord_down_padded, valid_mask_down)
        
        feat_down = self.proj(feat_down)
        if self.norm:
            feat_down = self.norm(feat_down)
        if self.act:
            feat_down = self.act(feat_down)
        
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
        
        feat_skip = self.proj_skip(parent_point.feat)
        if self.norm_skip:
            feat_skip = self.norm_skip(feat_skip)
        if self.act_skip:
            feat_skip = self.act_skip(feat_skip)
        
        feat_low = self.proj(point.feat)
        if self.norm:
            feat_low = self.norm(feat_low)
        if self.act:
            feat_low = self.act(feat_low)
        
        # Upsample via repeat
        feat_up_list = []
        start = 0
        for i, l_high in enumerate(lengths_up):
            l_low = lengths[i].item()
            if l_low == 0:
                feat_up_list.append(feat_low[start:start])
                continue
            curr = feat_low[start:start+l_low]
            idx = torch.arange(l_high.item(), device=curr.device) // stride
            idx = idx.clamp(max=l_low - 1)
            feat_up_list.append(curr[idx])
            start += l_low
        
        feat_up_sorted = torch.cat(feat_up_list, 0)
        feat_up_physical = perm.inverse(feat_up_sorted)
        
        parent_point.feat = feat_up_physical + feat_skip
        
        return parent_point, lengths_up


class EncodingBlock(nn.Module):
    def __init__(self, channels, depth, num_heads=4, win_size=32, dropout=0.0, 
                 ffn_ratio=3.0, is_encoder=True, downsample_stride=2, 
                 norm_layer=nn.BatchNorm1d, act_layer=nn.ReLU,
                 use_cross_window=False, use_checkpoint=False):  # Add this param
        super().__init__()
        self.is_encoder = is_encoder
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList([
            OPTBlock(
                channels, num_heads, win_size,
                shift_size=(0 if i % 2 == 0 else win_size//2),
                dropout=dropout, ffn_ratio=ffn_ratio, norm_layer=norm_layer,
                use_cross_window=use_cross_window
            ) for i in range(depth)
        ])
        
        if is_encoder:
            self.downsample = DownsamplingBlock(channels, channels, downsample_stride, norm_layer, act_layer)
        else:
            self.upsample = UpsamplingBlock(channels, channels, channels, norm_layer, act_layer)

    def forward(self, point, lengths, perm, upsample_perm=None):
        # Checkpoint individual blocks in encoder only
        for block in self.blocks:
            if self.use_checkpoint and self.training and self.is_encoder:
                point = checkpoint(block, point, lengths, perm, use_reentrant=False)
            else:
                point = block(point, lengths, perm)
        
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
# Backbone
# ==============================================================================
@MODELS.register_module("OPTNet")
class OPTNet(PointModule):
    def __init__(self, in_channels=6, embed_dim=128, enc_depths=(2, 2, 6, 2), 
                 dec_depths=(1, 1, 1, 1), num_heads=4, win_sizes=(32, 32, 32, 32),
                 pool_factors=(2, 2, 2, 2), dropout=0.0, ffn_ratio=3.0,
                 ordering_loss_weight=0.1, ordering_k=16,
                 enable_efpt_pos=True, use_checkpoint=True,
                 use_cross_window_stages=(False, False, True, True),
                 **kwargs):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.pool_factors = pool_factors
        self.ordering_loss_weight = ordering_loss_weight
        self.ordering_k = ordering_k
        self.use_checkpoint = use_checkpoint

     
        
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.patch_embed = PatchEmbedding(3, embed_dim, norm_layer=bn_layer)
        self.feat_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.sorter = PointSorter(embed_dim, hidden_dim=64)
        
        if enable_efpt_pos:
            self.gated_pos = GatedPosEmbedding(3, embed_dim)
        else:
            self.gated_pos = nn.Sequential(
                nn.Linear(3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True)
            )
        
        self.encoders = nn.ModuleList([
            EncodingBlock(
                embed_dim, enc_depths[si], num_heads, win_sizes[si],
                dropout, ffn_ratio, True, 
                pool_factors[si] if si < self.num_stages-1 else 1,
                bn_layer, nn.ReLU,
                use_cross_window=use_cross_window_stages[si],
                use_checkpoint=use_checkpoint  # Pass checkpoint flag
            ) for si in range(self.num_stages)
        ])

        self.decoders = nn.ModuleList([
            EncodingBlock(
                embed_dim, dec_depths[si], num_heads, win_sizes[si],
                dropout, ffn_ratio, False,
                norm_layer=bn_layer, act_layer=nn.ReLU,
                use_cross_window=False,
                use_checkpoint=False  # Never checkpoint decoder
            ) for si in range(self.num_stages-1)
        ])

        
        self.head_norm = nn.LayerNorm(embed_dim)

    def compute_ordering_loss(self, point: Point, scores: torch.Tensor):
        scores = scores.view(-1)
        offset = point.offset if (hasattr(point, "offset") and point.offset is not None) else torch.tensor([scores.numel()], device=scores.device)
        
        # KNN locality loss
        idx = pointops.knn_query(self.ordering_k, point.coord, offset)[0]
        neighbor_scores = scores[idx.long()]
        loss_locality = ((scores.unsqueeze(1) - neighbor_scores) ** 2).sum(dim=1).mean()
        
        # Distribution loss
        sorted_scores, _ = torch.sort(scores)
        target = torch.linspace(0, 1, steps=len(sorted_scores), device=scores.device)
        loss_dist = ((sorted_scores - target) ** 2).mean()
        
        return loss_locality + loss_dist

    def forward(self, data_dict):
        point = Point(data_dict)
        feat_raw = point.feat
        
        # Split features
        color_feat = feat_raw[:, :3] if feat_raw.shape[1] >= 6 else feat_raw
        disp_feat = feat_raw[:, -3:] if feat_raw.shape[1] >= 6 else torch.zeros_like(point.coord)
        
        point.feat = color_feat
        point = self.patch_embed(point)
        x = self.feat_mlp(point.feat)
        point.feat = x
        
        # Compute ordering scores
        scores = self.sorter(point)
        
        # Ordering loss
        total_aux_loss = 0.0
        if self.training and self.ordering_loss_weight > 0:
            total_aux_loss += self.compute_ordering_loss(point, scores) * self.ordering_loss_weight
            
        # Add positional encoding
        point.feat = x + self.gated_pos(disp_feat)
        
        # Get lengths
        offset = point.offset
        lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long() if offset is not None else torch.tensor([point.feat.shape[0]], device=point.feat.device)
        
        # Create hard permutation
        batch = point.batch.long()
        global_scores = scores + batch.float() * (scores.max().detach() + 2.0)
        order = torch.argsort(global_scores)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        perm = Permutation(mode='hard', hard_order=order, hard_inverse=inverse)
        
        initial_perm = perm
        current_perm = perm
        skip_points = []
        
        # Encoder
        for si, encoder in enumerate(self.encoders):
            if si < self.num_stages - 1:
                skip_points.append(Point(point))
                skip_points[-1].feat = point.feat.clone()
            
            # Checkpoint is now handled inside EncodingBlock
            point, lengths = encoder(point, lengths, current_perm)
            
            if si < self.num_stages - 1:
                N_curr = point.feat.shape[0]
                current_perm = Permutation(
                    mode='hard',
                    hard_order=torch.arange(N_curr, device=point.feat.device),
                    hard_inverse=torch.arange(N_curr, device=point.feat.device)
                )

        # Decoder (no changes needed)
        for si in reversed(range(self.num_stages - 1)):
            point['down_parent_point'] = skip_points[si]
            point['down_parent_lengths'] = torch.cat([
                skip_points[si].offset[0].unsqueeze(0),
                skip_points[si].offset[1:] - skip_points[si].offset[:-1]
            ])
            point['down_stride'] = self.pool_factors[si]
            
            # Permutations
            current_N = point.feat.shape[0]
            block_perm = Permutation(
                mode='hard',
                hard_order=torch.arange(current_N, device=point.feat.device),
                hard_inverse=torch.arange(current_N, device=point.feat.device)
            )
            
            if si == 0:
                upsample_perm = initial_perm
            else:
                target_N = skip_points[si].feat.shape[0]
                upsample_perm = Permutation(
                    mode='hard',
                    hard_order=torch.arange(target_N, device=point.feat.device),
                    hard_inverse=torch.arange(target_N, device=point.feat.device)
                )
            
            point, lengths = self.decoders[si](point, lengths, block_perm, upsample_perm)
        
        point.feat = self.head_norm(point.feat)
        
        if self.training:
            point.aux_loss = total_aux_loss
        
        return point


@MODELS.register_module("OPTNetSegmentor")
class OPTNetSegmentor(DefaultSegmentorV2):
    def __init__(self, backbone, criteria, num_classes, backbone_out_channels, 
                 use_adaptive_weighting=True, **kwargs):
        super().__init__(backbone=backbone, criteria=criteria, 
                        num_classes=num_classes, 
                        backbone_out_channels=backbone_out_channels, **kwargs)
        
        self.use_adaptive_weighting = use_adaptive_weighting
        
        if use_adaptive_weighting:
            # Learnable uncertainty parameters (log variance)
            # Initialize with reasonable values: exp(-0.5) ≈ 0.6 weight
            self.log_var_seg = nn.Parameter(torch.tensor([0.0]))
            self.log_var_order = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        
        # Extract auxiliary loss (unweighted)
        ordering_loss = getattr(point, "aux_loss", 0.0)
        if not isinstance(ordering_loss, torch.Tensor):
            ordering_loss = torch.tensor(0.0, device=point.feat.device)
        
        seg_logits = self.seg_head(point.feat)
        
        return_dict = {'point': point} if return_point else {}
        
        if self.training:
            # Compute segmentation loss
            seg_loss = self.criteria(seg_logits, input_dict["segment"])
            
            if self.use_adaptive_weighting and ordering_loss.abs() > 1e-8:
                # Adaptive uncertainty weighting
                # Formula: L = exp(-log_var) * loss + log_var
                # This balances losses automatically and prevents collapse
                
                # Weighted segmentation loss
                precision_seg = torch.exp(-self.log_var_seg)
                weighted_seg_loss = precision_seg * seg_loss + self.log_var_seg
                
                # Weighted ordering loss
                precision_order = torch.exp(-self.log_var_order)
                weighted_ordering_loss = precision_order * ordering_loss + self.log_var_order
                
                # Total loss
                total_loss = weighted_seg_loss + weighted_ordering_loss
                
                # For logging: compute effective weights
                with torch.no_grad():
                    eff_weight_seg = precision_seg.item()
                    eff_weight_order = precision_order.item()
            else:
                # Fixed weighting fallback
                total_loss = seg_loss + ordering_loss
                eff_weight_seg = 1.0
                eff_weight_order = 1.0
            
            # Return losses for logging
            return_dict["loss"] = total_loss
            return_dict["seg_loss"] = seg_loss
            return_dict["ordering_loss"] = ordering_loss
            
            # Log effective weights (will show in tensorboard)
            if self.use_adaptive_weighting:
                return_dict["weight_seg"] = torch.tensor(eff_weight_seg, device=seg_loss.device)
                return_dict["weight_order"] = torch.tensor(eff_weight_order, device=seg_loss.device)
                return_dict["log_var_seg"] = self.log_var_seg
                return_dict["log_var_order"] = self.log_var_order
            
        elif "segment" in input_dict:
            return_dict["loss"] = self.criteria(seg_logits, input_dict["segment"])
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        
        return return_dict
