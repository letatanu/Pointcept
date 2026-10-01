"""
Dual-Stream OPTNet.
- Stream A: Local Window Attention (Points <-> Neighbors in sorted window).
- Stream B: Global Centroid Attention (Window Centroid <-> Other Window Centroids).
- Uses Learnable PointSorter to optimize the 1D ordering (Adaptive Windowing).
- Integrates eFPT Gated Position Embedding.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.cuda.amp import autocast

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None

from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point
from pointcept.models.utils.serialization import encode
from pointcept.models.default import DefaultSegmentorV2

# ==============================================================================
# Helpers
# ==============================================================================

@torch.no_grad()
def compute_intrinsic_ordering_scores(xyz, batch_ids, strategy="hilbert"):
    """ Generates target scores [0, 1] for the PointSorter. """
    device = xyz.device
    batch_ids = batch_ids.long().flatten()
    if batch_ids.numel() == 0: return torch.zeros(0, device=device)

    # 1. Normalize
    num_batches = batch_ids.max().item() + 1
    inf = torch.full((num_batches, 3), float('inf'), device=device)
    neg_inf = torch.full((num_batches, 3), float('-inf'), device=device)
    
    mn = inf.scatter_reduce(0, batch_ids.unsqueeze(1).expand(-1, 3), xyz, reduce="amin", include_self=False)[batch_ids]
    mx = neg_inf.scatter_reduce(0, batch_ids.unsqueeze(1).expand(-1, 3), xyz, reduce="amax", include_self=False)[batch_ids]
    
    norm = (xyz - mn) / (mx - mn).clamp(min=1e-4)
    coords_int = (norm.clamp(0.0, 1.0) * 65535).long()

    # 2. Encode
    base_order = "z" if "z" in strategy else "hilbert"
    spatial_code = encode(coords_int, batch=None, depth=16, order=base_order)
    
    # 3. Rank
    full_code = (batch_ids << 48) | spatial_code.long()
    perm = torch.argsort(full_code)
    rank = torch.zeros_like(perm)
    rank.scatter_(0, perm, torch.arange(perm.size(0), device=device))
    rank = rank.float()
    
    # 4. Normalize Rank
    min_r = inf.new_full((num_batches,), float('inf')).scatter_reduce(0, batch_ids, rank, reduce="amin", include_self=False)[batch_ids]
    max_r = neg_inf.new_full((num_batches,), float('-inf')).scatter_reduce(0, batch_ids, rank, reduce="amax", include_self=False)[batch_ids]
    s = (rank - min_r) / (max_r - min_r).clamp(min=1e-4)
    
    if "inv" in strategy: s = 1.0 - s
    return s

def build_windows_1d(lengths, W, stride, device):
    """
    Partitions the sorted point cloud into windows.
    Returns: indices (M, W), mask (M, W)
    """
    M_idx, M_mask = [], []
    start_global = 0
    lengths_list = lengths.cpu().tolist()
    for L in lengths_list:
        L = int(L)
        if L <= 0: continue
        n_windows = (max(0, L - 1) // stride) + 1
        s = torch.arange(n_windows, device=device) * stride
        b = torch.minimum(torch.tensor(L, device=device), s + W)
        for i in range(n_windows):
            cur_s = s[i].item(); cur_b = b[i].item(); cur_len = cur_b - cur_s
            loc = torch.arange(cur_s, cur_b, device=device) + start_global
            if W - cur_len > 0:
                loc = torch.cat([loc, loc[-1].repeat(W - cur_len)], dim=0)
            mask = torch.zeros(W, dtype=torch.bool, device=device)
            mask[:cur_len] = True
            M_idx.append(loc); M_mask.append(mask)
        start_global += L
    if not M_idx: return torch.empty(0, W, dtype=torch.long, device=device), torch.empty(0, W, dtype=torch.bool, device=device)
    return torch.stack(M_idx, 0), torch.stack(M_mask, 0)

# ==============================================================================
# Modules
# ==============================================================================

class PointSorter(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, xyz):
        if x.shape[0] == 0: return torch.zeros(0, device=x.device)
        return self.sigmoid(self.mlp(torch.cat([x, xyz], dim=1))).squeeze(-1)

class GatedPosEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels, bias=False),
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
        with autocast(enabled=False):
            rel_coord = rel_coord.float()
            gate = self.mlp(rel_coord)
            modulated_coord = rel_coord * gate + gate
            out = self.enc_mlp(modulated_coord)
        return out

class SpCPE(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key=None, kernel_size=3):
        super().__init__()
        assert spconv is not None
        self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, bias=True, indice_key=indice_key)
        self.linear = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, point):
        shortcut = point.feat
        if "sparse_conv_feat" not in point or not hasattr(point.sparse_conv_feat, "replace_feature"):
             point.sparsify()
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.sparse_conv_feat = self.conv(point.sparse_conv_feat)
        feat = point.sparse_conv_feat.features
        feat = self.norm(self.linear(feat))
        point.feat = shortcut + feat
        return point

class CentroidAttention(nn.Module):
    """ Global Stream: Attention between Window Centroids. """
    def __init__(self, channels, num_heads=4, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(channels)

    def forward(self, centroids):
        # centroids: (M, C)
        M, C = centroids.shape
        x = self.norm(centroids)
        qkv = self.qkv(x).reshape(M, 3, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(M, C)
        return centroids + self.proj_drop(self.proj(x))

class DualStreamBlock(nn.Module):
    """
    Contains both Stream A (Window) and Stream B (Centroid).
    """
    def __init__(self, channels, win_size, stride=None, num_heads=4, dropout=0.0, 
                 win_chunk=256, ffn_ratio=2.0, indice_key=None, norm_layer=nn.BatchNorm1d, 
                 enable_spcpe=False):
        super().__init__()
        
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        if self.enable_spcpe:
            self.cpe = SpCPE(channels, channels, indice_key=indice_key)
        else:
            self.cpe = nn.Identity()
        
        self.W = int(win_size); self.stride = int(stride) if stride is not None else self.W
        self.win_chunk = int(win_chunk); self.h = int(num_heads)
        
        # Stream A: Window Attn
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.drop = nn.Dropout(dropout)
        
        # Stream B: Global Centroid Attn
        self.centroid_attn = CentroidAttention(channels, num_heads=num_heads, drop=dropout)
        
        # FFN
        hidden_dim = int(ffn_ratio * channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.LayerNorm(hidden_dim) if norm_layer == nn.LayerNorm else nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout)
        )

    def forward(self, point, lengths):
        if self.enable_spcpe:
            point = self.cpe(point)
        
        x = point.feat
        
        # 1. Window Partitioning (Points sorted by Sorter)
        win_idx, win_mask = build_windows_1d(lengths, self.W, self.stride, device=x.device)
        if win_idx.numel() == 0: return point

        shortcut = x
        x_in = self.norm1(x)
        y = torch.zeros_like(x_in); counts = torch.zeros_like(x_in[:, 0:1]) 
        M, W = win_idx.shape; C = x.shape[1]
        
        # Gather windows for processing: (M, W, C)
        x_windows = x_in[win_idx]
        
        # --- Stream A: Local Window Attention ---
        # We chunk this to avoid OOM if M*W is huge
        for s in range(0, M, self.win_chunk):
            e = min(M, s + self.win_chunk)
            idx = win_idx[s:e]; msk = win_mask[s:e]; m = idx.size(0)
            if m == 0: continue
            
            x_w_chunk = x_in[idx] # (m, W, C)
            qkv = self.qkv(x_w_chunk).reshape(m, W, 3, self.h, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            pad_mask = torch.zeros((m, 1, 1, W), device=x.device, dtype=q.dtype).masked_fill(~msk.view(m, 1, 1, W), float('-inf'))
            
            out_w = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask)
            out_w = out_w.transpose(1, 2).reshape(m, W, C)
            out_proj = self.drop(self.proj(out_w)) * msk.unsqueeze(-1)
            
            # Scatter A back
            y.scatter_add_(0, idx.view(-1, 1).expand(-1, C), out_proj.view(-1, C).to(y.dtype))
            counts.scatter_add_(0, idx.view(-1, 1), msk.view(-1, 1).to(counts.dtype))

        # --- Stream B: Global Centroid Attention ---
        # Calculate centroids (M, C)
        win_counts = win_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Apply mask to zero out padding before sum
        centroids = (x_windows * win_mask.unsqueeze(-1)).sum(dim=1) / win_counts
        
        # Global interaction
        centroids_out = self.centroid_attn(centroids)
        
        # Broadcast back to points
        # (M, C) -> (M, W, C)
        global_ctx = centroids_out.unsqueeze(1).expand(-1, W, -1) * win_mask.unsqueeze(-1)
        
        # Scatter B back
        y.scatter_add_(0, win_idx.view(-1, 1).expand(-1, C), global_ctx.view(-1, C).to(y.dtype))
        
        # FFN
        x_out = shortcut + (y / counts.clamp(min=1.0))
        x_out = x_out + self.drop(self.ffn(self.norm2(x_out)))
        
        point.feat = x_out
        
        if self.enable_spcpe:
             if "sparse_conv_feat" in point and hasattr(point.sparse_conv_feat, "replace_feature"):
                point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_channels, grid_size, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.grid_size = grid_size
        self.embed = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            norm_layer(embed_channels), 
            nn.ReLU()
        )

    def forward(self, point):
        if point.coord.shape[0] > 0:
            point.grid_coord = torch.div(point.coord - point.coord.min(0)[0], self.grid_size, rounding_mode='trunc').int()
        point.feat = self.embed(point.feat)
        return point

# ==============================================================================
# Backbone
# ==============================================================================

@MODELS.register_module("OPTNet")
class OPTNet(PointModule):
    def __init__(self, in_channels=6, embed_dim=128, enc_depths=(2, 2, 6, 2), dec_depths=(1, 1, 1, 1), 
                 num_heads=4, win_sizes=(64, 64, 64, 64), base_grid_size=0.02, pool_factors=(2, 2, 2, 2), 
                 dropout=0.0, ffn_ratio=3.0, win_chunk=256, ordering_loss_weight=0.5, warmup_epoch=1,
                 enable_efpt_pos=True, enable_spcpe=True, geo_input_dim=3):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.base_grid_size = base_grid_size; self.pool_factors = pool_factors
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        self.geo_input_dim = geo_input_dim
        
        self.ordering_loss_weight = ordering_loss_weight
        self.warmup_epoch = warmup_epoch
        self.strategies = ["z", "z-inv", "hilbert", "hilbert-inv"]
        
        norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.patch_embed = PatchEmbedding(3, embed_dim, base_grid_size, norm_layer=norm_layer) 
        self.feat_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.sorter = PointSorter(embed_dim, hidden_dim=64)
        
        if enable_efpt_pos:
            self.gated_pos = GatedPosEmbedding(geo_input_dim, embed_dim)
        else:
            self.gated_pos = nn.Sequential(nn.Linear(geo_input_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
            
        self.enc_blocks = nn.ModuleList(); self.enc_trans = nn.ModuleList()
        curr_dim = embed_dim
        
        for si in range(self.num_stages):
            self.enc_blocks.append(nn.ModuleList([
                DualStreamBlock(curr_dim, win_sizes[si], num_heads=num_heads, 
                         dropout=dropout, win_chunk=win_chunk, ffn_ratio=ffn_ratio, norm_layer=norm_layer, 
                         enable_spcpe=self.enable_spcpe, indice_key=f"enc_s{si}") 
                for _ in range(enc_depths[si])
            ]))
            if si < self.num_stages - 1: self.enc_trans.append(nn.Linear(curr_dim, curr_dim))

        self.dec_blocks = nn.ModuleList(); self.dec_trans = nn.ModuleList()
        for si in range(self.num_stages):
            if si < self.num_stages - 1: self.dec_trans.append(nn.Linear(curr_dim, curr_dim))
            self.dec_blocks.append(nn.ModuleList([
                DualStreamBlock(curr_dim, win_sizes[si], num_heads=num_heads, 
                         dropout=dropout, win_chunk=win_chunk, ffn_ratio=ffn_ratio, norm_layer=norm_layer,
                         enable_spcpe=self.enable_spcpe, indice_key=f"dec_s{si}") 
                for _ in range(dec_depths[si])
            ]))
        self.head_norm = nn.LayerNorm(curr_dim)

    def forward(self, data_dict):
        point = Point(data_dict)
        device = point.feat.device
        
        # --- 1. Split Features ---
        feat_raw = point.feat
        c = 3
        g = self.geo_input_dim
        
        if feat_raw.shape[1] >= (c + g):
            color_feat = feat_raw[:, :c]
            geo_feat = feat_raw[:, c:c+g]
        else:
            color_feat = feat_raw
            geo_feat = torch.zeros((point.coord.shape[0], g), device=device)

        point.feat = color_feat
        point = self.patch_embed(point)
        x = self.feat_mlp(point.feat)
        
        # --- 2. Sorting Supervision ---
        learner_scores = self.sorter(x, point.coord)
        
        current_epoch = data_dict.get("epoch", 0)
        total_aux_loss = 0.0
        
        if self.training:
            strategy = random.choice(self.strategies)
            with torch.no_grad():
                teacher_scores = compute_intrinsic_ordering_scores(point.coord, point.batch, strategy=strategy)
            
            total_aux_loss += F.mse_loss(learner_scores, teacher_scores) * self.ordering_loss_weight
            
            if current_epoch < self.warmup_epoch:
                score_to_sort = teacher_scores
            else:
                score_to_sort = learner_scores
        else:
            score_to_sort = learner_scores

        # Sort
        sort_key = score_to_sort + point.batch.float() * 2.0
        perm = torch.argsort(sort_key)
        
        point.apply_permutation(perm)
        geo_feat = geo_feat[perm]
        x = x[perm]
        
        if self.enable_spcpe: point.sparsify()
        
        # --- 3. Geometry Injection ---
        geom_emb = self.gated_pos(geo_feat)
        point.feat = x + geom_emb
        
        if self.enable_spcpe:
             point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        offset = point.offset
        if offset.numel() > 0:
            lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long()
        else:
            lengths = torch.tensor([point.feat.shape[0]], device=device).long()

        encoder_coords, encoder_lengths, encoder_grid_coords, skips, up_metadata = [], [], [], [], []
        
        # --- Encoder ---
        for si in range(self.num_stages):
            encoder_coords.append(point.coord)
            encoder_lengths.append(lengths)
            encoder_grid_coords.append(point.grid_coord)
            
            for blk in self.enc_blocks[si]:
                point = blk(point, lengths)
            
            skips.append(point.feat)
            
            if si < len(self.enc_trans):
                stride = self.pool_factors[si]
                if lengths.sum() > 0:
                    curr_offset = torch.cat([torch.tensor([0], device=device), torch.cumsum(lengths, dim=0)])
                    batch_indices = []
                    for b in range(len(lengths)):
                        idx = torch.arange(curr_offset[b], curr_offset[b+1], stride, device=device)
                        batch_indices.append(idx)
                    indices = torch.cat(batch_indices)
                else:
                    indices = torch.empty(0, device=device, dtype=torch.long)

                point.apply_permutation(indices)
                point.feat = self.enc_trans[si](point.feat)
                
                up_metadata.append((lengths.clone(), stride, None))
                lengths = (lengths + stride - 1) // stride
                point.offset = torch.cumsum(lengths, dim=0)
                
                if lengths.sum() > 0:
                    batch_list = [torch.full((l,), i, device=device, dtype=torch.long) for i, l in enumerate(lengths)]
                    point.batch = torch.cat(batch_list)
                else:
                    point.batch = torch.empty(0, device=device, dtype=torch.long)
                
                if self.enable_spcpe:
                     point.grid_coord = torch.div(point.coord - point.coord.min(0)[0], self.base_grid_size * (stride**(si+1)), rounding_mode='trunc').int()
                     point.sparsify()

        # --- Decoder ---
        curr_lengths = lengths 
        for si in reversed(range(self.num_stages)):
            if si < len(up_metadata):
                if si < len(self.dec_trans): point.feat = self.dec_trans[si](point.feat)
                
                lengths_high, stride, _ = up_metadata[si]
                offset_low = torch.cat([torch.tensor([0], device=device), torch.cumsum(curr_lengths, dim=0)])
                ids_low_list = []
                valid_mask_list = []
                
                for b in range(len(curr_lengths)):
                    n_high = lengths_high[b].item()
                    n_low = curr_lengths[b].item()
                    
                    if n_high > 0:
                        local_low_idx = torch.arange(n_high, device=device) // stride
                        if n_low > 0:
                            local_low_idx = local_low_idx.clamp(max=n_low - 1)
                            global_low_idx = local_low_idx + offset_low[b]
                            ids_low_list.append(global_low_idx)
                            valid_mask_list.append(torch.ones(n_high, device=device, dtype=torch.bool))
                        else:
                            ids_low_list.append(torch.zeros(n_high, device=device, dtype=torch.long)) 
                            valid_mask_list.append(torch.zeros(n_high, device=device, dtype=torch.bool))
                    else:
                        ids_low_list.append(torch.empty(0, device=device, dtype=torch.long))
                        valid_mask_list.append(torch.empty(0, device=device, dtype=torch.bool))
                
                ids_low = torch.cat(ids_low_list) if len(ids_low_list) > 0 else torch.empty(0, device=device, dtype=torch.long)
                valid_mask = torch.cat(valid_mask_list) if len(valid_mask_list) > 0 else torch.empty(0, device=device, dtype=torch.bool)
                
                if ids_low.numel() > 0 and point.feat.shape[0] > 0:
                    ids_low = ids_low.clamp(max=point.feat.shape[0] - 1)
                    x_up = point.feat[ids_low]
                    x_up = x_up * valid_mask.unsqueeze(-1)
                else:
                    x_up = torch.zeros((ids_low.shape[0], point.feat.shape[1]), device=device)

                x_up = x_up + skips[si]
                
                point.feat = x_up
                point.coord = encoder_coords[si]
                point.grid_coord = encoder_grid_coords[si]
                curr_lengths = lengths_high
                point.offset = torch.cumsum(curr_lengths, dim=0)
                
                # IMPORTANT FIX for Runtime Error in previous turn:
                # Must update point.batch to match the new high-res length
                point.batch = offset2batch(point.offset)
                
                if self.enable_spcpe:
                    point.sparsify()

            for blk in self.dec_blocks[si]:
                point = blk(point, curr_lengths)

        point.feat = self.head_norm(point.feat)
        
        if self.training:
            if not hasattr(point, "aux_loss"): point.aux_loss = 0.0
            point.aux_loss += total_aux_loss
            
        return point

@MODELS.register_module("OPTNetSegmentor")
class OPTNetSegmentor(DefaultSegmentorV2):
    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        
        aux_loss = 0.0
        if isinstance(point, Point) and hasattr(point, "aux_loss"):
            aux_loss = point.aux_loss

        if isinstance(point, Point): feat = point.feat
        else: feat = point
            
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point: return_dict["point"] = point
            
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss + aux_loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
            
        return return_dict