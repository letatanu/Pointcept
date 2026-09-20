"""
OPTNet Adapter for Pointcept with PTv3-style CPE and Learned Ordering
(Dense Version with xCPE and Gradient Safety - No SpConv)
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.cuda.amp import autocast

# Check for spconv
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
    device = xyz.device
    batch_ids = batch_ids.long().flatten()
    
    if batch_ids.numel() == 0:
        return torch.zeros(0, device=device)

    # 1. Normalize and Clamp
    num_batches = batch_ids.max().item() + 1
    inf = torch.full((num_batches, 3), float('inf'), device=device)
    neg_inf = torch.full((num_batches, 3), float('-inf'), device=device)
    
    mn = inf.scatter_reduce(0, batch_ids.unsqueeze(1).expand(-1, 3), xyz, reduce="amin", include_self=False)[batch_ids]
    mx = neg_inf.scatter_reduce(0, batch_ids.unsqueeze(1).expand(-1, 3), xyz, reduce="amax", include_self=False)[batch_ids]
    
    # Increased eps for stability
    rng = (mx - mn).clamp(min=1e-4)
    norm = (xyz - mn) / rng
    coords_int = (norm.clamp(0.0, 1.0) * 65535).long()

    # 2. Encode
    base_order = strategy.replace("-inv", "")
    spatial_code = encode(coords_int, batch=None, depth=16, order=base_order)
    
    # 3. Rank (Local to Batch)
    full_code = (batch_ids << 48) | spatial_code.long()
    
    perm = torch.argsort(full_code)
    rank = torch.zeros_like(perm)
    rank.scatter_(0, perm, torch.arange(perm.size(0), device=device))
    
    # Normalize Rank
    rank = rank.float()
    min_r = inf.new_full((num_batches,), float('inf')).scatter_reduce(0, batch_ids, rank, reduce="amin", include_self=False)[batch_ids]
    max_r = neg_inf.new_full((num_batches,), float('-inf')).scatter_reduce(0, batch_ids, rank, reduce="amax", include_self=False)[batch_ids]
    
    denom = (max_r - min_r).clamp(min=1e-4)
    s = (rank - min_r) / denom
    
    if "inv" in strategy: s = 1.0 - s

    return batch_ids.float() + s

def build_windows_1d(lengths, W, stride, device):
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

class GatedPosEmbedding(nn.Module):
    """
    Gated Position Embedding (eFPT style).
    Expects small, relative coordinates (displacement).
    """
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
        # Force FP32 for stability with geometric inputs
        with autocast(enabled=False):
            rel_coord = rel_coord.float()
            gate = self.mlp(rel_coord)
            modulated_coord = rel_coord * gate + gate
            out = self.enc_mlp(modulated_coord)
        return out

class SpCPE(nn.Module):
    """
    Sparse Convolution CPE with Voxel-Relative Gated Embedding.
    """
    def __init__(self, in_channels, out_channels, indice_key=None, kernel_size=3, use_gated=False, grid_size=0.02):
        super().__init__()
        assert spconv is not None, "spconv is required for SpCPE"
        
        self.use_gated = use_gated
        self.grid_size = grid_size
        
        if use_gated:
            self.gated_pos = GatedPosEmbedding(3, out_channels)

        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=True,
            indice_key=indice_key,
        )
        self.linear = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, point):
        shortcut = point.feat
        
        if "sparse_conv_feat" not in point or not hasattr(point.sparse_conv_feat, "replace_feature"):
             point.sparsify()

        # Update sparse features from dense
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        
        # Inject Internal Voxel-Relative Gated Embedding
        # We use modulo grid_size here because we don't have explicit displacement features for internal layers
        if self.use_gated:
            rel_coord = (point.coord % self.grid_size) - (self.grid_size * 0.5)
            rel_coord_norm = rel_coord / self.grid_size
            pos_emb = self.gated_pos(rel_coord_norm)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.sparse_conv_feat.features + pos_emb)

        # Contextual Conv
        point.sparse_conv_feat = self.conv(point.sparse_conv_feat)
        feat = point.sparse_conv_feat.features
        feat = self.norm(self.linear(feat))
        
        point.feat = shortcut + feat
        return point


class xCPE(nn.Module):
    def __init__(self, in_channels, embed_channels, grid_size=None, indice_key=None):
        super().__init__()
        self.xyz_proj = nn.Sequential(
            nn.Linear(3, embed_channels), 
            nn.LayerNorm(embed_channels), 
            nn.GELU()
        )
        self.pos_enc = nn.Conv1d(
            in_channels, embed_channels, 
            kernel_size=3, padding=1, 
            groups=embed_channels, bias=True
        )
        self.norm = nn.LayerNorm(embed_channels)
    
    def forward(self, point, lengths):
        x = point.feat
        xyz = point.coord
        if x.numel() == 0: return point
        
        x_out_list = []; start = 0
        for l in lengths.tolist():
            l = int(l)
            if l == 0: continue
            
            x_scene = x[start : start + l]
            xyz_scene = xyz[start : start + l]
            
            xyz_centered = xyz_scene - xyz_scene.mean(0, keepdim=True)
            pos_emb = self.xyz_proj(xyz_centered)
            
            feat_in = (x_scene + pos_emb).transpose(0, 1).unsqueeze(0)
            feat_out = self.pos_enc(feat_in)
            
            x_out_list.append(feat_out.squeeze(0).transpose(0, 1))
            start += l
            
        if not x_out_list: return point
        point.feat = self.norm(x + torch.cat(x_out_list, dim=0))
        return point

class GatedxCPE(nn.Module):
    """
    Dense CPE with Gated Voxel-Relative Embedding.
    """
    def __init__(self, in_channels, embed_channels, grid_size=0.02, indice_key=None):
        super().__init__()
        self.grid_size = grid_size
        self.xyz_proj = GatedPosEmbedding(3, embed_channels)
        
        self.pos_enc = nn.Conv1d(
            in_channels, embed_channels, 
            kernel_size=3, padding=1, 
            groups=embed_channels, bias=True
        )
        self.norm = nn.LayerNorm(embed_channels)
    
    def forward(self, point, lengths):
        x = point.feat
        xyz = point.coord
        if x.numel() == 0: return point
        
        rel_coord = (xyz % self.grid_size) - (self.grid_size * 0.5)
        rel_coord_norm = rel_coord / self.grid_size
        
        pos_emb_global = self.xyz_proj(rel_coord_norm)
        
        x_out_list = []; start = 0
        for l in lengths.tolist():
            l = int(l)
            if l == 0: continue
            
            x_scene = x[start : start + l]
            pos_emb_scene = pos_emb_global[start : start + l]
            
            feat_in = (x_scene + pos_emb_scene).transpose(0, 1).unsqueeze(0)
            feat_out = self.pos_enc(feat_in)
            
            x_out_list.append(feat_out.squeeze(0).transpose(0, 1))
            start += l
            
        if not x_out_list: return point
        point.feat = self.norm(x + torch.cat(x_out_list, dim=0))
        return point


class PointScorer(nn.Module):
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


class OPTBlock(nn.Module):
    def __init__(self, channels, win_size, stride=None, num_heads=4, dropout=0.0, 
                 win_chunk=256, ffn_ratio=2.0, indice_key=None, norm_layer=nn.BatchNorm1d, 
                 enable_spcpe=False, enable_gated_cpe=False, grid_size=0.02):
        super().__init__()
        
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        
        if self.enable_spcpe:
            self.cpe = SpCPE(channels, channels, indice_key=indice_key, use_gated=enable_gated_cpe, grid_size=grid_size)
        elif enable_gated_cpe:
            self.cpe = GatedxCPE(channels, channels, grid_size=grid_size, indice_key=indice_key)
        else:
            self.cpe = xCPE(channels, channels, grid_size=grid_size, indice_key=indice_key)
        
        self.W = int(win_size); self.stride = int(stride) if stride is not None else self.W
        self.win_chunk = int(win_chunk); self.h = int(num_heads)
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.drop = nn.Dropout(dropout)
        
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

    def forward(self, point, lengths, perm=None):
        if self.enable_spcpe:
            point = self.cpe(point) 
        else:
            point = self.cpe(point, lengths)
        
        x = point.feat
        x_ord = x[perm] if perm is not None else x 
        
        win_idx, win_mask = build_windows_1d(lengths, self.W, self.stride, device=x.device)
        if win_idx.numel() == 0: return point

        shortcut = x_ord
        x_in = self.norm1(x_ord)
        y = torch.zeros_like(x_in); counts = torch.zeros_like(x_in[:, 0:1]) 
        M, W = win_idx.shape; C = x.shape[1]
        
        for s in range(0, M, self.win_chunk):
            e = min(M, s + self.win_chunk)
            idx = win_idx[s:e]; msk = win_mask[s:e]; m = idx.size(0)
            if m == 0: continue
            
            # Ensure contiguous for SDPA
            qkv = self.qkv(x_in[idx]).reshape(m, W, 3, self.h, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()
            
            pad_mask = torch.zeros((m, 1, 1, W), device=x.device, dtype=q.dtype).masked_fill(~msk.view(m, 1, 1, W), float('-inf'))
            
            out_w = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask)
            
            out_w = out_w.transpose(1, 2).reshape(m, W, C)
            
            out_proj = self.drop(self.proj(out_w)) * msk.unsqueeze(-1)
            y.scatter_add_(0, idx.view(-1, 1).expand(-1, C), out_proj.view(-1, C).to(y.dtype)) 
            counts.scatter_add_(0, idx.view(-1, 1), msk.view(-1, 1).to(counts.dtype))

        x_out = shortcut + (y / counts.clamp(min=1.0))
        x_out = x_out + self.drop(self.ffn(self.norm2(x_out)))
        
        if perm is not None:
            point.feat = x_out[torch.argsort(perm)]
        else:
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
# OPTNet Backbone
# ==============================================================================

@MODELS.register_module("OPTNet")
class OPTNet(PointModule):
    def __init__(self, in_channels=6, embed_dim=128, enc_depths=(2, 2, 6, 2), dec_depths=(1, 1, 1, 1), 
                 num_heads=4, win_sizes=(64, 64, 64, 64), base_grid_size=0.02, pool_factors=(2, 2, 2, 2), 
                 dropout=0.0, ffn_ratio=3.0, win_chunk=256, ordering_loss_weight=0.1, warmup_epoch=0,
                 enable_efpt_pos=True, enable_spcpe=True, enable_gated_cpe=True):
        super().__init__()
        self.num_stages = len(enc_depths); self.ordering_loss_weight = ordering_loss_weight
        self.warmup_epoch = warmup_epoch; self.base_grid_size = base_grid_size; self.pool_factors = pool_factors
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        
        self.all_strategies = ["z", "z-trans", "hilbert", "hilbert-trans"]
        norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.init_scorer = PointScorer(embed_dim, hidden_dim=64)
        
        # Note: in_channels expected to cover color+displacement (3+3=6)
        # We process color (3) and displacement (3) separately
        self.patch_embed = PatchEmbedding(3, embed_dim, base_grid_size, norm_layer=norm_layer) # Input only color
        self.feat_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        
        # STEM: Use Gated Embedding on the displacement features
        if enable_efpt_pos:
            self.geom_mlp = GatedPosEmbedding(3, embed_dim)
        else:
            self.geom_mlp = nn.Sequential(nn.Linear(3, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        
        self.enc_blocks = nn.ModuleList(); self.enc_scorers = nn.ModuleList(); self.enc_trans = nn.ModuleList()
        curr_dim = embed_dim
        curr_grid = base_grid_size
        
        for si in range(self.num_stages):
            self.enc_scorers.append(PointScorer(curr_dim, hidden_dim=64))
            
            self.enc_blocks.append(nn.ModuleList([
                OPTBlock(curr_dim, win_sizes[si], indice_key=f"enc_s{si}", num_heads=num_heads, 
                         dropout=dropout, win_chunk=win_chunk, ffn_ratio=ffn_ratio, norm_layer=norm_layer, 
                         enable_spcpe=self.enable_spcpe, enable_gated_cpe=enable_gated_cpe, 
                         grid_size=curr_grid) 
                for _ in range(enc_depths[si])
            ]))
            
            if si < self.num_stages - 1: 
                self.enc_trans.append(nn.Linear(curr_dim, curr_dim))
                curr_grid *= pool_factors[si]

        self.dec_blocks = nn.ModuleList(); self.dec_scorers = nn.ModuleList(); self.dec_trans = nn.ModuleList()
        
        enc_grids = [base_grid_size]
        for p in pool_factors: enc_grids.append(enc_grids[-1] * p)
        
        for si in range(self.num_stages):
            self.dec_scorers.append(PointScorer(curr_dim, hidden_dim=64))
            if si < self.num_stages - 1: self.dec_trans.append(nn.Linear(curr_dim, curr_dim))
            
            target_grid = enc_grids[si]
            self.dec_blocks.append(nn.ModuleList([
                OPTBlock(curr_dim, win_sizes[si], indice_key=f"dec_s{si}", num_heads=num_heads, 
                         dropout=dropout, win_chunk=win_chunk, ffn_ratio=ffn_ratio, norm_layer=norm_layer,
                         enable_spcpe=self.enable_spcpe, enable_gated_cpe=enable_gated_cpe,
                         grid_size=target_grid) 
                for _ in range(dec_depths[si])
            ]))
            
        self.head_norm = nn.LayerNorm(curr_dim)

    def forward(self, data_dict):
        point = Point(data_dict)
        device = point.feat.device
        
        # --- Handle Input Splitting (Color vs Displacement) ---
        # Assume feat is [Color (3), Displacement (3)] if in_channels=6 in config
        # or [Color (N)] if no displacement. 
        # But user config implies displacement is present.
        
        feat_raw = point.feat
        if feat_raw.shape[1] >= 6:
            # Color is typically first, displacement last or part of it
            color_feat = feat_raw[:, :3]
            disp_feat = feat_raw[:, -3:] # Displacement is p - center
        else:
            # Fallback if config is wrong
            color_feat = feat_raw
            disp_feat = torch.zeros_like(point.coord)

        # Embed Color
        point.feat = color_feat
        point = self.patch_embed(point)
        
        if self.enable_spcpe:
            point.sparsify()
            
        # Embed Geometry using PRE-CALCULATED Displacement (Centroid-Aware)
        # This is exactly what eFPT does with "intra_pos_enc"
        if isinstance(self.geom_mlp, GatedPosEmbedding):
            geom_emb = self.geom_mlp(disp_feat)
        else:
            geom_emb = self.geom_mlp(disp_feat) # Fallback uses disp too for consistency
            
        point.feat = self.feat_mlp(point.feat) + geom_emb
        
        if self.enable_spcpe:
             if "sparse_conv_feat" in point and hasattr(point.sparse_conv_feat, "replace_feature"):
                point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        offset = point.offset
        
        if offset.numel() > 0:
            lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long()
        else:
            lengths = torch.tensor([point.feat.shape[0]], device=device).long()

        current_epoch = data_dict.get("epoch", 0)
        total_aux_loss = 0.0
        
        # --- 1. Initial Sorting ---
        learner_scores = self.init_scorer(point.feat, point.coord)
        if self.training:
            layer_strategies = [random.choice(self.all_strategies) for _ in range(self.num_stages+1)]
            with torch.no_grad():
                if point.coord.shape[0] > 0:
                    teacher_scores = compute_intrinsic_ordering_scores(point.coord, point.batch, strategy=layer_strategies[0])
                else:
                    teacher_scores = learner_scores.detach()
            
            if point.coord.shape[0] > 0:
                total_aux_loss += F.mse_loss(learner_scores, teacher_scores % 1.0) * self.ordering_loss_weight
                score_to_sort = teacher_scores if current_epoch < self.warmup_epoch else learner_scores
            else:
                score_to_sort = learner_scores
        else:
            score_to_sort = learner_scores

        init_perm = torch.argsort(score_to_sort + point.batch.float() * 2.0)
        
        point.apply_permutation(init_perm)
        
        if self.enable_spcpe:
            point.sparsify()

        encoder_coords, encoder_lengths, encoder_grid_coords, skips, up_metadata = [], [], [], [], []
        stride_accum = 1

        # --- Encoder ---
        for si in range(self.num_stages):
            encoder_coords.append(point.coord)
            encoder_lengths.append(lengths)
            encoder_grid_coords.append(point.grid_coord) 
            
            l_scores = self.enc_scorers[si](point.feat, point.coord)
            if self.training:
                with torch.no_grad(): 
                    if point.coord.shape[0] > 0:
                        t_scores = compute_intrinsic_ordering_scores(point.coord, point.batch, strategy=layer_strategies[si+1])
                    else:
                        t_scores = l_scores.detach()
                
                if point.coord.shape[0] > 0:
                    total_aux_loss += F.mse_loss(l_scores, t_scores % 1.0) * self.ordering_loss_weight
                    block_score = t_scores if current_epoch < self.warmup_epoch else l_scores
                else:
                    block_score = l_scores
            else:
                block_score = l_scores
            
            perm = torch.argsort(block_score + point.batch.float() * 2.0)
            
            for blk in self.enc_blocks[si]:
                point = blk(point, lengths, perm=perm)
            
            skips.append(point.feat)
            
            if si < len(self.enc_trans):
                stride = self.pool_factors[si]; stride_accum *= stride
                point.apply_permutation(perm) 
                
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
                
                up_metadata.append((lengths.clone(), stride, perm))
                lengths = (lengths + stride - 1) // stride
                point.offset = torch.cumsum(lengths, dim=0)
                
                if lengths.sum() > 0:
                    batch_list = [torch.full((l,), i, device=device, dtype=torch.long) for i, l in enumerate(lengths)]
                    point.batch = torch.cat(batch_list)
                else:
                    point.batch = torch.empty(0, device=device, dtype=torch.long)
                
                if self.enable_spcpe:
                     point.grid_coord = torch.div(point.coord - point.coord.min(0)[0], self.base_grid_size * stride_accum, rounding_mode='trunc').int()
                     point.sparsify()

        # --- Decoder ---
        curr_lengths = lengths 
        for si in reversed(range(self.num_stages)):
            if si < len(up_metadata):
                if si < len(self.dec_trans): point.feat = self.dec_trans[si](point.feat)
                
                lengths_high, stride, enc_perm = up_metadata[si]
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

                x_up = x_up[torch.argsort(enc_perm)] + skips[si]
                
                point.feat = x_up
                point.coord = encoder_coords[si]
                point.grid_coord = encoder_grid_coords[si]
                
                curr_lengths = lengths_high
                point.offset = torch.cumsum(curr_lengths, dim=0)
                if curr_lengths.sum() > 0:
                    batch_list = [torch.full((l,), i, device=device, dtype=torch.long) for i, l in enumerate(curr_lengths)]
                    point.batch = torch.cat(batch_list)
                else:
                    point.batch = torch.empty(0, device=device, dtype=torch.long)
                
                if self.enable_spcpe:
                    point.sparsify()

            d_scores = self.dec_scorers[si](point.feat, point.coord)
            if self.training:
                with torch.no_grad(): 
                    if point.coord.shape[0] > 0:
                        t_scores = compute_intrinsic_ordering_scores(point.coord, point.batch, strategy=layer_strategies[si+1])
                    else:
                        t_scores = d_scores.detach()
                        
                if point.coord.shape[0] > 0:
                    total_aux_loss += F.mse_loss(d_scores, t_scores % 1.0) * self.ordering_loss_weight
                    perm = torch.argsort((t_scores if current_epoch < self.warmup_epoch else d_scores) + point.batch.float()*2.0)
                else:
                    perm = torch.argsort(d_scores)
            else:
                perm = torch.argsort(d_scores + point.batch.float()*2.0)
            
            for blk in self.dec_blocks[si]:
                point = blk(point, curr_lengths, perm=perm)

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