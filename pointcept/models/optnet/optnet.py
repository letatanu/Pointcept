"""
OPTNet Adapter for Pointcept with PTv3-style CPE and Learned Ordering
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point
from pointcept.models.utils.serialization import encode
from pointcept.models.default import DefaultSegmentorV2

import spconv.pytorch as spconv


# ==============================================================================
# Helpers
# ==============================================================================

@torch.no_grad()
def compute_intrinsic_ordering_scores(xyz, batch_ids, strategy="hilbert"):
    device = xyz.device
    batch_ids = batch_ids.long().flatten()
    
    # ... [Normalization Code from previous answer] ...
    # 1. Normalize and Clamp (Critical for stability)
    num_batches = batch_ids.max().item() + 1
    inf = torch.full((num_batches, 3), float('inf'), device=device)
    neg_inf = torch.full((num_batches, 3), float('-inf'), device=device)
    
    mn = inf.scatter_reduce(0, batch_ids.unsqueeze(1).expand(-1, 3), xyz, reduce="amin", include_self=False)[batch_ids]
    mx = neg_inf.scatter_reduce(0, batch_ids.unsqueeze(1).expand(-1, 3), xyz, reduce="amax", include_self=False)[batch_ids]
    
    norm = (xyz - mn) / (mx - mn).clamp(min=1e-6)
    coords_int = (norm.clamp(0.0, 1.0) * 65535).long() # Strict clamp

    # 2. Encode
    base_order = strategy.replace("-inv", "")
    spatial_code = encode(coords_int, batch=None, depth=16, order=base_order)
    
    # 3. Rank (Local to Batch)
    # We add batch_ids << 48 to ensure the sort is primarily by Batch, secondly by Space
    full_code = (batch_ids << 48) | spatial_code.long()
    
    perm = torch.argsort(full_code)
    rank = torch.zeros_like(perm)
    rank.scatter_(0, perm, torch.arange(perm.size(0), device=device))
    
    # Normalize Rank to [0, 1]
    rank = rank.float()
    min_r = inf.new_full((num_batches,), float('inf')).scatter_reduce(0, batch_ids, rank, reduce="amin", include_self=False)[batch_ids]
    max_r = neg_inf.new_full((num_batches,), float('-inf')).scatter_reduce(0, batch_ids, rank, reduce="amax", include_self=False)[batch_ids]
    
    s = (rank - min_r) / (max_r - min_r + 1e-6)
    if "inv" in strategy: s = 1.0 - s

    # RETURN: Batch ID + Score
    # This prevents the downstream 'apply_permutation' from mixing batches
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
        return self.sigmoid(self.mlp(torch.cat([x, xyz], dim=1))).squeeze(-1)

class OPTBlock(nn.Module):
    def __init__(self, channels, win_size, stride=None, num_heads=4, dropout=0.0, win_chunk=256, ffn_ratio=2.0, indice_key=None, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.cpe = spconv.SparseSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=indice_key),
            nn.Linear(channels, channels),
            norm_layer(channels)
        )
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
        # 1. Apply CPE
        # Ensure sparse structure exists and is up to date
        if "sparse_conv_feat" not in point: 
            point.sparsify() 
        
        # Use 'sparse_conv_feat' as defined in Point class
        point.sparse_conv_feat = self.cpe(point.sparse_conv_feat)
        point.feat = point.sparse_conv_feat.features
        
        # 2. Window Attention
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
            qkv = self.qkv(x_in[idx]).reshape(m, W, 3, self.h, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            pad_mask = torch.zeros((m, 1, 1, W), device=x.device, dtype=q.dtype).masked_fill(~msk.view(m, 1, 1, W), float('-inf'))
            out_w = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask).transpose(1, 2).reshape(m, W, C)
            out_proj = self.drop(self.proj(out_w)) * msk.unsqueeze(-1)
            y.scatter_add_(0, idx.view(-1, 1).expand(-1, C), out_proj.view(-1, C).to(y.dtype)) 
            counts.scatter_add_(0, idx.view(-1, 1), msk.view(-1, 1).to(counts.dtype))

        x_out = shortcut + (y / counts.clamp(min=1.0))
        x_out = x_out + self.drop(self.ffn(self.norm2(x_out)))
        
        if perm is not None:
            point.feat = x_out[torch.argsort(perm)]
        else:
            point.feat = x_out
            
        return point

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_channels, grid_size, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.grid_size = grid_size
        self.embed = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, embed_channels, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_layer(embed_channels), nn.ReLU()
        )

    def forward(self, point):
        # 1. Quantize Coords -> point.grid_coord
        grid_coord = torch.div(point.coord - point.coord.min(0)[0], self.grid_size, rounding_mode='trunc').int()
        point.grid_coord = grid_coord
        
        # 2. Build Sparse Tensor (utilizing point.sparsify)
        # This creates point['sparse_conv_feat']
        point.sparsify()
        
        # 3. Run Conv on 'sparse_conv_feat'
        point.sparse_conv_feat = self.embed(point.sparse_conv_feat)
        
        # 4. Save Features
        point.feat = point.sparse_conv_feat.features
        # Store shape so sparsify() can reuse it if needed (though sparsify checks self.sparse_shape)
        point.sparse_shape = point.sparse_conv_feat.spatial_shape
        
        return point

# ==============================================================================
# OPTNet Backbone
# ==============================================================================

@MODELS.register_module("OPTNet")
class OPTNet(PointModule):
    def __init__(self, in_channels=6, embed_dim=128, enc_depths=(2, 2, 6, 2), dec_depths=(1, 1, 1, 1), 
                 num_heads=4, win_sizes=(64, 64, 64, 64), base_grid_size=0.02, pool_factors=(2, 2, 2, 2), 
                 dropout=0.0, ffn_ratio=3.0, win_chunk=256, ordering_loss_weight=0.1, warmup_epoch=0):
        super().__init__()
        self.num_stages = len(enc_depths); self.ordering_loss_weight = ordering_loss_weight
        self.warmup_epoch = warmup_epoch; self.base_grid_size = base_grid_size; self.pool_factors = pool_factors
        self.all_strategies = ["z", "z-trans", "hilbert", "hilbert-trans"]
        norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.init_scorer = PointScorer(embed_dim, hidden_dim=64)
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, base_grid_size, norm_layer=norm_layer)
        self.feat_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.geom_mlp = nn.Sequential(nn.Linear(3, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        
        self.enc_blocks = nn.ModuleList(); self.enc_scorers = nn.ModuleList(); self.enc_trans = nn.ModuleList()
        curr_dim = embed_dim
        for si in range(self.num_stages):
            self.enc_scorers.append(PointScorer(curr_dim, hidden_dim=64))
            self.enc_blocks.append(nn.ModuleList([
                OPTBlock(curr_dim, win_sizes[si], indice_key=f"enc_s{si}", num_heads=num_heads, 
                         dropout=dropout, win_chunk=win_chunk, ffn_ratio=ffn_ratio, norm_layer=norm_layer) 
                for _ in range(enc_depths[si])
            ]))
            if si < self.num_stages - 1: self.enc_trans.append(nn.Linear(curr_dim, curr_dim))

        self.dec_blocks = nn.ModuleList(); self.dec_scorers = nn.ModuleList(); self.dec_trans = nn.ModuleList()
        for si in range(self.num_stages):
            self.dec_scorers.append(PointScorer(curr_dim, hidden_dim=64))
            if si < self.num_stages - 1: self.dec_trans.append(nn.Linear(curr_dim, curr_dim))
            self.dec_blocks.append(nn.ModuleList([
                OPTBlock(curr_dim, win_sizes[si], indice_key=f"dec_s{si}", num_heads=num_heads, 
                         dropout=dropout, win_chunk=win_chunk, ffn_ratio=ffn_ratio, norm_layer=norm_layer) 
                for _ in range(dec_depths[si])
            ]))
        self.head_norm = nn.LayerNorm(curr_dim)

    def _ensure_sparse_validity(self, point, stride_factor=1.0):
        grid_size = self.base_grid_size * stride_factor
        point.grid_coord = torch.div(point.coord - point.coord.min(0)[0], grid_size, rounding_mode='trunc').int()
        
        # Invalidate old shape since grid_coord changed
        if "sparse_shape" in point: del point["sparse_shape"]
        
        # Rebuild sparse_conv_feat
        point.sparsify() 
        return point

    def forward(self, data_dict):
        point = Point(data_dict)
        device = point.feat.device
        point = self.patch_embed(point)
        point.feat = self.feat_mlp(point.feat) + self.geom_mlp(point.coord)
        offset = point.offset
        lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long()
        current_epoch = data_dict.get("epoch", 0)
        total_aux_loss = 0.0
        
        # --- 1. Initial Sorting ---
        learner_scores = self.init_scorer(point.feat, point.coord)
        if self.training:
            # Randomly select one strategy for each layer index
            layer_strategies = [random.choice(self.all_strategies) for _ in range(self.num_stages+1)]
            with torch.no_grad():
                teacher_scores = compute_intrinsic_ordering_scores(point.coord, point.batch, strategy=layer_strategies[0])
            total_aux_loss += F.mse_loss(learner_scores, teacher_scores % 1.0) * self.ordering_loss_weight
            score_to_sort = teacher_scores if current_epoch < self.warmup_epoch else learner_scores
        else:
            score_to_sort = learner_scores

        init_perm = torch.argsort(score_to_sort + point.batch.float() * 2.0)
        point.apply_permutation(init_perm)
        point = self._ensure_sparse_validity(point)

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
                    t_scores = compute_intrinsic_ordering_scores(point.coord, point.batch, strategy=layer_strategies[si+1])
                total_aux_loss += F.mse_loss(l_scores, t_scores % 1.0) * self.ordering_loss_weight
                block_score = t_scores if current_epoch < self.warmup_epoch else l_scores
            else:
                block_score = l_scores
            
            perm = torch.argsort(block_score + point.batch.float() * 2.0)
            
            for blk in self.enc_blocks[si]:
                point = blk(point, lengths, perm=perm)
            
            skips.append(point.feat)
            
            if si < len(self.enc_trans):
                stride = self.pool_factors[si]; stride_accum *= stride
                point.apply_permutation(perm) 
                
                indices = torch.arange(0, point.feat.shape[0], stride, device=device)                
                point.apply_permutation(indices)
                
                point.feat = self.enc_trans[si](point.feat)
                up_metadata.append((lengths.clone(), stride, perm))
                lengths = (lengths + stride - 1) // stride
                point.offset = torch.cumsum(lengths, dim=0)
                point = self._ensure_sparse_validity(point, stride_accum)

        # --- Decoder ---
        curr_lengths = lengths
        for si in reversed(range(self.num_stages)):
            if si < len(up_metadata):
                if si < len(self.dec_trans): point.feat = self.dec_trans[si](point.feat)
                lengths_high, stride, enc_perm = up_metadata[si]
                
                ids_high = torch.arange(lengths_high.sum(), device=device)
                ids_low = (ids_high // stride).clamp(max=point.feat.shape[0]-1)
                x_up = point.feat[ids_low]
                x_up = x_up[torch.argsort(enc_perm)] + skips[si]
                
                point.feat = x_up
                point.coord = encoder_coords[si]
                point.grid_coord = encoder_grid_coords[si]
                if "sparse_shape" in point: del point["sparse_shape"]
                
                point.sparsify() 
                point.sparse_conv_feat.replace_feature(x_up)
                
                curr_lengths = encoder_lengths[si]
                point.offset = torch.cumsum(curr_lengths, dim=0)
                point.batch = torch.arange(len(curr_lengths), device=device).repeat_interleave(curr_lengths)

            d_scores = self.dec_scorers[si](point.feat, point.coord)
            if self.training:
                with torch.no_grad(): 
                    t_scores = compute_intrinsic_ordering_scores(point.coord, point.batch, strategy=layer_strategies[si+1])
                total_aux_loss += F.mse_loss(d_scores, t_scores % 1.0) * self.ordering_loss_weight
                perm = torch.argsort((t_scores if current_epoch < self.warmup_epoch else d_scores) + point.batch.float()*2.0)
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