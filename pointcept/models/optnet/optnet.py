"""
Scatter-OPTNet with Cluster Attention.
- Uses PointSorter -> Quantize -> Scatter to create Adaptive Voxels.
- Implements explicit Query-Key-Value Cluster Attention.
- Integrates eFPT Gated Position Embedding.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.amp import autocast

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

# ==============================================================================
# Modules
# ==============================================================================

class PointSorter(nn.Module):
    """ Learns to assign a continuous score [0,1] for ordering/bucketing. """
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
    """ eFPT Gated Position Embedding """
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
        with autocast("cuda", enabled=False):
            rel_coord = rel_coord.float()
            gate = self.mlp(rel_coord)
            modulated_coord = rel_coord * gate + gate
            out = self.enc_mlp(modulated_coord)
        return out

class SpCPE(nn.Module):
    """ Sparse Convolution CPE """
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

class ClusterAttention(nn.Module):
    """
    Explicit Attention Block for ScatterOPT.
    Performs Point-to-Centroid Attention.
    """
    def __init__(self, channels, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.channels = channels

        self.q_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.k_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.v_proj = nn.Linear(channels, channels, bias=qkv_bias)
        
        # Gating network for Vector Attention (Simulating Softmax behavior in vector space)
        self.attn_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cluster_ids, total_buckets):
        """
        x: (N, C) - Point features
        cluster_ids: (N,) - Bucket ID per point
        total_buckets: int
        """
        N, C = x.shape
        
        # 1. Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Aggregate K and V to form Cluster Context (Centroids)
        # Initialize buffers
        k_sum = torch.zeros((total_buckets, C), device=x.device, dtype=x.dtype)
        v_sum = torch.zeros((total_buckets, C), device=x.device, dtype=x.dtype)
        counts = torch.zeros((total_buckets, 1), device=x.device, dtype=x.dtype)
        
        # Scatter Add
        k_sum.index_add_(0, cluster_ids, k)
        v_sum.index_add_(0, cluster_ids, v)
        counts.index_add_(0, cluster_ids, torch.ones((N, 1), device=x.device, dtype=x.dtype))
        
        # Mean pooling
        counts = counts.clamp(min=1.0)
        k_centroids = k_sum / counts
        v_centroids = v_sum / counts
        
        # 3. Gather Context back to Points
        k_context = k_centroids[cluster_ids]
        v_context = v_centroids[cluster_ids]
        
        # 4. Attention Mechanism (Vector Attention)
        # Standard Scalar Attention: A = Softmax(q @ k.T). But k is 1-to-1 here.
        # Vector Attention: A = Sigmoid(MLP(q - k)) or q * k
        # We use a Hadamard product + Gating to let query select features from the context
        
        # Calculate raw interaction
        interaction = (q * k_context) * self.scale
        
        # Apply gating (learns which parts of the centroid are relevant to this point)
        attn_weights = self.attn_gate(interaction)
        
        # 5. Output
        x_out = attn_weights * v_context
        
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        return x_out

class ScatterOPTBlock(nn.Module):
    """
    Scatter-Gather Block using ClusterAttention.
    """
    def __init__(self, channels, num_heads=4, dropout=0.0, ffn_ratio=2.0, indice_key=None, norm_layer=nn.BatchNorm1d, enable_spcpe=False):
        super().__init__()
        
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        if self.enable_spcpe:
            self.cpe = SpCPE(channels, channels, indice_key=indice_key)
        else:
            self.cpe = nn.Identity()
            
        self.norm1 = nn.LayerNorm(channels)
        
        # REPLACED Aggregator with Explicit Attention
        self.attn = ClusterAttention(channels, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        
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

    def forward(self, point, lengths, sorter_scores):
        if self.enable_spcpe:
            point = self.cpe(point)
            
        x = point.feat
        shortcut = x
        x = self.norm1(x)
        
        # 1. Generate Bucket IDs
        offset = torch.zeros_like(lengths)
        offset[1:] = torch.cumsum(lengths, dim=0)[:-1]
        
        # Adaptive bucketing approx 128 points
        target_win_size = 128
        n_buckets_per_batch = (lengths.float() / target_win_size).ceil().long()
        
        batch_ids = point.batch
        num_buckets_point = n_buckets_per_batch[batch_ids]
        batch_offset_point = torch.cat([torch.tensor([0], device=x.device), torch.cumsum(n_buckets_per_batch, dim=0)[:-1]])[batch_ids]
        
        # Modulo Quantization (as requested) + Batch Offset
        # sorter_scores are [0, 1]. Map to [0, n_buckets-1]
        local_bucket = (sorter_scores * num_buckets_point.float()).long()
        local_bucket = torch.min(local_bucket, num_buckets_point - 1)
        
        global_bucket = local_bucket + batch_offset_point
        total_buckets = n_buckets_per_batch.sum().item()
        
        # 2. Cluster Attention
        x_out = self.attn(x, global_bucket, total_buckets)
        
        # 3. FFN
        x = shortcut + x_out
        x = x + self.ffn(self.norm2(x))
        
        point.feat = x
        
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
# Scatter-OPTNet Backbone
# ==============================================================================

@MODELS.register_module("OPTNet")
class OPTNet(PointModule):
    def __init__(self, in_channels=6, embed_dim=128, enc_depths=(2, 2, 6, 2), dec_depths=(1, 1, 1, 1), 
                 num_heads=4, win_sizes=(64, 64, 64, 64), base_grid_size=0.02, pool_factors=(2, 2, 2, 2), 
                 dropout=0.0, ffn_ratio=3.0, win_chunk=256, ordering_loss_weight=0.5, warmup_epoch=1,
                 enable_efpt_pos=True, enable_spcpe=True):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.base_grid_size = base_grid_size; self.pool_factors = pool_factors
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        
        self.ordering_loss_weight = ordering_loss_weight
        self.warmup_epoch = warmup_epoch
        self.strategies = ["z", "z-inv", "hilbert", "hilbert-inv"]
        
        norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.patch_embed = PatchEmbedding(3, embed_dim, base_grid_size, norm_layer=norm_layer) 
        self.feat_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        
        # Learnable Sorter
        self.sorter = PointSorter(embed_dim, hidden_dim=64)
        
        # eFPT Geometry
        if enable_efpt_pos:
            self.gated_pos = GatedPosEmbedding(3, embed_dim)
        else:
            self.gated_pos = nn.Sequential(nn.Linear(3, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
            
        self.enc_blocks = nn.ModuleList(); self.enc_trans = nn.ModuleList()
        curr_dim = embed_dim
        
        for si in range(self.num_stages):
            self.enc_blocks.append(nn.ModuleList([
                ScatterOPTBlock(curr_dim, num_heads=num_heads, dropout=dropout, ffn_ratio=ffn_ratio, 
                                norm_layer=norm_layer, enable_spcpe=self.enable_spcpe, indice_key=f"enc_s{si}") 
                for _ in range(enc_depths[si])
            ]))
            if si < self.num_stages - 1: self.enc_trans.append(nn.Linear(curr_dim, curr_dim))

        self.dec_blocks = nn.ModuleList(); self.dec_trans = nn.ModuleList()
        for si in range(self.num_stages):
            if si < self.num_stages - 1: self.dec_trans.append(nn.Linear(curr_dim, curr_dim))
            self.dec_blocks.append(nn.ModuleList([
                ScatterOPTBlock(curr_dim, num_heads=num_heads, dropout=dropout, ffn_ratio=ffn_ratio,
                                norm_layer=norm_layer, enable_spcpe=self.enable_spcpe, indice_key=f"dec_s{si}") 
                for _ in range(dec_depths[si])
            ]))
        self.head_norm = nn.LayerNorm(curr_dim)

    def forward(self, data_dict):
        point = Point(data_dict)
        device = point.feat.device
        
        feat_raw = point.feat
        if feat_raw.shape[1] >= 6:
            color_feat = feat_raw[:, :3]
            disp_feat = feat_raw[:, -3:]
        else:
            color_feat = feat_raw
            disp_feat = torch.zeros_like(point.coord)

        point.feat = color_feat
        point = self.patch_embed(point)
        x = self.feat_mlp(point.feat)
        
        # 1. Predict Scores
        learner_scores = self.sorter(x, point.coord)
        
        current_epoch = data_dict.get("epoch", 0)
        total_aux_loss = 0.0
        
        if self.training:
            strategy = random.choice(self.strategies)
            with torch.no_grad():
                teacher_scores = compute_intrinsic_ordering_scores(point.coord, point.batch, strategy=strategy)
            
            total_aux_loss += F.mse_loss(learner_scores, teacher_scores) * self.ordering_loss_weight
            
            if current_epoch < self.warmup_epoch:
                bucket_scores = teacher_scores
            else:
                bucket_scores = learner_scores
        else:
            bucket_scores = learner_scores

        # 2. Geometry Injection
        geom_emb = self.gated_pos(disp_feat)
        point.feat = x + geom_emb
        
        if self.enable_spcpe: point.sparsify() 

        offset = point.offset
        if offset.numel() > 0:
            lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long()
        else:
            lengths = torch.tensor([point.feat.shape[0]], device=device).long()

        encoder_coords, encoder_lengths, encoder_scores, skips, up_metadata = [], [], [], [], []
        
        # No Permutation - Just slice downsampling
        curr_scores = bucket_scores
        
        # --- Encoder ---
        for si in range(self.num_stages):
            encoder_coords.append(point.coord)
            encoder_lengths.append(lengths)
            encoder_scores.append(curr_scores)
            
            for blk in self.enc_blocks[si]:
                point = blk(point, lengths, curr_scores)
            
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

                point.feat = point.feat[indices]
                point.coord = point.coord[indices]
                point.batch = point.batch[indices]
                curr_scores = curr_scores[indices]
                
                point.feat = self.enc_trans[si](point.feat)
                
                up_metadata.append((lengths.clone(), stride, indices))
                
                lengths = (lengths + stride - 1) // stride
                point.offset = torch.cumsum(lengths, dim=0)
                
                if self.enable_spcpe:
                     point.grid_coord = torch.div(point.grid_coord[indices], stride, rounding_mode='trunc').int()
                     point.sparsify()

        # --- Decoder ---
        curr_lengths = lengths 
        for si in reversed(range(self.num_stages)):
            if si < len(up_metadata):
                if si < len(self.dec_trans): point.feat = self.dec_trans[si](point.feat)
                
                lengths_high, stride, indices_down = up_metadata[si]
                
                x_up = torch.zeros((encoder_scores[si].shape[0], point.feat.shape[1]), device=device, dtype=point.feat.dtype)
                
                # Simple Index Scatter Upsampling (since no permutation occurred)
                # Map low-res indices back to high-res grid
                offset_low = torch.cat([torch.tensor([0], device=device), torch.cumsum(curr_lengths, dim=0)])
                idx_map_list = []
                for b in range(len(lengths)):
                    l_low = curr_lengths[b]
                    l_high = lengths_high[b]
                    if l_high == 0: 
                        idx_map_list.append(torch.empty(0, device=device, dtype=torch.long))
                        continue
                    local_idx = torch.arange(l_high, device=device) // stride
                    local_idx = local_idx.clamp(max=l_low - 1)
                    idx_map_list.append(local_idx + offset_low[b])
                
                global_map = torch.cat(idx_map_list)
                x_up = point.feat[global_map]

                x_up = x_up + skips[si]
                
                point.feat = x_up
                point.coord = encoder_coords[si]
                point.batch = offset2batch(torch.cumsum(lengths_high, dim=0))
                
                point.grid_coord = torch.div(point.coord - point.coord.min(0)[0], self.base_grid_size * (stride**si), rounding_mode='trunc').int()
                
                curr_scores = encoder_scores[si]
                curr_lengths = lengths_high
                point.offset = torch.cumsum(curr_lengths, dim=0)
                
                if self.enable_spcpe:
                    point.sparsify()

            for blk in self.dec_blocks[si]:
                point = blk(point, curr_lengths, curr_scores)

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