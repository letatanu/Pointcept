"""
Optimized Scatter-OPTNet with Self-Supervised Ordering.
Features:
- Self-supervised ordering using k-NN locality
- Contrastive learning for far-away points
- Separate DownsamplingBlock and UpsamplingBlock
- Pool modes: strided, mean, max
- Mixed precision training safe
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.amp import autocast
import spconv.pytorch as spconv
import pointops


from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point
from pointcept.models.default import DefaultSegmentorV2

# ==============================================================================
# Self-Supervised Ordering Loss
# ==============================================================================

class SelfSupervisedOrderingLoss(nn.Module):
    """
    Self-supervised loss for learning ordering scores.
    Uses k-NN locality and contrastive learning.
    
    Key principles:
    1. Neighboring points should have similar scores (locality)
    2. Far-away points should have different scores (contrastive)
    3. Scores should span the full [0, 1] range (distribution)
    """
    def __init__(self, k_near=8, k_far=16, temp_locality=0.1, temp_contrastive=0.5):
        super().__init__()
        self.k_near = k_near
        self.k_far = k_far
        self.temp_locality = temp_locality
        self.temp_contrastive = temp_contrastive
    
    def forward(self, scores, coords, batch_ids, offset=None):
        """
        Args:
            scores: (N,) predicted ordering scores in [0, 1]
            coords: (N, 3) point coordinates
            batch_ids: (N,) batch indices
            offset: (B,) cumulative point counts per batch
        
        Returns:
            total_loss: scalar loss value
            loss_dict: dict with individual loss components
        """
        N = scores.shape[0]
        device = scores.device
        dtype = scores.dtype
        
        if N < self.k_near:
            return torch.tensor(0.0, device=device, dtype=dtype), {}
        
        # Prepare offset if not provided
        if offset is None:
            batch_ids_long = batch_ids.long()
            num_batches = batch_ids_long.max().item() + 1
            lengths = torch.zeros(num_batches, device=device, dtype=torch.long)
            lengths.scatter_add_(0, batch_ids_long, torch.ones_like(batch_ids_long))
            offset = torch.cumsum(lengths, dim=0).int()
        else:
            offset = offset.int()
        
        # 1. Locality Loss
        loss_locality = self._compute_locality_loss(scores, coords, offset)
        
        # 2. Contrastive Loss
        loss_contrastive = self._compute_contrastive_loss(scores, coords, offset)
        
        # # 3. Distribution Loss
        loss_distribution = self._compute_distribution_loss(scores, batch_ids)
        # loss_distribution = 0.0  # Disabled for stability, can be re-enabled with proper tuning
        # 4. Smoothness Loss
        loss_smoothness = self._compute_smoothness_loss(scores, coords, offset)
        # loss_smoothness = 0.0  # Disabled for stability, can be re-enabled with proper tuning
        
        # Combine losses
        total_loss = (
            1.0 * loss_locality + 
            0.5 * loss_contrastive + 
            0.3 * loss_distribution +
            0.2 * loss_smoothness
        )
        
        loss_dict = {
            'locality': loss_locality.item(),
            'contrastive': loss_contrastive.item(),
            'distribution': loss_distribution.item() if isinstance(loss_distribution, torch.Tensor) else loss_distribution,
            'smoothness': loss_smoothness.item() if isinstance(loss_smoothness, torch.Tensor) else loss_smoothness
        }
        
        return total_loss, loss_dict
    
    def _compute_locality_loss(self, scores, coords, offset):
        """Neighboring points should have similar scores."""
        device = scores.device
        dtype = scores.dtype
        
        try:
            k_actual = min(self.k_near, scores.shape[0] - 1)
            if k_actual < 1:
                return torch.tensor(0.0, device=device, dtype=dtype)
            
            # if pointops is not None:
            idx = pointops.knn_query(k_actual, coords.contiguous(), offset)[0]
            # else:
            #     idx = self._manual_knn(coords, offset, k_actual)
            
            neighbor_scores = scores[idx.long()]
            score_diff = (scores.unsqueeze(1) - neighbor_scores).abs()
            
            # Weighted by distance
            neighbor_coords = coords[idx.long()]
            dists = torch.norm(coords.unsqueeze(1) - neighbor_coords, dim=2)
            weights = torch.exp(-dists / self.temp_locality)
            
            loss = (weights * score_diff.pow(2)).sum() / weights.sum().clamp(min=1e-8)
            
        except Exception as e:
            loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        return loss
    
    def _compute_contrastive_loss(self, scores, coords, offset):
        """Far-away points should have different scores."""
        device = scores.device
        dtype = scores.dtype
        N = scores.shape[0]
        
        k_near = min(self.k_near, N - 1)
        k_far = min(self.k_far, N - 1)
        
        if k_far <= k_near:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        idx_far_all = pointops.knn_query(k_far, coords.contiguous(), offset)[0]
        idx_near = idx_far_all[:, :k_near]
        idx_far = idx_far_all[:, k_near:]
 
        # Positive pairs (near): high similarity
        scores_near = scores[idx_near.long()]
        sim_pos = 1.0 - (scores.unsqueeze(1) - scores_near).abs()
        
        # Negative pairs (far): low similarity
        scores_far = scores[idx_far.long()]
        sim_neg = 1.0 - (scores.unsqueeze(1) - scores_far).abs()
        
        # Contrastive loss
        logits_pos = sim_pos / self.temp_contrastive
        logits_neg = sim_neg / self.temp_contrastive
        
        loss_pos = -torch.log(torch.sigmoid(logits_pos) + 1e-8).mean()
        loss_neg = -torch.log(1 - torch.sigmoid(logits_neg) + 1e-8).mean()
        
        loss = loss_pos + loss_neg
        

        
        return loss
    
    def _compute_distribution_loss(self, scores, batch_ids):
        """Scores should be uniformly distributed in [0, 1]."""
        device = scores.device
        dtype = scores.dtype
        batch_ids_long = batch_ids.long()
        num_batches = batch_ids_long.max().item() + 1
        
        total_loss = 0.0
        count = 0
        
        for b in range(num_batches):
            mask = (batch_ids_long == b)
            if mask.sum() < 2:
                continue
            
            scores_b = scores[mask]
            sorted_scores, _ = torch.sort(scores_b)
            
            n = sorted_scores.shape[0]
            target = torch.linspace(0, 1, n, device=device, dtype=dtype)
            
            loss_b = F.mse_loss(sorted_scores, target)
            total_loss += loss_b
            count += 1
        
        return total_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=device, dtype=dtype)
    
    def _compute_smoothness_loss(self, scores, coords, offset):
        """Encourage smooth score transitions."""
        device = scores.device
        dtype = scores.dtype
        k_smooth = min(8, scores.shape[0] - 1)
        if k_smooth < 1:
            return torch.tensor(0.0, device=device, dtype=dtype)

        idx = pointops.knn_query(k_smooth, coords.contiguous(), offset)[0]
        neighbor_scores = scores[idx.long()]
        neighbor_mean = neighbor_scores.mean(dim=1)
        
        loss = F.mse_loss(scores, neighbor_mean)
            

        
        return loss
    


# ==============================================================================
# Core Modules
# ==============================================================================

class PointSorter(nn.Module):
    """Learns to assign a continuous score [0,1] with self-supervised training."""
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
        
        self.ss_loss = SelfSupervisedOrderingLoss(
            k_near=16, 
            k_far=32,
            temp_locality=0.1,
            temp_contrastive=0.5
        )

    def forward(self, x, xyz):
        if x.shape[0] == 0: 
            return torch.zeros(0, device=x.device)
        scores = self.sigmoid(self.mlp(torch.cat([x, xyz], dim=1))).squeeze(-1)
        return scores
    
    def compute_loss(self, scores, coords, batch_ids, offset=None):
        """Compute self-supervised loss."""
        return self.ss_loss(scores, coords, batch_ids, offset)


class GatedPosEmbedding(nn.Module):
    """eFPT Gated Position Embedding"""
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
    """Sparse Convolution CPE"""
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
    """Point-to-Centroid Attention with Dual Aggregation - Fixed for gradient computation."""
    def __init__(self, channels, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.channels = channels

        self.q_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.k_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.v_proj = nn.Linear(channels, channels, bias=qkv_bias)
        
        self.attn_gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        
        self.v_combine = nn.Linear(channels * 2, channels)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, x, cluster_ids, total_buckets):
        """
        x: (N, C) - Point features
        cluster_ids: (N,) - Bucket ID per point (long tensor)
        total_buckets: int - Total number of buckets
        """
        N, C = x.shape
        
        if N == 0:
            return x
        
        device = x.device
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Get dtype from k/v after projection
        dtype = k.dtype
        
        # Create aggregation buffers
        k_sum = torch.zeros((total_buckets, C), device=device, dtype=dtype)
        v_sum = torch.zeros((total_buckets, C), device=device, dtype=dtype)
        k_max = torch.full((total_buckets, C), float('-inf'), device=device, dtype=dtype)
        v_max = torch.full((total_buckets, C), float('-inf'), device=device, dtype=dtype)
        counts = torch.zeros((total_buckets,), device=device, dtype=dtype)
        
        # Scatter operations
        k_sum.index_add_(0, cluster_ids, k)
        v_sum.index_add_(0, cluster_ids, v)
        counts.index_add_(0, cluster_ids, torch.ones(N, device=device, dtype=dtype))
        
        # Scatter max
        k_max = k_max.scatter_reduce(0, cluster_ids.unsqueeze(1).expand(-1, C), k, 
                                     reduce="amax", include_self=False)
        v_max = v_max.scatter_reduce(0, cluster_ids.unsqueeze(1).expand(-1, C), v, 
                                     reduce="amax", include_self=False)
        
        # CRITICAL FIX: Handle empty buckets WITHOUT in-place operations
        # Don't modify k_max/v_max in-place! Create new tensors instead
        mask = (counts == 0)
        if mask.any():
            # Use where() instead of in-place assignment
            k_max = torch.where(mask.unsqueeze(1), torch.zeros_like(k_max), k_max)
            v_max = torch.where(mask.unsqueeze(1), torch.zeros_like(v_max), v_max)

        # Mean pooling
        counts_safe = counts.clamp(min=1.0).unsqueeze(1)
        k_mean = k_sum / counts_safe
        v_mean = v_sum / counts_safe
        
        # Concatenate Mean + Max
        k_centroids = torch.cat([k_mean, k_max], dim=-1)  # (Buckets, 2*C)
        v_centroids = torch.cat([v_mean, v_max], dim=-1)  # (Buckets, 2*C)
        
        # Gather context back to points
        k_context = k_centroids[cluster_ids]  # (N, 2*C)
        v_context = v_centroids[cluster_ids]  # (N, 2*C)
        
        # Attention mechanism
        q_expanded = torch.cat([q, q], dim=-1)
        interaction = (q_expanded * k_context) * self.scale
        attn_weights = self.attn_gate(interaction)
        
        # Output
        v_combined = self.v_combine(v_context)
        x_out = attn_weights * v_combined
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        return x_out


class ScatterOPTBlock(nn.Module):
    """Scatter-Gather Block using ClusterAttention."""
    def __init__(self, channels, num_heads=4, dropout=0.0, ffn_ratio=2.0, indice_key=None, 
                 norm_layer=nn.BatchNorm1d, enable_spcpe=False):
        super().__init__()
        
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        if self.enable_spcpe:
            self.cpe = SpCPE(channels, channels, indice_key=indice_key)
        else:
            self.cpe = nn.Identity()
            
        self.norm1 = nn.LayerNorm(channels)
        self.attn = ClusterAttention(channels, num_heads=num_heads, qkv_bias=True, 
                                     attn_drop=dropout, proj_drop=dropout)
        
        hidden_dim = int(ffn_ratio * channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.LayerNorm(hidden_dim) if norm_layer == nn.LayerNorm else nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, point, lengths, sorter_scores):
        if self.enable_spcpe:
            point = self.cpe(point)
            
        x = point.feat
        if x.shape[0] == 0:
            return point
            
        shortcut = x
        x = self.norm1(x)
        
        device = x.device
        target_win_size = 128
        n_buckets_per_batch = (lengths.float() / target_win_size).ceil().long()
        
        batch_ids = point.batch
        num_buckets_point = n_buckets_per_batch[batch_ids]
        batch_offset_point = torch.cat([
            torch.tensor([0], device=device), 
            torch.cumsum(n_buckets_per_batch, dim=0)[:-1]
        ])[batch_ids]
        
        local_bucket = (sorter_scores * num_buckets_point.float()).long()
        local_bucket = torch.min(local_bucket, num_buckets_point - 1)
        global_bucket = local_bucket + batch_offset_point
        total_buckets = n_buckets_per_batch.sum().item()
        
        x_out = self.attn(x, global_bucket, total_buckets)
        
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
# Downsampling and Upsampling Blocks
# ==============================================================================

class DownsamplingBlock(nn.Module):
    """Downsampling block with multiple pooling strategies."""
    def __init__(self, in_channels, out_channels, stride=2, pool_mode='strided', norm_layer=None):
        super().__init__()
        self.stride = stride
        self.pool_mode = pool_mode
        
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = nn.ReLU()
    
    def forward(self, point, lengths, sorter_scores):
        device = point.feat.device
        
        if lengths.sum() == 0:
            point_down = Point({
                'feat': torch.empty(0, self.proj.out_features, device=device),
                'coord': torch.empty(0, 3, device=device),
                'batch': torch.empty(0, device=device, dtype=torch.long),
            })
            lengths_down = torch.zeros_like(lengths)
            return point_down, lengths_down, torch.empty(0, device=device), {}
        
        curr_offset = torch.cat([torch.tensor([0], device=device), torch.cumsum(lengths, dim=0)])
        
        if self.pool_mode == 'strided':
            indices = []
            for b in range(len(lengths)):
                idx = torch.arange(curr_offset[b], curr_offset[b+1], self.stride, device=device)
                indices.append(idx)
            indices = torch.cat(indices)
            
            feat_down = point.feat[indices]
            coord_down = point.coord[indices]
            batch_down = point.batch[indices]
            scores_down = sorter_scores[indices]
            lengths_down = (lengths + self.stride - 1) // self.stride
            
        elif self.pool_mode == 'mean':
            feat_down, coord_down, batch_down, scores_down, lengths_down = self._pool_mean(
                point.feat, point.coord, point.batch, sorter_scores, lengths, curr_offset
            )
            
        elif self.pool_mode == 'max':
            feat_down, coord_down, batch_down, scores_down, lengths_down = self._pool_max(
                point.feat, point.coord, point.batch, sorter_scores, lengths, curr_offset
            )
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")
        
        feat_down = self.proj(feat_down)
        feat_down = self.norm(feat_down)
        feat_down = self.act(feat_down)
        
        point_down = Point({
            'feat': feat_down,
            'coord': coord_down,
            'batch': batch_down,
            'offset': torch.cumsum(lengths_down, dim=0)
        })
        
        metadata = {
            'lengths_high': lengths.clone(),
            'stride': self.stride,
            'pool_mode': self.pool_mode
        }
        
        return point_down, lengths_down, scores_down, metadata
    
    def _pool_mean(self, feat, coord, batch, scores, lengths, curr_offset):
        device = feat.device
        B = len(lengths)
        
        feat_list, coord_list, batch_list, score_list = [], [], [], []
        lengths_down = torch.zeros_like(lengths)
        
        for b in range(B):
            start, end = curr_offset[b].item(), curr_offset[b+1].item()
            if start >= end:
                continue
                
            L = end - start
            n_windows = (L + self.stride - 1) // self.stride
            lengths_down[b] = n_windows
            
            for w in range(n_windows):
                w_start = start + w * self.stride
                w_end = min(w_start + self.stride, end)
                
                feat_list.append(feat[w_start:w_end].mean(dim=0, keepdim=True))
                coord_list.append(coord[w_start:w_end].mean(dim=0, keepdim=True))
                # FIX: For 1D tensor (scores), use mean() without keepdim or use unsqueeze
                score_list.append(scores[w_start:w_end].mean().unsqueeze(0))
                batch_list.append(torch.tensor([b], device=device))
        
        if len(feat_list) == 0:
            return (torch.empty(0, feat.shape[1], device=device),
                    torch.empty(0, 3, device=device),
                    torch.empty(0, device=device, dtype=torch.long),
                    torch.empty(0, device=device),
                    torch.zeros_like(lengths))
        
        return (torch.cat(feat_list, dim=0),
                torch.cat(coord_list, dim=0),
                torch.cat(batch_list, dim=0),
                torch.cat(score_list, dim=0),
                lengths_down)

    def _pool_max(self, feat, coord, batch, scores, lengths, curr_offset):
        device = feat.device
        B = len(lengths)
        
        feat_list, coord_list, batch_list, score_list = [], [], [], []
        lengths_down = torch.zeros_like(lengths)
        
        for b in range(B):
            start, end = curr_offset[b].item(), curr_offset[b+1].item()
            if start >= end:
                continue
                
            L = end - start
            n_windows = (L + self.stride - 1) // self.stride
            lengths_down[b] = n_windows
            
            for w in range(n_windows):
                w_start = start + w * self.stride
                w_end = min(w_start + self.stride, end)
                
                feat_list.append(feat[w_start:w_end].max(dim=0, keepdim=True)[0])
                coord_list.append(coord[w_start:w_end].mean(dim=0, keepdim=True))
                # FIX: For 1D tensor (scores), use mean() without keepdim or use unsqueeze
                score_list.append(scores[w_start:w_end].mean().unsqueeze(0))
                batch_list.append(torch.tensor([b], device=device))
        
        if len(feat_list) == 0:
            return (torch.empty(0, feat.shape[1], device=device),
                    torch.empty(0, 3, device=device),
                    torch.empty(0, device=device, dtype=torch.long),
                    torch.empty(0, device=device),
                    torch.zeros_like(lengths))
        
        return (torch.cat(feat_list, dim=0),
                torch.cat(coord_list, dim=0),
                torch.cat(batch_list, dim=0),
                torch.cat(score_list, dim=0),
                lengths_down)


class UpsamplingBlock(nn.Module):
    """Upsampling block with interpolation."""
    def __init__(self, in_channels, out_channels, upsample_mode='nearest', norm_layer=None):
        super().__init__()
        self.upsample_mode = upsample_mode
        
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = nn.ReLU()
    
    def forward(self, point_low, point_high, skip_feat, lengths_high, metadata):
        device = point_low.feat.device
        stride = metadata['stride']
        
        N_high = lengths_high.sum().item()
        
        if N_high == 0:
            point_up = Point({
                'feat': torch.empty(0, self.proj.out_features, device=device),
                'coord': torch.empty(0, 3, device=device),
                'batch': torch.empty(0, device=device, dtype=torch.long),
            })
            return point_up
        
        feat_low_proj = self.proj(point_low.feat)
        feat_low_proj = self.norm(feat_low_proj)
        feat_low_proj = self.act(feat_low_proj)
        
        feat_up = self._upsample_nearest(feat_low_proj, lengths_high, stride, device)
        feat_up = feat_up + skip_feat
        
        point_up = Point({
            'feat': feat_up,
            'coord': point_high.coord,
            'batch': point_high.batch,
            'offset': torch.cumsum(lengths_high, dim=0)
        })
        
        return point_up
    
    def _upsample_nearest(self, feat_low, lengths_high, stride, device):
        curr_lengths_low = (lengths_high + stride - 1) // stride
        offset_low = torch.cat([torch.tensor([0], device=device), torch.cumsum(curr_lengths_low, dim=0)])
        
        feat_list = []
        for b in range(len(lengths_high)):
            l_low = curr_lengths_low[b].item()
            l_high = lengths_high[b].item()
            
            if l_high == 0:
                continue
            
            local_idx = torch.arange(l_high, device=device) // stride
            local_idx = local_idx.clamp(max=l_low - 1)
            global_idx = local_idx + offset_low[b]
            
            feat_list.append(feat_low[global_idx])
        
        if len(feat_list) == 0:
            return torch.empty(0, feat_low.shape[1], device=device)
        
        return torch.cat(feat_list, dim=0)


# ==============================================================================
# OPTNet Backbone with Self-Supervised Ordering
# ==============================================================================

@MODELS.register_module("OPTNet")
class OPTNet(PointModule):
    def __init__(self, in_channels=6, embed_dim=128, enc_depths=(2, 2, 6, 2), dec_depths=(1, 1, 1, 1), 
                 num_heads=4, win_sizes=(64, 64, 64, 64), base_grid_size=0.02, pool_factors=(2, 2, 2, 2), 
                 dropout=0.0, ffn_ratio=3.0, ordering_loss_weight=0.5, 
                 enable_efpt_pos=True, enable_spcpe=False, pool_mode='strided', upsample_mode='nearest'):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.base_grid_size = base_grid_size
        self.pool_factors = pool_factors
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        self.pool_mode = pool_mode
        self.upsample_mode = upsample_mode
        
        self.ordering_loss_weight = ordering_loss_weight
        
        norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.patch_embed = PatchEmbedding(3, embed_dim, base_grid_size, norm_layer=norm_layer) 
        self.feat_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.LayerNorm(embed_dim), 
            nn.ReLU()
        )
        
        self.sorter = PointSorter(embed_dim, hidden_dim=64)
        
        if enable_efpt_pos:
            self.gated_pos = GatedPosEmbedding(3, embed_dim)
        else:
            self.gated_pos = nn.Sequential(
                nn.Linear(3, embed_dim), 
                nn.LayerNorm(embed_dim), 
                nn.ReLU()
            )
        
        self.enc_blocks = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        curr_dim = embed_dim
        
        for si in range(self.num_stages):
            self.enc_blocks.append(nn.ModuleList([
                ScatterOPTBlock(curr_dim, num_heads=num_heads, dropout=dropout, ffn_ratio=ffn_ratio, 
                                norm_layer=norm_layer, enable_spcpe=self.enable_spcpe, indice_key=f"enc_s{si}") 
                for _ in range(enc_depths[si])
            ]))
            if si < self.num_stages - 1:
                self.enc_down.append(
                    DownsamplingBlock(curr_dim, curr_dim, stride=pool_factors[si], 
                                    pool_mode=pool_mode, norm_layer=norm_layer)
                )

        self.dec_blocks = nn.ModuleList()
        self.dec_up = nn.ModuleList()
        for si in range(self.num_stages):
            if si < self.num_stages - 1:
                self.dec_up.append(
                    UpsamplingBlock(curr_dim, curr_dim, upsample_mode=upsample_mode, 
                                  norm_layer=norm_layer)
                )
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
        
        # Self-supervised ordering
        learner_scores = self.sorter(x, point.coord)
        
        total_aux_loss = 0.0
        
        if self.training:
            ordering_loss, loss_dict = self.sorter.compute_loss(
                learner_scores, 
                point.coord, 
                point.batch,
                point.offset
            )
            total_aux_loss += ordering_loss * self.ordering_loss_weight
        
        bucket_scores = learner_scores

        geom_emb = self.gated_pos(disp_feat)
        point.feat = x + geom_emb
        
        if self.enable_spcpe: 
            point.sparsify() 

        offset = point.offset
        if offset.numel() > 0:
            lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long()
        else:
            lengths = torch.tensor([point.feat.shape[0]], device=device).long()

        encoder_points = []
        encoder_lengths = []
        encoder_scores = []
        skips = []
        up_metadata = []
        
        curr_scores = bucket_scores
        
        # Encoder
        for si in range(self.num_stages):
            encoder_points.append(point)
            encoder_lengths.append(lengths)
            encoder_scores.append(curr_scores)
            
            for blk in self.enc_blocks[si]:
                point = blk(point, lengths, curr_scores)
            
            skips.append(point.feat)
            
            if si < len(self.enc_down):
                point, lengths, curr_scores, metadata = self.enc_down[si](point, lengths, curr_scores)
                up_metadata.append(metadata)
                
                if self.enable_spcpe:
                    stride = self.pool_factors[si]
                    point.grid_coord = torch.div(point.grid_coord, stride, rounding_mode='trunc').int()
                    point.sparsify()

        # Decoder
        for si in reversed(range(self.num_stages)):
            if si < len(up_metadata):
                point_high = encoder_points[si]
                skip_feat = skips[si]
                lengths_high = encoder_lengths[si]
                metadata = up_metadata[si]
                
                point = self.dec_up[si](point, point_high, skip_feat, lengths_high, metadata)
                
                lengths = lengths_high
                curr_scores = encoder_scores[si]
                
                if self.enable_spcpe:
                    stride = self.pool_factors[si]
                    point.grid_coord = torch.div(point.coord - point.coord.min(0)[0], 
                                                self.base_grid_size * (stride**si), rounding_mode='trunc').int()
                    point.sparsify()
            
            for blk in self.dec_blocks[si]:
                point = blk(point, lengths, curr_scores)

        point.feat = self.head_norm(point.feat)
        
        if self.training:
            if not hasattr(point, "aux_loss"): 
                point.aux_loss = 0.0
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

        if isinstance(point, Point): 
            feat = point.feat
        else: 
            feat = point
            
        seg_logits = self.seg_head(feat)
        return_dict = {}
        
        if return_point: 
            return_dict["point"] = point
            
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
