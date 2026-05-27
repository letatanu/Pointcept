"""
OPTNet (No SpConv, No Hilbert Warmup).

- Uses a self-supervised PointSorter with Geometric Initialization.
- No spconv import, no SpCPE.
- Uses 1D window self-attention on the serialized (sorted) point sequence.
- Sorting is fully learned via `compute_ordering_loss`.

This file is meant to replace your existing optnet.py entirely.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.amp import autocast

from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point
from pointcept.models.default import DefaultSegmentorV2
import pointops

# ==============================================================================
# Helpers
# ==============================================================================

def _apply_permutation_fallback(point: Point, perm: torch.Tensor):
    """
    Fallback if your Point class doesn't have apply_permutation().
    Permutes all per-point tensors.
    """
    # Determine N based on common keys
    N_old = -1
    for key in ["feat", "coord", "grid_coord", "batch"]:
        if key in point.keys() and isinstance(point[key], torch.Tensor):
            N_old = point[key].shape[0]
            break
            
    if N_old == -1:
        N_old = perm.shape[0]

    keys_to_update = []
    for k, v in point.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == N_old:
            keys_to_update.append(k)
            
    for k in keys_to_update:
        point[k] = point[k][perm]

    # remove sparse cache keys if any exist (they are invalid after permutation)
    for k in ["sparse", "sparse_conv_feat", "sparse_shape"]:
        if k in point:
            del point[k]
            
    return point

# ==============================================================================
# Modules
# ==============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_channels, norm_layer):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            norm_layer(embed_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, point: Point):
        point.feat = self.embed(point.feat)
        return point

class PointSorter(nn.Module):
    """
    Self-supervised Sorter with Geometric Initialization.
    Predicts a scalar score in [0,1] for sorting.
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        # Input is [coord(3), feat(in_channels)]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()
        
        # Initialize to behavior like a spatial sorter (Sort by Z+Y+X)
        self._init_geometric_weights()

    def _init_geometric_weights(self):
        # 1. Zero out feature contributions initially to ignore noise
        nn.init.constant_(self.mlp[0].weight[:, 3:], 0.0)
        
        # 2. Set coordinate weights to project onto diagonal (1, 1, 1)
        # This makes initial sort roughly (x + y + z)
        nn.init.normal_(self.mlp[0].weight[:, :3], std=0.1)
        with torch.no_grad():
            # Force strong dependence on coords for the first neuron
            self.mlp[0].weight[0, 0] = 1.0  # X
            self.mlp[0].weight[0, 1] = 1.0  # Y
            self.mlp[0].weight[0, 2] = 1.0  # Z

        # 3. Ensure signal passes through to output
        nn.init.normal_(self.mlp[3].weight, std=0.01)
        nn.init.normal_(self.mlp[6].weight, std=0.01)
        nn.init.constant_(self.mlp[6].bias, 0.0)

    def forward(self, point: Point):
        if point.feat is not None:
            # Detach features to prevent sorter from hacking the encoder
            inp = torch.cat([point.coord, point.feat.detach()], dim=1)
        else:
            inp = point.coord

        if inp.shape[0] == 0:
            return torch.zeros((0,), device=point.coord.device)
            
        return self.sigmoid(self.mlp(inp)).squeeze(-1)

class GatedPosEmbedding(nn.Module):
    """eFPT-style gated position embedding."""
    def __init__(self, in_channels, out_channels, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels, bias=False),
        )
        self.enc_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
        )

    def forward(self, rel_coord):
        with autocast("cuda", enabled=False):
            rel_coord = rel_coord.float()
            gate = self.mlp(rel_coord)
            modulated = rel_coord * gate + gate
            out = self.enc_mlp(modulated)
        return out

class MLPCPE(nn.Module):
    """Per-point CPE replacement (no sparse conv)."""
    def __init__(self, channels, hidden_ratio=2.0, drop=0.0):
        super().__init__()
        hidden = int(channels * hidden_ratio)
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, channels),
            nn.Dropout(drop),
        )

    def forward(self, point: Point):
        x = point.feat
        point.feat = x + self.net(self.norm(x))
        return point

class WindowAttention1D(nn.Module):
    """Standard MHA on (num_windows, win_size, C)."""
    def __init__(self, channels, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (Wn, K, C)
        Wn, K, C = x.shape
        qkv = self.qkv(x).reshape(Wn, K, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, Wn, H, K, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (Wn, H, K, K)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(Wn, K, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class ScatterOPTBlock(nn.Module):
    """
    1D-window attention block operating on the already-serialized (sorted) point sequence.
    """
    def __init__(
        self,
        channels,
        num_heads=4,
        win_size=32,
        shift_size=0,
        dropout=0.0,
        ffn_ratio=2.0,
        norm_layer=nn.BatchNorm1d,
        **kwargs,
    ):
        super().__init__()
        self.win_size = int(win_size)
        self.shift_size = int(shift_size)

        self.cpe = MLPCPE(channels, hidden_ratio=2.0, drop=dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention1D(
            channels, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout
        )

        hidden_dim = int(ffn_ratio * channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.LayerNorm(hidden_dim) if norm_layer == nn.LayerNorm else nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def _attend_one_batch(self, xb, shift):
        L, C = xb.shape
        if L == 0:
            return xb
            
        # shift
        if shift > 0:
            s = shift % L
            xb = torch.roll(xb, shifts=-s, dims=0)

        # pad
        pad = (-L) % self.win_size
        if pad > 0:
            xb = torch.cat([xb, xb[-1:].repeat(pad, 1)], dim=0)

        # attention
        xb_w = xb.view(-1, self.win_size, C)
        out_w = self.attn(xb_w).reshape(-1, C)
        out = out_w[:L]

        # unshift
        if shift > 0:
            out = torch.roll(out, shifts=s, dims=0)
        return out

    def forward(self, point: Point, lengths, sorter_scores=None):
        point = self.cpe(point)
        x = point.feat
        shortcut = x
        x = self.norm1(x)

        outs = []
        start = 0
        for l in lengths.tolist():
            xb = x[start : start + l]
            outb = self._attend_one_batch(xb, self.shift_size)
            outs.append(outb)
            start += l
            
        x_out = torch.cat(outs, dim=0) if len(outs) > 0 else x
        
        x = shortcut + x_out
        x = x + self.ffn(self.norm2(x))
        point.feat = x
        return point

# ==============================================================================
# OPTNet Backbone
# ==============================================================================

@MODELS.register_module("OPTNet")
class OPTNet(PointModule):
    def __init__(
        self,
        in_channels=6,
        embed_dim=128,
        enc_depths=(2, 2, 6, 2),
        dec_depths=(1, 1, 1, 1),
        num_heads=4,
        win_sizes=(32, 32, 32, 32),
        pool_factors=(2, 2, 2, 2),
        dropout=0.0,
        ffn_ratio=3.0,
        ordering_loss_weight=0.5,
        ordering_k=16,
        enable_efpt_pos=True,
        **kwargs,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.pool_factors = pool_factors
        self.ordering_loss_weight = ordering_loss_weight
        self.ordering_k = ordering_k
        
        # NOTE: Warmup removed. We rely on geometric init.

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.patch_embed = PatchEmbedding(3, embed_dim, norm_layer=bn_layer)
        self.feat_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Self-supervised sorter
        self.sorter = PointSorter(embed_dim, hidden_dim=64)

        if enable_efpt_pos:
            self.gated_pos = GatedPosEmbedding(3, embed_dim)
        else:
            self.gated_pos = nn.Sequential(
                nn.Linear(3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_trans = nn.ModuleList()
        curr_dim = embed_dim
        for si in range(self.num_stages):
            blocks = []
            for bi in range(enc_depths[si]):
                shift = 0 if (bi % 2 == 0) else (int(win_sizes[si]) // 2)
                blocks.append(
                    ScatterOPTBlock(
                        curr_dim,
                        num_heads=num_heads,
                        win_size=win_sizes[si],
                        shift_size=shift,
                        dropout=dropout,
                        ffn_ratio=ffn_ratio,
                        norm_layer=bn_layer,
                    )
                )
            self.enc_blocks.append(nn.ModuleList(blocks))
            if si < self.num_stages - 1:
                self.enc_trans.append(nn.Linear(curr_dim, curr_dim))

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.dec_trans = nn.ModuleList()
        for si in range(self.num_stages):
            if si < self.num_stages - 1:
                self.dec_trans.append(nn.Linear(curr_dim, curr_dim))
            blocks = []
            for bi in range(dec_depths[si]):
                shift = 0 if (bi % 2 == 0) else (int(win_sizes[si]) // 2)
                blocks.append(
                    ScatterOPTBlock(
                        curr_dim,
                        num_heads=num_heads,
                        win_size=win_sizes[si],
                        shift_size=shift,
                        dropout=dropout,
                        ffn_ratio=ffn_ratio,
                        norm_layer=bn_layer,
                    )
                )
            self.dec_blocks.append(nn.ModuleList(blocks))

        self.head_norm = nn.LayerNorm(curr_dim)

    @staticmethod
    def _make_serialized_order(batch: torch.Tensor, scores: torch.Tensor):
        # Add large offset to separate batches
        # Scores are [0,1], so +2.0 separation is safe
        batch = batch.long()
        global_scores = scores + batch.float() * (scores.detach().max() + 2.0)
        
        order = torch.argsort(global_scores)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return order, inverse

    def compute_ordering_loss(self, point: Point, scores: torch.Tensor):
        """
        ICPR-style ordering loss:
        1) Locality: KNN neighbors should have similar scores
        2) Distribution: scores should cover [0,1]
        """
        scores = scores.view(-1)
        if (not hasattr(point, "offset")) or (point.offset is None) or (point.offset.numel() == 0):
            offset = torch.tensor([scores.numel()], device=scores.device, dtype=torch.long)
        else:
            offset = point.offset
            
        # KNN query
        idx = pointops.knn_query(self.ordering_k, point.coord, offset)[0]  # (N, k)
        neighbor_scores = scores[idx.long()]
        
        # Locality Loss
        diff = scores.unsqueeze(1) - neighbor_scores
        loss_locality = (diff ** 2).sum(dim=1).mean()
        
        # Distribution Loss (Uniformity)
        sorted_scores, _ = torch.sort(scores)
        target = torch.linspace(0, 1, steps=len(sorted_scores), device=scores.device, dtype=scores.dtype)
        loss_dist = ((sorted_scores - target) ** 2).mean()
        
        return loss_locality + loss_dist

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

        # 1. Embed Features
        point.feat = color_feat
        point = self.patch_embed(point)
        x = self.feat_mlp(point.feat)
        point.feat = x
        
        # 2. Predict Sort Scores (Self-Supervised)
        # ----------------------------------------
        learner_scores = self.sorter(point)
        
        # Compute ordering loss
        total_aux_loss = 0.0
        if self.training and self.ordering_loss_weight > 0:
            total_aux_loss += self.compute_ordering_loss(point, learner_scores) * self.ordering_loss_weight

        # 3. Inject Geometry
        geom_emb = self.gated_pos(disp_feat)
        point.feat = x + geom_emb

        # 4. Handle Offsets/Lengths
        offset = point.offset
        if (offset is not None) and (offset.numel() > 0):
            lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long()
        else:
            lengths = torch.tensor([point.feat.shape[0]], device=device, dtype=torch.long)

        # 5. Serialize / Sort
        # -------------------
        order, inverse = self._make_serialized_order(point.batch, learner_scores)
        point["serialized_order"] = [order]
        point["serialized_inverse"] = [inverse]
        
        if hasattr(point, "apply_permutation"):
            point = point.apply_permutation(order)
        else:
            point = _apply_permutation_fallback(point, order)
            
        curr_scores = learner_scores[order]
        
        # 6. Encoder / Decoder Loop
        # -------------------------
        encoder_coords, encoder_lengths, encoder_scores = [], [], []
        skips, up_metadata = [], []

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
                # Strided subsample
                if lengths.sum() > 0:
                    curr_offset = torch.cat([torch.tensor([0], device=device), torch.cumsum(lengths, dim=0)])
                    batch_indices = []
                    for b in range(len(lengths)):
                        idx = torch.arange(curr_offset[b], curr_offset[b + 1], stride, device=device)
                        batch_indices.append(idx)
                    indices = torch.cat(batch_indices)
                else:
                    indices = torch.empty(0, device=device, dtype=torch.long)

                if hasattr(point, "apply_permutation"):
                    point = point.apply_permutation(indices)
                else:
                    point = _apply_permutation_fallback(point, indices)
                    
                curr_scores = curr_scores[indices]
                point.feat = self.enc_trans[si](point.feat)
                
                up_metadata.append((lengths.clone(), stride))
                lengths = (lengths + stride - 1) // stride
                point.offset = torch.cumsum(lengths, dim=0)

        # --- Decoder ---
        curr_lengths = lengths
        for si in reversed(range(self.num_stages)):
            if si < len(up_metadata):
                if si < len(self.dec_trans):
                    point.feat = self.dec_trans[si](point.feat)

                lengths_high, stride = up_metadata[si]
                
                # Unpool / Repeat
                offset_low = torch.cat([torch.tensor([0], device=device), torch.cumsum(curr_lengths, dim=0)])
                idx_map_list = []
                for b in range(len(lengths_high)):
                    l_low = int(curr_lengths[b].item())
                    l_high = int(lengths_high[b].item())
                    if l_high == 0:
                        idx_map_list.append(torch.empty(0, device=device, dtype=torch.long))
                        continue
                    if l_low <= 0:
                        idx_map_list.append(torch.zeros((l_high,), device=device, dtype=torch.long) + offset_low[b])
                        continue
                        
                    local_idx = torch.arange(l_high, device=device) // stride
                    local_idx = local_idx.clamp(max=l_low - 1)
                    idx_map_list.append(local_idx + offset_low[b])
                    
                global_map = torch.cat(idx_map_list) if len(idx_map_list) > 0 else torch.empty(0, device=device, dtype=torch.long)
                
                x_up = point.feat[global_map]
                x_up = x_up + skips[si]
                point.feat = x_up
                
                # Restore original coords/batch at this level
                point.coord = encoder_coords[si]
                point.batch = offset2batch(torch.cumsum(lengths_high, dim=0))
                point.offset = torch.cumsum(lengths_high, dim=0)
                
                curr_scores = encoder_scores[si]
                curr_lengths = lengths_high

            for blk in self.dec_blocks[si]:
                point = blk(point, curr_lengths, curr_scores)

        point.feat = self.head_norm(point.feat)

        # 7. Final Revert (Inverse Sort)
        # ------------------------------
        if "serialized_inverse" in point.keys() and len(point["serialized_inverse"]) > 0:
            inverse_0 = point["serialized_inverse"][0]
            # Must assume point.feat is full resolution again (N, C)
            point.feat = point.feat[inverse_0]
            # Optional: revert coords to match return_point expectations
            point.coord = point.coord[inverse_0]

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
        
        feat = point.feat if isinstance(point, Point) else point
        seg_logits = self.seg_head(feat)
        
        return_dict = dict()
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
