"""
OPTNet with Residual Coordinate Quantization (RCQ)
Replaces learned ordering with deterministic Hierarchical Residual Quantization IDs.
Integrates eFPT-style Centroid-Aware logic via displacement features.
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

def build_windows_1d(lengths, W, stride, device):
    """
    Generates window indices for 1D attention.
    Assumes points are already sorted/grouped by the RCQ-ID.
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

class ResidualCoordinateQuantizer(nn.Module):
    """
    Assigns a hierarchical 'Geometry ID' to points using Residual Quantization.
    Used for both Position Embedding and Deterministic Ordering.
    """
    def __init__(self, input_dim=3, embed_dim=128, codebook_size=16, num_layers=3, base_scale=50.0):
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        
        # Grid dimension per layer (e.g., codebook 64 -> 4x4x4 grid)
        self.grid_dim = int(math.pow(codebook_size, 1/3))
        # Ensure grid_dim is at least 2
        self.grid_dim = max(2, self.grid_dim)
        
        # Define scales: L0=50m, L1=50/4 m, L2=50/16 m ...
        self.scales = []
        curr = base_scale
        for _ in range(num_layers):
            self.scales.append(curr)
            curr /= self.grid_dim
            
        # Learnable embeddings for the codes
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim) for _ in range(num_layers)
        ])
        
        # MLP to fuse the quantized embeddings
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

    def forward(self, coord):
        """
        Args:
            coord: (N, 3) Global coordinates
        Returns:
            embedding: (N, C) Positional embedding
            sort_idx: (N, ) Integer ID for sorting
        """
        N = coord.shape[0]
        # Shift coords to positive for simpler hashing (assuming scene centered at 0 or roughly > -scale)
        residual = coord.clone()
        
        total_emb = 0
        combined_id = torch.zeros(N, device=coord.device, dtype=torch.long)
        
        # Primes for hashing 3D index to 1D codebook index
        p1, p2, p3 = 73856093, 19349663, 83492791
        
        for i in range(self.num_layers):
            scale = self.scales[i]
            
            # 1. Quantize
            grid_idx = torch.div(residual, scale, rounding_mode='floor').int()
            
            # 2. Hash to Codebook ID
            flat_idx = (grid_idx[:, 0] * p1 + grid_idx[:, 1] * p2 + grid_idx[:, 2] * p3) % self.codebook_size
            
            # 3. Embed
            total_emb = total_emb + self.embeddings[i](flat_idx.long())
            
            # 4. Update Residual (Geometric modulo)
            residual = residual % scale
            
            # 5. Accumulate ID for sorting
            combined_id = combined_id * self.codebook_size + flat_idx.long()
            
        return self.fusion(total_emb), combined_id

class SpCPE(nn.Module):
    """
    Sparse Convolution CPE.
    """
    def __init__(self, in_channels, out_channels, indice_key=None, kernel_size=3):
        super().__init__()
        assert spconv is not None, "spconv is required for SpCPE"
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
        
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.sparse_conv_feat = self.conv(point.sparse_conv_feat)
        feat = point.sparse_conv_feat.features
        feat = self.norm(self.linear(feat))
        point.feat = shortcut + feat
        return point

class OPTBlock(nn.Module):
    """
    Standard Window Attention Block.
    Relies on points being sorted (by RCQ-ID) for locality.
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

    def forward(self, point, lengths):
        # CPE
        if self.enable_spcpe:
            point = self.cpe(point)
        
        x = point.feat
        
        # Window Partitioning
        win_idx, win_mask = build_windows_1d(lengths, self.W, self.stride, device=x.device)
        if win_idx.numel() == 0: return point

        shortcut = x
        x_in = self.norm1(x)
        y = torch.zeros_like(x_in); counts = torch.zeros_like(x_in[:, 0:1]) 
        M, W = win_idx.shape; C = x.shape[1]
        
        # Chunked Attention
        for s in range(0, M, self.win_chunk):
            e = min(M, s + self.win_chunk)
            idx = win_idx[s:e]; msk = win_mask[s:e]; m = idx.size(0)
            if m == 0: continue
            
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
                 enable_efpt_pos=True, enable_spcpe=True):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.base_grid_size = base_grid_size; self.pool_factors = pool_factors
        self.enable_spcpe = enable_spcpe and (spconv is not None)
        
        norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # 1. Feature Embedding
        # Split color (3) and displacement (3)
        self.patch_embed = PatchEmbedding(3, embed_dim, base_grid_size, norm_layer=norm_layer) 
        self.feat_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        
        # 2. Residual Quantizer (Structure + Embedding)
        self.quantizer = ResidualCoordinateQuantizer(
            input_dim=3, 
            embed_dim=embed_dim, 
            codebook_size=64, 
            num_layers=3, 
            base_scale=100.0
        )
        
        # 3. Gated Displacement Embedding (eFPT)
        if enable_efpt_pos:
            self.disp_mlp = nn.Sequential(
                nn.Linear(3, 32, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 3), 
            )
            self.disp_enc = nn.Sequential(
                nn.Linear(3, embed_dim, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, embed_dim, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.Tanh()
            )
        else:
            self.disp_mlp = None
            
        self.enc_blocks = nn.ModuleList(); self.enc_trans = nn.ModuleList()
        curr_dim = embed_dim
        
        for si in range(self.num_stages):
            self.enc_blocks.append(nn.ModuleList([
                OPTBlock(curr_dim, win_sizes[si], num_heads=num_heads, 
                         dropout=dropout, win_chunk=win_chunk, ffn_ratio=ffn_ratio, norm_layer=norm_layer, 
                         enable_spcpe=self.enable_spcpe, indice_key=f"enc_s{si}") 
                for _ in range(enc_depths[si])
            ]))
            if si < self.num_stages - 1: self.enc_trans.append(nn.Linear(curr_dim, curr_dim))

        self.dec_blocks = nn.ModuleList(); self.dec_trans = nn.ModuleList()
        for si in range(self.num_stages):
            if si < self.num_stages - 1: self.dec_trans.append(nn.Linear(curr_dim, curr_dim))
            self.dec_blocks.append(nn.ModuleList([
                OPTBlock(curr_dim, win_sizes[si], num_heads=num_heads, 
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
        if feat_raw.shape[1] >= 6:
            color_feat = feat_raw[:, :3]
            disp_feat = feat_raw[:, -3:]
        else:
            color_feat = feat_raw
            disp_feat = torch.zeros_like(point.coord)

        # --- 2. Feature Embedding ---
        point.feat = color_feat
        point = self.patch_embed(point)
        if self.enable_spcpe: point.sparsify()
        
        x = self.feat_mlp(point.feat)
        
        # --- 3. Geometric Embedding (RCQ + Gated Disp) ---
        rcq_emb, sort_ids = self.quantizer(point.coord)
        
        if self.disp_mlp is not None:
            with autocast(enabled=False):
                d = disp_feat.float()
                gate = self.disp_mlp(d)
                d_gated = d * gate + gate
                disp_emb = self.disp_enc(d_gated)
        else:
            disp_emb = 0
            
        point.feat = x + rcq_emb + disp_emb
        
        if self.enable_spcpe:
             point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        offset = point.offset
        if offset.numel() > 0:
            lengths = torch.cat([offset[0].unsqueeze(0), offset[1:] - offset[:-1]]).long()
        else:
            lengths = torch.tensor([point.feat.shape[0]], device=device).long()

        # --- 4. Deterministic Sorting via RCQ IDs ---
        sort_score = point.batch.float() * (sort_ids.max().float() + 1.0) + sort_ids.float()
        perm = torch.argsort(sort_score)
        
        point.apply_permutation(perm)
        if self.enable_spcpe: point.sparsify()

        encoder_coords, encoder_lengths, encoder_grid_coords, skips, up_metadata = [], [], [], [], []
        
        # --- Encoder ---
        for si in range(self.num_stages):
            encoder_coords.append(point.coord)
            encoder_lengths.append(lengths)
            # FIX: Store grid_coord for restoration in decoder
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
                
                up_metadata.append((lengths.clone(), stride, perm))
                
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
                # FIX: Restore grid_coord so sparsify() uses the correct shape
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

            for blk in self.dec_blocks[si]:
                point = blk(point, curr_lengths)

        point.feat = self.head_norm(point.feat)
        return point

@MODELS.register_module("OPTNetSegmentor")
class OPTNetSegmentor(DefaultSegmentorV2):
    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        
        if isinstance(point, Point): feat = point.feat
        else: feat = point
            
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point: return_dict["point"] = point
            
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
            
        return return_dict