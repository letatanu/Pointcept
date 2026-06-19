"""
PTv3-NO Enhanced — Universal Shared NOGlobalBranch
  + GridPooling / GridUnpooling (from Sonata)
  + Relative position encoding in NOGlobalBranch
"""

import torch
import torch.nn as nn
import torch_scatter
from addict import Dict
from functools import partial

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Block, Embedding,
)

from mamba_ssm import Mamba



class MambaBlock(nn.Module):
    """
    Grid-Free Point Sequence Operator Block.
    Serializes irregular point clouds into 1D sequences via stable lexicographical 
    sorting, processes them using Bidirectional Mamba blocks, and remaps them back.
    """
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("Please install the official mamba-ssm package to use MambaBlock.")
            
        # Forward and Backward state-space models for bidirectional sequence scanning
        self.mamba_forward = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_backward = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.direction_fusion = nn.Linear(channels * 2, channels)

    def forward(self, feat: torch.Tensor, coord: torch.Tensor, offset: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            feat   : (N, C) point features (e.g., universal_dim channels)
            coord  : (N, 3) spatial coordinates
            offset : (B,) optional Pointcept cumulative batch point counts
        Returns:
            (N, C) sequence-coordinated global features
        """
        device = feat.device
        # Fallback to a single batch item if no offset tensor is passed
        if offset is None:
            offset = torch.tensor([feat.shape[0]], device=device, dtype=torch.long)
            
        num_batches = offset.shape[0]
        output_feats = torch.zeros_like(feat)
        
        start_idx = 0
        for b in range(num_batches):
            end_idx = offset[b].item()
            if start_idx == end_idx:
                continue
                
            b_feat = feat[start_idx:end_idx]   
            b_coord = coord[start_idx:end_idx] 
            
            # 1. Coordinate Serialization via Stable Lexicographical Sorting (Z -> Y -> X)
            sort_idx = torch.arange(b_feat.shape[0], device=device)
            for axis in [2, 1, 0]: 
                sort_idx = sort_idx[torch.argsort(b_coord[sort_idx, axis], stable=True)]
            inverse_idx = torch.argsort(sort_idx)
            
            # 2. Format to Mamba expected sequence shape: (Batch=1, Length, Channels)
            seq_feat = b_feat[sort_idx].unsqueeze(0) 
            
            # 3. Bidirectional context processing
            feat_fwd = self.mamba_forward(seq_feat)
            
            seq_feat_bwd = torch.flip(seq_feat, dims=[1])
            feat_bwd = self.mamba_backward(seq_feat_bwd)
            feat_bwd = torch.flip(feat_bwd, dims=[1])
            
            # 4. Concatenate directional tracks and project down
            fused_seq = torch.cat([feat_fwd, feat_bwd], dim=-1).squeeze(0) 
            fused_b_feat = self.direction_fusion(fused_seq)                          
            
            # 5. Invert sorting permutation map back to unstructured arrangement
            output_feats[start_idx:end_idx] = fused_b_feat[inverse_idx]
            start_idx = end_idx
            
        return output_feats

# ---------------------------------------------------------------------------
# GridPooling  (Fixed Two's Complement Bitwise Shift Bug)
# ---------------------------------------------------------------------------

class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride        = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce         = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable      = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        else:
            self.norm = None
        if act_layer is not None:
            self.act = PointSequential(act_layer())
        else:
            self.act = None

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[grid_coord] or [coord, grid_size] should be in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        
        # --- FIX: Shift grid coordinates to be strictly non-negative before bitwise manipulation ---
        coord_min = grid_coord.min(0, keepdim=True).values
        grid_coord = grid_coord - coord_min
        
        grid_coord = grid_coord.long()
        grid_coord = grid_coord | point.batch.view(-1, 1).long() << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord, sorted=True,
            return_inverse=True, return_counts=True, dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
        
        # Restore original coordinate space offsets safely
        grid_coord = grid_coord + coord_min.long()
        grid_coord = grid_coord.int()
        # --------------------------------------------------------------------------------------------
        
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        for key in ("origin_coord", "condition", "context", "name", "split"):
            if key in point.keys():
                point_dict[key] = (
                    torch_scatter.segment_csr(point[key][indices], idx_ptr, reduce="mean")
                    if key == "origin_coord" else point[key]
                )
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"]  = point
            point_dict["idx_ptr"]         = idx_ptr

        order = point.order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.serialization(order=order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        return point


# ---------------------------------------------------------------------------
# GridUnpooling
# ---------------------------------------------------------------------------

class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
    ):
        super().__init__()
        self.proj      = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point: Point):
        assert "pooling_parent"  in point.keys()
        assert "pooling_inverse" in point.keys()
        parent  = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat    = point.feat

        parent       = self.proj_skip(parent)
        parent.feat  = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
        return parent


# ---------------------------------------------------------------------------
# RelativePositionEncoding
# ---------------------------------------------------------------------------
class RelativePositionEncoding(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        coord_f   = coord.float()
        centroid  = coord_f.mean(dim=0, keepdim=True)          
        delta     = coord_f - centroid                          
        
        radius = torch.sqrt(torch.sum(delta ** 2, dim=1, keepdim=True) + 1e-12)
        scale = delta.abs().max(dim=0, keepdim=True).values.clamp(min=1e-6)
        delta_n   = delta / scale                               
        geo_feat  = torch.cat([delta_n, radius / (radius.max() + 1e-6)], dim=1)  
        return self.mlp(geo_feat)


# ---------------------------------------------------------------------------
# HybridNeuralOperatorBlock (Factorized FNO + Local CNN)
# ---------------------------------------------------------------------------

class HybridNeuralOperatorBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 8):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        
        self.w_x_real = nn.Parameter(scale * torch.rand(channels, channels, modes))
        self.w_x_imag = nn.Parameter(scale * torch.rand(channels, channels, modes))
        self.w_y_real = nn.Parameter(scale * torch.rand(channels, channels, modes))
        self.w_y_imag = nn.Parameter(scale * torch.rand(channels, channels, modes))
        self.w_z_real = nn.Parameter(scale * torch.rand(channels, channels, modes))
        self.w_z_imag = nn.Parameter(scale * torch.rand(channels, channels, modes))

        self.local_cnn = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1)
        )
        self.bypass = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bypass = self.bypass(x)
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        x_ft = torch.fft.rfftn(x_fp32, dim=[-3, -2, -1])
        out_ft = torch.zeros_like(x_ft)

        mx = min(self.modes, x_ft.shape[2])
        my = min(self.modes, x_ft.shape[3])
        mz = min(self.modes, x_ft.shape[4])

        wx = torch.complex(self.w_x_real, self.w_x_imag)
        wy = torch.complex(self.w_y_real, self.w_y_imag)
        wz = torch.complex(self.w_z_real, self.w_z_imag)

        out_x = torch.einsum("bcxyz,cdx->bdxyz", x_ft[:, :, :mx, :my, :mz], wx[:, :, :mx])
        out_y = torch.einsum("bcxyz,cdy->bdxyz", x_ft[:, :, :mx, :my, :mz], wy[:, :, :my])
        out_z = torch.einsum("bcxyz,cdz->bdxyz", x_ft[:, :, :mx, :my, :mz], wz[:, :, :mz])
        
        out_ft[:, :, :mx, :my, :mz] = out_x + out_y + out_z

        x_global = torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=[-3, -2, -1])
        x_global = x_global.to(orig_dtype)
        
        x_local = self.local_cnn(x)
        return x_global + x_local + x_bypass


# ---------------------------------------------------------------------------
# CNN3dBlock 
# ---------------------------------------------------------------------------

class CNN3dBlock(nn.Module):
    def __init__(self, channels: int, expansion_ratio: float = 2.0):
        super().__init__()
        hidden_dim = int(channels * expansion_ratio)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, channels, kernel_size=1)
        )
        
        self.local_cnn = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels)
        )
        self.bypass = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bypass = self.bypass(x)
        global_feat = self.global_pool(x)
        global_feat = self.global_mlp(global_feat) 
        local_feat = self.local_cnn(x) 
        out = local_feat + global_feat 
        return out + x_bypass


# ---------------------------------------------------------------------------
# MLP3dBlock
# ---------------------------------------------------------------------------

class MLP3dBlock(nn.Module):
    def __init__(self, channels: int, expansion_ratio: float = 2.0):
        super().__init__()
        hidden_dim = int(channels * expansion_ratio)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, channels, kernel_size=1)
        )
        
        self.local_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, channels, kernel_size=1)
        )
        self.bypass = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bypass = self.bypass(x)
        global_feat = self.global_pool(x)
        global_feat = self.global_mlp(global_feat) 
        local_feat = self.local_mlp(x) 
        out = local_feat + global_feat 
        return out + x_bypass


# ---------------------------------------------------------------------------
# WNO3dBlock (Wavelet Neural Operator Block)
# ---------------------------------------------------------------------------

class WNO3dBlock(nn.Module):
    def __init__(self, channels: int, levels: int = 3):
        super().__init__()
        self.levels = levels
        self.decomps = nn.ModuleList()
        self.recons = nn.ModuleList()
        self.mixers = nn.ModuleList()

        for _ in range(levels):
            self.decomps.append(
                nn.Conv3d(channels, channels, kernel_size=2, stride=2, groups=channels)
            )
            self.recons.append(
                nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2, groups=channels)
            )
            self.mixers.append(
                nn.Conv3d(channels, channels, kernel_size=1)
            )

        self.global_mix = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bypass = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bypass = self.bypass(x)
        coeffs = []
        curr_x = x
        for i in range(self.levels):
            curr_x = self.decomps[i](curr_x)
            coeffs.append(curr_x)

        curr_x = self.global_mix(curr_x)

        for i in reversed(range(self.levels)):
            mixed_coeff = self.mixers[i](coeffs[i])
            curr_x = curr_x + mixed_coeff
            curr_x = self.recons[i](curr_x)

        return curr_x + x_bypass


# ---------------------------------------------------------------------------
# FNO3dBlock
# ---------------------------------------------------------------------------

class FNO3dBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 8):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight_real = nn.Parameter(
            scale * torch.rand(channels, channels, modes, modes, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.rand(channels, channels, modes, modes, modes)
        )
        self.bypass = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z = x.shape
        orig_dtype = x.dtype
        x_fp32     = x.to(torch.float32)

        x_ft   = torch.fft.rfftn(x_fp32, dim=[-3, -2, -1])
        out_ft = torch.zeros_like(x_ft)

        mx = min(self.modes, x_ft.shape[2])
        my = min(self.modes, x_ft.shape[3])
        mz = min(self.modes, x_ft.shape[4])

        weight = torch.complex(self.weight_real, self.weight_imag)
        out_ft[:, :, :mx, :my, :mz] = torch.einsum(
            "bcxyz,cdxyz->bdxyz",
            x_ft[:, :, :mx, :my, :mz],
            weight[:, :, :mx, :my, :mz],
        )

        x_no = torch.fft.irfftn(out_ft, s=(X, Y, Z), dim=[-3, -2, -1])
        x_no = x_no.to(orig_dtype)
        del x_ft, out_ft

        return x_no + self.bypass(x)


# ---------------------------------------------------------------------------
# NOGlobalBranch
# ---------------------------------------------------------------------------

class NOGlobalBranch(nn.Module):
    """
    Single universal branch — operates at a fixed `channels` dimension (e.g. 64).
    Each calling layer is responsible for projecting its own features to/from
    this fixed dimension.

    Relative position encoding is fused into the feature before processing.
    """
    def __init__(
        self,
        channels: int,
        modes: int = 8,
        grid_size: tuple = (64, 64, 64),
        norm_layer=nn.LayerNorm,
        NO_type: str = "FNO",

        # Mamba-specific hyperparameters passed down to MambaBlock
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.NO_type = NO_type.upper()  
        self.grid_size = grid_size
        self.channels = channels
        
        # Dispatch Blocks Uniformly using uppercase keys
        if self.NO_type == "FNO":
            self.no = FNO3dBlock(channels, modes)
        elif self.NO_type == "WNO":
            self.no = WNO3dBlock(channels, levels=3)  
        elif self.NO_type == "MLP":
            self.no = MLP3dBlock(channels)
        elif self.NO_type == "CNN":
            self.no = CNN3dBlock(channels)
        elif self.NO_type == "MAMBA":
            self.no = MambaBlock(
                channels=channels, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
        else:
            raise ValueError(f"Unknown NO_type: {NO_type}. Supported: FNO, WNO, MLP, CNN, Mamba")
        
        # Post-operator normalization
        self.norm = norm_layer(channels)
        
        # --- FIX: Pre-operator normalization to stabilize inputs to Mamba ---
        self.pre_norm = norm_layer(channels)
        
        self.rel_pos_enc = RelativePositionEncoding(channels)

    def forward(self, feat: torch.Tensor, coord: torch.Tensor, offset: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            feat   : (N, C) — already projected to universal_dim by the caller
            coord  : (N, 3) — grid_coord or coord of the current stage
            offset : (B,)   — optional Pointcept batch displacement tracking array
        Returns:
            (N, C) — globally enriched features, same shape as input
        """
        C = feat.shape[1]
        device, dtype = feat.device, feat.dtype

        # --- 1. Fuse relative position encoding into features (Common to all paths) ---
        pos_enc = self.rel_pos_enc(coord).to(dtype)   # (N, C)
        feat = feat + pos_enc                         # in-place add, keeps grad

        # =========================================================================
        # PARADIGM A: GRID-FREE SEQUENTIAL PATH (Mamba)
        # =========================================================================
        if self.NO_type == "MAMBA":
            # --- FIX: Stabilize feature variance before executing selective scans ---
            feat = self.pre_norm(feat)
            
            # Pass directly to our standalone MambaBlock
            feat_out = self.no(feat, coord, offset)
            return self.norm(feat_out)

        # =========================================================================
        # PARADIGM B: TRADITIONAL DENSE 3D GRID PROCESSING PATH (FNO, WNO, CNN, MLP)
        # =========================================================================
        else:
            Gx, Gy, Gz = self.grid_size
            
            # --- 2. Scatter points onto a dense regular grid ---
            coord_f = coord.float()
            coord_min = coord_f.min(dim=0).values
            coord_max = coord_f.max(dim=0).values
            scale = (coord_max - coord_min).clamp(min=1.0)
            G_max = torch.tensor(
                [Gx - 1, Gy - 1, Gz - 1], device=device, dtype=torch.float32
            )
            
            # Safe voxel-index coordinate clamping per axis
            shifted = ((coord_f - coord_min) / scale * G_max).long()
            shifted_x = shifted[:, 0].clamp(0, Gx - 1)
            shifted_y = shifted[:, 1].clamp(0, Gy - 1)
            shifted_z = shifted[:, 2].clamp(0, Gz - 1)
            flat_idx = shifted_x * (Gy * Gz) + shifted_y * Gz + shifted_z

            grid_flat = torch.zeros(Gx * Gy * Gz, C, device=device, dtype=dtype)
            count = torch.zeros(Gx * Gy * Gz, 1, device=device, dtype=dtype)
            grid_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), feat)
            count.scatter_add_(0, flat_idx.unsqueeze(1),
                               torch.ones(feat.shape[0], 1, device=device, dtype=dtype))
            grid_flat = grid_flat / count.clamp(min=1.0)

            # --- 3. Run selected dense grid operator ---
            grid = grid_flat.view(1, Gx, Gy, Gz, C).permute(0, 4, 1, 2, 3).contiguous()
            grid_out = self.no(grid)                     # (1, C, Gx, Gy, Gz)

            # --- 4. Gather back to points ---
            feat_out = grid_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, C)[flat_idx]
            del grid_flat, count, grid, grid_out

            return self.norm(feat_out)

# ---------------------------------------------------------------------------
# NOFusedGridPooling (Fixed: Added point.offset passage)
# ---------------------------------------------------------------------------

class NOFusedGridPooling(PointModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_layer=None,
        act_layer=None,
        enable_no: bool = False,
        fusion: str = "add",
        reduce: str = "max",
        shuffle_orders: bool = True,
        universal_no_branch: nn.Module = None,
        universal_dim: int = 64,
    ):
        super().__init__()
        self.pool = GridPooling(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
            reduce=reduce,
            shuffle_orders=shuffle_orders,
            traceable=True,
        )
        self.enable_no  = enable_no
        self.fusion      = fusion

        if enable_no:
            assert universal_no_branch is not None, \
                "universal_no_branch must be provided when enable_no=True"
            self.no_branch = universal_no_branch

            self.down_proj = nn.Linear(in_channels, universal_dim)
            self.up_proj   = nn.Linear(universal_dim, out_channels)

            self.gate = nn.Parameter(torch.full((1,), -4.0))
            self.proj_concat = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.LayerNorm(out_channels),
            )

    def forward(self, point: Point) -> Point:
        if self.enable_no:
            # Compress to universal dim → NO → expand to out_channels
            feat_down   = self.down_proj(point.feat)
            # FIX: Explicitly pass point.offset to isolate batch elements in MambaBlock
            global_feat = self.no_branch(feat_down, point.grid_coord, offset=point.offset)
            feat_global = self.up_proj(global_feat)

        point = self.pool(point)

        if self.enable_no:
            inv            = point.pooling_inverse
            feat_no_coarse = torch_scatter.scatter_max(feat_global, inv, dim=0)[0]

            if self.fusion == "add":
                unused     = sum(p.sum() * 0 for p in self.proj_concat.parameters())
                point.feat = point.feat + self.gate.sigmoid() * feat_no_coarse + unused
            elif self.fusion == "concat":
                unused     = self.gate.sum() * 0
                point.feat = self.proj_concat(
                    torch.cat([point.feat, feat_no_coarse], dim=-1)
                ) + unused

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


# ---------------------------------------------------------------------------
# NOFusedGridUnpooling (Fixed: Added point.offset passage)
# ---------------------------------------------------------------------------

class NOFusedGridUnpooling(PointModule):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_layer=None,
        act_layer=None,
        enable_no: bool = False,
        use_skip: bool = True,
        fusion: str = "add",
        learnable_stage_weights: bool = False,
        universal_no_branch: nn.Module = None,
        universal_dim: int = 64,
    ):
        super().__init__()
        self.unpool = GridUnpooling(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            act_layer=act_layer,
            traceable=False,
        )
        self.enable_no              = enable_no
        self.use_skip               = use_skip
        self.fusion                 = fusion
        self.learnable_stage_weights = learnable_stage_weights

        if learnable_stage_weights:
            self.stage_weight = nn.Parameter(torch.ones(1))
        else:
            self.stage_weight = None

        if enable_no:
            assert universal_no_branch is not None, \
                "universal_no_branch must be provided when enable_no=True"
            self.no_branch = universal_no_branch

            self.down_proj = nn.Linear(in_channels, universal_dim)
            self.up_proj   = nn.Linear(universal_dim, out_channels)

            self.gate = nn.Parameter(torch.full((1,), -4.0))
            self.proj_concat = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.LayerNorm(out_channels),
            )

    def forward(self, point: Point) -> Point:
        if self.enable_no:
            feat_down   = self.down_proj(point.feat)
            # FIX: Explicitly pass point.offset to isolate batch elements in MambaBlock
            global_feat = self.no_branch(feat_down, point.grid_coord, offset=point.offset)
            feat_global = self.up_proj(global_feat)

        parent = point.pooling_parent
        if not self.use_skip:
            parent.feat = torch.zeros_like(parent.feat)
        elif self.learnable_stage_weights and self.stage_weight is not None:
            parent.feat = parent.feat * self.stage_weight

        inverse = point.pooling_inverse
        point   = self.unpool(point)

        if self.enable_no:
            feat_no_fine = feat_global[inverse]

            if self.fusion == "add":
                unused     = sum(p.sum() * 0 for p in self.proj_concat.parameters())
                point.feat = point.feat + self.gate.sigmoid() * feat_no_fine + unused
            elif self.fusion == "concat":
                unused     = self.gate.sum() * 0
                point.feat = self.proj_concat(
                    torch.cat([point.feat, feat_no_fine], dim=-1)
                ) + unused

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


# ---------------------------------------------------------------------------
# PointTransformerV3_NO
# ---------------------------------------------------------------------------

@MODELS.register_module("PT-v3m1-NO-SharedBranch")
class PointTransformerV3_NO(PointModule):
    def __init__(
        self,
        in_channels: int = 6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio: int = 4,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        no_stages=(True, True, True, True),
        fno_modes: int = 16,
        base_grid_size: tuple = (128, 128, 128),
        use_skip: bool = True,
        fusion: str = "concat",
        learnable_stage_weights: bool = True,
        share_no_branch: bool = True,
        universal_dim: int = 128,
        NO_type: str = "FNO",
        pool_reduce: str = "max",
    ):
        super().__init__()
        self.num_stages     = len(enc_depths)
        self.order          = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders

        bn_layer  = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer  = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, enc_channels[0], bn_layer, act_layer)
        dec_channels_  = list(dec_channels) + [enc_channels[-1]]

        if share_no_branch:
            self.universal_no_branch = NOGlobalBranch(
                channels=universal_dim,
                modes=fno_modes,
                grid_size=base_grid_size,
                norm_layer=ln_layer,
                NO_type=NO_type
            )
        else:
            self.universal_no_branch = None

        # Encoder Path
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    NOFusedGridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        enable_no=no_stages[s - 1],
                        fusion=fusion,
                        reduce=pool_reduce,
                        shuffle_orders=shuffle_orders,
                        universal_no_branch=self.universal_no_branch,
                        universal_dim=universal_dim,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.enc.add(module=enc, name=f"enc{s}")

        # Decoder Path
        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
        ]
        self.dec = PointSequential()
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[
                sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
            ][::-1]
            dec = PointSequential()

            dec.add(
                NOFusedGridUnpooling(
                    in_channels=dec_channels_[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels_[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                    enable_no=no_stages[s],
                    use_skip=use_skip,
                    fusion=fusion,
                    learnable_stage_weights=learnable_stage_weights,
                    universal_no_branch=self.universal_no_branch,
                    universal_dim=universal_dim,
                ),
                name="up",
            )

            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels_[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict: dict) -> Point:
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point