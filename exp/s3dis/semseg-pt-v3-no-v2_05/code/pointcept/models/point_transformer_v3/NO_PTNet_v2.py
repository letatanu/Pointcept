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




# ---------------------------------------------------------------------------
# GridPooling  (from Sonata / PT-v3m2)
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
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord, sorted=True,
            return_inverse=True, return_counts=True, dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
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
# GridUnpooling  (from Sonata / PT-v3m2)
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
    """
    Encodes normalized relative 3D coordinates into a channel vector.
    Inputs  : coord (N, 3) — raw or grid coordinates
    Output  : (N, out_channels)
    Features: (dx, dy, dz, radius) relative to the per-batch centroid.
    """
    def __init__(self, out_channels: int):
        super().__init__()
        # 4 raw geometry features → out_channels
        self.mlp = nn.Sequential(
            nn.Linear(4, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        # coord: (N, 3) float
        coord_f   = coord.float()
        centroid  = coord_f.mean(dim=0, keepdim=True)          # (1, 3)
        delta     = coord_f - centroid                          # (N, 3)
        radius    = delta.norm(dim=1, keepdim=True)             # (N, 1)
        # normalize to [-1, 1] for each axis
        scale = delta.abs().max(dim=0, keepdim=True).values.clamp(min=1e-6)
        delta_n   = delta / scale                               # (N, 3)
        geo_feat  = torch.cat([delta_n, radius / (radius.max() + 1e-6)], dim=1)  # (N, 4)
        return self.mlp(geo_feat)                               # (N, out_channels)


# ---------------------------------------------------------------------------
# MLP3dBlock (Global-Local Context MLP)
# Replaces WNO3dBlock/FNO3dBlock — Extremely fast, AMP safe, lowest memory
# ---------------------------------------------------------------------------

class MLP3dBlock(nn.Module):
    def __init__(self, channels: int, expansion_ratio: float = 2.0):
        super().__init__()
        hidden_dim = int(channels * expansion_ratio)
        
        # 1. Global Context MLP: Extracts the macro-shape of the room
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, channels, kernel_size=1)
        )
        
        # 2. Local Channel Mixer: Pointwise feature transformation
        self.local_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, channels, kernel_size=1)
        )
        
        # Bypass connection
        self.bypass = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, Gx, Gy, Gz)
        x_bypass = self.bypass(x)
        
        # --- Extract Global 3D Context ---
        # Pools (B, C, Gx, Gy, Gz) -> (B, C, 1, 1, 1)
        global_feat = self.global_pool(x)
        global_feat = self.global_mlp(global_feat) 
        
        # --- Transform Local Features ---
        local_feat = self.local_mlp(x) 
        
        # --- Fuse Global Context into Local Features ---
        # The (1,1,1) global_feat broadcasts naturally over (Gx, Gy, Gz)
        out = local_feat + global_feat 
        
        return out + x_bypass

# ---------------------------------------------------------------------------
# WNO3dBlock (Wavelet Neural Operator Block)
# Replaces FNO3dBlock — Drop-in replacement, fully AMP safe
# ---------------------------------------------------------------------------

class WNO3dBlock(nn.Module):
    def __init__(self, channels: int, levels: int = 3):
        """
        levels: Number of wavelet decomposition levels. 
                For a 64x64x64 grid, 3 levels decomposes down to an 8x8x8 
                global context grid, mimicking the lowest Fourier modes.
        """
        super().__init__()
        self.levels = levels
        
        self.decomps = nn.ModuleList()
        self.recons = nn.ModuleList()
        self.mixers = nn.ModuleList()

        # Learnable Wavelet Basis (Depthwise Convolutions)
        for _ in range(levels):
            # Analysis: Extract low/high frequency coefficients (like DWT)
            self.decomps.append(
                nn.Conv3d(channels, channels, kernel_size=2, stride=2, groups=channels)
            )
            # Synthesis: Reconstruct spatial signal (like Inverse DWT)
            self.recons.append(
                nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2, groups=channels)
            )
            # Operator: Mix wavelet coefficients across channels
            self.mixers.append(
                nn.Conv3d(channels, channels, kernel_size=1)
            )

        # Base operator for the deepest global context
        self.global_mix = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        
        # Spatial bypass connection
        self.bypass = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, Gx, Gy, Gz)
        x_bypass = self.bypass(x)

        # ---------------------------------------------------------
        # 1. Multi-scale Wavelet Decomposition (Analysis)
        # ---------------------------------------------------------
        coeffs = []
        curr_x = x
        for i in range(self.levels):
            curr_x = self.decomps[i](curr_x)
            coeffs.append(curr_x)

        # ---------------------------------------------------------
        # 2. Global Context Mixing (Lowest Frequency Band)
        # ---------------------------------------------------------
        curr_x = self.global_mix(curr_x)

        # ---------------------------------------------------------
        # 3. Wavelet Reconstruction (Synthesis)
        # ---------------------------------------------------------
        for i in reversed(range(self.levels)):
            # Mix the coefficients at this scale
            mixed_coeff = self.mixers[i](coeffs[i])
            # Add wavelet detail and upsample
            curr_x = curr_x + mixed_coeff
            curr_x = self.recons[i](curr_x)

        return curr_x + x_bypass

# ---------------------------------------------------------------------------
# FNO3dBlock  (AMP-safe, no internal bottleneck — universal dim is fixed)
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
# NOGlobalBranch  (Universal — fixed channels, with relative pos encoding)
# ---------------------------------------------------------------------------

class NOGlobalBranch(nn.Module):
    """
    Single universal branch — operates at a fixed `channels` dimension (e.g. 64).
    Each calling layer is responsible for projecting its own features to/from
    this fixed dimension.

    NEW: relative position encoding is fused into the feature before gridding.
    """
    def __init__(
        self,
        channels: int,
        modes: int = 8,
        grid_size: tuple = (64, 64, 64),
        norm_layer=nn.LayerNorm,
        NO_type: str = "FNO",
    ):
        super().__init__()
        if NO_type == "FNO":
            self.no = FNO3dBlock(channels, modes)
        elif NO_type == "WNO":
            self.no = WNO3dBlock(channels, levels=3)  # Using 3-level WNO as default alternative
        elif NO_type == "MLP":
            self.no = MLP3dBlock(channels)
        self.norm      = norm_layer(channels)
        self.grid_size = grid_size
        # Relative position encoding → fused into features before FFT
        self.rel_pos_enc = RelativePositionEncoding(channels)

    def forward(self, feat: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat  : (N, C) — already projected to universal_dim by the caller
            coord : (N, 3) — grid_coord or coord of the current stage
        Returns:
            (N, C) — spectral-enriched features, same shape as input
        """
        Gx, Gy, Gz  = self.grid_size
        C            = feat.shape[1]
        device, dtype = feat.device, feat.dtype

        # --- 1. Fuse relative position encoding into features ---
        pos_enc = self.rel_pos_enc(coord).to(dtype)   # (N, C)
        feat    = feat + pos_enc                       # in-place add, keeps grad

        # --- 2. Scatter points onto a dense regular grid ---
        coord_f   = coord.float()
        coord_min = coord_f.min(dim=0).values
        coord_max = coord_f.max(dim=0).values
        scale     = (coord_max - coord_min).clamp(min=1.0)
        G_max     = torch.tensor(
            [Gx - 1, Gy - 1, Gz - 1], device=device, dtype=torch.float32
        )
        shifted  = ((coord_f - coord_min) / scale * G_max).long()
        shifted  = shifted.clamp(
            min=torch.zeros(3, device=device, dtype=torch.long),
            max=G_max.long(),
        )
        flat_idx  = shifted[:, 0] * (Gy * Gz) + shifted[:, 1] * Gz + shifted[:, 2]

        grid_flat = torch.zeros(Gx * Gy * Gz, C, device=device, dtype=dtype)
        count     = torch.zeros(Gx * Gy * Gz, 1, device=device, dtype=dtype)
        grid_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), feat)
        count.scatter_add_(0, flat_idx.unsqueeze(1),
                           torch.ones(feat.shape[0], 1, device=device, dtype=dtype))
        grid_flat = grid_flat / count.clamp(min=1.0)

        # --- 3. FNO3dBlock on dense grid ---
        grid     = grid_flat.view(1, Gx, Gy, Gz, C).permute(0, 4, 1, 2, 3).contiguous()
        grid_out = self.no(grid)                     # (1, C, Gx, Gy, Gz)

        # --- 4. Gather back to points ---
        feat_out = grid_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, C)[flat_idx]
        del grid_flat, count, grid, grid_out

        return self.norm(feat_out)


# ---------------------------------------------------------------------------
# NOFusedGridPooling
# Replaces NOFusedPooling; uses GridPooling + Universal NO branch
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

            # Per-layer lightweight adapters (cheap)
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
            global_feat = self.no_branch(feat_down, point.grid_coord)
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
# NOFusedGridUnpooling
# Replaces NOFusedUnpooling; uses GridUnpooling + Universal NO branch
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
            global_feat = self.no_branch(feat_down, point.grid_coord)
            feat_global = self.up_proj(global_feat)

        # Optional skip-connection scaling
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
# PointTransformerV3_NO  — Universal Branch + Grid Pool/Unpool
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
        # ---- NO parameters ----
        no_stages=(True, True, True, True),
        fno_modes: int = 16,
        base_grid_size: tuple = (128, 128, 128),
        use_skip: bool = True,
        fusion: str = "concat",
        learnable_stage_weights: bool = True,
        share_no_branch: bool = True,
        universal_dim: int = 128,
        NO_type: str = "FNO",
        # ---- GridPooling parameters ----
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

        # ==============================================================
        # ONE Universal NOGlobalBranch — shared by ALL layers
        # ==============================================================
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

        # ------------------------------------------------------------------ #
        # Encoder
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # Decoder
        # ------------------------------------------------------------------ #
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