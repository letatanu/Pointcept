"""
PTv3-NO Enhanced — Shared NOGlobalBranch
=========================================
Key change: NOGlobalBranch instances are shared across encoder/decoder
transitions that operate on the **same channel dimension**, dramatically
reducing parameter count.

Sharing strategy:
  - A nn.ModuleDict keyed by channel count holds one NOGlobalBranch per
    unique active channel size.
  - Both NOFusedPooling and NOFusedUnpooling accept an optional
    `shared_no_branch` argument; when provided the external branch is used
    instead of creating a new one.
  - Projection heads (proj_no, gate, proj_concat) remain per-module because
    they map from the shared latent space to the local output dim.
"""

import torch
import torch.nn as nn
import torch_scatter
from functools import partial

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Block, Embedding, SerializedPooling, SerializedUnpooling,
)


# ---------------------------------------------------------------------------
# FNO3dBlock (With Bottleneck)
# ---------------------------------------------------------------------------

class FNO3dBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 8, max_fno_channels: int = 64):
        super().__init__()
        self.modes = modes
        
        # Bottleneck: cap the channels used in the expensive C x C x modes^3 multiplication
        self.use_bottleneck = channels > max_fno_channels
        self.fno_channels = max_fno_channels if self.use_bottleneck else channels

        if self.use_bottleneck:
            self.in_proj = nn.Conv3d(channels, self.fno_channels, 1)
            self.out_proj = nn.Conv3d(self.fno_channels, channels, 1)

        scale = 1.0 / (self.fno_channels * self.fno_channels)
        self.weight_real = nn.Parameter(
            scale * torch.rand(self.fno_channels, self.fno_channels, modes, modes, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.rand(self.fno_channels, self.fno_channels, modes, modes, modes)
        )
        
        self.bypass = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, X, Y, Z
        B, C, X, Y, Z = x.shape
        
        # Bottleneck projection
        x_fno = self.in_proj(x) if self.use_bottleneck else x

        # ==========================================================
        # AMP FIX: Force FFT and frequency math to float32
        # ==========================================================
        orig_dtype = x_fno.dtype
        x_fno_fp32 = x_fno.to(torch.float32)

        x_ft = torch.fft.rfftn(x_fno_fp32, dim=[-3, -2, -1])
        out_ft = torch.zeros_like(x_ft)

        mx = min(self.modes, x_ft.shape[2])
        my = min(self.modes, x_ft.shape[3])
        mz = min(self.modes, x_ft.shape[4])

        # Weights are already float32 parameters, so this yields ComplexFloat
        weight = torch.complex(self.weight_real, self.weight_imag)
        
        out_ft[:, :, :mx, :my, :mz] = torch.einsum(
            "bcxyz,cdxyz->bdxyz",
            x_ft[:, :, :mx, :my, :mz],
            weight[:, :, :mx, :my, :mz],
        )

        x_no = torch.fft.irfftn(out_ft, s=(X, Y, Z), dim=[-3, -2, -1])
        
        # Cast back to original dtype (float16 if AMP is enabled)
        x_no = x_no.to(orig_dtype)
        # ==========================================================

        # Reverse bottleneck
        if self.use_bottleneck:
            x_no = self.out_proj(x_no)

        x_bypass = self.bypass(x)
        return x_no + x_bypass


# ---------------------------------------------------------------------------
# NOGlobalBranch 
# ---------------------------------------------------------------------------

class NOGlobalBranch(PointModule):
    def __init__(
        self,
        channels: int,
        modes: int = 8,
        grid_size: tuple = (64, 64, 64),
        norm_layer=nn.LayerNorm,
        max_fno_channels: int = 64,  # NEW
    ):
        super().__init__()
        self.fno = FNO3dBlock(channels, modes, max_fno_channels)
        self.norm = norm_layer(channels)
        self.grid_size = grid_size 

    def forward(self, point: Point) -> torch.Tensor:
        feat = point.feat 
        coord = point.grid_coord 
        Gx, Gy, Gz = self.grid_size
        C = feat.shape[1]
        device, dtype = feat.device, feat.dtype

        coord_f = coord.float()
        coord_min = coord_f.min(dim=0).values
        coord_max = coord_f.max(dim=0).values
        scale = (coord_max - coord_min).clamp(min=1.0) 
        G_max = torch.tensor(
            [Gx - 1, Gy - 1, Gz - 1], device=device, dtype=torch.float32
        )

        shifted = ((coord_f - coord_min) / scale * G_max).long()
        shifted = shifted.clamp(
            min=torch.zeros(3, device=device, dtype=torch.long),
            max=G_max.long(),
        )

        flat_idx = shifted[:, 0] * (Gy * Gz) + shifted[:, 1] * Gz + shifted[:, 2]

        grid_flat = torch.zeros(Gx * Gy * Gz, C, device=device, dtype=dtype)
        count = torch.zeros(Gx * Gy * Gz, 1, device=device, dtype=dtype)

        grid_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), feat)
        count.scatter_add_(
            0,
            flat_idx.unsqueeze(1),
            torch.ones(feat.shape[0], 1, device=device, dtype=dtype),
        )

        grid_flat = grid_flat / count.clamp(min=1.0)
        grid = grid_flat.view(1, Gx, Gy, Gz, C).permute(0, 4, 1, 2, 3).contiguous()

        grid_out = self.fno(grid)
        feat_out = grid_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, C)[flat_idx]

        del grid_flat, count, grid, grid_out

        return self.norm(feat_out) 


# ---------------------------------------------------------------------------
# NOFusedPooling  — accepts optional shared branch
# ---------------------------------------------------------------------------

class NOFusedPooling(PointModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        modes: int = 8,
        grid_size: tuple = (64, 64, 64),
        norm_layer=None,
        act_layer=None,
        enable_no: bool = False,
        fusion: str = "add",
        # ---- NEW: pass an externally-owned branch to share weights ----
        shared_no_branch: NOGlobalBranch = None,
    ):
        super().__init__()
        self.pool = SerializedPooling(
            in_channels, out_channels, stride, norm_layer, act_layer
        )
        self.enable_no = enable_no
        self.fusion     = fusion

        if enable_no:
            # Reuse shared branch if provided, otherwise own one.
            if shared_no_branch is not None:
                # Register as non-parameter reference so it doesn't double-count.
                self.no_branch = shared_no_branch
            else:
                self.no_branch = NOGlobalBranch(
                    in_channels, modes, grid_size=grid_size,
                    norm_layer=norm_layer if norm_layer else nn.LayerNorm,
                )

            # Per-module projection + gate remain independent (cheap).
            self.proj_no = nn.Linear(in_channels, out_channels)
            self.gate     = nn.Parameter(torch.full((1,), -4.0))
            self.proj_concat = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.LayerNorm(out_channels),
            )

    def forward(self, point: Point) -> Point:
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))

        point = self.pool(point)

        if self.enable_no:
            inv          = point.pooling_inverse
            feat_no_coarse = torch_scatter.scatter_max(feat_global, inv, dim=0)[0]

            if self.fusion == "add":
                unused = sum(p.sum() * 0 for p in self.proj_concat.parameters())
                point.feat = (
                    point.feat + self.gate.sigmoid() * feat_no_coarse + unused
                )
            elif self.fusion == "concat":
                unused = self.gate.sum() * 0
                point.feat = (
                    self.proj_concat(
                        torch.cat([point.feat, feat_no_coarse], dim=-1)
                    ) + unused
                )

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


# ---------------------------------------------------------------------------
# NOFusedUnpooling  — accepts optional shared branch
# ---------------------------------------------------------------------------

class NOFusedUnpooling(PointModule):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        modes: int = 8,
        grid_size: tuple = (64, 64, 64),
        norm_layer=None,
        act_layer=None,
        enable_no: bool = False,
        use_skip: bool = True,
        fusion: str = "add",
        learnable_stage_weights: bool = False,
        # ---- NEW: pass an externally-owned branch to share weights ----
        shared_no_branch: NOGlobalBranch = None,
    ):
        super().__init__()
        self.unpool = SerializedUnpooling(
            in_channels, skip_channels, out_channels, norm_layer, act_layer
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
            if shared_no_branch is not None:
                self.no_branch = shared_no_branch
            else:
                self.no_branch = NOGlobalBranch(
                    in_channels, modes, grid_size=grid_size,
                    norm_layer=norm_layer if norm_layer else nn.LayerNorm,
                )

            self.proj_no     = nn.Linear(in_channels, out_channels)
            self.gate         = nn.Parameter(torch.full((1,), -4.0))
            self.proj_concat = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.LayerNorm(out_channels),
            )

    def forward(self, point: Point) -> Point:
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))

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
                unused = sum(p.sum() * 0 for p in self.proj_concat.parameters())
                point.feat = (
                    point.feat + self.gate.sigmoid() * feat_no_fine + unused
                )
            elif self.fusion == "concat":
                unused = self.gate.sum() * 0
                point.feat = (
                    self.proj_concat(
                        torch.cat([point.feat, feat_no_fine], dim=-1)
                    ) + unused
                )

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


# ---------------------------------------------------------------------------
# Helper: build the shared branch registry
# ---------------------------------------------------------------------------

def _build_shared_no_branches(
    enc_channels,
    dec_channels_,
    no_stages,
    num_stages,
    fno_modes,
    base_grid_size,
    norm_layer,
) -> nn.ModuleDict:
    """
    Returns a ModuleDict keyed by str(channel_count).
    One NOGlobalBranch per unique channel size that appears in any
    active (enable_no=True) encoder or decoder transition.

    Encoder pooling at stage s  processes  enc_channels[s-1]   (s in 1..num_stages-1)
    Decoder unpooling at stage s processes  dec_channels_[s+1]  (s in 0..num_stages-2)

    Branches with the same channel count are the SAME object, so enc & dec
    at the same channel resolution automatically share weights.
    """
    active_channels = set()

    for s in range(1, num_stages):
        if no_stages[s - 1]:
            active_channels.add(enc_channels[s - 1])

    for s in range(num_stages - 1):
        if no_stages[s]:
            active_channels.add(dec_channels_[s + 1])

    return nn.ModuleDict(
        {
            str(c): NOGlobalBranch(
                channels=c,
                modes=fno_modes,
                grid_size=base_grid_size,   # shared branch uses base (fixed) grid
                norm_layer=norm_layer,
            )
            for c in active_channels
        }
    )


# ---------------------------------------------------------------------------
# PointTransformerV3_NO  — with shared NOGlobalBranch
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
        fno_modes: int = 12,
        base_grid_size: tuple = (64, 64, 64),
        adaptive_grid: bool = True,
        use_skip: bool = True,
        fusion: str = "concat",
        learnable_stage_weights: bool = True,
        # ---- NEW: weight-sharing flag ----
        share_no_branch: bool = True,
    ):
        super().__init__()
        self.num_stages     = len(enc_depths)
        self.order          = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders

        bn_layer  = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer  = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, enc_channels[0], bn_layer, act_layer)

        # dec_channels_ is used consistently in both enc and dec loops.
        dec_channels_ = list(dec_channels) + [enc_channels[-1]]

        # ------------------------------------------------------------------ #
        # Shared branch registry
        #   Keys  : str(channel_count)
        #   Values: NOGlobalBranch (one per unique channel size)
        #
        # Encoder pooling at stage s uses channels=enc_channels[s-1].
        # Decoder unpooling at stage s uses channels=dec_channels_[s+1].
        # Any two transitions that share the same channel count will reference
        # the SAME NOGlobalBranch object → shared weights.
        # ------------------------------------------------------------------ #
        if share_no_branch:
            self.shared_no_branches = _build_shared_no_branches(
                enc_channels, dec_channels_, no_stages,
                self.num_stages, fno_modes, base_grid_size, ln_layer,
            )
        else:
            self.shared_no_branches = None   # each module builds its own

        # ------------------------------------------------------------------ #
        # Encoder
        # ------------------------------------------------------------------ #
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                in_ch = enc_channels[s - 1]

                # Adaptive grid: halve resolution with depth
                grid_size = (
                    tuple(max(8, g // (2 ** s)) for g in base_grid_size)
                    if adaptive_grid and self.shared_no_branches is None
                    else base_grid_size   # shared branch has fixed grid_size
                )

                # Retrieve shared branch (or None → module will build its own)
                shared_branch = None
                if self.shared_no_branches is not None and str(in_ch) in self.shared_no_branches:
                    shared_branch = self.shared_no_branches[str(in_ch)]

                enc.add(
                    NOFusedPooling(
                        in_channels=in_ch,
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        modes=fno_modes,
                        grid_size=grid_size,
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        enable_no=no_stages[s - 1],
                        fusion=fusion,
                        shared_no_branch=shared_branch,
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

            in_ch = dec_channels_[s + 1]

            grid_size = (
                tuple(max(8, g // (2 ** s)) for g in base_grid_size)
                if adaptive_grid and self.shared_no_branches is None
                else base_grid_size
            )

            # Retrieve shared branch for this channel dim (may overlap with enc)
            shared_branch = None
            if self.shared_no_branches is not None and str(in_ch) in self.shared_no_branches:
                shared_branch = self.shared_no_branches[str(in_ch)]

            dec.add(
                NOFusedUnpooling(
                    in_channels=in_ch,
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels_[s],
                    modes=fno_modes,
                    grid_size=grid_size,
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                    enable_no=no_stages[s],
                    use_skip=use_skip,
                    fusion=fusion,
                    learnable_stage_weights=learnable_stage_weights,
                    shared_no_branch=shared_branch,
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