"""
PTv3-NO: Point Transformer V3 + Neural Operator Global Branch
=============================================================
Key changes vs. original ptv3_no.py:
  1. NOGlobalBranch uses a FIXED grid allocation (grid_size param) instead of
     dynamic bounding-box-derived X,Y,Z.  This eliminates CUDA allocator
     fragmentation and DDP memory imbalance.
  2. Coordinates are normalised to the fixed canonical grid [0, G-1]^3 —
     the correct FNO formulation (function-to-function, resolution-independent).
  3. FNO3dBlock.bypass remains nn.Linear (avoids DDP grad-stride warning).
  4. Explicit del of large intermediates in both FNO3dBlock and NOGlobalBranch.
  5. grid_size is exposed as a model hyper-parameter (default 64^3 at each stage).
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
# FNO3dBlock
# ---------------------------------------------------------------------------

class FNO3dBlock(nn.Module):
    """
    3-D Fourier Neural Operator block.
    Operates on a dense grid  x : [B, C, X, Y, Z].

    The bypass uses nn.Linear (2-D contiguous weight) instead of
    nn.Conv3d(kernel_size=1) whose [C,C,1,1,1] weight is non-contiguous
    under DDP, causing grad-stride warnings and extra gradient copies.
    """

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
        # Contiguous 2-D weight  [C_out, C_in]  — DDP zero-copy safe
        self.bypass = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z = x.shape

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
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
        # Free large FFT buffers BEFORE the bypass to reduce peak memory
        del x_ft, out_ft

        # bypass: [..., C] for Linear, then restore grid layout
        x_bypass = self.bypass(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x_no + x_bypass


# ---------------------------------------------------------------------------
# NOGlobalBranch  —  fixed-grid version
# ---------------------------------------------------------------------------

class NOGlobalBranch(PointModule):
    """
    Maps point features to a FIXED dense 3-D grid, applies FNO3dBlock, then
    gathers output back to points via exact index lookup.

    Fixed grid design
    -----------------
    The current (broken) design derives grid dimensions X,Y,Z from each
    scene's bounding box at runtime.  This produces differently-sized tensors
    every forward pass, fragmenting the CUDA caching allocator and causing DDP
    memory imbalance.

    The correct Neural-Operator formulation operates on a canonical domain
    [0, 1]^3 (here discretised to [0, G-1]^3).  FNO learns a kernel in
    frequency space that is INDEPENDENT of physical scale — it does not need to
    know whether the grid represents 5 m or 50 m.  Normalising coordinates to
    a fixed grid is therefore both architecturally correct and memory-efficient:
    PyTorch's CUDA caching allocator sees the same allocation request every
    step and reuses the exact cached block → zero fragmentation.

    Parameters
    ----------
    channels  : feature dimension C
    modes     : number of Fourier modes per spatial dimension
    grid_size : (Gx, Gy, Gz)  — fixed canonical grid, same every forward pass.
                Tune per stage:
                  Stage 3 (÷8 coarsened): (64, 64, 64) is usually sufficient
                  Stage 4 (÷16 coarsened): (32, 32, 32) may suffice
                Memory scales as  Gx*Gy*Gz * C * 4 bytes:
                  64^3  × 256ch ≈  67 MB
                  128^3 × 256ch ≈ 537 MB
    """

    def __init__(
        self,
        channels: int,
        modes: int = 8,
        grid_size: tuple = (64, 64, 64),
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.fno = FNO3dBlock(channels, modes)
        self.norm = norm_layer(channels)
        self.grid_size = grid_size  # (Gx, Gy, Gz) — fixed, never changes

    def forward(self, point: Point) -> torch.Tensor:
        feat  = point.feat        # [N, C]
        coord = point.grid_coord  # [N, 3]  integer voxel coords from GridSample
        Gx, Gy, Gz = self.grid_size
        C = feat.shape[1]
        device, dtype = feat.device, feat.dtype

        # ---- Normalise coordinates to fixed canonical grid [0, G-1]^3 ----
        coord_f = coord.float()
        coord_min = coord_f.min(dim=0).values
        coord_max = coord_f.max(dim=0).values
        scale = (coord_max - coord_min).clamp(min=1.0)  # avoid div-by-zero
        G_max = torch.tensor(
            [Gx - 1, Gy - 1, Gz - 1], device=device, dtype=torch.float32
        )
        shifted = ((coord_f - coord_min) / scale * G_max).long()
        shifted = shifted.clamp(
            min=torch.zeros(3, device=device, dtype=torch.long),
            max=G_max.long(),
        )

        # ---- Scatter points → dense grid (mean aggregation) ----
        # flat_idx shape: [N]
        flat_idx = shifted[:, 0] * (Gy * Gz) + shifted[:, 1] * Gz + shifted[:, 2]

        # Fixed-size allocation — same shape EVERY call → CUDA block reuse
        grid_flat = torch.zeros(Gx * Gy * Gz, C, device=device, dtype=dtype)
        count     = torch.zeros(Gx * Gy * Gz, 1, device=device, dtype=dtype)

        grid_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), feat)
        count.scatter_add_(
            0,
            flat_idx.unsqueeze(1),
            torch.ones(feat.shape[0], 1, device=device, dtype=dtype),
        )
        grid_flat = grid_flat / count.clamp(min=1.0)

        # [1, C, Gx, Gy, Gz]
        grid = grid_flat.view(1, Gx, Gy, Gz, C).permute(0, 4, 1, 2, 3).contiguous()

        # ---- Apply global FNO operator ----
        grid_out = self.fno(grid)

        # ---- Gather back to points (exact lookup — no trilinear needed) ----
        feat_out = grid_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, C)[flat_idx]

        # Explicit free — prevents residual fragmentation
        del grid_flat, count, grid, grid_out

        return self.norm(feat_out)  # [N, C]


# ---------------------------------------------------------------------------
# NOFusedPooling
# ---------------------------------------------------------------------------

class NOFusedPooling(PointModule):
    """
    Wraps SerializedPooling with an optional parallel NO global branch.

    Flow (enable_no=True):
      1. Run NOGlobalBranch on FINE points  →  feat_global [N_fine, C_out]
      2. Run SerializedPooling              →  point_coarse
      3. Max-pool feat_global fine→coarse using pooling_inverse
      4. Add to point_coarse.feat  (additive fusion)
    """

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
    ):
        super().__init__()
        self.pool = SerializedPooling(
            in_channels, out_channels, stride, norm_layer, act_layer
        )
        self.enable_no = enable_no
        self.fusion = fusion
        if enable_no:
            self.no_branch = NOGlobalBranch(
                in_channels,
                modes,
                grid_size=grid_size,
                norm_layer=norm_layer if norm_layer else nn.LayerNorm,
            )
            self.proj_no = nn.Linear(in_channels, out_channels)

    def forward(self, point: Point) -> Point:
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))  # [N_fine, C_out]

        point = self.pool(point)  # SerializedPooling — unchanged

        if self.enable_no:
            inv = point.pooling_inverse
            # Max-pool fine NO features to coarse resolution
            feat_no_coarse = torch_scatter.scatter_max(feat_global, inv, dim=0)[0]
            if self.fusion == "add":
                point.feat = point.feat + feat_no_coarse
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point


# ---------------------------------------------------------------------------
# NOFusedUnpooling
# ---------------------------------------------------------------------------

class NOFusedUnpooling(PointModule):
    """
    Wraps SerializedUnpooling with an optional parallel NO global branch.

    Flow (enable_no=True):
      1. Run NOGlobalBranch on COARSE points  →  feat_global [N_coarse, C_out]
      2. (Ablation) Zero skip features if use_skip=False
      3. Run SerializedUnpooling              →  point_fine
      4. Broadcast feat_global coarse→fine using pooling_inverse
      5. Add to point_fine.feat
    """

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
    ):
        super().__init__()
        self.unpool = SerializedUnpooling(
            in_channels, skip_channels, out_channels, norm_layer, act_layer
        )
        self.enable_no = enable_no
        self.use_skip = use_skip
        self.fusion = fusion
        if enable_no:
            self.no_branch = NOGlobalBranch(
                in_channels,
                modes,
                grid_size=grid_size,
                norm_layer=norm_layer if norm_layer else nn.LayerNorm,
            )
            self.proj_no = nn.Linear(in_channels, out_channels)

    def forward(self, point: Point) -> Point:
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))  # [N_coarse, C_out]

        # Ablation: zero out encoder skip features to test NO as sole global path
        if not self.use_skip:
            parent = point.pooling_parent
            parent.feat = torch.zeros_like(parent.feat)

        # Capture inverse BEFORE unpool consumes it
        inverse = point.pooling_inverse
        point = self.unpool(point)  # returns FINE (parent) points

        if self.enable_no:
            # Broadcast coarse NO features to fine points via inverse mapping
            feat_no_fine = feat_global[inverse]
            if self.fusion == "add":
                point.feat = point.feat + feat_no_fine
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point


# ---------------------------------------------------------------------------
# PointTransformerV3_NO
# ---------------------------------------------------------------------------

@MODELS.register_module("PT-v3m1-NO")
class PointTransformerV3_NO(PointModule):
    """
    Standalone PTv3 backbone augmented with a parallel Neural Operator (NO)
    global branch.

    Architecture
    ------------
    - Serialized path (unchanged from PTv3):
        Embedding → SerializedPooling → Block → SerializedUnpooling
        Handles local patch-wise attention via space-filling curves.

    - NO branch (new, parallel):
        NOGlobalBranch: scatter to fixed 3-D grid → FNO3dBlock → gather back.
        Active only at deeper stages (no_stages) where the grid is coarse.
        Fused by addition into the serialized path at each active stage.

    Key parameters
    --------------
    no_stages   : tuple[bool] of length (num_stages - 1).
                  True means NOFusedPooling/Unpooling at that transition.
                  Default (False, False, True, True) — active at stages 3 & 4.
    fno_modes   : Fourier modes per dimension in FNO3dBlock.
    grid_size   : Fixed canonical grid (Gx, Gy, Gz) for NOGlobalBranch.
                  Exposed here so it can be swept in ablations.
    use_skip    : If False, encoder skip connections are zeroed before unpooling
                  (ablation: test whether NO can replace U-Net global path).
    fusion      : Currently "add" only.
    """

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
        no_stages=(False, False, True, True),
        fno_modes: int = 8,
        grid_size: tuple = (64, 64, 64),
        use_skip: bool = True,
        fusion: str = "add",
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, enc_channels[0], bn_layer, act_layer)

        # ------------------------------------------------------------------ #
        # Encoder                                                              #
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
                enc.add(
                    NOFusedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        modes=fno_modes,
                        grid_size=grid_size,
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        enable_no=no_stages[s - 1],
                        fusion=fusion,
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
        # Decoder                                                              #
        # ------------------------------------------------------------------ #
        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
        ]
        self.dec = PointSequential()
        dec_channels_ = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[
                sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
            ][::-1]
            dec = PointSequential()
            dec.add(
                NOFusedUnpooling(
                    in_channels=dec_channels_[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels_[s],
                    modes=fno_modes,
                    grid_size=grid_size,
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                    enable_no=no_stages[s],
                    use_skip=use_skip,
                    fusion=fusion,
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
