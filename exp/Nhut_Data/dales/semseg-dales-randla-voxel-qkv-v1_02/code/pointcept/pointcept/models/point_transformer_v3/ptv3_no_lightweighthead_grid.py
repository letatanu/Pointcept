"""
NOEncoder: PTv3-style encoder + Neural Operator global branches + lightweight
multi-scale upsampling head, with NO decoder and NO skip connections.

Design goals
------------
1. Keep the PTv3 encoder hierarchy and serialized attention blocks.
2. Inject global context with NOGlobalBranch during encoder downsampling.
3. Remove the entire decoder stage.
4. Replace decoder with a lightweight upsampling head:
   - project each encoder stage to a common channel width
   - nearest-neighbor upsample coarse stages to stage-0 resolution using
     pooling_inverse chains
   - sum fused multi-scale features
5. Return a finest-resolution Point object for DefaultSegmentorV2.

This file is self-contained and does not import PTv3 model classes.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

import flash_attn


from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


# -----------------------------------------------------------------------------
# Relative Positional Encoding
# -----------------------------------------------------------------------------

class RPE(nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)
        return out


class GridCellAttention(PointModule):
    """
    Strict Voxel-based Self-Attention.
    Groups points by their canonical fixed grid cell (e.g. 128x128x128)
    and applies FlashAttention strictly within each cell.
    """
    def __init__(
        self,
        channels,
        num_heads,
        grid_size=(128, 128, 128),
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.grid_size = grid_size

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop

        # Needs flash_attn for variable length sequences
        assert flash_attn is not None, "flash_attn is required for GridCellAttention"

    def forward(self, point: Point):
        feat = point.feat
        
        # 1. Retrieve cached grid indices
        grid_cache = point[f"fixed_grid_{self.grid_size}"]
        flat_idx = grid_cache.flat_idx
        order = grid_cache.order
        inverse = grid_cache.inverse

        # 2. Reorder features so cell members are contiguous
        sorted_feat = feat[order]
        sorted_idx = flat_idx[order]

        # 3. Find cell boundaries for variable-length FlashAttention
        _, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=feat.device),
            torch.cumsum(counts, dim=0, dtype=torch.int32)
        ])
        max_seqlen = counts.max().item()

        # 4. Compute QKV and apply Flash Attention
        qkv = self.qkv(sorted_feat)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.channels // self.num_heads)

        attn_out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.to(torch.bfloat16),
            cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        ).reshape(-1, self.channels).to(feat.dtype)

        # 5. Restore original point order and project
        feat_out = attn_out[inverse]
        feat_out = self.proj(feat_out)
        feat_out = self.proj_drop(feat_out)

        point.feat = feat_out
        return point

# -----------------------------------------------------------------------------
# Serialized Attention
# -----------------------------------------------------------------------------


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash

        if enable_flash:
            assert not enable_rpe
            assert not upcast_attention
            assert not upcast_softmax
            assert flash_attn is not None, "flash_attn is required when enable_flash=True"
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"

        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad

            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []

            for i in range(len(offset)):
                unpad[_offset[i]:_offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size): _offset_pad[i + 1] - self.patch_size
                    ]
                pad[_offset_pad[i]:_offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.cat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H)
                .permute(2, 0, 3, 1, 4)
                .unbind(dim=0)
            )
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)

        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


# -----------------------------------------------------------------------------
# Feed-forward
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -----------------------------------------------------------------------------
# PTv3 block
# -----------------------------------------------------------------------------

class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        grid_size=(128, 128, 128)
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        # self.attn = SerializedAttention(
        #     channels=channels,
        #     num_heads=num_heads,
        #     patch_size=patch_size,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=proj_drop,
        #     order_index=order_index,
        #     enable_rpe=enable_rpe,
        #     enable_flash=enable_flash,
        #     upcast_attention=upcast_attention,
        #     upcast_softmax=upcast_softmax,
        # )
        # In Block.__init__:
        self.norm1 = PointSequential(norm_layer(channels))
        
        # Alternate between Grid Cell Attention and Serialized Attention
        if order_index % 2 == 0:
            self.attn = GridCellAttention(
                channels=channels,
                num_heads=num_heads,
                grid_size=grid_size,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        else:
            self.attn = SerializedAttention(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    order_index=order_index,
                    enable_rpe=enable_rpe,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------

class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        return self.stem(point)


# -----------------------------------------------------------------------------
# Serialized Pooling
# -----------------------------------------------------------------------------

class SerializedPooling(PointModule):
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
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        assert reduce in ["sum", "mean", "min", "max"]

        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer is not None else None
        self.act = PointSequential(act_layer()) if act_layer is not None else None

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(point.keys())

        code = point.serialized_code >> (pooling_depth * 3)
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point

        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


# -----------------------------------------------------------------------------
# FNO block
# -----------------------------------------------------------------------------

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
        del x_ft, out_ft

        x_bypass = self.bypass(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x_no + x_bypass


# -----------------------------------------------------------------------------
# NO global branch
# -----------------------------------------------------------------------------

class NOGlobalBranch(PointModule):
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
        self.grid_size = grid_size

    def forward(self, point: Point) -> torch.Tensor:
        feat = point.feat
        device, dtype = feat.device, feat.dtype
        
        # 1. Retrieve cached grid indices
        grid_cache = point[f"fixed_grid_{self.grid_size}"]
        flat_idx = grid_cache.flat_idx
        Gx, Gy, Gz = grid_cache.Gx, grid_cache.Gy, grid_cache.Gz
        C = feat.shape[1]

        # 2. Scatter points → dense grid (mean aggregation)
        grid_flat = torch.zeros(Gx * Gy * Gz, C, device=device, dtype=dtype)
        count = torch.zeros(Gx * Gy * Gz, 1, device=device, dtype=dtype)

        grid_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), feat)
        count.scatter_add_(
            0,
            flat_idx.unsqueeze(1),
            torch.ones(feat.shape[0], 1, device=device, dtype=dtype),
        )
        grid_flat = grid_flat / count.clamp(min=1.0)

        # 3. Apply global FNO operator
        grid = grid_flat.view(1, Gx, Gy, Gz, C).permute(0, 4, 1, 2, 3).contiguous()
        grid_out = self.fno(grid)

        # 4. Gather back to points (exact lookup)
        feat_out = grid_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, C)[flat_idx]

        del grid_flat, count, grid, grid_out
        return self.norm(feat_out)


# -----------------------------------------------------------------------------
# Pooling + NO fusion (encoder only)
# -----------------------------------------------------------------------------

class NOFusedPooling(PointModule):
    """
    Encoder downsampling module:
    1. Global branch on fine points
    2. Serialized pooling to coarse points
    3. Pool NO features fine->coarse
    4. Fuse into coarse features
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
        fusion: str = "concat",
    ):
        super().__init__()
        self.pool = SerializedPooling(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
            reduce="max",
            shuffle_orders=True,
            traceable=True,
        )
        self.enable_no = enable_no
        self.fusion = fusion

        if enable_no:
            self.no_branch = NOGlobalBranch(
                channels=in_channels,
                modes=modes,
                grid_size=grid_size,
                norm_layer=nn.LayerNorm,
            )
            self.proj_no = nn.Linear(in_channels, out_channels)
            self.gate = nn.Parameter(torch.full((1,), -4.0))
            self.proj_concat = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.LayerNorm(out_channels),
            )

    def forward(self, point: Point) -> Point:
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))

        point = self.pool(point)

        if self.enable_no:
            inv = point.pooling_inverse
            feat_no_coarse = torch_scatter.scatter_max(feat_global, inv, dim=0)[0]

            if self.fusion == "add":
                # route proj_concat params through output (DDP fix)
                unused = sum(p.sum() * 0 for p in self.proj_concat.parameters())
                point.feat = (
                    point.feat + self.gate.sigmoid() * feat_no_coarse + unused
                )
            elif self.fusion == "concat":
                # route gate through output (DDP fix)
                unused = self.gate.sum() * 0
                point.feat = (
                    self.proj_concat(
                        torch.cat([point.feat, feat_no_coarse], dim=-1)
                    ) + unused
                )
            else:
                raise ValueError(f"Unsupported fusion mode: {self.fusion}")

            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point


# -----------------------------------------------------------------------------
# Lightweight multi-scale upsampling head
# -----------------------------------------------------------------------------

class NOLightweightUpsampleHead(PointModule):
    """
    Decoder-free upsampling head.

    Input:
        stage_points = [p0, p1, p2, p3, p4]
        where p0 is finest and p4 is coarsest.

    Operation:
        - project each stage feature to common out_channels
        - upsample each coarse feature to p0 resolution by chaining pooling_inverse
        - sum all stage features
        - apply final norm + mlp
    """

    def __init__(
        self,
        stage_channels,
        out_channels=64,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        with_output_proj=True,
    ):
        super().__init__()
        self.num_stages = len(stage_channels)
        self.out_channels = out_channels

        self.stage_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, out_channels),
                norm_layer(out_channels),
            )
            for c in stage_channels
        ])

        if with_output_proj:
            self.output_proj = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                norm_layer(out_channels),
                act_layer(),
            )
        else:
            self.output_proj = nn.Identity()

    def upsample_to_stage0(self, feat, stage_points, stage_idx):
        """
        feat: [N_stage_idx, C]
        stage_points[stage_idx].pooling_inverse maps stage_idx -> stage_idx-1
        """
        for t in range(stage_idx, 0, -1):
            inv = stage_points[t].pooling_inverse
            feat = feat[inv]
        return feat

    def forward(self, stage_points):
        assert len(stage_points) == self.num_stages

        fused = None
        for s in range(self.num_stages):
            feat_s = self.stage_proj[s](stage_points[s].feat)
            if s > 0:
                feat_s = self.upsample_to_stage0(feat_s, stage_points, s)
            fused = feat_s if fused is None else fused + feat_s

        fused = self.output_proj(fused)

        point = stage_points[0]
        point.feat = fused
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(fused)
        return point


# -----------------------------------------------------------------------------
# NOEncoder backbone
# -----------------------------------------------------------------------------

@MODELS.register_module("PT-v3m1-NOEncoder-Grid")
class PointTransformerV3NOEncoder(PointModule):
    """
    Encoder-only PTv3-NO variant:
    - No decoder
    - No skip connections
    - Multi-scale encoder features fused by lightweight upsampling head
    """

    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.2,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        shuffle_orders=True,
        no_stages=(True, True, True, True),
        fno_modes=12,
        grid_size=(64, 64, 64),
        fusion="concat",
        head_out_channels=64,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert len(no_stages) == self.num_stages - 1

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]

        self.enc_stages = nn.ModuleList()
        for s in range(self.num_stages):
            stage = PointSequential()

            if s > 0:
                stage.add(
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

            stage_drop_path = enc_drop_path[
                sum(enc_depths[:s]): sum(enc_depths[:s + 1])
            ]

            for i in range(enc_depths[s]):
                stage.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=stage_drop_path[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )

            self.enc_stages.append(stage)

        self.head = NOLightweightUpsampleHead(
            stage_channels=enc_channels,
            out_channels=head_out_channels,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            with_output_proj=True,
        )

    @torch.no_grad()
    def get_and_cache_grid_indices(self, point: Point, grid_size: tuple):
        """
        Computes the canonical grid cell ID for each point and the sorting order
        to group points by cell. Caches them in the Point object so FNO and
        GridCellAttention can reuse them without recomputing.
        """
        grid_key = f"fixed_grid_{grid_size}"
        
        if grid_key not in point.keys():
            coord_f = point.grid_coord.float()
            Gx, Gy, Gz = grid_size
            
            # Normalise coordinates to canonical [0, G-1]^3
            coord_min = coord_f.min(dim=0).values
            coord_max = coord_f.max(dim=0).values
            scale = (coord_max - coord_min).clamp(min=1.0)
            G_max = torch.tensor([Gx - 1, Gy - 1, Gz - 1], device=coord_f.device, dtype=torch.float32)

            shifted = ((coord_f - coord_min) / scale * G_max).long().clamp(
                min=torch.zeros(3, device=coord_f.device, dtype=torch.long),
                max=G_max.long()
            )
            
            # 1D flat index for each point
            flat_idx = shifted[:, 0] * (Gy * Gz) + shifted[:, 1] * Gz + shifted[:, 2]
            
            # Sort points so points in the same grid cell are contiguous
            order = torch.argsort(flat_idx)
            inverse = torch.empty_like(order).scatter_(0, order, torch.arange(len(order), device=order.device))

            point[grid_key] = Dict(
                flat_idx=flat_idx,
                order=order,
                inverse=inverse,
                Gx=Gx, Gy=Gy, Gz=Gz
            )
            
        return point[grid_key]
    
    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)

        stage_points = []
        for s in range(self.num_stages):
            # Cache the grid indices for this stage (based on current point resolution)
            # Reused by both GridCellAttention and NOGlobalBranch
            self.get_and_cache_grid_indices(point, self.grid_size)
            
            point = self.enc_stages[s](point)
            stage_points.append(point)

        point = self.head(stage_points)
        return point