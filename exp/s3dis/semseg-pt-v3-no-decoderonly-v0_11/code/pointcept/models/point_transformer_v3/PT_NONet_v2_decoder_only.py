"""
PTv3-NO Decoder-Only — Universal Shared NOGlobalBranch
  + GridPooling scaffold (traceable, builds pooling_inverse chain)
  + Coarse-to-fine: global WNO understanding at coarsest scale, then decode
  + QuatRPE at every transformer block
  + NOFusedGridUnpooling at each decoder stage
  
  Embedding (6ch XYZ+RGB)
    │
    ▼
PointNetPPEncoderStage x4   ← NEW: GridPool + local MLP (delta_xyz, delta_rgb, feat)
    │  stores scaffold[0..4]
    ▼
coarse_blocks (PTv3 + QuatRPE)
    │
    ▼  prev_mask_ids=None
MaskedNOFusedGridUnpooling   ← NEW: upsample → PointNetPP local agg → masked WNO per region
    │  returns (point, new_mask_ids)
    ▼  prev_mask_ids=new_mask_ids (passed to next stage)
MaskedNOFusedGridUnpooling   ← mask gets more refined each layer (Mask2Former cascade)
    ...
    ▼
output_proj → seg_head
  
"""

import torch
import torch.nn as nn
import torch_scatter
from addict import Dict
from functools import partial

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Block, Embedding,
)

# ---------------------------------------------------------------------------
# SerializedLocalMLP
# PointNet++-style local aggregation using the existing serialized windows.
# No kNN, no cdist — reuses point.serialized_order windows directly.
# For each window of K consecutive points in the space-filling curve order:
#   input  = [delta_xyz(3) | delta_rgb(3) | feat(C)]  relative to window centroid
#   shared MLP -> max-pool over window -> residual add back to each point
# ---------------------------------------------------------------------------

class SerializedLocalMLP(PointModule):
    def __init__(
        self,
        channels: int,
        patch_size: int = 48,
        order_index: int = 0,
        use_rgb: bool = True,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.order_index = order_index
        self.use_rgb     = use_rgb
        self.channels    = channels

        # BUILD TWO mlps: one with rgb, one without
        # select in forward based on actual data availability
        in_dim_rgb  = 3 + 3 + channels   # delta_xyz + delta_rgb + feat
        in_dim_geo  = 3 + channels        # delta_xyz + feat only

        self.mlp_rgb = nn.Sequential(
            nn.Linear(in_dim_rgb, channels),
            nn.LayerNorm(channels), nn.GELU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels), nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.mlp_geo = nn.Sequential(
            nn.Linear(in_dim_geo, channels),
            nn.LayerNorm(channels), nn.GELU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels), nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, point: Point) -> Point:
        assert "serialized_order" in point.keys(), \
            "Run point.serialization() before SerializedLocalMLP"

        K    = self.patch_size
        feat = point.feat          # (N, C)
        N, C = feat.shape
        device = feat.device

        offset   = point.offset
        bincount = offset2bincount(offset)
        B        = len(bincount)
        _offset  = nn.functional.pad(offset, (1, 0))  # (B+1,)

        global_order   = point.serialized_order[self.order_index]    # (N,) maps order-pos -> orig-idx
        global_inverse = point.serialized_inverse[self.order_index]  # (N,) maps orig-idx -> order-pos

        pad_list  = []
        unpad_pos = torch.empty(N, dtype=torch.long, device=device)

        cursor = 0
        for i in range(B):
            src_s = int(_offset[i].item())
            src_e = int(_offset[i + 1].item())
            n_i   = src_e - src_s

            # order positions of this scene's points (in serialized order)
            order_positions = global_inverse[src_s:src_e]        # (n_i,) positions in order
            order_positions_sorted, _ = order_positions.sort()   # sort so windows are contiguous
            scene_order = global_order[order_positions_sorted]   # (n_i,) orig-indices in SFC order

            # pad to next multiple of K (min K)
            np_i  = max(((n_i + K - 1) // K) * K, K)
            n_pad = np_i - n_i
            if n_pad > 0:
                extra = scene_order[torch.arange(n_pad, device=device) % n_i]
                scene_order_padded = torch.cat([scene_order, extra])
            else:
                scene_order_padded = scene_order

            # map each orig-space point to its position in the padded buffer
            unpad_pos[scene_order] = torch.arange(n_i, device=device) + cursor

            pad_list.append(scene_order_padded)
            cursor += np_i

        pad_tensor = torch.cat(pad_list, dim=0)  # (N_pad,)

        feat_ord  = feat[pad_tensor].reshape(-1, K, C)
        coord     = point.coord.float()
        coord_ord = coord[pad_tensor].reshape(-1, K, 3)

        centroid_coord = coord_ord.mean(dim=1, keepdim=True)
        delta_xyz      = coord_ord - centroid_coord

        has_color = self.use_rgb and "color" in point.keys()
        if has_color:
            color_ord      = point.color.float()[pad_tensor].reshape(-1, K, 3)
            centroid_color = color_ord.mean(dim=1, keepdim=True)
            delta_rgb      = color_ord - centroid_color
            local_input    = torch.cat([delta_xyz, delta_rgb, feat_ord], dim=-1)
            mlp_out        = self.mlp_rgb(local_input)
        else:
            local_input    = torch.cat([delta_xyz, feat_ord], dim=-1)
            mlp_out        = self.mlp_geo(local_input)

        agg_win    = mlp_out.max(dim=1).values                               # (W, C)
        agg_padded = agg_win.unsqueeze(1).expand(-1, K, -1).reshape(-1, C)  # (N_pad, C)
        agg        = agg_padded[unpad_pos]                                   # (N, C)

        point.feat = self.out_norm(feat + agg)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class PointNetPPEncoderStage(PointModule):
    """One encoder stage: GridPooling (traceable) + SerializedLocalMLP."""
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        patch_size=48,
        order_index=0,
        use_rgb=True,
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
        self.local_mlp = SerializedLocalMLP(
            channels=out_channels,
            patch_size=patch_size,
            order_index=order_index,
            use_rgb=use_rgb,
        )

    def forward(self, point: Point) -> Point:
        point = self.pool(point)
        point = self.local_mlp(point)
        return point

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
        grid_coord = grid_coord.long() | point.batch.view(-1, 1).long() << 48

        grid_coord, cluster, counts = torch.unique(
            grid_coord, sorted=True,
            return_inverse=True, return_counts=True, dim=0,
        )
        grid_coord = (grid_coord & ((1 << 48) - 1)).long()
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
# QuatRPE — Quaternion Rotational Position Embedding
# ---------------------------------------------------------------------------

class QuatRPE(nn.Module):
    def __init__(self, out_channels: int, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        in_dim = 4 + num_freqs
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)

    @staticmethod
    def _build_quaternion(delta: torch.Tensor) -> torch.Tensor:
        norm  = delta.norm(dim=1, keepdim=True).clamp(min=1e-6)
        axis  = delta / norm
        theta = torch.sigmoid(norm) * torch.pi
        half  = theta * 0.5
        w     = torch.cos(half)
        xyz   = torch.sin(half) * axis
        return torch.cat([w, xyz], dim=1)

    def forward(self, point: Point) -> torch.Tensor:
        # coord_f  = point.coord.float()
        coord_f  = point.coord.detach().float()
        centroid = coord_f.mean(dim=0, keepdim=True)
        delta    = coord_f - centroid
        radius   = delta.norm(dim=1, keepdim=True)
        r_norm   = radius / (radius.max() + 1e-6)
        quat     = self._build_quaternion(delta)
        r_freq   = torch.sin(r_norm * self.freqs.unsqueeze(0))
        feat     = torch.cat([quat, r_freq], dim=1)
        out      = self.proj(feat)
        if out.shape[1] % 4 == 0:
            out = apply_quat_rotation_to_features(out, quat)
        return out


# ---------------------------------------------------------------------------
# QuatRPEBlockWrapper
# ---------------------------------------------------------------------------

class QuatRPEBlockWrapper(PointModule):
    def __init__(self, block: nn.Module, channels: int, coord_drop: float = 0.0):
        super().__init__()
        self.block      = block
        self.rpe        = QuatRPE(channels, num_freqs=8)
        self.coord_drop = coord_drop

    def forward(self, point: Point) -> Point:
        if self.training and self.coord_drop > 0.0:
            mask       = (torch.rand(point.coord.shape[0], 1, device=point.coord.device)
                          > self.coord_drop)
            orig_coord  = point.coord
            point.coord = point.coord * mask.float()
            pos_enc     = self.rpe(point).to(point.feat.dtype)
            point.coord = orig_coord          # restore immediately
        else:
            pos_enc = self.rpe(point).to(point.feat.dtype)

        point.feat = point.feat + pos_enc
        if hasattr(point, "sparse_conv_feat") and point.sparse_conv_feat is not None:
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return self.block(point)


def apply_quat_rotation_to_features(feat: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    N, C = feat.shape
    assert C % 4 == 0
    f  = feat.view(N, C // 4, 4)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    fw, fx, fy, fz = f[..., 0], f[..., 1], f[..., 2], f[..., 3]
    rw = w.unsqueeze(1)*fw - x.unsqueeze(1)*fx - y.unsqueeze(1)*fy - z.unsqueeze(1)*fz
    rx = w.unsqueeze(1)*fx + x.unsqueeze(1)*fw + y.unsqueeze(1)*fz - z.unsqueeze(1)*fy
    ry = w.unsqueeze(1)*fy - x.unsqueeze(1)*fz + y.unsqueeze(1)*fw + z.unsqueeze(1)*fx
    rz = w.unsqueeze(1)*fz + x.unsqueeze(1)*fy - y.unsqueeze(1)*fx + z.unsqueeze(1)*fw
    return torch.stack([rw, rx, ry, rz], dim=-1).view(N, C)


# ---------------------------------------------------------------------------
# MLP3dBlock
# ---------------------------------------------------------------------------

class MLP3dBlock(nn.Module):
    def __init__(self, channels: int, expansion_ratio: float = 2.0):
        super().__init__()
        hidden_dim = int(channels * expansion_ratio)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_mlp  = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, channels, kernel_size=1),
        )
        self.local_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, channels, kernel_size=1),
        )
        self.bypass = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bypass    = self.bypass(x)
        global_feat = self.global_mlp(self.global_pool(x))
        local_feat  = self.local_mlp(x)
        return local_feat + global_feat + x_bypass


# ---------------------------------------------------------------------------
# WNO3dBlock
# ---------------------------------------------------------------------------

class WNO3dBlock(nn.Module):
    def __init__(self, channels: int, levels: int = 3):
        super().__init__()
        self.levels  = levels
        self.decomps = nn.ModuleList()
        self.recons  = nn.ModuleList()
        self.mixers  = nn.ModuleList()
        for _ in range(levels):
            self.decomps.append(nn.Conv3d(channels, channels, kernel_size=2, stride=2, groups=channels))
            self.recons.append(nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2, groups=channels))
            self.mixers.append(nn.Conv3d(channels, channels, kernel_size=1))
        self.global_mix = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bypass     = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bypass = self.bypass(x)
        coeffs   = []
        curr_x   = x
        for i in range(self.levels):
            curr_x = self.decomps[i](curr_x)
            coeffs.append(curr_x)
        curr_x = self.global_mix(curr_x)
        for i in reversed(range(self.levels)):
            curr_x = curr_x + self.mixers[i](coeffs[i])
            curr_x = self.recons[i](curr_x)
        return curr_x + x_bypass


# ---------------------------------------------------------------------------
# FNO3dBlock
# ---------------------------------------------------------------------------

class FNO3dBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 8):
        super().__init__()
        self.modes = modes
        scale      = 1.0 / (channels * channels)
        self.weight_real = nn.Parameter(scale * torch.rand(channels, channels, modes, modes, modes))
        self.weight_imag = nn.Parameter(scale * torch.rand(channels, channels, modes, modes, modes))
        self.bypass      = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z = x.shape
        orig_dtype     = x.dtype
        x_fp32         = x.to(torch.float32)
        x_ft           = torch.fft.rfftn(x_fp32, dim=[-3, -2, -1])
        out_ft         = torch.zeros_like(x_ft)
        mx = min(self.modes, x_ft.shape[2])
        my = min(self.modes, x_ft.shape[3])
        mz = min(self.modes, x_ft.shape[4])
        weight = torch.complex(self.weight_real, self.weight_imag)
        out_ft[:, :, :mx, :my, :mz] = torch.einsum(
            "bcxyz,cdxyz->bdxyz",
            x_ft[:, :, :mx, :my, :mz],
            weight[:, :, :mx, :my, :mz],
        )
        x_no = torch.fft.irfftn(out_ft, s=(X, Y, Z), dim=[-3, -2, -1]).to(orig_dtype)
        del x_ft, out_ft
        return x_no + self.bypass(x)


# ---------------------------------------------------------------------------
# NOGlobalBranch
# ---------------------------------------------------------------------------

class NOGlobalBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        modes: int = 8,
        grid_size: tuple = (64, 64, 64),
        norm_layer=nn.LayerNorm,
        NO_type: str = "WNO",
    ):
        super().__init__()
        if NO_type == "FNO":
            self.no = FNO3dBlock(channels, modes)
        elif NO_type == "WNO":
            self.no = WNO3dBlock(channels, levels=3)
        elif NO_type == "MLP":
            self.no = MLP3dBlock(channels)
        self.norm        = norm_layer(channels)
        self.grid_size   = grid_size
        self.rel_pos_enc = QuatRPE(channels, num_freqs=8)

    def forward(self, feat: torch.Tensor, point: Point) -> torch.Tensor:
        Gx, Gy, Gz    = self.grid_size
        C              = feat.shape[1]
        device, dtype  = feat.device, feat.dtype

        pos_enc = self.rel_pos_enc(point).to(dtype)
        feat    = feat + pos_enc

        coord     = point.grid_coord
        coord_f   = coord.float()
        coord_min = coord_f.min(dim=0).values
        coord_max = coord_f.max(dim=0).values
        scale     = (coord_max - coord_min).clamp(min=1.0)
        G_max     = torch.tensor([Gx - 1, Gy - 1, Gz - 1], device=device, dtype=torch.float32)
        shifted   = ((coord_f - coord_min) / scale * G_max).long()
        shifted   = shifted.clamp(
            min=torch.zeros(3, device=device, dtype=torch.long),
            max=G_max.long(),
        )
        flat_idx  = shifted[:, 0] * (Gy * Gz) + shifted[:, 1] * Gz + shifted[:, 2]

        grid_flat = torch.zeros(Gx * Gy * Gz, C, device=device, dtype=dtype)
        count     = torch.zeros(Gx * Gy * Gz, 1, device=device, dtype=dtype)
        grid_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), feat)
        count.scatter_add_(0, flat_idx.unsqueeze(1), torch.ones(feat.shape[0], 1, device=device, dtype=dtype))
        grid_flat = grid_flat / count.clamp(min=1.0)

        grid     = grid_flat.view(1, Gx, Gy, Gz, C).permute(0, 4, 1, 2, 3).contiguous()
        grid_out = self.no(grid)
        feat_out = grid_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, C)[flat_idx]
        del grid_flat, count, grid, grid_out

        return self.norm(feat_out)


# ---------------------------------------------------------------------------
# NOFusedGridUnpooling
# ---------------------------------------------------------------------------

class NOFusedGridUnpooling(PointModule):
    def __init__(self, in_channels, out_channels, enable_no=True,
                 fusion="add", universal_no_branch=None, universal_dim=64):
        super().__init__()
        self.enable_no = enable_no
        self.fusion    = fusion
        self.up_proj   = nn.Linear(in_channels, out_channels)

        if enable_no:
            assert universal_no_branch is not None
            self.no_branch = universal_no_branch
            self.down_proj = nn.Linear(out_channels, universal_dim)
            self.feat_proj = nn.Linear(universal_dim, out_channels)
            if fusion == "add":
                self.gate = nn.Parameter(torch.full((1,), -4.0))
            elif fusion == "concat":
                self.proj_concat = nn.Sequential(
                    nn.Linear(out_channels * 2, out_channels),
                    nn.LayerNorm(out_channels),
                )

    def forward(self, point_coarse: Point, point_fine: Point) -> Point:
        inv     = point_coarse.pooling_inverse
        feat_up = self.up_proj(point_coarse.feat)[inv]
        point_fine.feat = feat_up
        point_fine.sparse_conv_feat = point_fine.sparse_conv_feat.replace_feature(feat_up)

        if self.enable_no:
            feat_down   = self.down_proj(point_fine.feat)
            global_feat = self.no_branch(feat_down, point_fine)
            feat_no     = self.feat_proj(global_feat)

            if self.fusion == "add":
                point_fine.feat = point_fine.feat + self.gate.sigmoid() * feat_no
            elif self.fusion == "concat":
                point_fine.feat = self.proj_concat(
                    torch.cat([point_fine.feat, feat_no], dim=-1)
                )
            point_fine.sparse_conv_feat = point_fine.sparse_conv_feat.replace_feature(point_fine.feat)

        return point_fine


# ---------------------------------------------------------------------------
# Helper: assign each point to its closest query (Mask2Former-style mask)
# ---------------------------------------------------------------------------

def _build_mask_ids(feat: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """
    feat:    (N, D)
    queries: (Q, D)
    returns: (N,) LongTensor — index of closest query per point
    """
    sim = torch.einsum(
        "nd,qd->nq",
        nn.functional.normalize(feat.float(), dim=-1),
        nn.functional.normalize(queries.float(), dim=-1),
    )
    return sim.argmax(dim=-1)

class MaskedNOFusedGridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        enable_no=True,
        fusion="concat",
        universal_no_branch=None,
        universal_dim=64,
        num_queries=100,
        patch_size=48,       # CHANGED: was k_local=16
        order_index=0,       # CHANGED: was implicit
        use_rgb=True,
    ):
        super().__init__()
        self.enable_no   = enable_no
        self.fusion      = fusion
        self.up_proj     = nn.Linear(in_channels, out_channels)
        self.num_queries = num_queries

        # CHANGED: SerializedLocalMLP instead of PointNetPPLocalAgg (no kNN)
        self.local_mlp = SerializedLocalMLP(
            channels=out_channels,
            patch_size=patch_size,
            order_index=order_index,
            use_rgb=use_rgb,
        )

        if enable_no:
            assert universal_no_branch is not None
            self.no_branch  = universal_no_branch
            self.down_proj  = nn.Linear(out_channels, universal_dim)
            self.feat_proj  = nn.Linear(universal_dim, out_channels)
            self.queries    = nn.Embedding(num_queries, universal_dim)
            if fusion == "add":
                self.gate = nn.Parameter(torch.full((1,), -4.0))
            elif fusion == "concat":
                self.proj_concat = nn.Sequential(
                    nn.Linear(out_channels * 2, out_channels),
                    nn.LayerNorm(out_channels),
                )

    def forward(
        self,
        point_coarse: Point,
        point_fine: Point,
        prev_mask_ids=None,
    ):
        inv     = point_coarse.pooling_inverse
        feat_up = self.up_proj(point_coarse.feat)[inv]
        point_fine.feat = feat_up
        point_fine.sparse_conv_feat = point_fine.sparse_conv_feat.replace_feature(feat_up)

        # CHANGED: serialization-window local MLP (no kNN)
        point_fine = self.local_mlp(point_fine)

        if not self.enable_no:
            return point_fine, None

        feat_down = self.down_proj(point_fine.feat)
        Q         = self.queries.weight

        if prev_mask_ids is not None:
            mask_ids = prev_mask_ids
        else:
            mask_ids = _build_mask_ids(feat_down, Q)

        feat_no = torch.zeros_like(feat_down)
        for q_idx in range(self.num_queries):
            pts_mask = (mask_ids == q_idx).nonzero(as_tuple=True)[0]
            if pts_mask.numel() < 4:
                continue
            mini_feat  = feat_down[pts_mask]
            mini_coord = point_fine.grid_coord[pts_mask]
            mini_batch = point_fine.batch[pts_mask]
            mini_offset = torch.zeros(
                mini_batch.max().item() + 1,
                dtype=torch.long, device=mini_feat.device
            )
            mini_offset.scatter_add_(
                0, mini_batch,
                torch.ones(mini_batch.shape[0], dtype=torch.long, device=mini_batch.device)
            )
            mini_offset = mini_offset.cumsum(0)
            mini_pt = Point({
                "feat":       mini_feat,
                "coord":      point_fine.coord[pts_mask],
                "grid_coord": mini_coord,
                "batch":      mini_batch,
                "offset":     mini_offset,
            })
            feat_no[pts_mask] = self.no_branch(mini_feat, mini_pt).to(feat_no.dtype)

        feat_no      = self.feat_proj(feat_no)
        new_mask_ids = _build_mask_ids(feat_down, Q)

        if self.fusion == "add":
            point_fine.feat = point_fine.feat + self.gate.sigmoid() * feat_no
        elif self.fusion == "concat":
            point_fine.feat = self.proj_concat(
                torch.cat([point_fine.feat, feat_no], dim=-1)
            )
        point_fine.sparse_conv_feat = point_fine.sparse_conv_feat.replace_feature(point_fine.feat)
        return point_fine, new_mask_ids

# ---------------------------------------------------------------------------
# PointTransformerV3_NO_DecoderOnly
# ---------------------------------------------------------------------------

@MODELS.register_module("PT-v3m1-NO-DecoderOnly")
class PointTransformerV3_NO_DecoderOnly(PointModule):
    def __init__(
        self,
        in_channels: int = 6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 2, 2),
        dec_depths=(2, 3, 4, 4),        # independent, length = num_stages - 1, coarse->fine
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio: int = 4,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        no_stages=(True, True, True, True),
        fno_modes: int = 8,
        base_grid_size: tuple = (64, 64, 64),
        fusion: str = "concat",
        share_no_branch: bool = True,
        universal_dim: int = 64,
        NO_type: str = "WNO",
        pool_reduce: str = "max",
        head_out_channels: int = 64,
        head_fusion: str = "sum",
    ):
        super().__init__()
        self.num_stages     = len(enc_depths)
        self.order          = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders

        # Validate dec_depths
        self.dec_depths = list(dec_depths)
        assert len(self.dec_depths) == self.num_stages - 1, (
            f"dec_depths must have {self.num_stages - 1} entries "
            f"(one per decoder stage), got {len(self.dec_depths)}"
        )

        bn_layer  = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer  = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, enc_channels[0], bn_layer, act_layer)

        # -----------------------------------------------------------------
        # Encoder: enc_depths[s] transformer blocks THEN GridPooling per stage
        # (all stages except coarsest; coarsest handled by coarse_blocks)
        # -----------------------------------------------------------------
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]

        self.enc_blocks = nn.ModuleList()
        for s in range(self.num_stages - 1):   # stages 0 .. num_stages-2
            drop_paths_s = enc_drop_path[sum(enc_depths[:s]): sum(enc_depths[:s + 1])]
            stage_seq = PointSequential()
            for i in range(enc_depths[s]):
                base_block = Block(
                    channels=enc_channels[s],
                    num_heads=enc_num_head[s],
                    patch_size=enc_patch_size[s],
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_paths_s[i],
                    norm_layer=ln_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    order_index=i % len(self.order),
                    cpe_indice_key=f"enc_stage{s}",
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
                stage_seq.add(
                    QuatRPEBlockWrapper(base_block, enc_channels[s]),
                    name=f"block{i}",
                )
            self.enc_blocks.append(stage_seq)

        # -----------------------------------------------------------------
        # GridPooling scaffold (traceable, stores pooling_inverse chain)
        # -----------------------------------------------------------------
        # --- Scaffold: PointNetPP encoder stages ---
        # patch_size matches enc_patch_size[s], order_index cycles
        self.pre_pool_stages = nn.ModuleList()
        for s in range(1, self.num_stages):
            self.pre_pool_stages.append(
                PointNetPPEncoderStage(
                    in_channels=enc_channels[s - 1],
                    out_channels=enc_channels[s],
                    stride=stride[s - 1],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                    reduce=pool_reduce,
                    shuffle_orders=shuffle_orders,
                    patch_size=enc_patch_size[s],   # from config, e.g. 1024
                    order_index=s % len(self.order),
                    use_rgb=True,
                )
            )

        # -----------------------------------------------------------------
        # Shared WNO / NO branch
        # -----------------------------------------------------------------
        if share_no_branch:
            self.universal_no_branch = NOGlobalBranch(
                channels=universal_dim,
                modes=fno_modes,
                grid_size=base_grid_size,
                norm_layer=ln_layer,
                NO_type=NO_type,
            )
        else:
            self.universal_no_branch = None

        # -----------------------------------------------------------------
        # Coarsest-scale blocks (global scene understanding)
        # -----------------------------------------------------------------
        coarse_drop_paths = enc_drop_path[sum(enc_depths[:self.num_stages - 1]):]
        self.coarse_blocks = PointSequential()
        s = self.num_stages - 1
        for i in range(enc_depths[s]):
            base_block = Block(
                channels=enc_channels[s],
                num_heads=enc_num_head[s],
                patch_size=enc_patch_size[s],
                mlp_ratio=mlp_ratio,
                drop_path=coarse_drop_paths[i],
                norm_layer=ln_layer,
                act_layer=act_layer,
                pre_norm=pre_norm,
                order_index=i % len(self.order),
                cpe_indice_key=f"stage{s}",
                enable_flash=enable_flash,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
            )
            self.coarse_blocks.add(
                QuatRPEBlockWrapper(base_block, enc_channels[s]),
                name=f"block{i}",
            )

        # -----------------------------------------------------------------
        # Decoder stages (coarse -> fine)
        # dec_depths[idx] blocks per stage, independent drop-path schedule
        # -----------------------------------------------------------------
        dec_drop_path = [
            x.item()
            for x in torch.linspace(0, drop_path * 0.5, sum(self.dec_depths))
        ]

        self.dec_unpool = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for idx, s in enumerate(range(self.num_stages - 2, -1, -1)):
            self.dec_unpool.append(
                MaskedNOFusedGridUnpooling(
                    in_channels=enc_channels[s + 1],
                    out_channels=enc_channels[s],
                    enable_no=no_stages[s],
                    fusion=fusion,
                    universal_no_branch=self.universal_no_branch,
                    universal_dim=universal_dim,
                    num_queries=100,
                    patch_size=enc_patch_size[s],   # reuse enc patch size per stage
                    order_index=s % len(self.order),
                    use_rgb=True,
                )
            )
            
            drop_paths_s = dec_drop_path[
                sum(self.dec_depths[:idx]): sum(self.dec_depths[:idx + 1])
            ]
            stage_seq = PointSequential()
            for i in range(self.dec_depths[idx]):
                base_block = Block(
                    channels=enc_channels[s],
                    num_heads=enc_num_head[s],
                    patch_size=enc_patch_size[s],
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_paths_s[i],
                    norm_layer=ln_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    order_index=i % len(self.order),
                    cpe_indice_key=f"dec_stage{s}",
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
                stage_seq.add(
                    QuatRPEBlockWrapper(base_block, enc_channels[s]),
                    name=f"block{i}",
                )
                
            self.dec_blocks.append(stage_seq)

        # -----------------------------------------------------------------
        # Output projection enc_channels[0] -> head_out_channels
        # -----------------------------------------------------------------
        self.output_proj = nn.Sequential(
            nn.Linear(enc_channels[0], head_out_channels),
            nn.LayerNorm(head_out_channels),
            nn.GELU(),
        )

    def forward(self, data_dict: dict) -> Point:
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # 1. Embed at finest resolution
        point = self.embedding(point)

        # 2. Encoder: blocks -> pool -> blocks -> pool -> ...
        #    enc_blocks[s] runs BEFORE pool into stage s+1
        scaffold = []
        for s, (enc_blk, pool) in enumerate(
            zip(self.enc_blocks, self.pre_pool_stages)
        ):
            point = enc_blk(point)       # enc_depths[s] transformer blocks
            scaffold.append(point)       # save fine point AFTER blocks, BEFORE pool
            point = pool(point)          # pool into next scale

        # scaffold[0]=finest-after-enc ... scaffold[num_stages-2]=second-coarsest-after-enc
        # point is now at coarsest scale (after last pool)

        # 3. Coarsest-scale global understanding
        point = self.coarse_blocks(point)

        # 4. Decode: coarse -> fine
        prev_mask_ids = None
        for i, (unpool, blocks) in enumerate(zip(self.dec_unpool, self.dec_blocks)):
            fine_point = scaffold[-(i + 2)]
            point, prev_mask_ids = unpool(point, fine_point, prev_mask_ids)
            point = blocks(point)

        # 5. Output projection
        point.feat = self.output_proj(point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        # 6. Clear pooling chain
        for key in ("pooling_parent", "pooling_inverse"):
            if key in point.keys():
                point.pop(key)

        return point

# ---------------------------------------------------------------------------
# DefaultSegmentorV3
# ---------------------------------------------------------------------------

@MODELS.register_module()
class DefaultSegmentorV3(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone        = build_model(backbone)
        self.criteria        = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(input_dict)
        else:
            point = self.backbone(input_dict)

        aux_rpe_loss = point["aux_rpe_loss"] if "aux_rpe_loss" in point.keys() else None

        feat       = point.feat
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            if aux_rpe_loss is not None:
                loss = loss + 0.5 * aux_rpe_loss
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"]       = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict