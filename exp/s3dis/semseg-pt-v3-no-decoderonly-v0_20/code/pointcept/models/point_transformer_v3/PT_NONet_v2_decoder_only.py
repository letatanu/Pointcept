"""
PTv3-NO Decoder-Only — OPTNet Enhanced
  + PointSorter (Learned Serialization) with Self-Supervised Ordering Loss
  + GridPooling scaffold (traceable, builds pooling_inverse chain)
  + Coarse-to-fine: global WNO understanding at coarsest scale, then decode
  + OrderAwareFusedGridUnpooling at each decoder stage
"""

import torch
import torch.nn as nn
import torch_scatter
import pointops
from addict import Dict
from functools import partial

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
from pointcept.models.modules import PointModule, PointSequential

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Embedding,
)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob

# ---------------------------------------------------------------------------
# PointSorter & LearnedSerialization (from OPTNet)
# ---------------------------------------------------------------------------

class PointSorter(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64, num_orders: int = 4):
        super().__init__()
        self.num_orders = num_orders
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_orders),
        )

    def forward(self, point: Point):
        inp = torch.cat([point.coord.float(), point.feat], dim=-1)
        scores = self.mlp(inp)

        orders, inverses = [], []
        num_batches = int(point.batch.max().item()) + 1

        for order_idx in range(self.num_orders):
            score = scores[:, order_idx]
            batch_orders = []

            for batch_idx in range(num_batches):
                indices = torch.where(point.batch == batch_idx)[0]
                local_order = torch.argsort(score[indices], stable=True)
                batch_orders.append(indices[local_order])

            order = torch.cat(batch_orders, dim=0)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(
                order.numel(),
                device=order.device,
                dtype=order.dtype,
            )

            orders.append(order)
            inverses.append(inverse)

        return scores, torch.stack(orders, dim=0), torch.stack(inverses, dim=0)

class LearnedSerialization(PointModule):
    def __init__(self, in_channels: int, num_orders: int = 1):
        super().__init__()
        self.sorter = PointSorter(in_channels, num_orders=num_orders)

    def forward(self, point: Point) -> Point:
        scores, learned_order, learned_inverse = self.sorter(point)
        point.sort_scores = scores 
        point.learned_order = learned_order
        point.learned_inverse = learned_inverse
        return point


class PointSorterBlock(PointModule):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        order_index: int = 0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.order_index = order_index
        self.patch_size = patch_size
        self.pre_norm = pre_norm

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(channels * mlp_ratio), channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, point: Point) -> Point:
        order_id = self.order_index % point.learned_order.shape[0]
        order = point.learned_order[order_id]

        ordered_feat = point.feat[order]
        ordered_batch = point.batch[order]
        output = torch.empty_like(ordered_feat)

        for batch_idx in range(int(point.batch.max().item()) + 1):
            mask = ordered_batch == batch_idx
            feat = ordered_feat[mask]

            if feat.shape[0] == 0:
                continue

            num_points, channels = feat.shape
            patch_size = min(self.patch_size, num_points)
            num_patches = (num_points + patch_size - 1) // patch_size
            padded_size = num_patches * patch_size
            padding = padded_size - num_points

            if padding > 0:
                feat = torch.cat(
                    [feat, feat[-1:].expand(padding, -1)],
                    dim=0,
                )

            feat = feat.view(num_patches, patch_size, channels)

            if self.pre_norm:
                attn_input = self.norm1(feat)
                attn_out, _ = self.attn(
                    attn_input,
                    attn_input,
                    attn_input,
                    need_weights=False,
                )
                feat = feat + self.drop_path(attn_out)
                feat = feat + self.drop_path(self.mlp(self.norm2(feat)))
            else:
                attn_out, _ = self.attn(feat, feat, feat, need_weights=False)
                feat = self.norm1(feat + self.drop_path(attn_out))
                feat = self.norm2(feat + self.drop_path(self.mlp(feat)))

            output[mask] = feat.reshape(padded_size, channels)[:num_points]

        point.feat = output[point.learned_inverse[order_id]]

        if (
            hasattr(point, "sparse_conv_feat")
            and point.sparse_conv_feat is not None
        ):
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(
                point.feat
            )

        return point

#---------------------------------------------------------------------------
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
        coord_f = point.coord.float()
        batch = point.batch
        delta = torch.empty_like(coord_f)
        r_norm = torch.empty(
            (coord_f.shape[0], 1),
            device=coord_f.device,
            dtype=coord_f.dtype,
        )

        for batch_idx in range(int(batch.max().item()) + 1):
            mask = batch == batch_idx
            coords = coord_f[mask]
            centered = coords - coords.mean(dim=0, keepdim=True)
            radius = centered.norm(dim=1, keepdim=True)
            delta[mask] = centered
            r_norm[mask] = radius / radius.max().clamp_min(1e-6)

        quat = self._build_quaternion(delta)
        r_freq = torch.sin(r_norm * self.freqs.unsqueeze(0))
        feat = torch.cat([quat, r_freq], dim=1)
        out = self.proj(feat)

        if out.shape[1] % 4 == 0:
            out = apply_quat_rotation_to_features(out, quat)

        return out

class QuatRPEBlockWrapper(PointModule):
    def __init__(self, block: nn.Module, channels: int, coord_drop: float = 0.1):
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
# GridPooling
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

        point = Point(point_dict)

        if self.norm is not None:
            point = self.norm(point)

        if self.act is not None:
            point = self.act(point)

        point.sparsify()
        return point

# ---------------------------------------------------------------------------
# Core Wavelet / FNO / MLP Blocks (Unchanged)
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


class WNO3dBlock(nn.Module):
    """
    Lightweight wavelet block with channel-wise, level-specific frequency gates.

    Gate initialization at zero means sigmoid(0)=0.5, allowing every wavelet
    scale to participate initially while training learns its importance.
    """

    def __init__(self, channels: int, levels: int = 3):
        super().__init__()
        self.levels = levels

        self.decomps = nn.ModuleList()
        self.recons = nn.ModuleList()
        self.mixers = nn.ModuleList()
        self.freq_gates = nn.ParameterList()

        for _ in range(levels):
            self.decomps.append(
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=2,
                    stride=2,
                    groups=channels,
                )
            )
            self.recons.append(
                nn.ConvTranspose3d(
                    channels,
                    channels,
                    kernel_size=2,
                    stride=2,
                    groups=channels,
                )
            )
            self.mixers.append(nn.Conv3d(channels, channels, kernel_size=1))

            # Per-channel gate for this wavelet level: negligible parameter cost.
            self.freq_gates.append(
                nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
            )

        self.global_mix = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1),
        )
        self.bypass = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.bypass(x)

        coeffs = []
        curr_x = x

        # Wavelet-like multiscale decomposition.
        for decomp in self.decomps:
            curr_x = decomp(curr_x)
            coeffs.append(curr_x)

        curr_x = self.global_mix(curr_x)

        # Coarse-to-fine reconstruction with learned scale selection.
        for i in reversed(range(self.levels)):
            gate = torch.sigmoid(self.freq_gates[i])
            detail = gate * self.mixers[i](coeffs[i])
            curr_x = self.recons[i](curr_x + detail)

        return curr_x + residual


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

    def forward(self, feat: torch.Tensor, point: Point) -> torch.Tensor:
        gx, gy, gz = self.grid_size
        channels = feat.shape[1]
        device, dtype = feat.device, feat.dtype
        output = torch.empty_like(feat)

        for batch_idx in range(int(point.batch.max().item()) + 1):
            mask = point.batch == batch_idx
            batch_feat = feat[mask]
            coord_f = point.grid_coord[mask].float()

            coord_min = coord_f.min(dim=0).values
            coord_max = coord_f.max(dim=0).values
            scale = (coord_max - coord_min).clamp_min(1.0)
            grid_max = torch.tensor(
                [gx - 1, gy - 1, gz - 1],
                device=device,
                dtype=coord_f.dtype,
            )

            shifted = ((coord_f - coord_min) / scale * grid_max).long()
            shifted = torch.minimum(
                torch.maximum(
                    shifted,
                    torch.zeros(3, device=device, dtype=torch.long),
                ),
                grid_max.long(),
            )

            flat_idx = (
                shifted[:, 0] * (gy * gz)
                + shifted[:, 1] * gz
                + shifted[:, 2]
            )

            grid_flat = torch.zeros(
                gx * gy * gz, channels, device=device, dtype=dtype
            )
            count = torch.zeros(
                gx * gy * gz, 1, device=device, dtype=dtype
            )

            grid_flat.scatter_add_(
                0,
                flat_idx.unsqueeze(1).expand(-1, channels),
                batch_feat,
            )
            count.scatter_add_(
                0,
                flat_idx.unsqueeze(1),
                torch.ones(
                    batch_feat.shape[0], 1, device=device, dtype=dtype
                ),
            )

            grid = (
                (grid_flat / count.clamp_min(1.0))
                .view(1, gx, gy, gz, channels)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )

            grid_out = self.no(grid)
            output[mask] = (
                grid_out.squeeze(0)
                .permute(1, 2, 3, 0)
                .reshape(-1, channels)[flat_idx]
            )

        return self.norm(output)

# ---------------------------------------------------------------------------
# OrderAwareFusedGridUnpooling
# ---------------------------------------------------------------------------

class OrderAwareFusedGridUnpooling(PointModule):
    def __init__(self, in_channels, out_channels, enable_no=True,
                 fusion="concat", universal_no_branch=None, universal_dim=64, num_orders=4):
        super().__init__()
        self.enable_no = enable_no
        self.fusion    = fusion
        self.up_proj   = nn.Linear(in_channels, out_channels)

        # Gate to weigh skip connection based on score differences
        self.score_gate = nn.Sequential(
            nn.Linear(2 * num_orders, 16),
            nn.GELU(),
            nn.Linear(16, out_channels),
            nn.Sigmoid(),
        )

        if enable_no:
            assert universal_no_branch is not None

            self.no_branch = universal_no_branch
            self.down_proj = nn.Linear(out_channels, universal_dim)
            self.feat_proj = nn.Linear(universal_dim, out_channels)

            if fusion == "add":
                gate_hidden = max(out_channels // 4, 16)
                self.gate_mlp = nn.Sequential(
                    nn.Linear(out_channels, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, out_channels),
                )
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.constant_(self.gate_mlp[-1].bias, -4.0)

            elif fusion == "concat":
                self.proj_concat = nn.Sequential(
                    nn.Linear(out_channels * 2, out_channels),
                    nn.LayerNorm(out_channels),
                )

            else:
                raise ValueError(f"Unsupported fusion: {fusion}")

    def forward(self, point_coarse: Point, point_fine: Point) -> Point:
        inv     = point_coarse.pooling_inverse
        feat_up = self.up_proj(point_coarse.feat)[inv]

        # --- Order-Aware Skip Gating ---
        if "sort_scores" in point_coarse.keys() and "sort_scores" in point_fine.keys():
            c_scores = point_coarse.sort_scores[inv]
            f_scores = point_fine.sort_scores

            if c_scores.dim() == 1:
                c_scores = c_scores.unsqueeze(-1)

            if f_scores.dim() == 1:
                f_scores = f_scores.unsqueeze(-1)

            score_feat = torch.cat(
                [
                    (c_scores - f_scores).abs(),
                    0.5 * (c_scores + f_scores),
                ],
                dim=-1,
            )
            
            feat_up = feat_up * self.score_gate(score_feat)

        point_fine.feat = feat_up
        if hasattr(point_fine, "sparse_conv_feat") and point_fine.sparse_conv_feat is not None:
            point_fine.sparse_conv_feat = point_fine.sparse_conv_feat.replace_feature(feat_up)

        # --- Standard NO Fusion ---
        if self.enable_no:
            feat_down   = self.down_proj(point_fine.feat)
            global_feat = self.no_branch(feat_down, point_fine)
            feat_no     = self.feat_proj(global_feat)

            if self.fusion == "add":
                # Point-wise and channel-wise gating:
                # [N, C] rather than one gate shared by the whole scene.
                gate = torch.sigmoid(self.gate_mlp(point_fine.feat))

                # Preserve an explicit residual path from the point-based decoder.
                point_fine.feat = point_fine.feat + gate * feat_no
                
            elif self.fusion == "concat":
                point_fine.feat = self.proj_concat(
                    torch.cat([point_fine.feat, feat_no], dim=-1)
                )
            if hasattr(point_fine, "sparse_conv_feat") and point_fine.sparse_conv_feat is not None:
                point_fine.sparse_conv_feat = point_fine.sparse_conv_feat.replace_feature(point_fine.feat)

        return point_fine

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
        dec_depths=(2, 3, 4, 4),        
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
        ordering_loss_weight: float = 0.02,
        ordering_k: int = 16,
        warmup_epoch: int = 0,
    ):
        super().__init__()
        self.num_stages     = len(enc_depths)
        self.order          = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders

        self.dec_depths = list(dec_depths)
        assert len(self.dec_depths) == self.num_stages - 1

        self.ordering_loss_weight = ordering_loss_weight
        self.ordering_k = ordering_k
        self.warmup_epoch = warmup_epoch

        bn_layer  = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer  = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, enc_channels[0], bn_layer, act_layer)

        # 1. OPTNet-style Learned Serialization
        self.learned_serializers = nn.ModuleList(
            [
                LearnedSerialization(
                    in_channels=channels,
                    num_orders=len(self.order),
                )
                for channels in enc_channels
            ]
        )

        # 2. Encoder: GridPooling scaffold
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.pre_pool_stages = nn.ModuleList(
            [
                GridPooling(
                    in_channels=enc_channels[s - 1],
                    out_channels=enc_channels[s],
                    stride=stride[s - 1],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                    reduce=pool_reduce,
                    shuffle_orders=shuffle_orders,
                    traceable=True,
                )
                for s in range(1, self.num_stages)
            ]
        )

        # 3. Shared WNO / NO branch
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

        # 4. Coarsest-scale blocks
        coarse_drop_paths = enc_drop_path[
            sum(enc_depths[:self.num_stages - 1]):
        ]
        self.coarse_blocks = PointSequential()
        s = self.num_stages - 1

        for i in range(enc_depths[s]):
            base_block = PointSorterBlock(
                channels=enc_channels[s],
                num_heads=enc_num_head[s],
                patch_size=enc_patch_size[s],
                mlp_ratio=mlp_ratio,
                drop_path=coarse_drop_paths[i],
                order_index=i % len(self.order),
                pre_norm=pre_norm,
            )
            self.coarse_blocks.add(
                QuatRPEBlockWrapper(base_block, enc_channels[s]),
                name=f"block{i}",
            )

        # 5. Decoder stages
        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path * 0.5, sum(self.dec_depths))
        ]
        self.dec_unpool = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        self.decoder_order_offset = 0

        for idx, s in enumerate(range(self.num_stages - 2, -1, -1)):
            self.dec_unpool.append(
                OrderAwareFusedGridUnpooling(
                    in_channels=enc_channels[s + 1],
                    out_channels=enc_channels[s],
                    enable_no=no_stages[s],
                    fusion=fusion,
                    universal_no_branch=self.universal_no_branch,
                    universal_dim=universal_dim,
                    num_orders=len(self.order)
                )
            )

            drop_paths_s = dec_drop_path[
                sum(self.dec_depths[:idx]): sum(self.dec_depths[:idx + 1])
            ]

            stage_seq = PointSequential()

            for i in range(self.dec_depths[idx]):
                global_decoder_block_idx = self.decoder_order_offset + i
                order_index = global_decoder_block_idx % len(self.order)

                base_block = PointSorterBlock(
                    channels=enc_channels[s],
                    num_heads=enc_num_head[s],
                    patch_size=enc_patch_size[s],
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_paths_s[i],
                    order_index=order_index,
                    pre_norm=pre_norm,
                )

                stage_seq.add(
                    QuatRPEBlockWrapper(base_block, enc_channels[s]),
                    name=f"block{i}",
                )

            self.decoder_order_offset += self.dec_depths[idx]
            self.dec_blocks.append(stage_seq)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(enc_channels[0], head_out_channels),
            nn.LayerNorm(head_out_channels),
            nn.GELU(),
        )

    def compute_ordering_loss(self, point: Point, scores: torch.Tensor):
        # scores: (N, K), one learned ranking score for each of K orders.
        if scores.dim() == 1:
            scores = scores.unsqueeze(-1)

        # KNN is restricted to each room because the same cumulative offsets
        # are passed for source and query points.
        idx, _ = pointops.knn_query(
            self.ordering_k,
            point.coord.float().contiguous(),
            point.offset.float().contiguous(),
        )

        neighbor_scores = scores[idx.long()]                  # (N, k, K)
        loss_locality = (
            scores.unsqueeze(1) - neighbor_scores
        ).square().mean()

        # Enforce a non-collapsed score range within every room and every order.
        distribution_losses = []
        start = 0
        for end in point.offset.tolist():
            room_scores = scores[start:end]                   # (Nr, K)
            sorted_scores = room_scores.sort(dim=0).values
            target = torch.linspace(
                0.0,
                1.0,
                steps=room_scores.shape[0],
                device=scores.device,
                dtype=scores.dtype,
            ).unsqueeze(-1)
            distribution_losses.append((sorted_scores - target).square().mean())
            start = end

        loss_distribution = torch.stack(distribution_losses).mean()

        # Decorrelation prevents K sorters from learning the same point order.
        if scores.shape[1] > 1:
            centered = scores - scores.mean(dim=0, keepdim=True)
            normalized = centered / centered.norm(dim=0, keepdim=True).clamp_min(1e-6)
            correlation = normalized.transpose(0, 1) @ normalized
            identity = torch.eye(
                scores.shape[1], device=scores.device, dtype=scores.dtype
            )
            loss_diversity = (correlation - identity).square().mean()
        else:
            loss_diversity = scores.new_zeros(())

        return loss_locality + loss_distribution + 0.05 * loss_diversity

    def forward(self, data_dict: dict) -> Point:
        point = Point(data_dict)
        current_epoch = data_dict.get("epoch", float('inf')) if self.training else float('inf')

        # 1. Sparse representation only; no PTv3 serialization.
        point.sparsify()

        # 2. Input embedding.
        point = self.embedding(point)

        # 3. Finest-scale PointSorter.
        ordering_losses = []

        point = self.learned_serializers[0](point)
        if self.training and self.ordering_loss_weight > 0:
            ordering_losses.append(
                self.compute_ordering_loss(point, point.sort_scores)
            )

        # 4. Grid hierarchy plus scale-specific PointSorter.
        scaffold = []

        for stage_idx, pool in enumerate(self.pre_pool_stages):
            scaffold.append(point)
            point = pool(point)
            point = self.learned_serializers[stage_idx + 1](point)

            if self.training and self.ordering_loss_weight > 0:
                ordering_losses.append(
                    self.compute_ordering_loss(point, point.sort_scores)
                )

        ordering_loss = (
            torch.stack(ordering_losses).mean()
            * self.ordering_loss_weight
            if ordering_losses
            else point.feat.new_zeros(())
        )
        
        # FIX: Do not attach it to the `point` variable yet, as `point` will be overwritten.
        # Save it in a distinct local variable.
        final_ordering_loss = ordering_loss

        # 5. Coarsest blocks.
        point = self.coarse_blocks(point)

        # 6. Decoder.
        for i, (unpool, blocks) in enumerate(zip(self.dec_unpool, self.dec_blocks)):
            fine_point = scaffold[-(i + 1)]
            point = unpool(point, fine_point)
            point = blocks(point)

        # 7. Output
        point.feat = self.output_proj(point.feat)
        if hasattr(point, "sparse_conv_feat") and point.sparse_conv_feat is not None:
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        for key in ("pooling_parent", "pooling_inverse"):
            if key in point.keys():
                point.pop(key)

        #  Attach the saved ordering loss
        point["ordering_loss"] = final_ordering_loss
        
        # --- The Nuclear DDP Anchor ---
        # Physically wire the MLP parameters to the output feature graph
        # to guarantee DDP never flags them as unused, regardless of config weights.
        if self.training:
            dummy_anchor = 0.0
            for serializer in self.learned_serializers:
                for param in serializer.sorter.mlp.parameters():
                    dummy_anchor = dummy_anchor + param.sum() * 0.0
            point.feat = point.feat + dummy_anchor

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
            if num_classes > 0 else nn.Identity()
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

        feat       = point.feat
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["seg_loss"] = loss
            
            ordering_loss = point.get("ordering_loss", None)
            if ordering_loss is not None:
                loss = loss + ordering_loss
                return_dict["ordering_loss"] = ordering_loss

            return_dict["loss"] = loss
            
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
            
        return return_dict