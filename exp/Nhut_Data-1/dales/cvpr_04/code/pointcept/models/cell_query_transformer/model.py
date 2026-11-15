"""Convolution-free, point-preserving hierarchical cell-query transformer.

Attention is local M x N inside occupied cells, followed by P x P attention
between pooled cells. No quaternion encoding and no sparse convolution.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from .geometry import (
    batch_from_data,
    centered_coordinates,
    group_mean,
    make_voxels,
    make_patches,
)


def safe_sdpa(q, k, v, attn_mask):
    """Exclude cuDNN SDPA, whose backward can reject ragged/masked shapes.

    Backend selection happens in forward; autograd retains the chosen operator
    for backward. No process-wide backend settings or parameter changes.
    """
    with sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        return F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=attn_mask.contiguous(),
        )


def mlp(cin, cout, expansion=2):
    return nn.Sequential(
        nn.LayerNorm(cin),
        nn.Linear(cin, cin * expansion),
        nn.GELU(),
        nn.Linear(cin * expansion, cout),
    )


class CellLocalAttention(nn.Module):
    def __init__(self, channels, heads, ratio, max_queries, chunk_size):
        super().__init__()
        self.heads, self.dim = heads, channels // heads
        self.ratio, self.max_queries, self.chunk_size = ratio, max_queries, chunk_size
        self.norm = nn.LayerNorm(channels)
        self.position = nn.Sequential(
            nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, 2 * channels)
        self.output = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, feature, coord, layout, cell_size):
        counts = layout.counts
        # Stable sorting preserves point identity; gathers alone change order.
        order = layout.inverse.argsort(stable=True)
        starts = counts.cumsum(0) - counts
        centers, _ = group_mean(coord, layout.inverse, len(counts))
        normalized = self.norm(feature)
        position = self.position(
            ((coord - centers[layout.inverse]) / cell_size).to(feature.dtype)
        )
        q_input = normalized + position
        updates, targets = [], []
        # Occupancy buckets avoid padding every cell to the densest cell.
        buckets = torch.floor(torch.log2(counts.float())).long()
        for bucket in torch.unique(buckets).tolist():
            cells = (buckets == bucket).nonzero(as_tuple=True)[0]
            for start in range(0, len(cells), self.chunk_size):
                ids = cells[start : start + self.chunk_size]
                n = counts[ids]
                quotas = (
                    torch.ceil(n.float() * self.ratio).long().clamp(1, self.max_queries)
                )
                width, queries = int(n.max()), int(quotas.max())
                rank = torch.arange(width, device=feature.device)[None]
                valid_k = rank < n[:, None]
                members = order[(starts[ids, None] + rank).clamp_max(len(order) - 1)]
                slots = torch.arange(queries, device=feature.device)[None]
                valid_q = slots < quotas[:, None]
                low = slots * n[:, None] // quotas[:, None]
                high = (slots + 1) * n[:, None] // quotas[:, None]
                fraction = (
                    torch.rand(low.shape, device=feature.device)
                    if self.training
                    else 0.5
                )
                positions = low + torch.floor((high - low) * fraction).long()
                positions = torch.minimum(positions, n[:, None] - 1)
                selected = members.gather(1, positions)
                b = len(ids)
                q = (
                    self.q(q_input[selected])
                    .reshape(b, queries, self.heads, self.dim)
                    .transpose(1, 2)
                )
                # Cartesian offsets condition the local keys and values.
                kv = self.kv(q_input[members]).reshape(
                    b, width, 2, self.heads, self.dim
                )
                k = kv[:, :, 0].transpose(1, 2)
                v = kv[:, :, 1].transpose(1, 2)
                out = safe_sdpa(q, k, v, attn_mask=valid_k[:, None, None, :])
                out = self.output(out.transpose(1, 2).reshape(b, queries, -1))
                updates.append(out[valid_q])
                targets.append(selected[valid_q])
        selected = torch.cat(targets)
        updated = feature[selected] + self.scale * torch.cat(updates)
        return feature.index_copy(0, selected, updated.to(feature.dtype)), selected


class CellPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.value = mlp(channels, channels)
        self.fuse = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.LayerNorm(channels), nn.GELU()
        )

    def forward(self, feature, coord, mass, layout):
        group, size = layout.inverse, len(layout.counts)
        value = self.value(feature)
        mean, _ = group_mean(value, group, size)
        maximum = value.new_full((size, value.shape[1]), -torch.inf)
        maximum.scatter_reduce_(
            0, group[:, None].expand_as(value), value, reduce="amax", include_self=True
        )
        pooled = self.fuse(torch.cat((mean, maximum), -1))
        with torch.no_grad():
            center, mass = group_mean(coord, group, size, mass)
        return pooled, center, mass, layout.indices[:, 0]


class SerializedCellBlock(nn.Module):
    def __init__(self, channels, heads, chunk_size, ffn_ratio):
        super().__init__()
        self.heads, self.dim, self.chunk_size = heads, channels // heads, chunk_size
        self.norm = nn.LayerNorm(channels)
        self.position = nn.Sequential(
            nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.qkv = nn.Linear(channels, channels * 3)
        self.output = nn.Linear(channels, channels)
        self.ffn = mlp(channels, channels, ffn_ratio)
        self.attention_scale = nn.Parameter(torch.full((channels,), 0.1))
        self.ffn_scale = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, feature, coord, patches, spatial_scale):
        normalized = self.norm(feature)
        updates, targets = [], []
        for start in range(0, len(patches.members), self.chunk_size):
            members = patches.members[start : start + self.chunk_size]
            valid = patches.valid[start : start + self.chunk_size]
            idx = members.clamp_min(0)
            xyz = coord[idx]
            # Patch-relative Cartesian encoding; intentionally not equivariant.
            center = (xyz * valid[..., None]).sum(1, keepdim=True) / valid.sum(1)[
                :, None, None
            ]
            positioned = normalized[idx] + self.position(
                ((xyz - center) / spatial_scale).to(feature.dtype)
            )
            b, p = members.shape
            qkv = self.qkv(positioned).reshape(b, p, 3, self.heads, self.dim)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            out = safe_sdpa(q, k, v, attn_mask=valid[:, None, None, :])
            out = self.output(out.transpose(1, 2).reshape(b, p, -1))
            updates.append(out[valid])
            targets.append(members[valid])
        target = torch.cat(targets)
        feature = feature.index_copy(
            0,
            target,
            (feature[target] + self.attention_scale * torch.cat(updates)).to(
                feature.dtype
            ),
        )
        return feature + self.ffn_scale * self.ffn(feature)


class DecoderFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.project = mlp(channels, channels)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * channels), nn.Linear(2 * channels, channels), nn.Sigmoid()
        )
        self.refine = mlp(channels, channels)

    def forward(self, coarse, skip, inverse):
        lifted = self.project(coarse)[inverse]
        mixed = skip + self.gate(torch.cat((skip, lifted), -1)) * lifted
        return mixed + self.refine(mixed)


class CellQueryTransformer(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=64,
        channels=(48, 96, 192, 256),
        heads=(3, 6, 12, 16),
        depths=(2, 2, 3, 2),
        cell_sizes=(0.4, 0.8, 1.6, 3.2),
        query_ratios=(0.25, 0.25, 0.5, 1.0),
        max_queries=(4, 4, 8, 8),
        patch_sizes=(128, 128, 128, 64),
        ffn_ratio=2,
        serialization_resolution=0.05,
        axis_orders=((0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0)),
        chunk_size=64,
    ):
        super().__init__()
        stages = len(channels)
        if stages < 1 or any(
            len(v) != stages
            for v in (heads, depths, cell_sizes, query_ratios, max_queries, patch_sizes)
        ):
            raise ValueError("Stage settings must have the same nonzero length")
        if any(c < 1 or h < 1 or c % h for c, h in zip(channels, heads)):
            raise ValueError("Channels must be positive and divisible by heads")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (*cell_sizes, serialization_resolution)
        ):
            raise ValueError("Spatial scales must be finite and positive")
        if any(b <= a for a, b in zip(cell_sizes, cell_sizes[1:])):
            raise ValueError("Cell sizes must increase across stages")
        if any(not 0 < r <= 1 for r in query_ratios) or any(
            v < 1 for v in (*depths, *max_queries, *patch_sizes, chunk_size, ffn_ratio)
        ):
            raise ValueError("Invalid query ratio or dimensions")
        if not axis_orders or any(sorted(a) != [0, 1, 2] for a in axis_orders):
            raise ValueError("axis_orders must be XYZ permutations")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.cell_sizes, self.patch_sizes = tuple(cell_sizes), tuple(patch_sizes)
        self.resolution, self.axis_orders = serialization_resolution, tuple(
            tuple(a) for a in axis_orders
        )
        self.stem = mlp(in_channels, channels[0])
        self.local = nn.ModuleList(
            [
                CellLocalAttention(c, h, r, m, chunk_size)
                for c, h, r, m in zip(channels, heads, query_ratios, max_queries)
            ]
        )
        self.local_ffn = nn.ModuleList([mlp(c, c, ffn_ratio) for c in channels])
        self.pools = nn.ModuleList([CellPooling(c) for c in channels])
        self.inter = nn.ModuleList(
            [
                nn.ModuleList(
                    [SerializedCellBlock(c, h, chunk_size, ffn_ratio) for _ in range(d)]
                )
                for c, h, d in zip(channels, heads, depths)
            ]
        )
        self.down = nn.ModuleList([mlp(a, b) for a, b in zip(channels, channels[1:])])
        self.up = nn.ModuleList([mlp(b, a) for a, b in zip(channels, channels[1:])])
        self.decoders = nn.ModuleList([DecoderFusion(c) for c in channels])
        self.head = mlp(channels[0], out_channels)

    def forward(self, data, return_debug=False):
        coord, raw = data["coord"], data["feat"]
        if (
            coord.ndim != 2
            or coord.shape[1] != 3
            or not len(coord)
            or raw.shape != (len(coord), self.in_channels)
        ):
            raise ValueError("Expected nonempty coord (N,3) and feat (N,in_channels)")
        if (
            coord.device != raw.device
            or not coord.is_floating_point()
            or not raw.is_floating_point()
        ):
            raise ValueError("Floating coordinates and features must share a device")
        if not bool(torch.isfinite(coord).all()) or not bool(torch.isfinite(raw).all()):
            raise ValueError("Inputs must be finite")
        batch = batch_from_data(data, len(coord), coord.device)
        with torch.no_grad():
            coord = centered_coordinates(coord.float(), batch)
        feature = self.stem(raw)
        mass = coord.new_ones(len(coord))
        records, debug = [], []
        for s, size in enumerate(self.cell_sizes):
            layout = make_voxels(coord, batch, size)
            feature, selected = self.local[s](feature, coord, layout, size)
            # Cheap FFN updates every point, including unselected points.
            feature = feature + self.local_ffn[s](feature)
            records.append((feature, layout.inverse))
            feature, coord, mass, batch = self.pools[s](feature, coord, mass, layout)
            layouts = {}
            for b, block in enumerate(self.inter[s]):
                axes = self.axis_orders[(s + b) % len(self.axis_orders)]
                if axes not in layouts:
                    layouts[axes] = make_patches(
                        coord, batch, self.resolution, axes, self.patch_sizes[s]
                    )
                feature = block(feature, coord, layouts[axes], size)
            if return_debug:
                debug.append(
                    dict(
                        stage=s,
                        selected=selected.detach(),
                        cell_inverse=layout.inverse.detach(),
                        cell_counts=layout.counts.detach(),
                        cell_batch=batch.detach(),
                        patches=[p.members.detach() for p in layouts.values()],
                    )
                )
            if s < len(self.down):
                feature = self.down[s](feature)
        for s in reversed(range(len(records))):
            skip, inverse = records[s]
            feature = self.decoders[s](feature, skip, inverse)
            if s > 0:
                feature = self.up[s - 1](feature)
        output = self.head(feature)
        return (output, debug) if return_debug else output
