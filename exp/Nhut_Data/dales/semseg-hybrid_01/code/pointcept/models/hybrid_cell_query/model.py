"""Fine-grid sparse CNN + selected-query attention over neighboring cells."""

from itertools import product
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from .geometry import batch_from_data, centered_coordinates, group_mean, make_voxels
from .sparse import SparseContext


def mlp(cin, cout, ratio=2):
    return nn.Sequential(
        nn.LayerNorm(cin),
        nn.Linear(cin, cin * ratio),
        nn.GELU(),
        nn.Linear(cin * ratio, cout),
    )


@torch.no_grad()
def cell_order(layout):
    order = layout.inverse.argsort(stable=True)
    starts = layout.counts.cumsum(0) - layout.counts
    return order, starts


@torch.no_grad()
def sample_members(layout, order, starts, width, ratio, training):
    """One sample per disjoint index stratum; fixed padded width, no chunk maxima."""
    counts = layout.counts
    quota = torch.ceil(counts.float() * ratio).long().clamp(1, width)
    slots = torch.arange(width, device=counts.device)[None]
    low = slots * counts[:, None] // quota[:, None]
    high = (slots + 1) * counts[:, None] // quota[:, None]
    fraction = torch.rand(low.shape, device=counts.device) if training else 0.5
    rank = low + torch.floor((high - low) * fraction).long()
    rank = torch.minimum(rank, counts[:, None] - 1)
    members = order[starts[:, None] + rank]
    return members, slots < quota[:, None]


@torch.no_grad()
def neighboring_cells(layout):
    """Exact batch-safe lookup of the 27 adjacent occupied query cells."""
    shape = layout.spatial_shape
    if math.prod(shape) * layout.batch_size >= 2**63:
        raise ValueError("Neighbor hash exceeds int64")

    def encode(index):
        return (
            (index[..., 0] * shape[0] + index[..., 1]) * shape[1] + index[..., 2]
        ) * shape[2] + index[..., 3]

    index = layout.indices
    offsets = index.new_tensor([(0, *delta) for delta in product((-1, 0, 1), repeat=3)])
    wanted = index[:, None, :] + offsets[None]
    valid = ((wanted[..., 1:] >= 0) & (wanted[..., 1:] < index.new_tensor(shape))).all(
        -1
    )
    keys, permutation = encode(index).sort()
    query = encode(wanted)
    location = torch.searchsorted(keys, query.contiguous()).clamp_max(len(keys) - 1)
    valid = valid & (keys[location] == query)
    return permutation[location].masked_fill(~valid, -1)


class NeighborAttention(nn.Module):
    def __init__(
        self, channels, heads, max_queries, keys_per_cell, query_ratio, chunk_size
    ):
        super().__init__()
        self.heads, self.dim = heads, channels // heads
        self.max_queries, self.keys_per_cell = max_queries, keys_per_cell
        self.query_ratio, self.chunk_size = query_ratio, chunk_size
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.position = nn.Sequential(
            nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.output = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.full((channels,), 0.1))

    def forward(
        self,
        feature,
        coord,
        layout,
        order,
        starts,
        neighbors,
        cell_size,
        return_debug=False,
    ):
        query, qvalid = sample_members(
            layout, order, starts, self.max_queries, self.query_ratio, self.training
        )
        # Independent capped key sampling; no raw points are dropped from skips/pooling.
        key, kvalid = sample_members(
            layout, order, starts, self.keys_per_cell, 1.0, self.training
        )
        normalized = self.norm(feature)
        # Project real points once BEFORE padded neighborhood gathering.
        q_all = self.q(normalized)
        k_all, v_all = self.kv(normalized).chunk(2, -1)
        updates, targets = [], []
        for begin in range(0, len(query), self.chunk_size):
            end = begin + self.chunk_size
            adjacent = neighbors[begin:end]
            neighbor_valid = adjacent >= 0
            adjacent = adjacent.clamp_min(0)
            members = key[adjacent].flatten(1)
            mask = (kvalid[adjacent] & neighbor_valid[..., None]).flatten(1)
            selected, valid = query[begin:end], qvalid[begin:end]
            # Shared query-cell origin preserves pair geometry without large phases.
            origin = layout.indices[begin:end, None, 1:].to(coord.dtype) * cell_size
            qp = self.position(
                ((coord[selected] - origin) / cell_size).to(feature.dtype)
            )
            kp = self.position(
                ((coord[members] - origin) / cell_size).to(feature.dtype)
            )
            b, m = selected.shape
            width = members.shape[1]
            q = (
                (q_all[selected] + qp)
                .reshape(b, m, self.heads, self.dim)
                .transpose(1, 2)
            )
            k = (
                (k_all[members] + kp)
                .reshape(b, width, self.heads, self.dim)
                .transpose(1, 2)
            )
            v = v_all[members].reshape(b, width, self.heads, self.dim).transpose(1, 2)
            # Center cell always supplies a valid key, even for singleton clouds.
            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ):
                out = F.scaled_dot_product_attention(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    attn_mask=mask[:, None, None, :],
                )
            out = self.output(out.transpose(1, 2).reshape(b, m, -1))
            updates.append(out[valid])
            targets.append(selected[valid])
        target = torch.cat(targets)
        output = feature.index_copy(
            0,
            target,
            (feature[target] + self.scale * torch.cat(updates)).to(feature.dtype),
        )
        debug = (
            dict(
                selected=target.detach(),
                query=query.detach(),
                query_valid=qvalid.detach(),
                keys=key.detach(),
                key_valid=kvalid.detach(),
            )
            if return_debug
            else None
        )
        return output, debug


class Block(nn.Module):
    def __init__(self, channels, heads, ratio, queries, keys, backend, chunk_size):
        super().__init__()
        self.spatial = SparseContext(channels, backend, True)
        self.attention = NeighborAttention(
            channels, heads, queries, keys, ratio, chunk_size
        )
        self.ffn = mlp(channels, channels)
        self.scale = nn.Parameter(torch.full((channels,), 0.1))

    def forward(
        self, feature, coord, fine, cells, order, starts, neighbors, size, debug
    ):
        feature = self.spatial(feature, fine)
        feature, record = self.attention(
            feature, coord, cells, order, starts, neighbors, size, debug
        )
        return feature + self.scale * self.ffn(feature), record


class Pool(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.value = mlp(cin, cout)
        self.fuse = nn.Sequential(
            nn.Linear(cout * 2, cout), nn.LayerNorm(cout), nn.GELU()
        )

    def forward(self, feature, coord, mass, layout):
        group, count = layout.inverse, len(layout.counts)
        value = self.value(feature)
        mean, _ = group_mean(value, group, count)
        maximum = value.new_full((count, value.shape[1]), -torch.inf)
        maximum.scatter_reduce_(
            0, group[:, None].expand_as(value), value, reduce="amax", include_self=True
        )
        with torch.no_grad():
            coord, mass = group_mean(coord, group, count, mass)
        return (
            self.fuse(torch.cat((mean, maximum), -1)),
            coord,
            mass,
            layout.indices[:, 0],
        )


class Fusion(nn.Module):
    def __init__(self, fine, coarse):
        super().__init__()
        self.project = mlp(coarse, fine)
        self.gate = nn.Sequential(
            nn.LayerNorm(fine * 2), nn.Linear(fine * 2, fine), nn.Sigmoid()
        )
        self.refine = mlp(fine, fine)

    def forward(self, coarse, skip, inverse):
        lifted = self.project(coarse)[inverse]
        mixed = skip + self.gate(torch.cat((skip, lifted), -1)) * lifted
        return mixed + self.refine(mixed)


class HybridCellQuery(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=64,
        channels=(48, 96, 192, 256),
        heads=(3, 6, 12, 16),
        depths=(2, 2, 3, 2),
        voxel_sizes=(0.2, 0.4, 0.8, 1.6),
        cell_sizes=(0.4, 0.8, 1.6, 3.2),
        query_ratios=(0.25, 0.5, 0.5, 1.0),
        max_queries=(4, 4, 8, 8),
        keys_per_cell=(4, 4, 4, 4),
        chunk_size=512,
        sparse_backend="spconv",
    ):
        super().__init__()
        stages = len(channels)
        settings = (
            heads,
            depths,
            voxel_sizes,
            cell_sizes,
            query_ratios,
            max_queries,
            keys_per_cell,
        )
        if not stages or any(len(v) != stages for v in settings):
            raise ValueError("All stage settings must have equal nonzero length")
        if any(
            not isinstance(v, int) or v < 1
            for v in (
                *channels,
                *heads,
                *depths,
                *max_queries,
                *keys_per_cell,
                chunk_size,
            )
        ):
            raise ValueError(
                "Widths, depths, sample counts and chunk size must be positive integers"
            )
        if any(c % h for c, h in zip(channels, heads)) or any(
            not 0 < r <= 1 for r in query_ratios
        ):
            raise ValueError("Invalid head widths or query ratios")
        if any(not math.isfinite(v) or v <= 0 for v in (*voxel_sizes, *cell_sizes)):
            raise ValueError("Spatial sizes must be finite and positive")
        if any(v >= c for v, c in zip(voxel_sizes, cell_sizes)):
            raise ValueError("Fine voxel sizes must be smaller than query cells")
        if any(b <= a for a, b in zip(cell_sizes, cell_sizes[1:])):
            raise ValueError("Cell sizes must increase")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.voxel_sizes, self.cell_sizes = tuple(voxel_sizes), tuple(cell_sizes)
        self.stem = mlp(in_channels, channels[0])
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [Block(c, h, r, q, k, sparse_backend, chunk_size) for _ in range(d)]
                )
                for c, h, r, q, k, d in zip(
                    channels, heads, query_ratios, max_queries, keys_per_cell, depths
                )
            ]
        )
        self.pools = nn.ModuleList([Pool(a, b) for a, b in zip(channels, channels[1:])])
        self.decoders = nn.ModuleList(
            [Fusion(a, b) for a, b in zip(channels, channels[1:])]
        )
        self.head = mlp(channels[0], out_channels)

    def forward(self, data, return_debug=False):
        coord, raw = data["coord"], data["feat"]
        if (
            coord.ndim != 2
            or coord.shape[1] != 3
            or not len(coord)
            or raw.shape != (len(coord), self.in_channels)
        ):
            raise ValueError("Expected coord (N,3), feat (N,in_channels), N>0")
        if (
            coord.device != raw.device
            or not coord.is_floating_point()
            or not raw.is_floating_point()
        ):
            raise ValueError("Floating coordinates/features must share a device")
        if not bool(torch.isfinite(coord).all()) or not bool(torch.isfinite(raw).all()):
            raise ValueError("Inputs must be finite")
        batch = batch_from_data(data, len(coord), coord.device)
        with torch.no_grad():
            coord = centered_coordinates(coord.float(), batch)
        feature = self.stem(raw)
        mass = coord.new_ones(len(coord))
        records, debug = [], []
        for s, blocks in enumerate(self.blocks):
            # Build spatial metadata ONCE per stage, reuse in every block.
            fine = make_voxels(coord, batch, self.voxel_sizes[s])
            cells = make_voxels(coord, batch, self.cell_sizes[s])
            order, starts = cell_order(cells)
            neighbors = neighboring_cells(cells)
            for block in blocks:
                feature, record = block(
                    feature,
                    coord,
                    fine,
                    cells,
                    order,
                    starts,
                    neighbors,
                    self.cell_sizes[s],
                    return_debug,
                )
                if return_debug:
                    record.update(
                        stage=s,
                        cell_inverse=cells.inverse.detach(),
                        cell_indices=cells.indices.detach(),
                        counts=cells.counts.detach(),
                        neighbors=neighbors.detach(),
                    )
                    debug.append(record)
            if s < len(self.pools):
                records.append((feature, cells.inverse))
                feature, coord, mass, batch = self.pools[s](feature, coord, mass, cells)
        for s in reversed(range(len(records))):
            skip, inverse = records[s]
            feature = self.decoders[s](feature, skip, inverse)
        output = self.head(feature)
        return (output, debug) if return_debug else output
