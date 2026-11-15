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
    return nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, cin * ratio), nn.GELU(), nn.Linear(cin * ratio, cout))


from .sampling import cell_order, sample_members, neighboring_cells
from .packed_attention import PackedNeighborAttention as NeighborAttention


class Block(nn.Module):
    def __init__(self, channels, heads, ratio, queries, keys, backend, chunk_size, attention_backend):
        super().__init__()
        self.spatial = SparseContext(channels, backend, True)
        self.attention = NeighborAttention(channels, heads, queries, keys, ratio, chunk_size, attention_backend)
        self.ffn = mlp(channels, channels)
        self.scale = nn.Parameter(torch.full((channels,), .1))

    def forward(self, feature, coord, fine, cells, order, starts, neighbors, size, debug):
        feature = self.spatial(feature, fine)
        feature, record = self.attention(feature, coord, cells, order, starts, neighbors, size, debug)
        return feature + self.scale * self.ffn(feature), record


class Pool(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.value = mlp(cin, cout)
        self.fuse = nn.Sequential(nn.Linear(cout * 2, cout), nn.LayerNorm(cout), nn.GELU())

    def forward(self, feature, coord, mass, layout):
        group, count = layout.inverse, len(layout.counts)
        value = self.value(feature)
        mean, _ = group_mean(value, group, count)
        maximum = value.new_full((count, value.shape[1]), -torch.inf)
        maximum.scatter_reduce_(0, group[:, None].expand_as(value), value, reduce="amax", include_self=True)
        with torch.no_grad():
            coord, mass = group_mean(coord, group, count, mass)
        return self.fuse(torch.cat((mean, maximum), -1)), coord, mass, layout.indices[:, 0]


class Fusion(nn.Module):
    def __init__(self, fine, coarse):
        super().__init__()
        self.project = mlp(coarse, fine)
        self.gate = nn.Sequential(nn.LayerNorm(fine * 2), nn.Linear(fine * 2, fine), nn.Sigmoid())
        self.refine = mlp(fine, fine)

    def forward(self, coarse, skip, inverse):
        lifted = self.project(coarse)[inverse]
        mixed = skip + self.gate(torch.cat((skip, lifted), -1)) * lifted
        return mixed + self.refine(mixed)


class PackedHybridCellQuery(nn.Module):
    def __init__(self, in_channels=4, out_channels=64,
                 channels=(48, 96, 192, 256), heads=(3, 6, 12, 16), depths=(2, 2, 3, 2),
                 voxel_sizes=(.2, .4, .8, 1.6), cell_sizes=(.4, .8, 1.6, 3.2),
                 query_ratios=(.25, .5, .5, 1.), max_queries=(4, 4, 8, 8),
                 keys_per_cell=(4, 4, 4, 4), chunk_size=512, sparse_backend="spconv", attention_backend="flash_varlen"):
        super().__init__()
        stages = len(channels)
        settings = (heads, depths, voxel_sizes, cell_sizes, query_ratios, max_queries, keys_per_cell)
        if not stages or any(len(v) != stages for v in settings):
            raise ValueError("All stage settings must have equal nonzero length")
        if any(not isinstance(v, int) or v < 1 for v in (*channels, *heads, *depths, *max_queries, *keys_per_cell, chunk_size)):
            raise ValueError("Widths, depths, sample counts and chunk size must be positive integers")
        if any(c % h for c, h in zip(channels, heads)) or any(not 0 < r <= 1 for r in query_ratios):
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
        self.blocks = nn.ModuleList([nn.ModuleList([
            Block(c, h, r, q, k, sparse_backend, chunk_size, attention_backend) for _ in range(d)])
            for c, h, r, q, k, d in zip(channels, heads, query_ratios, max_queries, keys_per_cell, depths)])
        self.pools = nn.ModuleList([Pool(a, b) for a, b in zip(channels, channels[1:])])
        self.decoders = nn.ModuleList([Fusion(a, b) for a, b in zip(channels, channels[1:])])
        self.head = mlp(channels[0], out_channels)

    def forward(self, data, return_debug=False):
        coord, raw = data["coord"], data["feat"]
        if coord.ndim != 2 or coord.shape[1] != 3 or not len(coord) or raw.shape != (len(coord), self.in_channels):
            raise ValueError("Expected coord (N,3), feat (N,in_channels), N>0")
        if coord.device != raw.device or not coord.is_floating_point() or not raw.is_floating_point():
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
                feature, record = block(feature, coord, fine, cells, order, starts, neighbors, self.cell_sizes[s], return_debug)
                if return_debug:
                    record.update(stage=s, cell_inverse=cells.inverse.detach(), cell_indices=cells.indices.detach(),
                                  counts=cells.counts.detach(), neighbors=neighbors.detach())
                    debug.append(record)
            if s < len(self.pools):
                records.append((feature, cells.inverse))
                feature, coord, mass, batch = self.pools[s](feature, coord, mass, cells)
        for s in reversed(range(len(records))):
            skip, inverse = records[s]
            feature = self.decoders[s](feature, skip, inverse)
        output = self.head(feature)
        return (output, debug) if return_debug else output
