"""Point-preserving sparse CNN + selected-query QuatRoPE encoder/decoder.

No wavelet branch and no decoder self-attention. Every cell member enters
learned pooling; all original points survive through saved fine-stage skips.
"""
import math

import torch
from torch import nn

from .geometry import (batch_from_data, centered_coordinates, group_mean,
                       make_patches, make_voxels)
from .attention import SelectedQueryAttention
from .sparse import SparseContext


def mlp(cin, cout, hidden=None):
    return nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, hidden or cout), nn.GELU(),
                         nn.Linear(hidden or cout, cout))


class EncoderBlock(nn.Module):
    def __init__(self, channels, heads, query_ratio, sparse_backend, use_sparse_cnn,
                 position_encoding, ffn_ratio, attention_dropout, patch_chunk_size,
                 max_distance, checkpoint_attention):
        super().__init__()
        self.spatial = SparseContext(channels, sparse_backend, use_sparse_cnn)
        self.attention = SelectedQueryAttention(
            channels, heads, query_ratio, position_encoding, attention_dropout,
            patch_chunk_size, max_distance, checkpoint_attention)
        self.ffn = mlp(channels, channels, int(channels * ffn_ratio))
        self.scale = nn.Parameter(torch.full((channels,), .1))

    def forward(self, feature, coord, voxels, patches):
        feature = self.spatial(feature, voxels)
        feature, selected = self.attention(feature, coord, patches)
        # Local attention and its FFN update only selected query points.
        updated = feature[selected] + self.scale * self.ffn(feature[selected])
        return feature.index_copy(0, selected, updated.to(feature.dtype)), selected


class AllPointPooling(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.value = mlp(cin, cout)
        self.score = nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, 1))
        self.output = nn.Sequential(nn.LayerNorm(cout), nn.GELU())

    def forward(self, feature, coord, mass, batch, layout):
        group, size = layout.inverse, len(layout.indices)
        score = self.score(feature).squeeze(-1).float()
        maximum = score.new_full((size,), -torch.inf)
        maximum.scatter_reduce_(0, group, score.detach(), reduce="amax", include_self=True)
        exp = (score - maximum[group]).exp()
        denominator = exp.new_zeros(size).index_add(0, group, exp)
        weight = exp / denominator[group].clamp_min(1e-12)
        value = self.value(feature)
        pooled = torch.zeros((size, value.shape[1]), device=value.device, dtype=torch.float32)
        pooled.index_add_(0, group, weight[:, None] * value.float())
        with torch.no_grad():
            centroid, total_mass = group_mean(coord, group, size, mass)
        return (self.output(pooled.to(value.dtype)), centroid, total_mass,
                layout.indices[:, 0], weight)


class DecoderFusion(nn.Module):
    def __init__(self, fine, coarse):
        super().__init__()
        self.project = mlp(coarse, fine)
        self.gate = nn.Sequential(nn.LayerNorm(fine * 2), nn.Linear(fine * 2, fine), nn.Sigmoid())
        self.refine = mlp(fine, fine, fine * 2)
        self.scale = nn.Parameter(torch.full((fine,), .1))

    def forward(self, coarse, skip, inverse):
        lifted = self.project(coarse)[inverse]
        mixed = skip + self.gate(torch.cat((skip, lifted), dim=-1)) * lifted
        return mixed + self.scale * self.refine(mixed)


class SparseQueryQuatRoPE(nn.Module):
    def __init__(self, in_channels=4, out_channels=64,
                 channels=(48, 96, 192, 256), depths=(2, 2, 3, 2),
                 num_heads=(3, 6, 12, 16), patch_sizes=(64, 64, 64, 32),
                 query_ratios=(.25, .5, .5, 1.), voxel_sizes=(.2, .4, .8, 1.6),
                 pool_sizes=(.4, .8, 1.6), stage_scales=(.5, 1., 2., 4.),
                 serialization_resolution=.05,
                 axis_orders=((0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0)),
                 sparse_backend="spconv", use_sparse_cnn=True,
                 position_encoding="quaternion", ffn_ratio=2., attention_dropout=0.,
                 max_neighbor_distances=(None, None, None, None),
                 patch_chunk_size=128, checkpoint_attention=True):
        super().__init__()
        stages = len(channels)
        if stages < 2 or len(pool_sizes) != stages - 1:
            raise ValueError("Need at least two stages and stages-1 pool_sizes")
        for name, values in dict(depths=depths, num_heads=num_heads, patch_sizes=patch_sizes,
                                 query_ratios=query_ratios, voxel_sizes=voxel_sizes,
                                 stage_scales=stage_scales, max_neighbor_distances=max_neighbor_distances).items():
            if len(values) != stages:
                raise ValueError(name + " must have one entry per stage")
        if any(not isinstance(v, int) or v < 1 for v in (*channels, *depths, *num_heads, *patch_sizes)):
            raise ValueError("Widths, depths, heads and patch sizes must be positive integers")
        if any(not math.isfinite(v) or v <= 0 for v in (*voxel_sizes, *pool_sizes, *stage_scales, serialization_resolution)):
            raise ValueError("All spatial scales must be finite and positive")
        if not axis_orders or any(sorted(a) != [0, 1, 2] for a in axis_orders):
            raise ValueError("axis_orders must contain XYZ permutations")
        if in_channels < 1 or out_channels < 1 or ffn_ratio <= 0 or not 0 <= attention_dropout < 1:
            raise ValueError("Invalid feature dimensions, FFN ratio or dropout")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.patch_sizes, self.voxel_sizes = tuple(patch_sizes), tuple(voxel_sizes)
        self.pool_sizes, self.stage_scales = tuple(pool_sizes), tuple(stage_scales)
        self.resolution = serialization_resolution
        self.axis_orders = tuple(tuple(a) for a in axis_orders)
        self.stem = mlp(in_channels, channels[0])
        self.encoders = nn.ModuleList()
        for s in range(stages):
            guard = max_neighbor_distances[s]
            if guard is not None and (not math.isfinite(guard) or guard <= 0):
                raise ValueError("Distance guards must be finite positive meters or None")
            self.encoders.append(nn.ModuleList([
                EncoderBlock(channels[s], num_heads[s], query_ratios[s], sparse_backend,
                             use_sparse_cnn, position_encoding, ffn_ratio, attention_dropout,
                             patch_chunk_size, None if guard is None else guard / stage_scales[s],
                             checkpoint_attention) for _ in range(depths[s])
            ]))
        self.pools = nn.ModuleList([AllPointPooling(channels[s], channels[s+1]) for s in range(stages-1)])
        self.decoders = nn.ModuleList([DecoderFusion(channels[s], channels[s+1]) for s in range(stages-1)])
        self.head = mlp(channels[0], out_channels)

    def forward(self, data, return_debug=False):
        coord, raw = data["coord"], data["feat"]
        if coord.ndim != 2 or coord.shape[1] != 3 or len(coord) == 0:
            raise ValueError("coord must be nonempty (N,3)")
        if raw.shape != (len(coord), self.in_channels) or raw.device != coord.device:
            raise ValueError("feat must have shape (N,in_channels) and share coord's device")
        if not coord.is_floating_point() or not raw.is_floating_point():
            raise ValueError("coord and feat must be floating-point tensors")
        if not bool(torch.isfinite(coord).all()) or not bool(torch.isfinite(raw).all()):
            raise ValueError("Input coordinates and features must be finite")
        batch = batch_from_data(data, len(coord), coord.device)
        with torch.no_grad():
            # Fixed per-cloud origin throughout the hierarchy. Pointwise raw
            # features retain the preprocessing selected by the data pipeline.
            local_coord = centered_coordinates(coord.float(), batch)
        feature = self.stem(raw)
        mass = coord.new_ones(len(coord), dtype=torch.float32)
        records, debug = [], []
        for s, blocks in enumerate(self.encoders):
            voxels = make_voxels(local_coord, batch, self.voxel_sizes[s])
            # Each block changes Morton axes. Point arrays themselves stay in
            # their current order; only patch gathers are serialized.
            layouts = {}
            for b, block in enumerate(blocks):
                axes = self.axis_orders[(s + b) % len(self.axis_orders)]
                if axes not in layouts:
                    layouts[axes] = make_patches(local_coord, batch, self.resolution, axes, self.patch_sizes[s])
                patches = layouts[axes]
                feature, selected = block(feature, local_coord / self.stage_scales[s], voxels, patches)
                if return_debug:
                    debug.append(dict(stage=s, block=b, selected=selected.detach(),
                                      members=patches.members.detach(), batch=batch.detach(),
                                      sparse_inverse=voxels.inverse.detach()))
            if s < len(self.pools):
                pooling = make_voxels(local_coord, batch, self.pool_sizes[s])
                skip = feature
                feature, local_coord, mass, batch, weight = self.pools[s](
                    feature, local_coord, mass, batch, pooling)
                records.append((skip, pooling.inverse))
                if return_debug:
                    debug[-1].update(pool_group=pooling.inverse.detach(), pool_weight=weight.detach())
        for s in reversed(range(len(records))):
            skip, inverse = records[s]
            feature = self.decoders[s](feature, skip, inverse)
        output = self.head(feature)
        return (output, debug) if return_debug else output
