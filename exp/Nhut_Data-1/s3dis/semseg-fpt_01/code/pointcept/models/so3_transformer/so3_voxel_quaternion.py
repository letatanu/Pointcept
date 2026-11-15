"""Experimental sparse voxel/quaternion backbone for Pointcept.

Import after so3_quaternion_dual_attention. Grid grouping is rotation dependent;
continuous geometry uses invariant scalar and equivariant vector operations.
No custom CUDA dependency. Sorted-key lookup is a reference implementation, not
an O(1) hash kernel. Auxiliary heads are unsupported (point/voxel label domains).
"""
import itertools
import math
import torch
from torch import nn
from .so3_quaternion_dual_attention import (
    MODELS, Point, SO3QuaternionDualNet, QuaternionRadiusAttention,
    RadiusGraph, StageCache, counts, segment_mean, segment_sum,
    patch_layout, safe_norm,
)


@torch.no_grad()
def voxel_partition(coord, offset, size):
    """Batch-safe inverse map, representatives, centroids and cumulative sizes."""
    batch = torch.repeat_interleave(torch.arange(len(offset), device=coord.device), counts(offset))
    cells = torch.floor(coord.float() / size).long()
    keys, parent = torch.unique(torch.cat([batch[:, None], cells], 1), dim=0, return_inverse=True)
    m = len(keys)
    representatives = torch.full((m,), len(coord), device=coord.device, dtype=torch.long)
    representatives.scatter_reduce_(0, parent, torch.arange(len(coord), device=coord.device), reduce='amin')
    centroid = segment_mean(coord.float(), parent, m)
    coarse_offset = torch.bincount(keys[:, 0], minlength=len(offset)).cumsum(0).to(offset.dtype)
    return parent, representatives, centroid, coarse_offset


@torch.no_grad()
def voxel_graph(coord, offset, size, radius, chunk_size=4096):
    """All occupied adjacent cells, with a centroid-radius filter; self included.

    At most one token per cell is required. No cross-cloud edges. The 27-cell
    stencil has fixed grid support; radius is a further filter, not a promise
    of a complete radius graph outside this stencil.
    """
    batch = torch.repeat_interleave(torch.arange(len(offset), device=coord.device), counts(offset))
    cells = torch.floor(coord.float() / size).long()
    cells = cells - cells.amin(0) + 1
    widths = cells.amax(0) + 2
    wx, wy, wz = [int(x) for x in widths]
    if (len(offset) + 1) * wx * wy * wz >= 2**62:
        raise ValueError('Voxel key overflow; increase voxel size or reduce scene extent')
    def encode(b, c):
        return ((b * wx + c[..., 0]) * wy + c[..., 1]) * wz + c[..., 2]
    keys = encode(batch, cells)
    sorted_keys, order = keys.sort()
    if len(keys.unique()) != len(keys):
        raise ValueError('voxel_graph requires one token per occupied cell')
    shifts = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), device=coord.device)
    dsts, srcs = [], []
    for start in range(0, len(coord), chunk_size):
        end = min(start + chunk_size, len(coord))
        candidates = cells[start:end, None] + shifts
        code = encode(batch[start:end, None], candidates)
        pos = torch.searchsorted(sorted_keys, code).clamp_max(len(keys) - 1)
        valid = sorted_keys[pos] == code
        dd = torch.arange(start, end, device=coord.device)[:, None].expand_as(pos)
        ss = order[pos]
        valid &= (batch[ss] == batch[dd])
        valid &= ((cells[ss] - cells[dd]).abs() <= 1).all(-1)
        valid &= ((coord[ss] - coord[dd]).square().sum(-1) <= radius**2)
        dsts.append(dd[valid]); srcs.append(ss[valid])
    return RadiusGraph(torch.cat(dsts), torch.cat(srcs), radius)


class LiteQuaternionAttention(QuaternionRadiusAttention):
    """Original quaternion scores; node-wise values plus scalar edge gates."""
    def __init__(self, channels, vector_channels, **kwargs):
        super().__init__(channels, vector_channels, **kwargs)
        self.scalar_value = nn.Linear(channels, channels)
        # Keep a small invariant gate per vector group instead of channel-wise
        # scalar/value vectors and quaternion relation messages on every edge.
        self.values = nn.Linear(3 * self.cq + 1, self.groups)
        del self.relation_vector

    def _message_chunk(self, aq, bk, uq, wk, value, dst, src, rel, weights):
        invariant, _ = self._relation(aq, bk, uq, wk, dst, src, rel)
        gate = self.values(invariant.to(aq.dtype)).tanh().float()
        expanded = weights.repeat_interleave(self.cv // self.groups, -1)
        displacement_gate = gate.repeat_interleave(self.cv // self.groups, -1)
        vv, sv = value
        vector = expanded[..., None] * (
            vv[src].float() + displacement_gate[..., None] * rel[:, None])
        scalar = sv[src].float() * weights.mean(-1, keepdim=True)
        return scalar, vector

    def forward(self, feat, vec, coord, graph):
        # Parent forward uses vector_value(vec); carry node scalar values alongside
        # it locally, without module state (safe for checkpoint recomputation).
        aq, bk = self.real_qk(feat).chunk(2, -1)
        uq, wk = self.vector_q(vec), self.vector_k(vec)
        value = (self.vector_value(vec), self.scalar_value(feat))
        from .so3_quaternion_dual_attention import segment_softmax
        from torch.utils.checkpoint import checkpoint
        dst, src = graph.dst, graph.src
        rel = ((coord[src] - coord[dst]) / graph.radius).detach()
        enabled = self.training and self.checkpoint_edges and torch.is_grad_enabled()
        def run(fn, *args):
            return checkpoint(fn, *args, use_reentrant=False) if enabled else fn(*args)
        logits = torch.cat([run(self._logit_chunk, aq, bk, uq, wk,
            dst[a:a+self.chunk_size], src[a:a+self.chunk_size], rel[a:a+self.chunk_size])
            for a in range(0, len(dst), self.chunk_size)])
        weights = torch.nn.functional.dropout(segment_softmax(logits, dst, len(feat)), self.attn_drop, self.training)
        scalar = torch.zeros_like(feat, dtype=torch.float32)
        vector = torch.zeros_like(vec, dtype=torch.float32)
        for a in range(0, len(dst), self.chunk_size):
            dd, ss, rr = dst[a:a+self.chunk_size], src[a:a+self.chunk_size], rel[a:a+self.chunk_size]
            si, vi = run(self._message_chunk, aq, bk, uq, wk, value, dd, ss, rr, weights[a:a+self.chunk_size])
            scalar = scalar.index_add(0, dd, si)
            vector = vector.index_add(0, dd, vi)
        return self.scalar_out(scalar.to(feat.dtype)), self.vector_out(vector.to(vec.dtype))


@MODELS.register_module('Voxel-Quaternion-Dual-v1')
class VoxelQuaternionDualNet(SO3QuaternionDualNet):
    def __init__(self, voxel_sizes=(0.2, 0.4, 0.8, 1.6), quaternion_stages=(2, 3), **kwargs):
        if not kwargs.get('use_quaternion_attention', True):
            raise ValueError('Use quaternion_stages=() to disable quaternion blocks')
        super().__init__(**kwargs)
        if len(voxel_sizes) != len(self.channels) or any(not math.isfinite(s) or s <= 0 for s in voxel_sizes):
            raise ValueError('voxel_sizes must give one finite positive size per stage')
        if any(abs(voxel_sizes[i+1] / voxel_sizes[i] - round(voxel_sizes[i+1] / voxel_sizes[i])) > 1e-6 or voxel_sizes[i+1] <= voxel_sizes[i] for i in range(len(voxel_sizes)-1)):
            raise ValueError('voxel sizes must grow by integer factors for nested pooling')
        if any(i < 0 or i >= len(self.channels) for i in quaternion_stages):
            raise ValueError('Invalid quaternion stage')
        self.voxel_sizes = tuple(voxel_sizes)
        for level, stage in list(enumerate(self.stages)) + list(zip(reversed(range(len(self.channels)-1)), self.decoder)):
            for block in stage.blocks:
                block.use_quaternion = level in quaternion_stages
                if hasattr(block, 'quaternion_attention'):
                    del block.quaternion_attention
                if not block.use_quaternion:
                    for name in ('vector_norm', 'vector_update', 'geometry_gate'):
                        delattr(block, name)
                if block.use_quaternion:
                    block.quaternion_attention = LiteQuaternionAttention(
                        self.channels[level], self.vector_channels[level],
                        quaternion_channels=kwargs.get('quaternion_channels', 4),
                        groups=kwargs.get('geometry_groups', 4),
                        geometry_width=kwargs.get('geometry_width', 32),
                        chunk_size=kwargs.get('edge_chunk_size', 65536),
                        checkpoint_edges=kwargs.get('checkpoint_edges', True),
                        attn_drop=kwargs.get('attn_drop', 0.0))
        # Norm-only intra-voxel geometry avoids a Cartesian position MLP.
        self.point_refine = nn.Sequential(nn.Linear(self.head_channels + max(1, len(self._scalar_indices)) + 1, self.head_channels), nn.GELU(), nn.Linear(self.head_channels, self.head_channels))

    @torch.no_grad()
    def _cache(self, state, level):
        size = self.voxel_sizes[level]
        graph = voxel_graph(state.coord, state.offset, size, self.radii[level])
        pooling_size = self.voxel_sizes[min(level+1, len(self.voxel_sizes)-1)]
        parent, centers, _, _ = voxel_partition(state.coord, state.offset, pooling_size)
        priority = torch.arange(len(state.coord), device=state.coord.device)
        layout = patch_layout(parent, priority, self.patch_sizes[level])
        return StageCache(graph, parent, centers, layout)

    def forward(self, data_dict, return_stages=False):
        if return_stages:
            raise ValueError('Disable deep_supervision for voxel backbone; auxiliary labels require voxel pooling first')
        coord, feat, offset = data_dict['coord'], data_dict['feat'], data_dict['offset']
        if coord.ndim != 2 or coord.shape[1] != 3 or not len(coord) or not bool(torch.isfinite(coord).all()):
            raise ValueError('coord must be finite nonempty [N,3]')
        if offset.ndim != 1 or offset.dtype not in (torch.int32, torch.int64) or not len(offset) or int(offset[-1]) != len(coord) or bool((counts(offset) <= 0).any()):
            raise ValueError('offset must give positive cumulative cloud sizes ending at N')
        if feat.shape != (len(coord), self.in_channels) or not feat.is_floating_point() or not bool(torch.isfinite(feat).all()):
            raise ValueError('Invalid input features')
        if feat.device != coord.device or offset.device != coord.device:
            raise ValueError('Input tensors must share a device')
        parent, centers, centroid, coarse_offset = voxel_partition(coord.detach(), offset, self.voxel_sizes[0])
        pooled = segment_mean(feat, parent, len(centers))
        voxel = super().forward(dict(coord=centroid, feat=pooled, offset=coarse_offset))
        scalar = feat.index_select(1, self._scalar_indices) if self._scalar_indices.numel() else feat.new_ones(len(feat), 1)
        radius = safe_norm(coord.detach().float()-centroid[parent], keepdim=True).to(feat.dtype) / self.voxel_sizes[0]
        decoded = voxel.feat[parent]
        point = Point(dict(data_dict))
        point.feat = decoded + self.point_refine(torch.cat([decoded, scalar, radius], -1))
        point['vector_feat'] = voxel.vector_feat[parent]
        return point
