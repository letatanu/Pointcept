"""Torch-only geometry operators. Coordinates/radii use the same metric units.

No KNN, dense all-pairs distances, torch_cluster, or new CUDA extension.
Radius enumeration is exact before a uniform random training-time neighbor cap.
Evaluation selection is deterministic for a fixed input order.
"""

import itertools
import math

import torch
from torch import nn


def polar_features(delta, mode="cylindrical"):
    """Five seam-free features; undefined directions are zero, including poles."""
    if mode not in ("spherical", "cylindrical"):
        raise ValueError("embedding_mode must be spherical or cylindrical")
    xyz = delta.float()
    rho = torch.linalg.vector_norm(xyz[..., :2], dim=-1)
    radius = torch.linalg.vector_norm(xyz, dim=-1)
    valid = rho > 1e-8
    sin_theta = torch.where(valid, xyz[..., 1] / rho.clamp_min(1e-8), 0.0)
    cos_theta = torch.where(valid, xyz[..., 0] / rho.clamp_min(1e-8), 0.0)
    if mode == "spherical":
        sin_phi = xyz[..., 2] / radius.clamp_min(1e-8)
        cos_phi = rho / radius.clamp_min(1e-8)
        return torch.stack((radius.log1p(), sin_theta, cos_theta, sin_phi, cos_phi), -1)
    return torch.stack(
        (rho.log1p(), sin_theta, cos_theta, xyz[..., 2], radius.log1p()), -1
    )


def hamilton_product(a, b):
    """Left multiplication, scalar-first convention, broadcasting supported."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


class PolarQuaternionMessage(nn.Module):
    """Polar embedding plus optional unit-quaternion mixing of latent channels.

    This is not an SO(3)-equivariant network. The two coordinate modes affect
    the learned polar branch; the quaternion branch uses the same displacement
    direction and metric radius in either mode, for a controlled ablation.
    """

    def __init__(self, channels, mode="cylindrical", use_quaternion=True):
        super().__init__()
        if mode not in ("spherical", "cylindrical"):
            raise ValueError("Unsupported embedding mode: " + str(mode))
        if channels % 4:
            raise ValueError("Geometry channels must be divisible by four")
        self.mode = mode
        self.use_quaternion = use_quaternion
        self.position = nn.Sequential(
            nn.Linear(5, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.register_buffer("frequencies", torch.logspace(-1, 0.6, channels // 4))

    def forward(self, delta, feature):
        delta = delta.float()
        position = self.position(polar_features(delta, self.mode))
        if not self.use_quaternion:
            return feature + position.to(feature.dtype)
        radius = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        axis = delta / radius.clamp_min(1e-8)
        phase = radius * self.frequencies.float() * 0.5
        rotor = torch.cat(
            (phase.cos().unsqueeze(-1), phase.sin().unsqueeze(-1) * axis.unsqueeze(-2)),
            -1,
        )
        mixed = hamilton_product(
            rotor, feature.float().reshape(*feature.shape[:-1], -1, 4)
        )
        return mixed.flatten(-2).to(feature.dtype) + position.to(feature.dtype)


def segment_sum(value, group, size):
    return value.new_zeros((size,) + value.shape[1:]).index_add_(0, group, value)


def segment_softmax(logit, group, size):
    """FP32 scatter softmax, including safe handling of unoccupied groups."""
    x = logit.float()
    maximum = x.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, group, x.detach(), reduce="amax", include_self=True)
    exp = (x - maximum[group]).exp()
    return exp / segment_sum(exp, group, size)[group].clamp_min(1e-12)


@torch.no_grad()
def capped_members(group, size, cap, training):
    """Uniform sample without replacement within each group when training."""
    if cap < 1:
        raise ValueError("cap must be positive")
    permutation = (
        torch.randperm(group.numel(), device=group.device)
        if training
        else torch.arange(group.numel(), device=group.device)
    )
    ordered = permutation[torch.argsort(group[permutation], stable=True)]
    counts = torch.bincount(group, minlength=size)
    starts = counts.cumsum(0) - counts
    rank = torch.arange(group.numel(), device=group.device) - starts[group[ordered]]
    return ordered[rank < cap]


@torch.no_grad()
def sample_queries(batch, ratio, training):
    if not 0 < ratio <= 1:
        raise ValueError("query_ratio must be in (0, 1]")
    selected = []
    for cloud in batch.unique(sorted=True):
        ids = torch.where(batch == cloud)[0]
        count = max(1, math.ceil(ids.numel() * ratio))
        if training:
            take = torch.randperm(ids.numel(), device=batch.device)[:count]
        else:
            take = torch.div(
                torch.arange(count, device=batch.device) * ids.numel(),
                count,
                rounding_mode="floor",
            )
        selected.append(ids[take])
    return torch.cat(selected)


@torch.no_grad()
def radius_edges(
    coord,
    batch,
    query,
    radius,
    max_neighbors=16,
    training=False,
    query_chunk_size=128,
    max_candidate_edges=262144,
):
    """Return (query-row, support-index) edges, with a per-query neighbor cap.

    Radius-sized hash cells and 27 adjacent cells enumerate every candidate.
    Packed keys are collision-free after range checks. No gradient through
    search. Chunking bounds candidate memory, not total runtime. An oversized
    single-query candidate set raises a descriptive error instead of silently
    approximating radius search. Self is always retained; the remaining slots
    are uniformly sampled from valid other neighbors during training. Eval
    uses fixed cell traversal and support-index order, and is input-order dependent.
    """
    if (
        radius <= 0
        or max_neighbors < 1
        or query_chunk_size < 1
        or max_candidate_edges < 1
    ):
        raise ValueError(
            "radius, max_neighbors, chunk size and edge budget must be positive"
        )
    if query.numel() == 0:
        return query, query
    xyz = coord.detach().float()
    cell = torch.floor((xyz - xyz.amin(0)) / radius).long() + 1
    width = cell.amax(0) + 2
    wx, wy, wz = (int(v) for v in width)
    batches = int(batch.max()) + 1
    if batches * wx * wy * wz >= torch.iinfo(torch.int64).max:
        raise ValueError(
            "Radius hash range exceeds int64; center coordinates or increase radius"
        )

    def encode(c, b):
        return ((b * wx + c[..., 0]) * wy + c[..., 1]) * wz + c[..., 2]

    keys = encode(cell, batch.long())
    sorted_keys, support_order = keys.sort(stable=True)
    shifts = torch.tensor(
        list(itertools.product((-1, 0, 1), repeat=3)),
        device=coord.device,
        dtype=torch.long,
    )
    all_rows, all_neighbors = [], []
    start = 0
    while start < query.numel():
        stop = min(query.numel(), start + query_chunk_size)
        while True:
            q = query[start:stop]
            lookup = encode(cell[q, None, :] + shifts, batch[q, None].long()).flatten()
            lo = torch.searchsorted(sorted_keys, lookup)
            hi = torch.searchsorted(sorted_keys, lookup, right=True)
            lengths = hi - lo
            total = int(lengths.sum())
            if total <= max_candidate_edges:
                break
            if stop - start == 1:
                raise RuntimeError(
                    "One radius query exceeds max_candidate_edges. Reduce "
                    "radius or increase radius_max_candidate_edges."
                )
            stop = start + max(1, (stop - start) // 2)
        bins = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=coord.device), lengths
        )
        starts = lengths.cumsum(0) - lengths
        offsets = torch.arange(total, device=coord.device) - torch.repeat_interleave(
            starts, lengths
        )
        neighbors = support_order[lo[bins] + offsets]
        rows = torch.div(bins, 27, rounding_mode="floor")
        distance2 = (xyz[neighbors] - xyz[q[rows]]).square().sum(-1)
        keep = (distance2 <= radius * radius) & (neighbors != q[rows])
        rows, neighbors = rows[keep], neighbors[keep]
        if max_neighbors > 1:
            take = capped_members(rows, q.numel(), max_neighbors - 1, training)
            all_rows.append(rows[take] + start)
            all_neighbors.append(neighbors[take])
        all_rows.append(torch.arange(start, stop, device=coord.device))
        all_neighbors.append(q)
        start = stop
    return torch.cat(all_rows), torch.cat(all_neighbors)


def rotary_qkv(qkv, coord, scale):
    """Axial 3D RoPE on Q/K; V unchanged. Any even head dimension is supported.

    Pairs cycle through x/y/z. This avoids head_dim % 6 restrictions. These
    commuting planar rotations preserve relative translations algebraically.
    This is the attention baseline, not quaternion or polar pairwise attention.
    """
    dimension = qkv.shape[-1]
    if dimension % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    pairs = dimension // 2
    index = torch.arange(pairs, device=qkv.device)
    frequency = 10000.0 ** (-(index // 3).float() / max(1, math.ceil(pairs / 3)))
    phase = coord.float()[:, index % 3] / scale * frequency
    cos, sin = phase.cos()[:, None, :], phase.sin()[:, None, :]

    def rotate(x):
        pair = x.float().reshape(x.shape[0], x.shape[1], pairs, 2)
        a, b = pair.unbind(-1)
        return (
            torch.stack((a * cos - b * sin, a * sin + b * cos), -1)
            .flatten(-2)
            .to(x.dtype)
        )

    return torch.stack((rotate(qkv[:, 0]), rotate(qkv[:, 1]), qkv[:, 2]), dim=1)
