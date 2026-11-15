"""Batch-safe Morton serialization, occupied voxels, and bounded patches."""

from dataclasses import dataclass, field
import math

import torch
from .serialization import serialized_order


def group_mean(value, group, size, weight=None):
    """FP32 accumulation, preserving gradients to all member features."""
    dtype = value.dtype
    weight = (
        value.new_ones(len(value), dtype=torch.float32)
        if weight is None
        else weight.float()
    )
    shape = (len(value),) + (1,) * (value.ndim - 1)
    total = torch.zeros(
        (size,) + value.shape[1:], device=value.device, dtype=torch.float32
    )
    total.index_add_(0, group, value.float() * weight.reshape(shape))
    mass = torch.zeros(size, device=value.device, dtype=torch.float32)
    mass.index_add_(0, group, weight)
    return (total / mass.clamp_min(1).reshape((size,) + (1,) * (value.ndim - 1))).to(
        dtype
    ), mass


@torch.no_grad()
def batch_from_data(data, n, device):
    if "offset" in data:
        offset = data["offset"].to(device=device, dtype=torch.long)
        if offset.ndim != 1 or len(offset) == 0:
            raise ValueError("offset must be a nonempty vector")
        counts = torch.diff(offset, prepend=offset.new_zeros(1))
        if int(offset[-1]) != n or bool((counts <= 0).any()):
            raise ValueError(
                "offset must describe contiguous nonempty clouds and end at N"
            )
        return torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
    if "batch" in data:
        batch = data["batch"].to(device=device, dtype=torch.long)
        if (
            batch.shape != (n,)
            or int(batch[0]) != 0
            or bool((batch[1:] < batch[:-1]).any())
        ):
            raise ValueError("batch must be sorted, start at zero, and have shape (N,)")
        if bool((torch.bincount(batch) == 0).any()):
            raise ValueError("batch IDs must be consecutive")
        return batch
    return torch.zeros(n, device=device, dtype=torch.long)


@torch.no_grad()
def centered_coordinates(coord, batch):
    origin = coord.new_full((int(batch.max()) + 1, 3), torch.inf)
    origin.scatter_reduce_(
        0, batch[:, None].expand(-1, 3), coord, reduce="amin", include_self=True
    )
    return coord - origin[batch]


def _spread(x):
    x = x & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    return (x | x << 2) & 0x1249249249249249


@torch.no_grad()
def morton_order(local_coord, batch, resolution, axes):
    integer = torch.floor(local_coord / resolution).long()
    if bool(((integer < 0) | (integer >= 2**21)).any()):
        raise ValueError("Morton key overflow; increase serialization_resolution")
    x, y, z = (integer[:, a] for a in axes)
    code = _spread(x) | _spread(y) << 1 | _spread(z) << 2
    order = code.argsort(stable=True)
    return order[batch[order].argsort(stable=True)]


@dataclass
class Voxels:
    indices: torch.Tensor  # unique [batch, x, y, z], sorted lexicographically
    inverse: torch.Tensor  # point -> occupied voxel
    counts: torch.Tensor
    spatial_shape: tuple
    batch_size: int
    neighbor_pairs: object = None
    indice_dict: dict = field(default_factory=dict)


@torch.no_grad()
def make_voxels(local_coord, batch, voxel_size):
    integer = torch.floor(local_coord / voxel_size).long()
    indices, inverse, counts = torch.unique(
        torch.cat((batch[:, None], integer), dim=1),
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    spatial_shape = tuple(max(3, int(v) + 1) for v in indices[:, 1:].amax(0))
    if max(spatial_shape) >= 2**31:
        raise ValueError("Sparse grid exceeds int32 coordinate range")
    return Voxels(indices, inverse, counts, spatial_shape, int(batch.max()) + 1)


@dataclass
class Patches:
    members: torch.Tensor  # [number of patches, P], -1 denotes padding
    valid: torch.Tensor
    counts: torch.Tensor


@torch.no_grad()
def make_patches(local_coord, batch, resolution, axes, patch_size, backend="reference"):
    """No cloud crossing, wrapped tails, or duplicate padding points."""
    order = (
        serialized_order(local_coord, batch, resolution, axes, backend)
        if isinstance(axes, str)
        else morton_order(local_coord, batch, resolution, axes)
    )
    cloud_counts = torch.bincount(batch)
    counts = torch.div(cloud_counts + patch_size - 1, patch_size, rounding_mode="floor")
    cloud_start = cloud_counts.cumsum(0) - cloud_counts
    patch_start = counts.cumsum(0) - counts
    sorted_batch = batch[order]
    rank = torch.arange(len(batch), device=batch.device) - cloud_start[sorted_batch]
    patch_id = patch_start[sorted_batch] + rank // patch_size
    members = torch.full(
        (int(counts.sum()), patch_size), -1, device=batch.device, dtype=torch.long
    )
    members[patch_id, rank % patch_size] = order
    valid = members >= 0
    return Patches(members, valid, valid.sum(-1))


@torch.no_grad()
def select_queries(patches, ratio, training):
    """One selected query per occupied contiguous stratum, with bounded width.

    Stratification spreads queries across the patch. Training randomly chooses
    within each stratum; evaluation takes its midpoint. ratio=1 selects all.
    """
    width = patches.members.shape[1]
    slots = max(1, math.ceil(width * ratio))
    quota = torch.ceil(patches.counts.float() * ratio).long().clamp_min(1)
    slot = torch.arange(slots, device=patches.members.device)[None].expand(
        len(quota), -1
    )
    low = torch.div(
        slot * patches.counts[:, None], quota[:, None], rounding_mode="floor"
    )
    high = torch.div(
        (slot + 1) * patches.counts[:, None], quota[:, None], rounding_mode="floor"
    )
    fraction = (
        torch.rand(low.shape, device=low.device)
        if training
        else torch.full_like(low, 0.5, dtype=torch.float32)
    )
    positions = low + torch.floor(fraction * (high - low)).long()
    valid = slot < quota[:, None]
    query = patches.members.gather(1, positions.clamp(0, width - 1))
    return query, valid
