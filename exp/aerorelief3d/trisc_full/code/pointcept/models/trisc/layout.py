"""Cloud-safe patch and voxel neighborhoods with stratified sampling."""
from dataclasses import dataclass
from math import prod
import torch


@dataclass
class CrossLayout:
    queries: int
    keys: int
    group: torch.Tensor
    slot: torch.Tensor
    groups: int


@torch.no_grad()
def sample_rows(members, budget, width, training):
    """One random sample per index stratum; midpoint selection in eval."""
    counts = (members >= 0).sum(-1)
    positions, valid = sample_positions(counts, budget, width, training)
    return members.gather(1, positions).masked_fill(~valid, -1)


def sample_positions(counts, budget, width, training):
    quota = torch.minimum(counts, torch.as_tensor(budget, device=counts.device)).clamp_min(1)
    slots = torch.arange(width, device=counts.device)[None]
    low = slots * counts[:, None] // quota[:, None]
    high = (slots + 1) * counts[:, None] // quota[:, None]
    fraction = torch.rand(low.shape, device=counts.device) if training else .5
    positions = low + torch.floor((high - low) * fraction).long()
    positions = torch.minimum(positions, counts[:, None] - 1)
    return positions, slots < quota[:, None]


@dataclass
class GridCrossLayout:
    cells: torch.Tensor
    inverse: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    members: torch.Tensor
    upper: torch.Tensor
    strides: torch.Tensor
    codes: torch.Tensor
    lookup: torch.Tensor
    max_occupancy: int


@torch.no_grad()
def make_grid_cross_layout(grid, batch):
    cells, inverse, counts = torch.unique(
        torch.cat((batch[:, None], grid), dim=1), dim=0, sorted=True,
        return_inverse=True, return_counts=True)
    sx, sy, sz = (int(v) + 1 for v in cells[:, 1:].amax(dim=0).tolist())
    if (int(batch[-1]) + 1) * prod((sx, sy, sz)) >= 2**63:
        raise ValueError("Grid cross-attention cell codes exceed int64")
    strides = cells.new_tensor((sx * sy * sz, sy * sz, sz, 1))
    codes, lookup = (cells * strides).sum(dim=1).sort()
    return GridCrossLayout(cells, inverse, counts, counts.cumsum(0) - counts,
                           inverse.argsort(stable=True), cells.new_tensor((sx - 1, sy - 1, sz - 1)),
                           strides, codes, lookup, int(counts.max()))


def sample_grid_rows(layout, budget, training):
    positions, valid = sample_positions(layout.counts, budget, budget, training)
    return layout.members[layout.starts[:, None] + positions].masked_fill(~valid, -1)


def compact_samples(samples):
    valid = samples >= 0
    slots = (valid.flatten().cumsum(0) - 1).view_as(samples).masked_fill(~valid, -1)
    return samples[valid], slots


def grid_cross_tables(layout, window_size, dilation, q, source_keys, chunk_size):
    """Yield windows of sampled token indices in bounded chunks."""
    radius = window_size // 2
    axis = torch.arange(-radius, radius + 1, device=q.device)
    offsets = torch.cartesian_prod(axis, axis, axis)
    offsets = offsets[(offsets != 0).any(dim=1)] * dilation
    for begin in range(0, len(layout.cells), chunk_size):
        cells = layout.cells[begin:begin + chunk_size]
        neighbors = cells[:, None, 1:] + offsets
        in_bounds = ((neighbors >= 0) & (neighbors <= layout.upper)).all(dim=-1)
        neighbors = torch.minimum(neighbors.clamp_min(0), layout.upper)
        codes = (cells[:, :1] * layout.strides[0]
                 + (neighbors * layout.strides[1:]).sum(dim=-1))
        index = torch.searchsorted(layout.codes, codes).clamp_max(len(layout.cells) - 1)
        valid = in_bounds & (layout.codes[index] == codes)
        active = valid.any(dim=1).nonzero().flatten()
        if active.numel():
            neighbors = layout.lookup[index.index_select(0, active)]
            k = source_keys[neighbors].masked_fill(~valid.index_select(0, active)[..., None], -1)
            yield q[begin + active], k.flatten(1)


@torch.no_grad()
def cross_layout(patches, batch, group_size, dilation, queries, keys):
    patch_batch = batch[patches.members[:, 0]]
    counts = torch.bincount(patch_batch)
    starts = counts.cumsum(0) - counts
    rank = torch.arange(len(patch_batch), device=batch.device) - starts[patch_batch]
    # For G=2,D=2: groups [0,2], [1,3], [4,6], [5,7].
    code = rank // (group_size * dilation) * dilation + rank % dilation
    unique, group = torch.unique(torch.stack((patch_batch, code), -1), dim=0, return_inverse=True)
    slot = rank % (group_size * dilation) // dilation
    return CrossLayout(queries, keys, group, slot, len(unique))


@torch.no_grad()
def sample_cross(patches, layout, group_size, training):
    m, k = layout.queries, layout.keys
    width_q = (m + group_size - 1) // group_size
    budgets = m // group_size + (layout.slot < m % group_size).long()
    per_patch_q = sample_rows(patches.members, budgets, width_q, training)
    per_patch_k = sample_rows(patches.members, k, k, training)
    q = per_patch_q.new_full((layout.groups, group_size * width_q), -1)
    key = per_patch_k.new_full((layout.groups, group_size * k), -1)
    qcols = layout.slot[:, None] * width_q + torch.arange(width_q, device=q.device)[None]
    kcols = layout.slot[:, None] * k + torch.arange(k, device=q.device)[None]
    q[layout.group[:, None], qcols] = per_patch_q
    key[layout.group[:, None], kcols] = per_patch_k
    return q, key


def packed_table(members):
    valid = members >= 0
    counts = valid.sum(-1)
    cumulative = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
    return members[valid], cumulative


@torch.no_grad()
def sample_central_cross(patches, batch, window_size, dilation, queries, keys, training,
                         include_center=True):
    """Each patch owns Q; sliding, dilated surrounding patches supply K/V.

    Query IDs are unique across rows. Key IDs may be reused across windows,
    but are unique inside each row. No wrapped padding or cloud crossing.
    Budgets are Q per CENTRAL PATCH and K per SOURCE PATCH (unlike grouped).
    """
    if window_size < 3 or window_size % 2 != 1:
        raise ValueError("central cross_window_size must be odd and >=3")
    q = sample_rows(patches.members, queries, queries, training)
    per_patch_k = sample_rows(patches.members, keys, keys, training)
    patch_batch = batch[patches.members[:, 0]]
    centers = torch.arange(len(patch_batch), device=batch.device)
    offsets = torch.arange(-(window_size // 2), window_size // 2 + 1,
                           device=batch.device) * dilation
    if not include_center:
        offsets = torch.cat((offsets[:window_size // 2], offsets[window_size // 2 + 1:]))
    neighbors = centers[:, None] + offsets[None]
    bounded = neighbors.clamp(0, len(patch_batch) - 1)
    valid = ((neighbors >= 0) & (neighbors < len(patch_batch)) &
             (patch_batch[bounded] == patch_batch[:, None]))
    k = per_patch_k[bounded].masked_fill(~valid[..., None], -1)
    if not include_center:
        # Packed attention requires at least one valid key per query row.
        # Isolated patches receive no cross update.
        active = valid.any(dim=1)
        q, k = q[active], k[active]
    return q, k.flatten(1)
