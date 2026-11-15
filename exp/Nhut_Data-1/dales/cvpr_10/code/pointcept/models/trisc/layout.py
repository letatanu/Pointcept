"""Disjoint, cloud-safe dilated patch groups and stratified sampling."""
from dataclasses import dataclass
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
    quota = torch.minimum(counts, torch.as_tensor(budget, device=counts.device)).clamp_min(1)
    slots = torch.arange(width, device=counts.device)[None]
    low = slots * counts[:, None] // quota[:, None]
    high = (slots + 1) * counts[:, None] // quota[:, None]
    fraction = torch.rand(low.shape, device=counts.device) if training else .5
    positions = low + torch.floor((high - low) * fraction).long()
    positions = torch.minimum(positions, counts[:, None] - 1)
    selected = members.gather(1, positions)
    return selected.masked_fill(slots >= quota[:, None], -1)


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
