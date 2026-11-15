from itertools import product
import math
import torch

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
    fraction = torch.rand(low.shape, device=counts.device) if training else .5
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
        return ((index[..., 0] * shape[0] + index[..., 1]) * shape[1] + index[..., 2]) * shape[2] + index[..., 3]
    index = layout.indices
    offsets = index.new_tensor([(0, *delta) for delta in product((-1, 0, 1), repeat=3)])
    wanted = index[:, None, :] + offsets[None]
    valid = ((wanted[..., 1:] >= 0) & (wanted[..., 1:] < index.new_tensor(shape))).all(-1)
    keys, permutation = encode(index).sort()
    query = encode(wanted)
    location = torch.searchsorted(keys, query.contiguous()).clamp_max(len(keys) - 1)
    valid = valid & (keys[location] == query)
    return permutation[location].masked_fill(~valid, -1)


