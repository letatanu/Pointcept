"""3x3x3 submanifold convolution on occupied voxels, then point gathering.

spconv is the training backend. reference is an exact, slow PyTorch operator
for CPU checks and tiny inputs, not a dense convolution or an attention proxy.
"""
from itertools import product
import math

import torch
from torch import nn

from .geometry import group_mean


@torch.no_grad()
def reference_pairs(layout):
    if layout.neighbor_pairs is not None:
        return layout.neighbor_pairs
    shape = layout.spatial_shape
    if math.prod(shape) * layout.batch_size >= 2**63:
        raise ValueError("Reference sparse-coordinate key exceeds int64")
    indices = layout.indices

    def encode(coords):
        return ((coords[:, 0] * shape[0] + coords[:, 1]) * shape[1] + coords[:, 2]) * shape[2] + coords[:, 3]

    keys = encode(indices)
    sorted_keys, permutation = keys.sort()
    result = []
    for delta in product((-1, 0, 1), repeat=3):
        target = indices.clone()
        target[:, 1:] += indices.new_tensor(delta)
        valid = ((target[:, 1:] >= 0) & (target[:, 1:] < indices.new_tensor(shape))).all(-1)
        destination = valid.nonzero(as_tuple=True)[0]
        wanted = encode(target[valid])
        location = torch.searchsorted(sorted_keys, wanted).clamp_max(len(keys) - 1)
        hit = sorted_keys[location] == wanted
        result.append((destination[hit], permutation[location[hit]]))
    layout.neighbor_pairs = result
    return result


class ReferenceSubMConv3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(27, channels, channels))  # offset, input, output
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.uniform_(self.weight, -1 / math.sqrt(27 * channels), 1 / math.sqrt(27 * channels))

    def forward(self, feature, layout):
        dtype = feature.dtype
        output = torch.zeros(feature.shape, device=feature.device, dtype=torch.float32)
        for weight, (dst, src) in zip(self.weight, reference_pairs(layout)):
            # Empty edge sets still participate in autograd, as in a true kernel.
            message = feature[src] @ weight.to(dtype)
            output.index_add_(0, dst, message.float())
        return (output + self.bias.float()).to(dtype)


class SparseContext(nn.Module):
    def __init__(self, channels, backend="spconv", enabled=True):
        super().__init__()
        self.enabled = enabled
        self.backend = backend
        if not enabled:
            return
        if backend == "spconv":
            try:
                import spconv.pytorch as spconv
            except ImportError as exc:
                raise ImportError("Install spconv for your CUDA/PyTorch environment, or use sparse_backend='reference' for tiny checks") from exc
            self.conv = spconv.SubMConv3d(channels, channels, 3, padding=1,
                                         bias=True, indice_key="sq_subm3")
        elif backend == "reference":
            self.conv = ReferenceSubMConv3d(channels)
        else:
            raise ValueError("sparse_backend must be spconv or reference")
        self.project = nn.Sequential(nn.Linear(channels, channels), nn.LayerNorm(channels))
        self.scale = nn.Parameter(torch.full((channels,), .1))

    def forward(self, feature, layout):
        if not self.enabled:
            return feature
        voxel, _ = group_mean(feature, layout.inverse, len(layout.indices))
        if self.backend == "spconv":
            import spconv.pytorch as spconv
            if not feature.is_cuda:
                raise ValueError("Use CUDA tensors for the spconv backend; reference supports CPU")
            # Keep coordinate order x,y,z consistent with spatial_shape.
            sparse = spconv.SparseConvTensor(voxel.contiguous(), layout.indices.int().contiguous(),
                                            list(layout.spatial_shape), layout.batch_size)
            sparse.indice_dict = layout.indice_dict
            convolved = self.conv(sparse)
            layout.indice_dict = convolved.indice_dict
            value = convolved.features
        else:
            value = self.conv(voxel, layout)
        context = self.project(value)
        return feature + self.scale * context[layout.inverse]
