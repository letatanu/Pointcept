"""Pointcept backbone: random point sampling, learned voxel pooling, serialized SDPA.
Requires PyTorch >=2.1. No spconv, torch_scatter, pointops, or custom CUDA ops.
Serialization uses batch-safe lexicographic axis orders, not PTv3 Morton/Hilbert.
"""

import itertools
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point


def mean_pool(x, parent, size):
    dtype = x.dtype
    out = torch.zeros(size, x.shape[1], device=x.device, dtype=torch.float32)
    out.index_add_(0, parent, x.float())
    count = torch.bincount(parent, minlength=size).clamp_min(1)
    return (out / count[:, None]).to(dtype)


def max_pool(x, parent, size):
    out = x.new_full((size, x.shape[1]), -torch.inf)
    out.scatter_reduce_(
        0, parent[:, None].expand_as(x), x, reduce="amax", include_self=True
    )
    return out


def mlp(cin, cout):
    return nn.Sequential(
        nn.Linear(cin, cout), nn.LayerNorm(cout), nn.GELU(), nn.Linear(cout, cout)
    )


@torch.no_grad()
def partition(keys):
    return torch.unique(keys, dim=0, sorted=True, return_inverse=True)


@torch.no_grad()
def serial_order(keys, axes):
    # Stable sorts avoid overflow-prone packed coordinate keys; batch is primary.
    order = torch.arange(len(keys), device=keys.device)
    for col in reversed((0,) + tuple(a + 1 for a in axes)):
        order = order[torch.argsort(keys[order, col], stable=True)]
    return order


@torch.no_grad()
def stencil(keys):
    # Collision-free occupied-neighbor lookup. Built once per decoder resolution.
    low = keys[:, 1:].amin(0)
    xyz = keys[:, 1:] - low + 1
    widths = xyz.amax(0) + 2
    wx, wy, wz = [int(t) for t in widths]
    if (int(keys[:, 0].max()) + 1) * wx * wy * wz >= 2**62:
        raise ValueError("Coordinate range too large for voxel neighbor lookup")

    def encode(batch, p):
        return ((batch * wx + p[:, 0]) * wy + p[:, 1]) * wz + p[:, 2]

    code, order = encode(keys[:, 0], xyz).sort()
    ids = torch.arange(len(keys), device=keys.device)
    dst, src = [], []
    for shift in itertools.product((-1, 0, 1), repeat=3):
        candidate = xyz + xyz.new_tensor(shift)
        target = encode(keys[:, 0], candidate)
        pos = torch.searchsorted(code, target).clamp_max(len(code) - 1)
        found = order[pos]
        valid = (code[pos] == target) & (keys[found, 0] == keys[:, 0])
        valid &= (xyz[found] == candidate).all(-1)
        dst.append(ids[valid])
        src.append(found[valid])
    return torch.cat(dst), torch.cat(src)


class SerializedBlock(nn.Module):
    def __init__(self, width, heads, patch_size, ffn_ratio):
        super().__init__()
        self.heads, self.patch_size = heads, patch_size
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)
        )

    def forward(self, x, order, batch_counts, shifted):
        # Patches never cross cloud boundaries. Full patches use batched SDPA;
        # the final partial patch is processed separately without padding keys.
        sorted_x = self.norm1(x)[order]
        outputs, begin = [], 0
        p = self.patch_size
        for n in batch_counts:
            cloud = sorted_x[begin : begin + n]
            shift = min(p // 2, n // 2) if shifted else 0
            cloud = cloud.roll(-shift, 0)
            pad = (-n) % p
            if pad:
                cloud = torch.cat([cloud, cloud[-1:].expand(pad, -1)], 0)
            qkv = self.qkv(cloud).reshape(
                -1, p, 3, self.heads, x.shape[1] // self.heads
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            # Mask repeat-padding keys for the last patch only. Split the tail so
            # full patches keep the fused unmasked attention path.
            if pad:
                chunks = []
                if q.shape[0] > 1:
                    chunks.append(
                        F.scaled_dot_product_attention(q[:-1], k[:-1], v[:-1])
                    )
                valid = p - pad
                tail = F.scaled_dot_product_attention(
                    q[-1:, :, :valid], k[-1:, :, :valid], v[-1:, :, :valid]
                )
                # Concatenate valid tokens after flattening (tail has shorter length).
                flat = [a.transpose(1, 2).reshape(-1, x.shape[1]) for a in chunks]
                flat.append(tail.transpose(1, 2).reshape(valid, x.shape[1]))
                y = torch.cat(flat, 0)
            else:
                y = (
                    F.scaled_dot_product_attention(q, k, v)
                    .transpose(1, 2)
                    .reshape(n, x.shape[1])
                )
            outputs.append(y.roll(shift, 0))
            begin += n
        y = torch.cat(outputs, 0)
        restored = torch.empty_like(y).index_copy(0, order, y)
        x = x + self.proj(restored)
        return x + self.ffn(self.norm2(x))


class NeighborFusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Position is encoded per edge; feature projection is once per voxel.
        self.feature = nn.Linear(width, width, bias=False)
        self.position = mlp(3, width)
        self.fuse = mlp(2 * width, width)

    def forward(self, x, keys, edges):
        dst, src = edges
        relative = keys[src, 1:] - keys[dst, 1:]
        code = (relative[:, 0] + 1) * 9 + (relative[:, 1] + 1) * 3 + relative[:, 2] + 1
        offsets = x.new_tensor(list(itertools.product((-1, 0, 1), repeat=3)))
        # Encode just 27 offsets, not a separate MLP evaluation for every edge.
        message = self.feature(x)[src] + self.position(offsets)[code]
        context = mean_pool(message, dst, len(x))
        return x + self.fuse(torch.cat([x, context], -1))


@MODELS.register_module("SerializedVoxelLite")
class SerializedVoxelLite(nn.Module):
    def __init__(
        self,
        in_channels=4,
        channels=(32, 64, 96, 128),
        depths=(1, 1, 2, 2),
        num_heads=(2, 4, 4, 4),
        voxel_size=0.3,
        sample_ratio=0.5,
        patch_size=64,
        point_width=16,
        out_channels=64,
        ffn_ratio=2.0,
        neighbor_context=False,
        checkpoint_blocks=True,
    ):
        super().__init__()
        if not (len(channels) == len(depths) == len(num_heads)) or not channels:
            raise ValueError(
                "channels/depths/num_heads must have matching nonzero lengths"
            )
        if not 0 < sample_ratio <= 1 or voxel_size <= 0 or patch_size < 1:
            raise ValueError("Invalid sample ratio, voxel size, or patch size")
        if any(c % h or h < 1 or d < 1 for c, h, d in zip(channels, num_heads, depths)):
            raise ValueError("Invalid widths, heads, or depths")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.size, self.ratio = voxel_size, sample_ratio
        self.checkpoint_blocks = checkpoint_blocks
        self.point_embed = mlp(in_channels, point_width)
        self.local = mlp(point_width + 4, channels[0])
        self.pool_proj = nn.Linear(2 * channels[0], channels[0])
        self.down = nn.ModuleList(
            [mlp(2 * channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [SerializedBlock(c, h, patch_size, ffn_ratio) for _ in range(d)]
                )
                for c, h, d in zip(channels, num_heads, depths)
            ]
        )
        self.position = nn.ModuleList([mlp(3, c) for c in channels])
        self.up = nn.ModuleList(
            [
                mlp(channels[i] + channels[i + 1] + 3, channels[i])
                for i in range(len(channels) - 1)
            ]
        )
        self.neighbors = nn.ModuleList(
            [
                NeighborFusion(c) if neighbor_context else nn.Identity()
                for c in channels[:-1]
            ]
        )
        self.neighbor_context = neighbor_context
        self.restore = mlp(channels[0] + point_width + 4, out_channels)

    def forward(self, data_dict):
        coord, feat, offset = data_dict["coord"], data_dict["feat"], data_dict["offset"]
        if len(coord) == 0 or feat.shape != (len(coord), self.in_channels):
            raise ValueError(
                "Expected nonempty coordinates and [N,in_channels] features"
            )
        counts = torch.diff(offset, prepend=offset.new_zeros(1)).long()
        if int(offset[-1]) != len(coord) or bool((counts <= 0).any()):
            raise ValueError("Invalid cloud offsets")
        batch = torch.repeat_interleave(
            torch.arange(len(offset), device=coord.device), counts
        )
        with torch.no_grad():
            cells = torch.floor(coord.float() / self.size).long()
            keys, parent = partition(torch.cat([batch[:, None], cells], 1))
            center = mean_pool(coord.float(), parent, len(keys))
            relative = (coord.float() - center[parent]) / self.size
            geom = torch.cat([relative, relative.norm(dim=-1, keepdim=True)], -1)
            # Global Bernoulli sampling plus one random representative per occupied
            # voxel. Coverage repair prevents missing decoder support for originals.
            if self.training:
                keep = torch.rand(len(coord), device=coord.device) < self.ratio
                priority = torch.rand(len(coord), device=coord.device)
                minimum = priority.new_full((len(keys),), torch.inf)
                minimum.scatter_reduce_(
                    0, parent, priority, reduce="amin", include_self=True
                )
                keep |= priority == minimum[parent]
            else:
                # Deterministic approximate-ratio sampling + first point coverage.
                ids = torch.arange(len(coord), device=coord.device)
                keep = ((ids + 1) * self.ratio).floor() > (ids * self.ratio).floor()
                first = ids.new_full((len(keys),), len(coord))
                first.scatter_reduce_(0, parent, ids, reduce="amin", include_self=True)
                keep[first] = True
            selected = keep.nonzero(as_tuple=True)[0]
        point = self.point_embed(feat)
        local = self.local(
            torch.cat([point[selected], geom[selected].to(point.dtype)], -1)
        )
        group = parent[selected]
        x = self.pool_proj(
            torch.cat(
                [mean_pool(local, group, len(keys)), max_pool(local, group, len(keys))],
                -1,
            )
        )
        saved = []
        axes = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        for level, blocks in enumerate(self.blocks):
            # Bounded local position; no large absolute-coordinate positional MLP.
            local_center = center / (self.size * 2**level) - keys[:, 1:].float() - 0.5
            x = x + self.position[level](local_center.to(x.dtype))
            orders = {}
            cloud_counts = torch.bincount(keys[:, 0], minlength=len(offset)).tolist()
            for j, block in enumerate(blocks):
                axis = axes[(level + j) % len(axes)]
                if axis not in orders:
                    orders[axis] = serial_order(keys, axis)
                args = (x, orders[axis], cloud_counts, bool(j % 2))
                x = (
                    checkpoint(block, *args, use_reentrant=False)
                    if self.training and self.checkpoint_blocks
                    else block(*args)
                )
            if level < len(self.blocks) - 1:
                coarse_input = torch.cat(
                    [keys[:, :1], torch.div(keys[:, 1:], 2, rounding_mode="floor")], 1
                )
                coarse_keys, link = partition(coarse_input)
                edges = stencil(keys) if self.neighbor_context else None
                saved.append((x, keys, center, link, edges))
                x = self.down[level](
                    torch.cat(
                        [
                            mean_pool(x, link, len(coarse_keys)),
                            max_pool(x, link, len(coarse_keys)),
                        ],
                        -1,
                    )
                )
                center = mean_pool(center, link, len(coarse_keys))
                keys = coarse_keys
        for level in reversed(range(len(saved))):
            skip, fine_keys, fine_center, link, edges = saved[level]
            relative_center = (fine_center - center[link]) / (
                self.size * 2 ** (level + 1)
            )
            x = skip + self.up[level](
                torch.cat([skip, x[link], relative_center.to(x.dtype)], -1)
            )
            if self.neighbor_context:
                x = self.neighbors[level](x, fine_keys, edges)
            keys, center = fine_keys, fine_center
        result = self.restore(torch.cat([x[parent], point, geom.to(point.dtype)], -1))
        output = Point(dict(data_dict))
        output.feat = result
        return output
