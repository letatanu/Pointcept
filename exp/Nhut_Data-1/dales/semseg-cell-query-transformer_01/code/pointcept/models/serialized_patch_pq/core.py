"""Independent serialized sampled-point encoder/decoder. PyTorch >= 2.1.

No PTv3/Pointcept imports, sparse convolution, radius search, geometric KNN,
torch_scatter, FlashAttention package, or dense all-pairs distances.
Every input point survives to the output. Sorting, grouping, sampled sequence
neighbors, polar/quaternion messages, pooling and gated skips are implemented
here. Sequence neighbors are not Euclidean nearest neighbors.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def sum_by_group(value, group, size):
    """Accumulate in FP32 under AMP to reduce scatter overflow/roundoff."""
    result = torch.zeros((size,) + value.shape[1:], device=value.device, dtype=torch.float32)
    result.index_add_(0, group, value.float())
    return result.to(value.dtype)


def softmax_by_group(score, group, size):
    value = score.float()
    maximum = value.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, group, value.detach(), reduce="amax", include_self=True)
    exp = (value - maximum[group]).exp()
    total = sum_by_group(exp, group, size)
    return exp / total[group].clamp_min(1e-12)


def point_mlp(cin, cout, hidden=None):
    return nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, hidden or cout), nn.GELU(),
                         nn.Linear(hidden or cout, cout))


@torch.no_grad()
def cloud_batch(offset, n, device):
    offset = offset.to(device=device, dtype=torch.long)
    count = torch.diff(offset, prepend=offset.new_zeros(1))
    if offset.numel() == 0 or int(offset[-1]) != n or bool((count <= 0).any()):
        raise ValueError("offset must describe contiguous nonempty clouds and end at N")
    batch = torch.repeat_interleave(torch.arange(len(count), device=device), count)
    return batch, count


@torch.no_grad()
def quantize_for_keys(coord, batch, resolution, clouds):
    """Quantization only builds sort/group keys; it never removes points."""
    origin = coord.new_full((clouds, 3), torch.inf)
    origin.scatter_reduce_(0, batch[:, None].expand(-1, 3), coord,
                           reduce="amin", include_self=True)
    return torch.floor((coord - origin[batch]) / resolution).long()


def _spread_bits21(x):
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    return (x | (x << 2)) & 0x1249249249249249


@torch.no_grad()
def morton_permutation(coord, batch, resolution, axes=(0, 1, 2)):
    clouds = int(batch.max()) + 1
    integer = quantize_for_keys(coord, batch, resolution, clouds)
    if bool(((integer < 0) | (integer >= 2**21)).any()):
        raise ValueError("Invalid/Morton-overflow coordinate; check finite XYZ or increase serialization_resolution")
    x, y, z = (integer[:, axis] for axis in axes)
    key = _spread_bits21(x) | (_spread_bits21(y) << 1) | (_spread_bits21(z) << 2)
    order = torch.argsort(key, stable=True)
    # Separate stable sort keeps batch primary without packing batch into int64.
    order = order[torch.argsort(batch[order], stable=True)]
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order), device=order.device)
    return order, inverse


@dataclass
class Layout:
    group: torch.Tensor
    counts: torch.Tensor
    patch: torch.Tensor
    batch: torch.Tensor
    contiguous_groups: bool
    maximum_group_size: int


@torch.no_grad()
def build_layout(coord, batch, cloud_counts, patch_size, stride, grouping, cell_size):
    start = cloud_counts.cumsum(0) - cloud_counts
    rank = torch.arange(len(coord), device=coord.device) - start[batch]
    patch_counts = torch.div(cloud_counts + patch_size - 1, patch_size, rounding_mode="floor")
    patch_start = patch_counts.cumsum(0) - patch_counts
    patch = patch_start[batch] + torch.div(rank, patch_size, rounding_mode="floor")
    if grouping == "patch":
        group_counts = torch.div(cloud_counts + stride - 1, stride, rounding_mode="floor")
        group_start = group_counts.cumsum(0) - group_counts
        group = group_start[batch] + torch.div(rank, stride, rounding_mode="floor")
        counts = torch.bincount(group, minlength=int(group_counts.sum()))
        return Layout(group, counts, patch, batch, True, stride)
    if grouping != "grid":
        raise ValueError("grouping must be patch or grid")
    integer = quantize_for_keys(coord, batch, cell_size, len(cloud_counts))
    key = torch.cat((batch[:, None], integer), -1)
    _, group, counts = torch.unique(key, dim=0, sorted=True, return_inverse=True, return_counts=True)
    return Layout(group, counts, patch, batch, False, 0)


@torch.no_grad()
def choose_enhanced(layout, samples, training):
    """Sample within every pooling subgroup AFTER serialization/grouping.

    Patch path uses a small padded matrix and topk (no random global sort).
    Grid path uses a segmented ordering to avoid padding very dense cells.
    At least one point per group is selected, including incomplete tails.
    """
    counts = layout.counts
    starts = counts.cumsum(0) - counts
    if layout.contiguous_groups:
        width = layout.maximum_group_size
        members = starts[:, None] + torch.arange(width, device=counts.device)
        valid = torch.arange(width, device=counts.device)[None] < counts[:, None]
        cap = min(samples, width)
        if training:
            priority = torch.rand(members.shape, device=counts.device).masked_fill(~valid, -1)
            positions = priority.topk(cap, dim=1).indices
            keep = valid.gather(1, positions)
        else:
            quota = counts.clamp_max(cap)
            slot = torch.arange(cap, device=counts.device)[None].expand(len(counts), -1)
            positions = torch.div(slot * counts[:, None], quota[:, None], rounding_mode="floor")
            keep = slot < quota[:, None]
        return (starts[:, None] + positions)[keep]
    n = len(layout.group)
    if training:
        priority = torch.randperm(n, device=counts.device)
        order = priority[torch.argsort(layout.group[priority], stable=True)]
        rank = torch.arange(n, device=counts.device) - starts[layout.group[order]]
        return order[rank < samples]
    order = torch.argsort(layout.group, stable=True)
    quota = counts.clamp_max(samples)
    slot = torch.arange(samples, device=counts.device)[None].expand(len(counts), -1)
    position = starts[:, None] + torch.div(slot * counts[:, None], quota[:, None], rounding_mode="floor")
    return order[position[slot < quota[:, None]]]


@torch.no_grad()
def sequence_neighbors(coord, layout, selected, k_each_side, max_distance=None):
    """K preceding + self + K following, clipped at patch/cloud boundaries."""
    offsets = torch.arange(-k_each_side, k_each_side + 1, device=coord.device)
    candidate = selected[:, None] + offsets
    valid = (candidate >= 0) & (candidate < len(coord))
    neighbor = candidate.clamp(0, len(coord) - 1)
    valid &= layout.patch[neighbor] == layout.patch[selected, None]
    valid &= layout.batch[neighbor] == layout.batch[selected, None]
    delta = coord[neighbor] - coord[selected, None]
    if max_distance is not None:
        valid &= delta.square().sum(-1) <= max_distance * max_distance
    # Self is present even for isolated points; no all-masked softmax rows.
    valid[:, k_each_side] = True
    return neighbor, valid, delta


def polar_coordinates(delta, mode):
    xyz = delta.float()
    horizontal = torch.linalg.vector_norm(xyz[..., :2], dim=-1)
    radius = torch.linalg.vector_norm(xyz, dim=-1)
    valid = horizontal > 1e-8
    sin_a = torch.where(valid, xyz[..., 1] / horizontal.clamp_min(1e-8), 0.)
    cos_a = torch.where(valid, xyz[..., 0] / horizontal.clamp_min(1e-8), 0.)
    if mode == "cylindrical":
        return torch.stack((horizontal.log1p(), sin_a, cos_a, xyz[..., 2], radius.log1p()), -1)
    if mode == "spherical":
        return torch.stack((radius.log1p(), sin_a, cos_a,
                            xyz[..., 2] / radius.clamp_min(1e-8),
                            horizontal / radius.clamp_min(1e-8)), -1)
    raise ValueError("embedding_mode must be cylindrical or spherical")


def quaternion_left(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), -1)


class PolarQuaternion(nn.Module):
    def __init__(self, width, mode, enabled):
        super().__init__()
        if width % 4:
            raise ValueError("Quaternion message width must be divisible by four")
        self.mode, self.enabled = mode, enabled
        self.position = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.register_buffer("frequency", torch.logspace(-1, .6, width // 4))

    def forward(self, feature, delta):
        delta = delta.float()
        position = self.position(polar_coordinates(delta, self.mode))
        if self.enabled:
            radius = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            direction = delta / radius.clamp_min(1e-8)
            phase = radius * self.frequency.float() * .5
            rotor = torch.cat((phase.cos()[..., None], phase.sin()[..., None] * direction[..., None, :]), -1)
            feature = quaternion_left(rotor, feature.float().unflatten(-1, (-1, 4))).flatten(-2).to(feature.dtype)
        return feature + position.to(feature.dtype)


class SampledEnhancement(nn.Module):
    def __init__(self, width, edge_width, mode, quaternion, expansion, residual_scale):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.value = nn.Linear(width, edge_width)
        self.geometry = PolarQuaternion(edge_width, mode, quaternion)
        self.score = nn.Sequential(nn.Linear(edge_width, edge_width // 2), nn.GELU(),
                                   nn.Linear(edge_width // 2, 1))
        self.output = nn.Linear(edge_width, width)
        self.ffn = point_mlp(width, width, int(width * expansion))
        self.local_scale = nn.Parameter(torch.full((width,), residual_scale))
        self.ffn_scale = nn.Parameter(torch.full((width,), residual_scale))

    def forward(self, feature, selected, neighbor, valid, relative):
        value = self.value(self.norm(feature))
        edge = value[neighbor] - value[selected, None]
        message = self.geometry(edge, relative)
        score = self.score(message).squeeze(-1).float().masked_fill(~valid, -torch.inf)
        weight = score.softmax(-1).to(message.dtype)
        aggregate = (weight[..., None] * message).sum(1)
        updated = feature[selected] + self.local_scale * self.output(aggregate)
        updated = updated + self.ffn_scale * self.ffn(updated)
        return feature.index_copy(0, selected, updated.to(feature.dtype))


class SelectedPooling(nn.Module):
    """Only selected/enhanced points enter learned feature pooling."""
    def __init__(self, cin, cout, mode, quaternion):
        super().__init__()
        self.value = point_mlp(cin, cout)
        self.geometry = PolarQuaternion(cout, mode, quaternion)
        self.score = nn.Sequential(nn.Linear(cout, cout // 2), nn.GELU(), nn.Linear(cout // 2, 1))
        self.output = nn.Sequential(nn.LayerNorm(cout), nn.GELU(), nn.Linear(cout, cout))

    def forward(self, enhanced, coord, mass, layout, selected, scale):
        size = len(layout.counts)
        with torch.no_grad():
            total_mass = sum_by_group(mass, layout.group, size)
            centroid = sum_by_group(coord.float() * mass[:, None], layout.group, size)
            centroid = centroid / total_mass[:, None]
            group = layout.group[selected]
            relative = (coord[selected] - centroid[group]) / scale
        selected_feature = enhanced[selected]
        value = self.geometry(self.value(selected_feature), relative)
        weight = softmax_by_group(self.score(value).squeeze(-1), group, size)
        # No unselected feature can enter either learned pooled values or context.
        pooled = sum_by_group(value.float() * weight[:, None], group, size).to(value.dtype)
        context = sum_by_group(selected_feature.float() * weight[:, None], group, size).to(enhanced.dtype)
        return self.output(pooled), centroid, total_mass, context, weight


class WeightedSkip(nn.Module):
    """A shared pointwise MLP weights enhanced context versus base fine detail."""
    def __init__(self, fine_width, coarse_width, mode, quaternion, learned=True):
        super().__init__()
        self.mode, self.learned = mode, learned
        self.coarse = point_mlp(coarse_width, fine_width)
        if learned:
            self.position = nn.Sequential(nn.Linear(5, fine_width), nn.GELU(), nn.Linear(fine_width, fine_width))
            self.norm_base, self.norm_enhanced = nn.LayerNorm(fine_width), nn.LayerNorm(fine_width)
            self.weight = nn.Sequential(nn.Linear(2 * fine_width + 6, fine_width), nn.GELU(),
                                        nn.Linear(fine_width, 2))
        self.refine = point_mlp(fine_width, fine_width, fine_width * 2)
        self.scale = nn.Parameter(torch.full((fine_width,), .1))

    def forward(self, coarse, record):
        base, group = record["base"], record["layout"].group
        lifted = self.coarse(coarse)[group]
        if self.learned:
            descriptor = polar_coordinates(record["relative"], self.mode)
            field = record["context"][group] + self.position(descriptor)
            field = field.index_copy(0, record["selected"], record["enhanced_selected"])
            flag = base.new_zeros((len(base), 1))
            flag[record["selected"]] = 1
            logits = self.weight(torch.cat((self.norm_base(base), self.norm_enhanced(field),
                                           descriptor.to(base.dtype), flag), -1))
            weights = logits.float().softmax(-1).to(base.dtype)
            skip = weights[:, :1] * base + weights[:, 1:] * field
        else:
            # Parameter-matched claims require a separate ablation, not assumed here.
            skip = base
            weights = None
        combined = lifted + skip
        return combined + self.scale * self.refine(combined), weights


class SerializedPatchPQ(nn.Module):
    def __init__(self, in_channels=4, out_channels=64,
                 channels=(48, 96, 192, 256), depths=(2, 2, 3, 2),
                 edge_channels=(32, 48, 64, 64), patch_sizes=(64, 64, 64, 32),
                 pool_strides=(8, 4, 4), samples_per_group=(2, 2, 2, 2),
                 k_each_side=(8, 8, 8, 8), stage_scales=(.5, 1., 2., 4.),
                 max_neighbor_distances=(2., 4., 8., 16.),
                 serialization_resolution=.05,
                 axis_orders=((0,1,2), (1,0,2), (1,2,0), (2,1,0)),
                 grouping="patch", grid_pool_sizes=(1., 2., 4., 8.),
                 embedding_mode="cylindrical", use_quaternion=True,
                 learned_skip=True, ffn_expansion=2., residual_scale=.1,
                 checkpoint_blocks=False):
        super().__init__()
        channels, depths = tuple(channels), tuple(depths)
        edge_channels, patch_sizes = tuple(edge_channels), tuple(patch_sizes)
        pool_strides, samples_per_group = tuple(pool_strides), tuple(samples_per_group)
        k_each_side, stage_scales = tuple(k_each_side), tuple(stage_scales)
        max_neighbor_distances = tuple(max_neighbor_distances)
        grid_pool_sizes = tuple(grid_pool_sizes)
        stages = len(channels)
        for name, values in (("depths",depths),("edge_channels",edge_channels),
                             ("patch_sizes",patch_sizes),("samples_per_group",samples_per_group),
                             ("k_each_side",k_each_side),("stage_scales",stage_scales),
                             ("max_neighbor_distances",max_neighbor_distances),
                             ("grid_pool_sizes",grid_pool_sizes)):
            if len(values) != stages:
                raise ValueError(name + " must have one entry per stage")
        if len(pool_strides) != stages-1 or stages < 2:
            raise ValueError("Need >=2 stages and one fewer pool_strides")
        if grouping not in ("patch","grid") or embedding_mode not in ("spherical","cylindrical"):
            raise ValueError("Invalid grouping or embedding_mode")
        if serialization_resolution <= 0 or any(x <= 0 for x in stage_scales + grid_pool_sizes):
            raise ValueError("Spatial scales must be positive")
        if not axis_orders or any(sorted(a) != [0,1,2] for a in axis_orders):
            raise ValueError("axis_orders must contain permutations of (0,1,2)")
        if any(c < 4 or c % 4 for c in channels + edge_channels):
            raise ValueError("Channel widths must be positive multiples of four")
        if any(d < 1 for d in depths) or any(s < 1 for s in samples_per_group):
            raise ValueError("depths and samples_per_group must be positive")
        if any(k < 0 for k in k_each_side):
            raise ValueError("k_each_side cannot be negative")
        self.group_strides = tuple(pool_strides) + (pool_strides[-1],)
        if any(p < 1 or s < 1 or p % s for p,s in zip(patch_sizes,self.group_strides)):
            raise ValueError("Each patch_size must be divisible by its group stride")
        if any(d is not None and d <= 0 for d in max_neighbor_distances):
            raise ValueError("Neighbor distance must be positive or None")
        self.channels, self.patch_sizes = channels, patch_sizes
        self.samples_per_group, self.k_each_side = samples_per_group, k_each_side
        self.stage_scales, self.max_neighbor_distances = stage_scales, max_neighbor_distances
        self.resolution, self.axis_orders = serialization_resolution, axis_orders
        self.grouping, self.grid_pool_sizes = grouping, grid_pool_sizes
        self.checkpoint_blocks = checkpoint_blocks
        self.in_channels = in_channels
        self.stem = point_mlp(in_channels, channels[0], channels[0])
        self.encoders = nn.ModuleList([
            nn.ModuleList([SampledEnhancement(c,e,embedding_mode,use_quaternion,
                                            ffn_expansion,residual_scale) for _ in range(d)])
            for c,e,d in zip(channels,edge_channels,depths)
        ])
        self.pools = nn.ModuleList([SelectedPooling(channels[s],channels[s+1],embedding_mode,use_quaternion)
                                   for s in range(stages-1)])
        self.decoders = nn.ModuleList([WeightedSkip(channels[s],channels[s+1],embedding_mode,
                                                  use_quaternion,learned_skip)
                                      for s in range(stages-1)])
        self.bottleneck = point_mlp(channels[-1],channels[-1],channels[-1]*2)
        self.head = point_mlp(channels[0],out_channels,out_channels)

    def forward(self, data, return_debug=False):
        coord, raw = data["coord"].float(), data["feat"]
        if coord.ndim != 2 or coord.shape[1] != 3 or len(coord) == 0:
            raise ValueError("coord must be a nonempty N-by-3 tensor")
        if raw.shape != (len(coord),self.in_channels):
            raise ValueError("Unexpected input feature shape")
        offset = data.get("offset")
        if offset is None:
            if "batch" in data:
                b = data["batch"].long()
                if bool((b[1:] < b[:-1]).any()):
                    raise ValueError("batch must be contiguous and sorted")
                offset = torch.bincount(b).cumsum(0)
            else:
                offset = torch.tensor([len(coord)],device=coord.device)
        batch, cloud_counts = cloud_batch(offset,len(coord),coord.device)
        # First operation on points is serialization. No sampling/voxel reduction.
        feature = None
        mass = coord.new_ones(len(coord))
        records, debug = [], []
        for stage, blocks in enumerate(self.encoders):
            order,inverse = morton_permutation(coord,batch,self.resolution,
                                               self.axis_orders[stage % len(self.axis_orders)])
            coord,batch,mass = coord[order],batch[order],mass[order]
            if stage == 0:
                feature = self.stem(raw[order])
            else:
                feature = feature[order]
            base = feature
            layout = build_layout(coord,batch,cloud_counts,self.patch_sizes[stage],
                                  self.group_strides[stage],self.grouping,self.grid_pool_sizes[stage])
            selected = choose_enhanced(layout,self.samples_per_group[stage],self.training)
            neighbor,valid,delta = sequence_neighbors(coord,layout,selected,self.k_each_side[stage],
                                                     self.max_neighbor_distances[stage])
            relative = delta / self.stage_scales[stage]
            for block in blocks:
                if self.checkpoint_blocks and self.training:
                    feature = checkpoint(block,feature,selected,neighbor,valid,relative,use_reentrant=False)
                else:
                    feature = block(feature,selected,neighbor,valid,relative)
            if return_debug:
                debug.append(dict(selected=selected,group=layout.group,counts=layout.counts,
                                  neighbor=neighbor,valid=valid,patch=layout.patch,batch=batch,
                                  coord=coord,inverse=inverse))
            if stage < len(self.pools):
                pooled,centroid,total_mass,context,pool_weight = self.pools[stage](
                    feature,coord,mass,layout,selected,self.stage_scales[stage])
                records.append(dict(base=base,layout=layout,selected=selected,
                                    enhanced_selected=feature[selected],context=context,
                                    relative=(coord-centroid[layout.group])/self.stage_scales[stage],
                                    inverse=inverse))
                if return_debug:
                    debug[-1]["pool_weight"] = pool_weight
                # Group IDs are cloud-contiguous in both grouping modes.
                first = torch.full((len(layout.counts),),len(coord),device=coord.device,dtype=torch.long)
                first.scatter_reduce_(0,layout.group,torch.arange(len(coord),device=coord.device),
                                      reduce="amin",include_self=True)
                batch = batch[first]
                cloud_counts = torch.bincount(batch,minlength=len(cloud_counts))
                feature,coord,mass = pooled,centroid,total_mass
            else:
                feature = self.bottleneck(feature)[inverse]
        for stage in reversed(range(len(records))):
            feature,weights = self.decoders[stage](feature,records[stage])
            if return_debug:
                debug[stage]["skip_weight"] = weights
            feature = feature[records[stage]["inverse"]]
        feature = self.head(feature)
        return (feature,debug) if return_debug else feature
