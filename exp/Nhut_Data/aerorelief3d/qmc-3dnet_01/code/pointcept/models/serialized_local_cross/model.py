"""Convolution-free MLP -> local attention -> sampled cross-patch attention."""
from functools import partial
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .geometry import batch_from_data, centered_coordinates, make_patches, make_voxels, group_mean
from .layout import cross_layout, sample_cross
from .attention import PackedAttention


def mlp(cin, cout, ratio=2):
    return nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, cin * ratio), nn.GELU(), nn.Linear(cin * ratio, cout))


class LearnedPooling(nn.Module):
    """Small scalar score per point; channel projection only AFTER pooling."""
    def __init__(self, cin, cout, score_hidden=16):
        super().__init__()
        self.norm = nn.LayerNorm(cin)
        self.score = nn.Sequential(nn.Linear(cin + 3, score_hidden), nn.GELU(), nn.Linear(score_hidden, 1))
        self.project = mlp(cin, cout)

    def forward(self, feature, coord, mass, cells, size, return_weights=False):
        group, count = cells.inverse, len(cells.counts)
        with torch.no_grad():
            center, total_mass = group_mean(coord, group, count, mass)
        relative = ((coord - center[group]) / size).to(feature.dtype)
        score = self.score(torch.cat((self.norm(feature), relative), -1)).squeeze(-1).float()
        maximum = score.new_full((count,), -torch.inf)
        maximum.scatter_reduce_(0, group, score.detach(), reduce="amax", include_self=True)
        exp = (score - maximum[group]).exp()
        denominator = exp.new_zeros(count).index_add(0, group, exp)
        weight = exp / denominator[group].clamp_min(1e-12)
        # Do not create a full FP32 N x Cin weighted tensor: accumulate per
        # feature chunk. Autograd still saves required backward information.
        pooled = feature.new_zeros((count, feature.shape[1]), dtype=torch.float32)
        for begin in range(0, len(feature), 65536):
            end = begin + 65536
            pooled.index_add_(0, group[begin:end], weight[begin:end, None] * feature[begin:end].float())
        output = self.project(pooled.to(feature.dtype))
        result = (output, center, total_mass, cells.indices[:, 0])
        return (*result, weight) if return_weights else result


class Fusion(nn.Module):
    def __init__(self, fine, coarse):
        super().__init__()
        self.project = mlp(coarse, fine)
        self.gate = nn.Sequential(nn.LayerNorm(fine * 2), nn.Linear(fine * 2, fine), nn.Sigmoid())
        self.refine = mlp(fine, fine)

    def forward(self, coarse, skip, inverse):
        lifted = self.project(coarse)[inverse]
        mixed = skip + self.gate(torch.cat((skip, lifted), -1)) * lifted
        return mixed + self.refine(mixed)


class Block(nn.Module):
    def __init__(self, channels, heads, ffn_ratio, local_mode, cross_mode, enable_cross,
                 group_size, queries, keys, dilation, backend, chunk_size, min_freq, max_freq):
        super().__init__()
        self.ffn = mlp(channels, channels, ffn_ratio)
        self.scale = nn.Parameter(torch.full((channels,), .1))
        self.local = PackedAttention(channels, heads, local_mode, backend, chunk_size, min_freq, max_freq)
        self.cross = PackedAttention(channels, heads, cross_mode, backend, chunk_size, min_freq, max_freq) if enable_cross else None
        self.group_size, self.queries, self.keys, self.dilation = group_size, queries, keys, dilation

    def forward(self, feature, coord, batch, patches, spatial_scale, return_debug=False):
        feature = feature + self.scale.to(feature.dtype) * self.ffn(feature)
        p = patches.members.shape[1]
        feature = self.local(feature, coord, patches.members, patches.members, p, p, spatial_scale)
        record = dict(local_members=patches.members.detach()) if return_debug else None
        if self.cross is not None:
            layout = cross_layout(patches, batch, self.group_size, self.dilation, self.queries, self.keys)
            q, k = sample_cross(patches, layout, self.group_size, self.training)
            feature = self.cross(feature, coord, q, k, self.queries, self.group_size * self.keys, spatial_scale)
            if return_debug:
                record.update(cross_queries=q.detach(), cross_keys=k.detach(), patch_group=layout.group.detach(),
                              patch_slot=layout.slot.detach(), dilation=self.dilation)
        return (feature, record) if return_debug else feature


class SerializedLocalCross(nn.Module):
    def __init__(self, in_channels=4, out_channels=64, channels=(48, 96, 192, 256),
                 heads=(3, 6, 12, 16), depths=(2, 2, 3, 2), patch_sizes=(128,128,128,128),
                 pool_sizes=(.4,.8,1.6), position_scales=(.4,.8,1.6,3.2),
                 cross_group_size=2, cross_queries=(16,16,16,16), cross_keys=(32,32,32,32),
                 cross_every=2, cross_dilations=(1,2), local_rotary="none", cross_rotary="qk",
                 ffn_ratio=2, score_hidden=16, chunk_size=128, attention_backend="flash_varlen",
                 serialization_resolution=.05,
                 order=("z","z-trans","hilbert","hilbert-trans"),shuffle_orders=True,
                 serialization_backend="pointcept",axis_orders=None,
                 rotary_min_frequency=.25, rotary_max_frequency=2., checkpoint_blocks=False):
        super().__init__()
        stages = len(channels)
        if stages < 1 or any(len(v) != stages for v in (heads,depths,patch_sizes,position_scales,cross_queries,cross_keys)) or len(pool_sizes) != stages-1:
            raise ValueError("Stage setting lengths do not match")
        integers = (*channels,*heads,*depths,*patch_sizes,*cross_queries,*cross_keys,*cross_dilations,
                    cross_group_size,cross_every,ffn_ratio,score_hidden,chunk_size,in_channels,out_channels)
        if any(not isinstance(v,int) or v < 1 for v in integers):
            raise ValueError("Dimensions, depths, sampling counts, intervals must be positive integers")
        if not cross_dilations or cross_group_size < 2 or any(m < cross_group_size for m in cross_queries):
            raise ValueError("Need >=2 patches/group and >=1 query budget per member patch")
        if any(c % h for c,h in zip(channels,heads)):
            raise ValueError("Channels must be divisible by heads")
        if attention_backend not in ("flash_varlen","reference"):
            raise ValueError("Unknown attention backend")
        if attention_backend == "flash_varlen" and any((c//h)%8 for c,h in zip(channels,heads)):
            raise ValueError("FlashAttention head dimensions must be multiples of 8")
        if local_rotary not in ("none","qk","qkv") or cross_rotary not in ("none","qk","qkv"):
            raise ValueError("Unknown rotary mode")
        if any(not math.isfinite(v) or v<=0 for v in (*pool_sizes,*position_scales,serialization_resolution)):
            raise ValueError("Spatial scales must be finite and positive")
        if any(b<=a for a,b in zip(pool_sizes,pool_sizes[1:])):
            raise ValueError("Pooling sizes must increase")
        if axis_orders is not None:
            if not axis_orders or any(sorted(a)!=[0,1,2] for a in axis_orders):
                raise ValueError("Invalid legacy serialization axes")
        elif not order or any(o not in ("z","z-trans","hilbert","hilbert-trans") for o in order):
            raise ValueError("Invalid serialization order names")
        if serialization_backend not in ("pointcept","reference"):
            raise ValueError("Unknown serialization backend")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.patch_sizes, self.pool_sizes, self.position_scales = patch_sizes,pool_sizes,position_scales
        self.resolution,self.axis_orders,self.checkpoint_blocks=serialization_resolution,axis_orders,checkpoint_blocks
        self.order,self.shuffle_orders,self.serialization_backend=tuple(order),shuffle_orders,serialization_backend
        self.stem = mlp(in_channels,channels[0])
        stages_list, cross_index = [], 0
        for s,c in enumerate(channels):
            blocks=[]
            for b in range(depths[s]):
                enabled = (b+1)%cross_every == 0
                dilation=cross_dilations[cross_index%len(cross_dilations)]
                blocks.append(Block(c,heads[s],ffn_ratio,local_rotary,cross_rotary,enabled,
                    cross_group_size,cross_queries[s],cross_keys[s],dilation,attention_backend,
                    chunk_size,rotary_min_frequency,rotary_max_frequency))
                cross_index += int(enabled)
            stages_list.append(nn.ModuleList(blocks))
        self.stages=nn.ModuleList(stages_list)
        self.pools=nn.ModuleList([LearnedPooling(a,b,score_hidden) for a,b in zip(channels,channels[1:])])
        self.decoders=nn.ModuleList([Fusion(a,b) for a,b in zip(channels,channels[1:])])
        self.head=mlp(channels[0],out_channels)

    def forward(self,data,return_debug=False):
        coord,raw=data["coord"],data["feat"]
        if coord.ndim!=2 or coord.shape[1]!=3 or not len(coord) or raw.shape!=(len(coord),self.in_channels):
            raise ValueError("Expected coord (N,3), feat (N,in_channels), N>0")
        if coord.device!=raw.device or not coord.is_floating_point() or not raw.is_floating_point():
            raise ValueError("Floating features and coordinates must share a device")
        if not bool(torch.isfinite(coord).all()) or not bool(torch.isfinite(raw).all()):
            raise ValueError("Inputs must be finite")
        batch=batch_from_data(data,len(coord),coord.device)
        with torch.no_grad():
            coord=centered_coordinates(coord.float(),batch)
        feature=self.stem(raw)
        mass=coord.new_ones(len(coord))
        records,debug=[],[]
        for s,blocks in enumerate(self.stages):
            layouts={}
            available=self.axis_orders if self.axis_orders is not None else self.order
            if self.shuffle_orders and self.training:
                permutation=torch.randperm(len(available)).tolist()
                available=tuple(available[i] for i in permutation)
            for b,block in enumerate(blocks):
                axes=available[(s+b)%len(available)]
                if axes not in layouts:
                    layouts[axes]=make_patches(coord,batch,self.resolution,axes,self.patch_sizes[s],self.serialization_backend)
                operation=partial(block,coord=coord,batch=batch,patches=layouts[axes],spatial_scale=self.position_scales[s])
                if return_debug:
                    feature,record=operation(feature,return_debug=True)
                    record.update(stage=s,block=b,batch=batch.detach())
                    record["serialization_order"]=axes
                    debug.append(record)
                elif self.checkpoint_blocks and self.training:
                    feature=checkpoint(operation,feature,use_reentrant=False,preserve_rng_state=True)
                else:
                    feature=operation(feature)
            if s<len(self.pools):
                cells=make_voxels(coord,batch,self.pool_sizes[s])
                records.append((feature,cells.inverse))
                feature,coord,mass,batch=self.pools[s](feature,coord,mass,cells,self.pool_sizes[s])
        for s in reversed(range(len(records))):
            skip,inverse=records[s]
            feature=self.decoders[s](feature,skip,inverse)
        output=self.head(feature)
        return (output,debug) if return_debug else output
