"""Memory-oriented v2: radial voxel attention, coarse quaternion geometry,
checkpointed query chunks, and point-to-logit decoding without an N x C feature.
This is FPT-inspired, not a reproduction or an optimized sparse CUDA kernel.
"""
import math
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .so3_quaternion_dual_attention import (
    MODELS, Point, counts, segment_mean, VectorSeed, VNRMSNorm,
    VectorReadout, SO3QuaternionDualSegmentor,
)
from .so3_voxel_quaternion import voxel_partition, voxel_graph, LiteQuaternionAttention


def recompute(fn, *args, enabled=True):
    return checkpoint(fn, *args, use_reentrant=False) if enabled and torch.is_grad_enabled() else fn(*args)


class RadialVoxelBlock(nn.Module):
    """Query-dependent cosine weights against shared radial basis embeddings.

    No edge MLP, key projection, neighborhood softmax or patch QKV. Each channel
    group shares a score. Distances are continuous centroid distances, preserving
    rotation behavior on a fixed graph. Radial-only fine geometry is deliberate;
    angular geometric reasoning is supplied by coarse quaternion blocks.
    """
    def __init__(self, channels, groups=4, width=4, query_chunk=4096, checkpointing=True):
        super().__init__()
        if channels % groups or query_chunk < 1:
            raise ValueError('channels must divide groups; query_chunk must be positive')
        self.groups, self.width = groups, width
        self.query_chunk, self.checkpointing = query_chunk, checkpointing
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, groups * width)
        self.value = nn.Linear(channels, channels, bias=False)
        self.radial = nn.Parameter(torch.randn(4, groups, width) / math.sqrt(width))
        self.out = nn.Linear(channels, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 2*channels), nn.GELU(), nn.Linear(2*channels, channels))

    def _chunk(self, x, coord, dst, src, begin, end, radius):
        local = dst - begin
        q = F.normalize(self.query(self.norm(x[begin:end])).float().reshape(-1, self.groups, self.width), dim=-1)
        distance = (coord[src]-coord[dst]).float().norm(dim=-1) / radius
        basis = torch.stack([torch.ones_like(distance), distance, distance.square(), torch.exp(-distance.square())], -1)
        position = F.normalize(torch.einsum('eb,bgw->egw', basis, self.radial.float()), dim=-1)
        weights = (q[local] * position).sum(-1)
        values = self.value(self.norm(x[src])).reshape(-1, self.groups, x.shape[-1]//self.groups)
        message = values * weights.to(values.dtype)[..., None]
        # Occupancy normalization stabilizes scale; not softmax attention.
        count = torch.bincount(local, minlength=end-begin).clamp_min(1).to(message.dtype)
        agg = message.new_zeros(end-begin, self.groups, message.shape[-1]).index_add(0, local, message)
        y = x[begin:end] + self.out((agg / count[:, None, None]).flatten(1))
        return y + self.ffn(self.ffn_norm(y))

    def forward(self, x, coord, graph):
        ends = list(range(0, len(x), self.query_chunk)) + [len(x)]
        boundaries = torch.searchsorted(graph.dst, torch.tensor(ends, device=x.device)).tolist()
        chunks = []
        for j, (a, b) in enumerate(zip(ends[:-1], ends[1:])):
            dd, ss = graph.dst[boundaries[j]:boundaries[j+1]], graph.src[boundaries[j]:boundaries[j+1]]
            chunks.append(recompute(self._chunk, x, coord, dd, ss, a, b, graph.radius,
                enabled=self.training and self.checkpointing))
        return torch.cat(chunks)


class CoarseQuaternionBlock(nn.Module):
    def __init__(self, channels, vectors, query_chunk, checkpointing):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.vnorm = VNRMSNorm(vectors)
        self.attn = LiteQuaternionAttention(channels, vectors, quaternion_channels=2,
            groups=2, geometry_width=16, chunk_size=query_chunk*8, checkpoint_edges=checkpointing)
        self.readout = VectorReadout(vectors, proj=2, triple=False)
        self.to_scalar = nn.Linear(self.readout.out_dim, channels)

    def forward(self, x, v, coord, graph):
        s, dv = self.attn(self.norm(x), self.vnorm(v), coord, graph)
        v = v + .1*dv
        return x + s + self.to_scalar(self.readout(v)), v


@MODELS.register_module('Voxel-Quaternion-Lite-v2')
class VoxelQuaternionLite(nn.Module):
    def __init__(self, in_channels=1, scalar_feature_indices=(0,),
                 channels=(32, 64, 128, 192), depths=(1, 1, 2, 2),
                 voxel_sizes=(.4, .8, 1.6, 3.2), radii=(.8, 1.6, 3.2, 6.4),
                 quaternion_stages=(2, 3), vector_channels=4, head_channels=32,
                 query_chunk=4096, checkpoint_blocks=True, diagnostics=False):
        super().__init__()
        n = len(channels)
        if not n or any(len(s)!=n for s in (depths, voxel_sizes, radii)):
            raise ValueError('Stage sequence lengths must match')
        if any(c < 4 or c % 4 for c in channels) or any(d < 1 for d in depths):
            raise ValueError('Widths must be positive multiples of four; depths positive')
        if any(not math.isfinite(s) or s <= 0 for s in (*voxel_sizes, *radii)):
            raise ValueError('Sizes and radii must be finite and positive')
        if any(voxel_sizes[i+1] <= voxel_sizes[i] or abs(voxel_sizes[i+1]/voxel_sizes[i]-round(voxel_sizes[i+1]/voxel_sizes[i]))>1e-6 for i in range(n-1)):
            raise ValueError('Voxel sizes must grow by integer factors')
        if any(i < 0 or i >= n for i in quaternion_stages) or vector_channels < 2 or vector_channels % 2 or query_chunk < 1:
            raise ValueError('Invalid quaternion stages, vector width or chunk size')
        indices = tuple(scalar_feature_indices)
        if in_channels < 1 or any(i<0 or i>=in_channels for i in indices) or len(set(indices)) != len(indices):
            raise ValueError('Invalid scalar input indices')
        self.in_channels, self.head_channels = in_channels, head_channels
        self.channels, self.voxel_sizes, self.radii = tuple(channels), tuple(voxel_sizes), tuple(radii)
        self.indices, self.query_chunk = indices, query_chunk
        self.checkpoint_blocks, self.diagnostics = checkpoint_blocks, diagnostics
        self.last_stats = {}
        self.stem = nn.Linear(max(1,len(indices)), channels[0])
        self.stages = nn.ModuleList([nn.ModuleList([RadialVoxelBlock(c, query_chunk=query_chunk, checkpointing=checkpoint_blocks) for _ in range(d)]) for c,d in zip(channels,depths)])
        self.down = nn.ModuleList([nn.Linear(a,b) for a,b in zip(channels[:-1],channels[1:])])
        self.up = nn.ModuleList([nn.Linear(channels[i+1]+channels[i], channels[i]) for i in range(n-1)])
        self.geometry = nn.ModuleDict({str(i):CoarseQuaternionBlock(channels[i],vector_channels,query_chunk,checkpoint_blocks) for i in quaternion_stages})
        self.seeds = nn.ModuleDict({str(i):VectorSeed(channels[i],vector_channels,hidden=8,chunk_size=query_chunk*8) for i in quaternion_stages})
        self.output = nn.Linear(channels[0],head_channels)

    def scalars(self, feat):
        return feat[:, self.indices] if self.indices else feat.new_ones(len(feat),1)

    def _linear_chunks(self, layer, x):
        return torch.cat([recompute(layer,x[a:a+self.query_chunk],enabled=self.training and self.checkpoint_blocks) for a in range(0,len(x),self.query_chunk)])

    def forward_voxels(self, data):
        coord, feat, offset = data['coord'].detach().float(), data['feat'], data['offset']
        if coord.ndim!=2 or coord.shape[1]!=3 or not len(coord) or not bool(torch.isfinite(coord).all()):
            raise ValueError('coord must be finite nonempty [N,3]')
        if feat.shape!=(len(coord),self.in_channels) or not feat.is_floating_point() or not bool(torch.isfinite(feat).all()):
            raise ValueError('Invalid input features')
        if offset.dtype not in (torch.int32,torch.int64) or offset.ndim!=1 or not len(offset) or int(offset[-1])!=len(coord) or bool((counts(offset)<=0).any()):
            raise ValueError('Invalid offsets')
        if feat.device!=coord.device or offset.device!=coord.device:
            raise ValueError('Input devices must match')
        inverse, _, c, off = voxel_partition(coord,offset,self.voxel_sizes[0])
        x = self._linear_chunks(self.stem,segment_mean(self.scalars(feat),inverse,len(c)))
        fine_centroid = c
        skips, maps, stats = [], [], []
        for i, blocks in enumerate(self.stages):
            graph = voxel_graph(c,off,self.voxel_sizes[i],self.radii[i],chunk_size=self.query_chunk)
            stats.append(dict(voxels=len(c),edges=len(graph.dst),channels=x.shape[1]))
            for block in blocks:
                x = block(x,c,graph)
            if str(i) in self.geometry:
                # Seed vectors only at coarse resolutions; no fine vector tensors.
                def geometry(x, c, graph, key):
                    v = self.seeds[key](x,c,graph)
                    return self.geometry[key](x,v,c,graph)[0]
                x = recompute(geometry,x,c,graph,str(i),enabled=self.training and self.checkpoint_blocks)
            skips.append(x)
            if i < len(self.down):
                parent, _, c, off = voxel_partition(c,off,self.voxel_sizes[i+1])
                maps.append(parent)
                x = self._linear_chunks(self.down[i],segment_mean(x,parent,len(c)))
        for i in reversed(range(len(maps))):
            def decode(coarse, fine, parent, a, b, level):
                return F.gelu(self.up[level](torch.cat([fine[a:b],coarse[parent[a:b]]],-1)))
            x = torch.cat([recompute(decode,x,skips[i],maps[i],a,min(a+self.query_chunk,len(skips[i])),i,enabled=self.training and self.checkpoint_blocks) for a in range(0,len(skips[i]),self.query_chunk)])
        x = self._linear_chunks(self.output,x)
        self.last_stats = dict(points=len(coord),stages=stats)
        if self.diagnostics:
            import logging
            logging.getLogger('pointcept').info('VoxelQuaternionLite counts: %s',self.last_stats)
        return x, inverse, fine_centroid

    def forward(self, data, return_stages=False):
        if return_stages:
            raise ValueError('Auxiliary supervision is unsupported')
        x, inverse, _ = self.forward_voxels(data)
        point = Point(dict(data)); point.feat = x[inverse]
        return point


@MODELS.register_module('VoxelQuaternionLiteSegmentor')
class VoxelQuaternionLiteSegmentor(SO3QuaternionDualSegmentor):
    """Checkpoint refinement AND classifier together; retain N x classes only.

    Full-scene logits are kept for exact existing CE/Lovasz criteria. Loss is
    not averaged across chunks, which would change global Lovasz supervision.
    """
    def __init__(self, point_chunk=32768, **kwargs):
        if kwargs.get('deep_supervision',False):
            raise ValueError('Disable deep_supervision for lite segmentor')
        super().__init__(**kwargs)
        if point_chunk < 1:
            raise ValueError('point_chunk must be positive')
        self.point_chunk = point_chunk
        h = self.backbone.head_channels
        self.refine = nn.Sequential(nn.Linear(h+max(1,len(self.backbone.indices))+1,h),nn.GELU(),nn.Linear(h,h))

    def _point_logits(self, x, inverse, centroid, coord, feat, a, b):
        decoded = x[inverse[a:b]]
        radius = (coord[a:b].detach().float()-centroid[inverse[a:b]]).norm(dim=-1,keepdim=True) / self.backbone.voxel_sizes[0]
        scalars = self.backbone.scalars(feat[a:b])
        refined = decoded + self.refine(torch.cat([decoded,scalars.to(decoded.dtype),radius.to(decoded.dtype)],-1))
        return self.seg_head(refined)

    def forward(self,input_dict,return_point=False):
        target = input_dict.get('segment')
        if self.training and target is None:
            raise ValueError('Training requires segment labels')
        if target is not None:
            if target.dtype!=torch.long or target.shape!=(len(input_dict['coord']),):
                raise ValueError('segment must be long [N]')
            if bool(((target!=self.ignore_index)&((target<0)|(target>=self.num_classes))).any()):
                raise ValueError('Invalid segment labels')
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            x,inverse,centroid = self.backbone.forward_voxels(input_dict)
        n = len(inverse)
        logits = torch.cat([recompute(self._point_logits,x,inverse,centroid,input_dict['coord'],input_dict['feat'],a,min(a+self.point_chunk,n),enabled=self.training) for a in range(0,n,self.point_chunk)])
        out = {} if self.training else dict(seg_logits=logits)
        if target is not None:
            out['loss'] = self._loss(logits,target)
        if return_point:
            # Deliberately expose logits, without expanding hidden point features.
            point = Point(dict(input_dict)); point.feat = logits
            out['point'] = point
        return out
