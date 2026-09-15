"""Standalone local vector-attention / global Haar-WNO segmentation model.

All embedding, attention, pooling, fusion and head operations are defined here.
Runtime default: PyTorch only. No PTv3, Pointcept, spconv, torch_scatter, pointops,
or FlashAttention dependency. This is a research architecture, not pretrained.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
if __package__:
    from .spatial_ordering import curve_neighbors, exact_neighbors
    from .edge_encoding import QuaternionEdgeEncoding
    from .wavelet import WNO3dBlock
else:
    from spatial_ordering import curve_neighbors, exact_neighbors
    from edge_encoding import QuaternionEdgeEncoding
    from wavelet import WNO3dBlock

def masked_neighbor_softmax(logits, valid):
    """Channel-wise softmax over neighbors, including safely empty rows."""
    x = logits if logits.dtype == torch.float64 else logits.float()
    masked = x.masked_fill(~valid[...,None], float('-inf'))
    masked = torch.where(valid.any(dim=1)[:,None,None], masked, torch.zeros_like(masked))
    return (masked.softmax(dim=1) * valid[...,None]).to(logits.dtype)


class CartesianEdgeEncoding(nn.Module):
    """Standard positional MLP baseline on x_i-x_j, without quaternions."""
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3,channels),nn.LayerNorm(channels),
                                 nn.GELU(),nn.Linear(channels,channels))

    def encode_edges(self, coord, neighbors, query_index, batch=None):
        valid = (neighbors>=0) & (neighbors!=query_index[:,None])
        delta = coord[query_index,None,:]-coord[neighbors.clamp_min(0)]
        out = self.mlp(delta.to(self.mlp[0].weight.dtype))
        return out*valid[...,None], valid


def make_position_encoder(kind, channels, distance_scale):
    if kind == 'quaternion':
        return QuaternionEdgeEncoding(channels,distance_scale=distance_scale)
    if kind == 'cartesian':
        return CartesianEdgeEncoding(channels)
    if kind == 'none':
        return None
    raise ValueError("position must be quaternion, cartesian or none")


class Embedding(nn.Module):
    """Explicit pointwise stem: Linear -> LayerNorm -> GELU -> Linear.

    No absolute coordinates are appended internally; input features are explicit.
    Supply RGB alone, XYZ+RGB, or other attributes through in_channels and feat.
    """
    def __init__(self, in_channels, channels):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_channels,channels),nn.LayerNorm(channels),
                               nn.GELU(),nn.Linear(channels,channels))

    def forward(self, features):
        return self.net(features)


class LocalVectorAttention(nn.Module):
    """PTv1-style vector attention with edge position in BOTH branches.

    a_ij=softmax_j gamma(phi(f_i)-psi(f_j)+delta_ij)
    y_i=sum_j a_ij * (alpha(f_j)+delta_ij), where * is elementwise.
    Full channel-wise weights; not dot-product or FlashAttention.
    """
    def __init__(self, channels, position='quaternion', chunk_size=1024,
                 distance_scale=1.0):
        super().__init__()
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        self.channels,self.chunk_size=channels,chunk_size
        self.query=nn.Linear(channels,channels,bias=False)
        self.key=nn.Linear(channels,channels,bias=False)
        self.value=nn.Linear(channels,channels,bias=False)
        self.position=make_position_encoder(position,channels,distance_scale)
        hidden=max(channels//4,8)
        self.weight_mlp=nn.Sequential(nn.Linear(channels,hidden),nn.LayerNorm(hidden),
                                      nn.GELU(),nn.Linear(hidden,channels))
        self.output=nn.Linear(channels,channels,bias=False)

    def forward(self, feat, coord, batch, neighbors):
        q,k,v=self.query(feat),self.key(feat),self.value(feat)
        outputs=[]
        for start in range(0,len(feat),self.chunk_size):
            ids=torch.arange(start,min(start+self.chunk_size,len(feat)),device=feat.device)
            graph=neighbors[ids]
            valid=(graph>=0)&(graph!=ids[:,None])
            safe=graph.clamp_min(0)
            if self.position is None:
                delta=torch.zeros_like(k[safe])
            else:
                delta,valid=self.position.encode_edges(coord,graph,ids,batch)
                delta=delta.to(q.dtype)
            weights=masked_neighbor_softmax(self.weight_mlp(q[ids,None,:]-k[safe]+delta),valid)
            outputs.append((weights*(v[safe]+delta)).sum(dim=1))
        if not outputs:
            return feat.new_empty((0,self.channels))
        return self.output(torch.cat(outputs))


class Block(nn.Module):
    """Explicit pre-normalized residual attention + feed-forward block."""
    def __init__(self, channels, mlp_ratio=2.0, dropout=0.0, **attention_kwargs):
        super().__init__()
        self.norm1,self.norm2=nn.LayerNorm(channels),nn.LayerNorm(channels)
        self.attention=LocalVectorAttention(channels,**attention_kwargs)
        self.dropout=nn.Dropout(dropout)
        hidden=max(1,int(channels*mlp_ratio))
        self.mlp=nn.Sequential(nn.Linear(channels,hidden),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(hidden,channels))

    def forward(self, feat, coord, batch, neighbors):
        feat=feat+self.dropout(self.attention(self.norm1(feat),coord,batch,neighbors))
        return feat+self.dropout(self.mlp(self.norm2(feat)))


def scatter_mean(values, inverse, count):
    """Feature/coordinate mean with float32 accumulation under mixed precision."""
    dtype=values.dtype
    work=values if dtype==torch.float64 else values.float()
    out=work.new_zeros((count,work.shape[1])).index_add(0,inverse,work)
    degree=torch.bincount(inverse,minlength=count).to(work.dtype).clamp_min(1)
    return (out/degree[:,None]).to(dtype)


class VoxelPool(nn.Module):
    """Per-scene voxel pooling with traceable fine->coarse assignments."""
    def __init__(self, in_channels, out_channels, cell_size, reduce='max'):
        super().__init__()
        if cell_size<=0 or reduce not in ('max','mean'):
            raise ValueError('Positive cell_size and max/mean reduction required')
        self.cell_size,self.reduce=cell_size,reduce
        self.project=nn.Linear(in_channels,out_channels)
        self.norm=nn.LayerNorm(out_channels)
        self.act=nn.GELU()

    def forward(self, feat, coord, batch):
        cells=torch.empty_like(coord,dtype=torch.long)
        for scene in torch.unique(batch,sorted=True):
            ids=torch.nonzero(batch==scene,as_tuple=False).flatten()
            origin=coord[ids].amin(dim=0)
            cells[ids]=((coord[ids]-origin)/self.cell_size).floor().long()
        keys=torch.cat((batch[:,None],cells),dim=1)
        unique,inverse=torch.unique(keys,dim=0,sorted=True,return_inverse=True)
        projected=self.project(feat)
        if self.reduce=='mean':
            pooled=scatter_mean(projected,inverse,len(unique))
        else:
            pooled=projected.new_zeros((len(unique),projected.shape[1]))
            pooled=pooled.scatter_reduce(0,inverse[:,None].expand_as(projected),projected,
                                          reduce='amax',include_self=False)
        coords=scatter_mean(coord,inverse,len(unique))
        return self.act(self.norm(pooled)),coords,unique[:,0],inverse


class PointWaveletContext(nn.Module):
    """Shared point->Haar grid->point context with per-scene grids."""
    def __init__(self, channels, grid_size, levels, global_rank, global_mixing,
                 position, chunk_size, distance_scale):
        super().__init__()
        if len(grid_size)!=3 or any(g<2**levels or g%(2**levels) for g in grid_size):
            raise ValueError('Grid dimensions must be multiples of 2**levels')
        self.grid_size,self.chunk_size=tuple(grid_size),chunk_size
        self.position=make_position_encoder(position,channels,distance_scale)
        self.operator=WNO3dBlock(channels,levels,global_rank,global_mixing)
        self.norm=nn.LayerNorm(channels)

    def forward(self, feat, coord, batch, neighbors):
        if self.position is not None:
            enc=[]
            for start in range(0,len(feat),self.chunk_size):
                ids=torch.arange(start,min(start+self.chunk_size,len(feat)),device=feat.device)
                edge,valid=self.position.encode_edges(coord,neighbors[ids],ids,batch)
                enc.append(edge.sum(1)/valid.sum(1,keepdim=True).clamp_min(1))
            feat=feat+torch.cat(enc).to(feat.dtype)
        gx,gy,gz=self.grid_size
        cells=gx*gy*gz
        output,indices=[],[]
        for scene in torch.unique(batch,sorted=True):
            ids=torch.nonzero(batch==scene,as_tuple=False).flatten()
            xyz=coord[ids].float()
            xyz=xyz-xyz.amin(0)
            # Axis-wise scaling matches the earlier WNO bridge; not SO(3)-equivariant.
            voxel=(xyz/xyz.amax(0).clamp_min(1e-6)*xyz.new_tensor([gx-1,gy-1,gz-1])).clamp_min(0).long()
            flat=voxel[:,0]*(gy*gz)+voxel[:,1]*gz+voxel[:,2]
            grid=scatter_mean(feat[ids].float(),flat,cells)
            grid=grid.reshape(1,gx,gy,gz,-1).permute(0,4,1,2,3).contiguous()
            context=self.operator(grid).permute(0,2,3,4,1).reshape(cells,-1)[flat]
            output.append(self.norm(context).to(feat.dtype))
            indices.append(ids)
        return torch.cat(output)[torch.argsort(torch.cat(indices))]


class MultiScaleHead(nn.Module):
    """Project every stage and restore original point order via pooling inverses."""
    def __init__(self, channels, out_channels):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(c,out_channels),nn.LayerNorm(out_channels)) for c in channels])
        self.stage_logits=nn.Parameter(torch.zeros(len(channels)))
        self.output=nn.Sequential(nn.Linear(out_channels,out_channels),nn.LayerNorm(out_channels),nn.GELU())

    def forward(self, stages, inverses):
        weights=self.stage_logits.softmax(0)
        combined=None
        for s,(feat,projection) in enumerate(zip(stages,self.projections)):
            up=projection(feat)
            for level in range(s-1,-1,-1):
                up=up[inverses[level]]
            weighted=weights[s]*up
            combined=weighted if combined is None else combined+weighted
        return self.output(combined)

class AerialWaveletNet(nn.Module):
    """Complete model: point stem -> local encoder + wavelet transitions -> head.

    Input {'coord': N,3; 'feat': N,in_channels; 'batch': N} or cumulative 'offset'.
    Output raw N,num_classes logits in original input-point order.
    'curve_knn' means approximate kNN from multiple spatial orderings.
    'exact_knn' is a controlled baseline with no spatial-curve dependence.
    """
    def __init__(self, in_channels=6, num_classes=5, 
                 channels=(32,64,128,256),
                 depths=(2,2,4,2), 
                 strides=(2,2,2), 
                 base_voxel_size=0.5,
                 k=16, 
                 neighbor_mode='curve_knn',
                 exact_backend='torch',
                 orders=('hilbert:xyz','morton:yzx','serpentine:zxy'),
                 curve_bits=10, 
                 window_radius=16, 
                 chunk_size=1024,
                 position='quaternion', distance_scale=1.0, mlp_ratio=2.0,dropout=0.0,
                 wavelet_stages=None, wavelet_dim=64, grid_size=(64,64,64),
                 wavelet_levels=3, wavelet_rank=16, wavelet_global_mixing=True,
                 context_position=True, head_channels=64, pool_reduce='max',
                 checkpoint_blocks=False):
        super().__init__()
        n=len(channels)
        if n<1 or len(depths)!=n or len(strides)!=n-1 or any(d<1 for d in depths):
            raise ValueError('channels/depths must match, with one stride per transition')
        if any(s<1 for s in strides) or k<1 or chunk_size<1 or num_classes<1:
            raise ValueError('Positive strides/k/chunk_size/classes required')
        if neighbor_mode not in ('curve_knn','exact_knn'):
            raise ValueError('neighbor_mode must be curve_knn or exact_knn')
        wavelet_stages=(True,)*(n-1) if wavelet_stages is None else tuple(wavelet_stages)
        if len(wavelet_stages)!=n-1:
            raise ValueError('wavelet_stages needs one flag per transition')
        self.in_channels,self.head_channels=in_channels,head_channels
        self.k,self.neighbor_mode,self.exact_backend=k,neighbor_mode,exact_backend
        self.orders,self.curve_bits,self.window_radius=tuple(orders),curve_bits,window_radius
        self.chunk_size,self.checkpoint_blocks=chunk_size,checkpoint_blocks
        self.embedding=Embedding(in_channels,channels[0])
        self.encoder=nn.ModuleList([nn.ModuleList([
            Block(c,mlp_ratio,dropout,position=position,chunk_size=chunk_size,distance_scale=distance_scale)
            for _ in range(depth)]) for c,depth in zip(channels,depths)])
        self.pooling=nn.ModuleList()
        size=base_voxel_size
        for i,stride in enumerate(strides):
            size*=stride
            self.pooling.append(VoxelPool(channels[i],channels[i+1],size,pool_reduce))
        self.wavelet_stages=wavelet_stages
        self.context=PointWaveletContext(wavelet_dim,grid_size,wavelet_levels,wavelet_rank,
            wavelet_global_mixing,position if context_position else 'none',chunk_size,distance_scale) if any(wavelet_stages) else None
        self.context_in=nn.ModuleDict({str(s):nn.Linear(channels[s],wavelet_dim) for s,on in enumerate(wavelet_stages) if on})
        self.context_out=nn.ModuleDict({str(s):nn.Linear(wavelet_dim,channels[s+1]) for s,on in enumerate(wavelet_stages) if on})
        self.gates=nn.ParameterDict({str(s):nn.Parameter(torch.tensor(-2.0)) for s,on in enumerate(wavelet_stages) if on})
        self.head=MultiScaleHead(channels,head_channels)
        self.classifier=nn.Linear(head_channels,num_classes)

    def neighbors(self,coord,batch):
        if self.neighbor_mode=='curve_knn':
            return curve_neighbors(coord,batch,self.k,self.orders,self.curve_bits,self.window_radius,self.chunk_size)
        return exact_neighbors(coord,batch,self.k,self.exact_backend,min(self.chunk_size,512))

    def _inputs(self,data):
        coord,feat=data['coord'],data['feat']
        if coord.ndim!=2 or coord.shape[1]!=3 or feat.shape!=(len(coord),self.in_channels) or len(coord)==0:
            raise ValueError('Expected nonempty coord=(N,3), feat=(N,in_channels)')
        if coord.device!=feat.device or not bool(torch.isfinite(coord).all()):
            raise ValueError('Finite coordinates and matching coord/feat devices required')
        if 'batch' in data:
            batch=data['batch'].to(device=coord.device,dtype=torch.long)
        elif 'offset' in data:
            offset=data['offset'].to(device=coord.device,dtype=torch.long)
            if offset.ndim!=1 or not len(offset) or offset[-1]!=len(coord):
                raise ValueError('offset must contain cumulative counts ending at N')
            counts=torch.diff(offset,prepend=offset.new_zeros(1))
            if bool((counts<0).any()):
                raise ValueError('offset must be nondecreasing')
            batch=torch.repeat_interleave(torch.arange(len(offset),device=coord.device),counts)
        else:
            batch=torch.zeros(len(coord),device=coord.device,dtype=torch.long)
        if batch.shape!=(len(coord),):
            raise ValueError('batch must have shape (N,)')
        # Geometry stays float32 even under AMP. Center large GIS coords upstream.
        return coord.float(),feat.to(self.embedding.net[0].weight.dtype),batch

    def forward_features(self,data):
        coord,feat,batch=self._inputs(data)
        feat=self.embedding(feat)
        stages,inverses=[],[]
        for s,blocks in enumerate(self.encoder):
            graph=self.neighbors(coord,batch)  # reused by all blocks and context at this scale
            for block in blocks:
                if self.checkpoint_blocks and self.training:
                    feat=checkpoint(block,feat,coord,batch,graph,use_reentrant=False)
                else:
                    feat=block(feat,coord,batch,graph)
            stages.append(feat)
            if s<len(self.pooling):
                context=None
                if self.wavelet_stages[s]:
                    context=self.context(self.context_in[str(s)](feat),coord,batch,graph)
                feat,coord,batch,inverse=self.pooling[s](feat,coord,batch)
                inverses.append(inverse)
                if context is not None:
                    context=scatter_mean(context,inverse,len(feat))
                    feat=feat+self.gates[str(s)].sigmoid()*self.context_out[str(s)](context)
        return self.head(stages,inverses)

    def forward(self,data):
        return self.classifier(self.forward_features(data))
