"""Frame-free scalar/Vector Neuron segmentation for Pointcept.
Register by importing this module from pointcept/models/__init__.py.
No additive epsilon is used. Geometry has no coordinate gradients.
"""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Optional
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
try:
    import pointops
except ImportError:
    pointops = None
import warnings
_WARNED = set()
def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg, RuntimeWarning)
class _timer:
    def __init__(self, *args): pass
    def __enter__(self): pass
    def __exit__(self, *args): pass

def divide_or_zero(numerator, denominator):
    """Broadcastable nonnegative denominator; zero branch has zero gradient."""
    nonzero = denominator > 0
    safe = torch.where(nonzero, denominator, torch.ones_like(denominator))
    return (numerator / safe) * nonzero.to(numerator.dtype)

def safe_norm(x, dim=-1, keepdim=False):
    # torch.linalg.vector_norm gives a finite zero subgradient at zero.
    return torch.linalg.vector_norm(x.float(), dim=dim, keepdim=keepdim)

class ExactLayerNorm(nn.Module):
    """LayerNorm with explicit zero-variance branch, rather than additive eps."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
    def forward(self, x):
        y = x.float() - x.float().mean(-1, keepdim=True)
        std = safe_norm(y, keepdim=True) / math.sqrt(x.shape[-1])
        y = divide_or_zero(y, std)
        return (y * self.weight.float() + self.bias.float()).to(x.dtype)

class VNActivation(nn.Module):
    """Learned half-space activation with leak 0.2; no fixed spatial bias."""
    def __init__(self, channels, leak=0.2):
        super().__init__()
        self.direction = VLinear(channels, channels)
        self.leak = leak
    def forward(self, v):
        vf = v.float()
        d = self.direction(vf)
        unit = divide_or_zero(d, safe_norm(d, keepdim=True))
        dot = (vf * unit).sum(-1, keepdim=True)
        y = vf - (1 - self.leak) * dot.clamp_max(0) * unit
        return y.to(v.dtype)

class VNRMSNorm(nn.Module):
    """One invariant RMS scale per point; no additive epsilon."""
    def __init__(self, channels):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))
    def forward(self, v):
        vf = v.float()
        rms = safe_norm(vf.flatten(-2), keepdim=True) / math.sqrt(v.shape[-2])
        y = divide_or_zero(vf, rms[..., None])
        return (y * self.gain.float()[..., None]).to(v.dtype)

class VNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = VLinear(channels, channels)
        self.norm = VNRMSNorm(channels)
        self.activation = VNActivation(channels)
    def forward(self, v):
        return self.activation(self.norm(self.linear(v)))

class VNMaxPool(nn.Module):
    """Select a whole vector using invariant learned-direction scores.
    Exact score ties are averaged, avoiding arbitrary neighbor tie selection.
    """
    def __init__(self, channels):
        super().__init__()
        self.direction = VLinear(channels, channels)
    def forward(self, vectors, mask):
        # vectors [N,K,C,3]; mask [N,K]
        vf = vectors.float()
        mean = divide_or_zero((vf * mask[..., None, None]).sum(1),
                              mask.sum(1)[:, None, None].float())
        direction = self.direction(mean)
        scores = (vf * direction[:, None]).sum(-1)
        scores = scores.masked_fill(~mask[..., None], -torch.inf)
        chosen = (scores == scores.amax(1, keepdim=True)) & mask[..., None]
        return divide_or_zero((vf * chosen[..., None]).sum(1),
                              chosen.sum(1)[..., None].float()).to(vectors.dtype)

@torch.no_grad()
def vector_geometry(rel, dist, mask):
    rel, dist = rel.float(), dist.float()
    rho = dist.masked_fill(~mask, 0).amax(1, keepdim=True)
    z = divide_or_zero(rel, rho[..., None]) * mask[..., None]
    mean = divide_or_zero(z.sum(1), mask.sum(1)[:, None].float())
    centered = (z - mean[:, None]) * mask[..., None]
    cov = divide_or_zero(torch.einsum('nki,nkj->nij', centered, centered),
                         mask.sum(1)[:, None, None].float())
    lam = torch.linalg.eigvalsh(cov).flip(-1).clamp_min(0)
    ell = divide_or_zero(lam, lam.sum(-1, keepdim=True))
    return dict(z=z, delta=safe_norm(z), rho=rho, ell=ell)

class GeometryEncoding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(9, width), nn.SiLU(),
                                 nn.Linear(width, width), ExactLayerNorm(width))
    def forward(self, geo, idx):
        n, k = idx.shape
        g = torch.cat([geo['delta'][..., None],
                       torch.log1p(geo['rho'])[:, None].expand(n,k,1),
                       torch.log1p(geo['delta'] * geo['rho'])[..., None],
                       geo['ell'][:, None].expand(n,k,3), geo['ell'][idx]], -1)
        return self.mlp(g)

class VectorStem(nn.Module):
    def __init__(self, scalar_channels, vector_channels, pos_width):
        super().__init__()
        self.key = nn.Linear(scalar_channels, pos_width)
        self.coef = nn.Sequential(nn.Linear(3*pos_width, pos_width), nn.SiLU(),
                                  nn.Linear(pos_width, vector_channels))
        self.block = VNBlock(vector_channels)
    def forward(self, feat, idx, pos, z, mask):
        h = self.key(feat)
        edge = torch.cat([h[:,None].expand(-1,idx.shape[1],-1),h[idx],pos],-1)
        coef = self.coef(edge)
        msg = coef[...,None].float() * z[:,:,None]
        pooled = divide_or_zero((msg * mask[...,None,None]).sum(1),
                                 mask.sum(1)[:,None,None].float())
        return self.block(pooled.to(feat.dtype))

class GeometryAttention(nn.Module):
    """PTv2 ordering: channel multiplier/bias BEFORE grouped weight encoding."""
    def __init__(self, channels, groups, pos_width, attention_type, vector_channels,
                 position_multiplier=True, position_bias=True, geometry_in_values=False,
                 chunk_size=256, checkpoint_edges=True, attn_drop=0.,
                 directional_conditioning=True):
        super().__init__()
        if attention_type not in ('grouped_vector','scalar'): raise ValueError('Invalid attention_type')
        if channels % groups or (vector_channels and vector_channels % groups): raise ValueError('Channel counts must divide groups')
        self.channels,self.groups,self.cv=channels,groups,vector_channels
        self.kind,self.chunk_size,self.checkpoint_edges=attention_type,chunk_size,checkpoint_edges
        self.directional=bool(directional_conditioning and vector_channels)
        self.qkv=nn.Linear(channels,3*channels,bias=False)
        xin=9+(2*vector_channels if self.directional else 0)
        def mlp(out): return nn.Sequential(nn.Linear(xin,pos_width),nn.SiLU(),nn.Linear(pos_width,out))
        self.multiplier=mlp(channels) if position_multiplier else None
        self.bias=mlp(channels) if position_bias else None
        self.pos_value=mlp(channels) if geometry_in_values else None
        if self.kind=='grouped_vector':
            self.group_weight=nn.Parameter(torch.randn(groups,channels//groups)/math.sqrt(channels//groups))
            self.weight_encoder=nn.Sequential(ExactLayerNorm(groups) if groups>1 else nn.Identity(),nn.SiLU(),nn.Linear(groups,groups))
        if vector_channels:
            if self.directional: self.vq,self.vk=VLinear(vector_channels,vector_channels),VLinear(vector_channels,vector_channels)
            self.vv=VLinear(vector_channels,vector_channels)
            self.vcoef=nn.Sequential(nn.Linear(2*channels+xin,pos_width),nn.SiLU(),nn.Linear(pos_width,2*vector_channels))
            self.vout=VLinear(vector_channels,vector_channels)
        self.dropout=nn.Dropout(attn_drop)
        self.out=nn.Linear(channels,channels)
    def _chunk(self,q,k,v,vq,vk,vv,feat_i,feat,idx,geo,mask,z):
        n,kc=idx.shape;g,d=self.groups,self.channels//self.groups
        xi=geo
        if self.directional:
            dq=(vq[:,None].float()*z[:,:,None].float()).sum(-1)
            dk=(vk[idx].float()*z[:,:,None].float()).sum(-1)
            xi=torch.cat([geo,dq.to(geo.dtype),dk.to(geo.dtype)],-1)
        qi=q.reshape(n,g,d)[:,None];kj=k[idx].reshape(n,kc,g,d)
        relation=qi-kj if self.kind=='grouped_vector' else qi*kj
        if self.multiplier is not None: relation=relation*self.multiplier(xi).reshape(n,kc,g,d)
        if self.bias is not None: relation=relation+self.bias(xi).reshape(n,kc,g,d)
        logits=self.weight_encoder((relation*self.group_weight).sum(-1)) if self.kind=='grouped_vector' else relation.sum(-1)/math.sqrt(d)
        weight=self.dropout(logits.float().masked_fill(~mask[...,None],-torch.inf).softmax(1))
        values=v[idx].reshape(n,kc,g,d)
        if self.pos_value is not None: values=values+self.pos_value(xi).reshape(n,kc,g,d)
        so=(weight[...,None].to(values.dtype)*values).sum(1).reshape(n,self.channels)
        if not self.cv: return so,None
        coef_in=torch.cat([feat_i[:,None].expand(-1,kc,-1),feat[idx],xi],-1)
        A,B=self.vcoef(coef_in).chunk(2,-1)
        msg=A[...,None].float()*vv[idx].float()+B[...,None].float()*z[:,:,None].float()
        vo=(weight.repeat_interleave(self.cv//g,-1)[...,None]*msg).sum(1).to(vv.dtype)
        return so,vo
    def forward(self,feat,vec,idx,pos,mask,z):
        q,k,v=self.qkv(feat).chunk(3,-1)
        vq=self.vq(vec) if self.directional else None
        vk=self.vk(vec) if self.directional else None
        vv=self.vv(vec) if self.cv else None
        ss=[];vs=[]
        for start in range(0,len(feat),self.chunk_size):
            end=min(start+self.chunk_size,len(feat))
            so,vo=_run(self._chunk,self.training and self.checkpoint_edges,
                       q[start:end],k,v,vq[start:end] if vq is not None else None,vk,vv,
                       feat[start:end],feat,idx[start:end],pos[start:end],mask[start:end],z[start:end])
            ss.append(so)
            if vo is not None: vs.append(vo)
        return self.out(torch.cat(ss)),self.vout(torch.cat(vs)) if self.cv else None

class Block(nn.Module):
    def __init__(self,channels,groups,pos_width,vector_channels,mlp_ratio=2.,proj_drop=0.,**attention):
        super().__init__()
        self.cv=vector_channels
        self.norm1=ExactLayerNorm(channels);self.norm2=ExactLayerNorm(channels)
        self.attn=GeometryAttention(channels,groups,pos_width,vector_channels=vector_channels,**attention)
        if self.cv:
            self.vnorm=VNRMSNorm(vector_channels);self.vblock=VNBlock(vector_channels)
            self.readout=VectorReadout(vector_channels)
            self.v2s=nn.Sequential(ExactLayerNorm(self.readout.out_dim),nn.Linear(self.readout.out_dim,channels))
        hidden=int(channels*mlp_ratio)
        self.ffn=nn.Sequential(nn.Linear(channels,hidden),nn.GELU(),nn.Dropout(proj_drop),nn.Linear(hidden,channels),nn.Dropout(proj_drop))
    def forward(self,feat,vec,idx,pos,mask,z):
        so,vo=self.attn(self.norm1(feat),self.vnorm(vec) if self.cv else None,idx,pos,mask,z)
        feat=feat+so
        if self.cv:
            vec=vec+self.vblock(vo)
            feat=feat+self.v2s(self.readout(vec))
        return feat+self.ffn(self.norm2(feat)),vec

@dataclass
class VState:
    coord: torch.Tensor
    feat: torch.Tensor
    vec: Optional[torch.Tensor]
    offset: torch.Tensor
    sample_index: torch.Tensor

class Stage(nn.Module):
    def __init__(self,channels,depth,groups,k,pos_width,vector_channels,initialize_vectors=False,**block_options):
        super().__init__()
        self.k=k;self.chunk_size=block_options['chunk_size']
        self.geometry=GeometryEncoding(pos_width) if initialize_vectors and vector_channels else None
        self.cv=vector_channels
        self.seed=VectorStem(channels,vector_channels,pos_width) if initialize_vectors and vector_channels else None
        self.blocks=nn.ModuleList([Block(channels,groups,pos_width,vector_channels,**block_options) for _ in range(depth)])
    def forward(self,state,cache=None,name='stage'):
        if cache is None or cache[0].shape[1]!=self.k:
            idx,rel,dist,mask=graph(state.coord,state.offset,self.k)
            geo=vector_geometry(rel,dist,mask)
            cache=(idx,mask,geo)
        idx,mask,geo=cache
        # Geometry features require query and source eigenvalues, so slice only outputs.
        n,k=idx.shape
        parts=[]
        for s in range(0,n,self.chunk_size):
            e=min(s+self.chunk_size,n)
            g=torch.cat([geo['delta'][s:e,...,None],torch.log1p(geo['rho'][s:e])[:,None].expand(-1,k,1),
                         torch.log1p(geo['delta'][s:e]*geo['rho'][s:e])[...,None],
                         geo['ell'][s:e,None].expand(-1,k,3),geo['ell'][idx[s:e]]],-1)
            parts.append(g)
        pos=torch.cat(parts)
        vec=state.vec
        if vec is None and self.cv:
            if self.seed is None: raise RuntimeError("Vector initialization must occur in first encoder stage")
            embedding=torch.cat([_run(self.geometry.mlp,self.training and self.blocks[0].attn.checkpoint_edges,pos[s:s+self.chunk_size]) for s in range(0,n,self.chunk_size)])
            vec=self._seed(state.feat,idx,embedding,geo['z'],mask)
        feat=state.feat
        for block in self.blocks: feat,vec=block(feat,vec,idx,pos,mask,geo['z'])
        return VState(state.coord,feat,vec,state.offset,state.sample_index),cache
    def _seed(self,feat,idx,pos,z,mask):
        # Query features and source features need separate indexing for chunking.
        h=self.seed.key(feat);out=[]
        for s in range(0,len(idx),self.chunk_size):
            e=min(s+self.chunk_size,len(idx))
            def run(hi,hall,ii,p,zz,mm):
                edge=torch.cat([hi[:,None].expand(-1,ii.shape[1],-1),hall[ii],p],-1)
                coef=self.seed.coef(edge)
                msg=coef[...,None].float()*zz[:,:,None]
                return self.seed.block(divide_or_zero((msg*mm[...,None,None]).sum(1),mm.sum(1)[:,None,None].float()).to(feat.dtype))
            out.append(_run(run,self.training and self.blocks[0].attn.checkpoint_edges,h[s:e],h,idx[s:e],pos[s:e],z[s:e],mask[s:e]))
        return torch.cat(out)
def _offset_to_counts(offset: torch.Tensor) -> torch.Tensor:
    offset = offset.long()
    return torch.diff(F.pad(offset, (1, 0), value=0))


def _counts_to_offset(counts: torch.Tensor, dtype=torch.int32) -> torch.Tensor:
    return torch.cumsum(counts.long(), dim=0).to(dtype)


def _batch_ranges(offset: torch.Tensor):
    start = 0
    for end in offset.long().tolist():
        yield start, end
        start = end


def _knn_fallback(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    k: int,
    new_xyz: Optional[torch.Tensor] = None,
    new_offset: Optional[torch.Tensor] = None,
    chunk_size: int = 2048,
):
    """
    Debug fallback. It avoids allocating a full global NxN matrix, but the
    arithmetic cost is still O(N*M). Use pointops for real training.
    """
    if new_xyz is None:
        new_xyz = xyz
        new_offset = offset
    assert new_offset is not None

    all_idx, all_dist = [], []
    for (s0, e0), (s1, e1) in zip(_batch_ranges(offset), _batch_ranges(new_offset)):
        src = xyz[s0:e0]
        qry = new_xyz[s1:e1]
        kk = min(k, src.shape[0])
        idx_parts, dist_parts = [], []

        for q0 in range(0, qry.shape[0], chunk_size):
            q = qry[q0 : q0 + chunk_size]
            d = torch.cdist(
                q.float(), src.float(), compute_mode="donot_use_mm_for_euclid_dist"
            )
            dist, idx = torch.topk(d, k=kk, dim=-1, largest=False, sorted=True)

            if kk < k:
                pad_n = k - kk
                idx = torch.cat([idx, idx[:, -1:].expand(-1, pad_n)], dim=1)
                dist = torch.cat([dist, dist[:, -1:].expand(-1, pad_n)], dim=1)

            idx_parts.append(idx + s0)
            dist_parts.append(dist)

        all_idx.append(torch.cat(idx_parts, 0))
        all_dist.append(torch.cat(dist_parts, 0))

    return torch.cat(all_idx, 0).long(), torch.cat(all_dist, 0).to(xyz.dtype)


def knn_query(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    k: int,
    new_xyz: Optional[torch.Tensor] = None,
    new_offset: Optional[torch.Tensor] = None,
):
    """Compatibility wrapper for common Pointcept/pointops API variants."""
    xyz = xyz.contiguous()
    offset_i = offset.to(torch.int32).contiguous()

    if new_xyz is None:
        new_xyz = xyz
        new_offset = offset
    new_xyz = new_xyz.contiguous()
    new_offset_i = new_offset.to(torch.int32).contiguous()

    if (
        pointops is None
        or not xyz.is_cuda
        or bool((_offset_to_counts(offset) < k).any())
    ):
        if xyz.is_cuda:
            _warn_once(
                "knn",
                "pointops kNN unavailable or a cloud has < k points: "
                "using slow torch.cdist fallback",
            )
        return _knn_fallback(xyz, offset, k, new_xyz, new_offset)

    errors = []

    # Newer Pointcept API is commonly:
    #   knn_query(k, xyz, offset, new_xyz=None, new_offset=None)
    if hasattr(pointops, "knn_query"):
        calls = [
            lambda: pointops.knn_query(k, xyz, offset_i, new_xyz, new_offset_i),
            lambda: pointops.knn_query(
                k, xyz, offset_i, new_xyz=new_xyz, new_offset=new_offset_i
            ),
            # Legacy argument order:
            lambda: pointops.knn_query(k, xyz, new_xyz, offset_i, new_offset_i),
        ]
        for call in calls:
            try:
                out = call()
                if isinstance(out, (tuple, list)):
                    idx, dist = out[0], out[1]
                else:
                    idx, dist = out, None
                idx = idx.long()
                if (idx < 0).any():
                    raise RuntimeError("pointops returned invalid neighbor indices")
                rel = xyz[idx] - new_xyz[:, None, :]
                dist = torch.linalg.vector_norm(rel, dim=-1)
                return idx, dist.to(xyz.dtype)
            except (TypeError, RuntimeError) as exc:
                errors.append(exc)

    # Older pointops package:
    #   knnquery(k, xyz, new_xyz, offset, new_offset)
    if hasattr(pointops, "knnquery"):
        try:
            idx, dist = pointops.knnquery(k, xyz, new_xyz, offset_i, new_offset_i)
            idx = idx.long()
            if (idx < 0).any():
                raise RuntimeError("pointops returned invalid neighbor indices")
            dist = torch.linalg.vector_norm(xyz[idx] - new_xyz[:, None, :], dim=-1)
            return idx, dist.to(xyz.dtype)
        except (TypeError, RuntimeError) as exc:
            errors.append(exc)

    raise RuntimeError(
        "A pointops module was found, but no supported k-NN API worked. "
        "Edit knn_query() for the pointops version installed in this Pointcept "
        f"environment. Last errors: {[str(e) for e in errors[-2:]]}"
    )


def _fps_fallback(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
):
    """Slow debug FPS. Use Pointcept CUDA pointops in training."""
    sampled = []
    new_counts = _offset_to_counts(new_offset).tolist()

    for (s, e), m in zip(_batch_ranges(offset), new_counts):
        pts = xyz[s:e]
        n = pts.shape[0]
        m = min(int(m), n)
        if m == n:
            sampled.append(torch.arange(s, e, device=xyz.device))
            continue

        # Deterministic initialization independent of orientation: first point.
        chosen = torch.empty(m, dtype=torch.long, device=xyz.device)
        chosen[0] = 0
        min_d2 = torch.full((n,), float("inf"), device=xyz.device)

        for t in range(1, m):
            p = pts[chosen[t - 1]]
            d2 = ((pts - p) ** 2).sum(-1)
            min_d2 = torch.minimum(min_d2, d2)
            chosen[t] = torch.argmax(min_d2)

        sampled.append(chosen + s)

    return torch.cat(sampled, 0)


def farthest_point_sampling(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
):
    xyz = xyz.contiguous()
    offset_i = offset.to(torch.int32).contiguous()
    new_offset_i = new_offset.to(torch.int32).contiguous()

    if pointops is None or not xyz.is_cuda:
        if xyz.is_cuda:
            _warn_once(
                "fps",
                "pointops FPS unavailable: using Python-loop FPS "
                "(extremely slow for >10k points). Install/compile pointops.",
            )
        return _fps_fallback(xyz, offset, new_offset)

    names = ("farthest_point_sampling", "furthestsampling")
    errors = []
    for name in names:
        if hasattr(pointops, name):
            try:
                return getattr(pointops, name)(xyz, offset_i, new_offset_i).long()
            except (TypeError, RuntimeError) as exc:
                errors.append(exc)

    raise RuntimeError(
        "A pointops module was found, but no supported FPS API worked. "
        "Edit farthest_point_sampling() for the installed pointops version. "
        f"Errors: {[str(e) for e in errors[-2:]]}"
    )


@dataclass
class State:
    coord: torch.Tensor
    feat: torch.Tensor
    offset: torch.Tensor
    sample_index: torch.Tensor


def _run(fn, enabled, *args):
    if enabled and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


@torch.no_grad()
def graph(source, offset, k, query=None, query_offset=None):
    """Distance graph, sorted by Euclidean distance; padded indices are masked.

    Geometry is treated as input data: no coordinate gradients are supported.
    Sorting/FPS can resolve exact ties differently across devices or permutations.
    """
    query = source if query is None else query
    query_offset = offset if query_offset is None else query_offset
    idx, _ = knn_query(source, offset, k, query, query_offset)
    rel = source[idx].float() - query[:, None].float()
    dist = torch.linalg.vector_norm(rel, dim=-1)
    order = dist.argsort(dim=1, stable=True)
    idx = idx.gather(1, order)
    dist = dist.gather(1, order)
    rel = rel.gather(1, order[..., None].expand(-1, -1, 3))
    # Mark only the first occurrence of a repeated source index. K is small.
    mask = torch.ones_like(idx, dtype=torch.bool)
    for j in range(1, idx.shape[1]):
        mask[:, j] = (idx[:, j : j + 1] != idx[:, :j]).all(1)
    return idx, rel, dist, mask


class VLinear(nn.Module):
    """Channel mixing of 3D vectors, no bias: V[N,Cin,3] -> [N,Cout,3]. Equivariant."""

    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin))
        nn.init.normal_(self.weight, std=cin**-0.5)

    def forward(self, v):
        return torch.einsum("oc,nci->noi", self.weight.to(v.dtype), v)


class VectorReadout(nn.Module):
    """Invariants: per-channel norms + Gram matrix of a small projection
    (+ optional triple products for handedness)."""

    def __init__(self, cv, proj=4, triple=False):
        super().__init__()
        self.proj = VLinear(cv, proj)
        self.triple = triple
        iu = torch.triu_indices(proj, proj)
        self.register_buffer("iu", iu, persistent=False)
        self.out_dim = cv + iu.shape[1] + (proj if triple else 0)

    def forward(self, v):
        norms = torch.linalg.vector_norm(v.float(), dim=-1)
        p = self.proj(v).float()
        gram = torch.einsum("nci,ndi->ncd", p, p)[:, self.iu[0], self.iu[1]]
        feats = [norms, gram]
        if self.triple:
            pr = p.roll(1, 1)
            feats.append((p * torch.cross(pr, p.roll(2, 1), dim=-1)).sum(-1))
        return torch.cat(feats, -1).to(v.dtype)


class Downsample(nn.Module):
    """FPS + scalar max pool + VN vector max pool (whole-vector selection) + centroid coord."""

    def __init__(
        self, cin, cout, ratio, k, chunk_size, checkpoint_edges, vector_channels, vector_pooling="max"
    ):
        super().__init__()
        self.ratio, self.k, self.chunk_size = ratio, k, chunk_size
        self.checkpoint_edges, self.cv = checkpoint_edges, vector_channels
        self.proj = nn.Sequential(ExactLayerNorm(cin), nn.Linear(cin, cout), nn.GELU())
        if vector_channels > 0:
            self.vproj = VNBlock(vector_channels)
            self.vector_pooling = vector_pooling
            self.vpool = VNMaxPool(vector_channels) if vector_pooling == "max" else None

    @staticmethod
    def _pool(feat, idx, mask):
        return feat[idx].masked_fill(~mask[..., None], float("-inf")).amax(1)

    def forward(self, state):
        counts = (
            torch.ceil(_offset_to_counts(state.offset).float() * self.ratio)
            .long()
            .clamp_min(1)
        )
        offset = _counts_to_offset(counts, state.offset.dtype)
        selected = farthest_point_sampling(state.coord, state.offset, offset)
        anchors = state.coord[selected]
        idx, _, _, mask = graph(state.coord, state.offset, self.k, anchors, offset)
        with torch.no_grad(), torch.autocast(
            device_type=state.coord.device.type, enabled=False
        ):
            valid = mask.to(torch.float32)[..., None]
            coord = divide_or_zero((state.coord[idx].float() * valid).sum(1), valid.sum(1))
        feat = self.proj(state.feat)
        pooled = []
        for s in range(0, len(idx), self.chunk_size):
            e = min(s + self.chunk_size, len(idx))
            pooled.append(
                _run(
                    self._pool,
                    self.training and self.checkpoint_edges,
                    feat,
                    idx[s:e],
                    mask[s:e],
                )
            )
        vec = None
        if self.cv > 0:
            vec = torch.cat([self.vpool(state.vec[idx[s:s+self.chunk_size]], mask[s:s+self.chunk_size]) if self.vpool is not None else divide_or_zero((state.vec[idx[s:s+self.chunk_size]] * mask[s:s+self.chunk_size,...,None,None]).sum(1), mask[s:s+self.chunk_size].sum(1)[:,None,None].to(state.vec.dtype))
                             for s in range(0,len(idx),self.chunk_size)])
            vec = self.vproj(vec)
        return VState(
            coord, torch.cat(pooled), vec, offset, state.sample_index[selected]
        )


def interp_weights(coarse, fine, k=3):
    idx, _, dist, mask = graph(coarse.coord, coarse.offset, k, fine.coord, fine.offset)
    zero = (dist == 0) & mask
    has_zero = zero.any(1, keepdim=True)
    inverse = divide_or_zero(mask.float(), dist)
    ordinary = divide_or_zero(inverse, inverse.sum(1, keepdim=True))
    exact = divide_or_zero(zero.float(), zero.sum(1, keepdim=True).float())
    return idx, torch.where(has_zero, exact, ordinary)



def interpolate(coarse, fine, feat, k=3, chunk_size=65536, weights=None):
    """Inverse-distance interpolation; works for [M,C] or [M,Cv,3] (weights invariant).
    Pass weights=(idx, w) from interp_weights() to avoid recomputing the kNN."""
    idx, weight = weights if weights is not None else interp_weights(coarse, fine, k)
    weight = weight.to(feat.dtype)
    extra = (1,) * (feat.ndim - 1)
    return torch.cat(
        [
            (
                weight[s : s + chunk_size].reshape(
                    *weight[s : s + chunk_size].shape, *extra
                )
                * feat[idx[s : s + chunk_size]]
            ).sum(1)
            for s in range(0, len(idx), chunk_size)
        ]
    )


class Upsample(nn.Module):
    def __init__(self, coarse, skip, out, k, chunk_size, vector_channels):
        super().__init__()
        self.k, self.chunk_size, self.cv = k, chunk_size, vector_channels
        self.coarse = nn.Linear(coarse, out)
        self.skip = nn.Linear(skip, out)
        self.fuse = nn.Sequential(nn.Linear(2 * out, out), ExactLayerNorm(out), nn.GELU())
        if vector_channels > 0:
            self.vfuse = VLinear(2 * vector_channels, vector_channels)

    def forward(self, coarse, fine):
        w = interp_weights(coarse, fine, self.k)  # one kNN for feat + vec
        up = interpolate(
            coarse, fine, self.coarse(coarse.feat), self.k, self.chunk_size, w
        )
        feat = self.fuse(torch.cat([up, self.skip(fine.feat)], -1))
        vec = None
        if self.cv > 0:
            vup = interpolate(coarse, fine, coarse.vec, self.k, self.chunk_size, w)
            vec = self.vfuse(torch.cat([vup, fine.vec], 1))
        return VState(fine.coord, feat, vec, fine.offset, fine.sample_index)


@MODELS.register_module("SO3-VN-PTv2-v1")
class SO3VectorNeuronNet(nn.Module):
    """Frame-free invariant scalar / equivariant vector encoder-decoder."""

    def __init__(
        self,
        in_channels=6,
        scalar_feature_indices=None,
        channels=(32, 64, 128, 256, 384),
        depths=(2, 2, 2, 4, 2),
        heads=(2, 4, 8, 16, 16),
        neighbors=(8, 8, 12, 12, 16),
        sample_ratios=(0.25, 0.25, 0.25, 0.25),
        head_channels=64,
        attention_type="grouped_vector",
        pos_channels=16,
        vector_channels=16,
        readout_triple=False,
        position_multiplier=True,
        position_bias=True,
        geometry_in_values=False,
        use_decoder=True,
        decoder_depths=None,
        mlp_ratio=2.0,
        attn_drop=0.0,
        proj_drop=0.0,
        interp_k=3,
        edge_chunk_size=16384,
        checkpoint_edges=True,
        share_geometry=True,
        directional_conditioning=True,
        vector_pooling="max",
    ):
        super().__init__()
        n = len(channels)
        self.share_geometry = share_geometry
        if n < 1 or any(len(x) != n for x in (depths, heads, neighbors)):
            raise ValueError(
                "channels, depths, heads, neighbors must have equal nonzero length"
            )
        if len(sample_ratios) != n - 1 or any(not 0 < r <= 1 for r in sample_ratios):
            raise ValueError("sample_ratios must have nstages-1 entries in (0,1]")
        if any(c % h for c, h in zip(channels, heads)):
            raise ValueError("every channels[i] must be divisible by heads[i]")
        if vector_pooling not in ("max", "mean"): raise ValueError("Invalid vector_pooling")
        if vector_channels < 0 or any(vector_channels % h for h in heads):
            raise ValueError("vector_channels must be nonnegative and divisible by all heads")
        if not position_bias and not position_multiplier and not geometry_in_values:
            raise ValueError("Enable at least one geometry path")
        if scalar_feature_indices is None:
            raise ValueError(
                "Set scalar_feature_indices explicitly: RGB-only (0,1,2); XYZRGB (3,4,5)"
            )
        indices = list(scalar_feature_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(i < 0 or i >= in_channels for i in indices)
        ):
            raise ValueError(
                "scalar_feature_indices must be distinct valid input channel indices"
            )
        self.register_buffer(
            "_scalar_indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.in_channels, self.head_channels = in_channels, head_channels
        self.channels, self.use_decoder, self.cv = (
            tuple(channels),
            use_decoder,
            vector_channels,
        )
        self.chunk_size, self.interp_k = edge_chunk_size, interp_k
        self.stem = nn.Sequential(
            nn.Linear(len(indices), channels[0]), ExactLayerNorm(channels[0]), nn.GELU()
        )
        block = dict(
            directional_conditioning=directional_conditioning,
            attention_type=attention_type,
            position_multiplier=position_multiplier,
            position_bias=position_bias,
            geometry_in_values=geometry_in_values,
            chunk_size=edge_chunk_size,
            checkpoint_edges=checkpoint_edges,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.stages = nn.ModuleList(
            [
                Stage(
                    channels[i],
                    depths[i],
                    heads[i],
                    neighbors[i],
                    pos_channels,
                    vector_channels,
                    initialize_vectors=(i == 0),
                    **block,
                )
                for i in range(n)
            ]
        )
        self.down = nn.ModuleList(
            [
                Downsample(
                    channels[i],
                    channels[i + 1],
                    sample_ratios[i],
                    neighbors[i],
                    edge_chunk_size,
                    checkpoint_edges,
                    vector_channels,
                    vector_pooling=vector_pooling,
                )
                for i in range(n - 1)
            ]
        )
        if use_decoder:
            dd = list(decoder_depths) if decoder_depths is not None else [1] * (n - 1)
            if len(dd) != n - 1 or any(not isinstance(x, int) or x < 1 for x in dd):
                raise ValueError("decoder_depths must have nstages-1 positive entries")
            self.up, self.decoder = nn.ModuleList(), nn.ModuleList()
            for i in reversed(range(n - 1)):
                self.up.append(
                    Upsample(
                        channels[i + 1],
                        channels[i],
                        channels[i],
                        interp_k,
                        edge_chunk_size,
                        vector_channels,
                    )
                )
                self.decoder.append(
                    Stage(
                        channels[i],
                        dd[i],
                        heads[i],
                        neighbors[i],
                        pos_channels,
                        vector_channels,
                            **block,
                    )
                )
            out_channels = channels[0]
        else:
            self.scale_proj = nn.ModuleList(
                [nn.Linear(c, head_channels) for c in channels]
            )
            self.scale_logits = nn.Parameter(torch.zeros(n))
            out_channels = head_channels
        if vector_channels > 0:
            self.final_readout = VectorReadout(vector_channels, triple=readout_triple)
            out_channels += self.final_readout.out_dim
        self.output = nn.Sequential(
            nn.Linear(out_channels, head_channels),
            ExactLayerNorm(head_channels),
            nn.GELU(),
        )

    def forward(self, data_dict, return_stages=False):
        point = Point(dict(data_dict))
        coord, feat, offset = point.coord.float(), point.feat, point.offset
        if coord.ndim != 2 or coord.shape[1] != 3 or not torch.isfinite(coord).all():
            raise ValueError("coord must be finite [N,3]")
        if (
            feat.ndim != 2
            or feat.shape != (len(coord), self.in_channels)
            or not feat.is_floating_point()
        ):
            raise ValueError(f"feat must be floating [N,{self.in_channels}]")
        if (
            offset.ndim != 1
            or len(offset) == 0
            or int(offset[-1]) != len(coord)
            or (_offset_to_counts(offset) <= 0).any()
        ):
            raise ValueError(
                "offset must contain cumulative positive point counts ending at N"
            )
        s = self.stem(feat.index_select(1, self._scalar_indices))
        # First stage initializes vectors using an equivariant relative-geometry stem.
        vec = None
        state = VState(
            coord, s, vec, offset, torch.arange(len(coord), device=coord.device)
        )
        stages, caches = [], []
        dev = coord.device
        for i, stage in enumerate(self.stages):
            state, cache = stage(state, name=f"enc{i}")
            stages.append(state)
            caches.append(cache if self.share_geometry else None)
            if i < len(self.down):
                with _timer(f"down{i}", dev):
                    state = self.down[i](state)
        if self.use_decoder:
            levels = list(reversed(range(len(stages) - 1)))
            for lvl, up, stage, skip in zip(
                levels, self.up, self.decoder, reversed(stages[:-1])
            ):
                with _timer(f"up{lvl}", dev):
                    state = up(state, skip)
                state, _ = stage(state, cache=caches[lvl], name=f"dec{lvl}")
            out, vout = state.feat, state.vec
        else:
            weights = self.scale_logits.softmax(0)
            out, vout = 0, 0
            for i, st in enumerate(stages):
                value = self.scale_proj[i](st.feat)
                v = st.vec
                if i:
                    w = interp_weights(st, stages[0], self.interp_k)
                    value = interpolate(
                        st, stages[0], value, self.interp_k, self.chunk_size, w
                    )
                    if self.cv > 0:
                        v = interpolate(
                            st, stages[0], v, self.interp_k, self.chunk_size, w
                        )
                out = out + weights[i] * value
                if self.cv > 0:
                    vout = vout + weights[i] * v
        if self.cv > 0:
            out = torch.cat([out, self.final_readout(vout)], -1)
        point.feat = self.output(out)
        if self.cv > 0:
            point["vector_feat"] = vout  # equivariant [N,Cv,3], for analysis
        return (point, stages) if return_stages else point


@MODELS.register_module("VNPTv2Segmentor")
class VNPTv2Segmentor(nn.Module):
    """Pointcept segmentor. Optional auxiliary encoder supervision; no paired pass.

    Logits are built from invariant features only. Previous backbone checkpoints are incompatible.
    Default loss comes from the caller's Pointcept criteria configuration.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone,
        criteria,
        ignore_index=-1,
        deep_supervision=False,
        aux_loss_weight=0.2,
        aux_stage_weights=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        if self.backbone.head_channels != backbone_out_channels:
            raise ValueError("backbone_out_channels must equal backbone.head_channels")
        if not math.isfinite(aux_loss_weight) or aux_loss_weight < 0:
            raise ValueError("aux_loss_weight must be finite and nonnegative")
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.aux_loss_weight = aux_loss_weight
        n = len(self.backbone.channels)
        weights = (
            tuple(aux_stage_weights) if aux_stage_weights is not None else (1.0,) * n
        )
        if len(weights) != n or any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError(
                "aux_stage_weights must have nstages nonnegative finite entries"
            )
        self.aux_weights = weights
        self.aux_heads = nn.ModuleDict()
        if deep_supervision and aux_loss_weight > 0:
            for i, (c, w) in enumerate(zip(self.backbone.channels, weights)):
                if w > 0:
                    self.aux_heads[str(i)] = nn.Linear(c, num_classes)
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def _loss(self, logits, target):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            if not (target != self.ignore_index).any():
                return logits.float().sum() * 0
            return self.criteria(logits.float(), target)

    def forward(self, input_dict, return_point=False):
        target = input_dict.get("segment")
        if target is not None:
            if target.ndim != 1 or target.dtype != torch.long:
                raise ValueError("segment must be a one-dimensional torch.long tensor")
            invalid = (target != self.ignore_index) & (
                (target < 0) | (target >= self.num_classes)
            )
            if invalid.any():
                raise ValueError(
                    "segment contains a label outside num_classes/ignore_index"
                )
        elif self.training:
            raise ValueError("Training requires segment labels")
        need = self.training and len(self.aux_heads) > 0
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            if need:
                point, stages = self.backbone(input_dict, return_stages=True)
            else:
                point = self.backbone(input_dict)
        logits = self.seg_head(point.feat)
        if target is not None and len(target) != len(logits):
            raise ValueError("segment and output point counts differ")
        if self.training:
            final = self._loss(logits, target)
            out = {"loss": final}
            if need:
                aux = final * 0
                for key, head in self.aux_heads.items():
                    i = int(key)
                    loss = self._loss(
                        head(stages[i].feat), target[stages[i].sample_index]
                    )
                    aux = aux + self.aux_weights[i] * loss
                    out[f"loss_aux_stage_{i}"] = loss.detach()
                out.update(loss_final=final.detach(), loss_aux=aux.detach())
                out["loss"] = final + self.aux_loss_weight * aux
        else:
            out = {"seg_logits": logits}
            if target is not None:
                out["loss"] = self._loss(logits, target)
        if return_point:
            out["point"] = point
        return out
