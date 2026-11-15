"""SO(3) scalar/vector segmentation with independent geometry attention.

Standalone reconstruction of the previously inspected Pointcept model, not a
byte-for-byte copy. Import this module from pointcept/models/__init__.py.

Architecture:
    feature grouped attention + geometry grouped attention + geometry bias
    FPS/kNN encoder, inverse-distance decoder, invariant scalar readout

Coordinates and all discrete neighborhood operations have no gradients.
Scalar inputs MUST be rotation-invariant (e.g. RGB, not raw XYZ/normals).
Symmetry is mathematical, subject to floating-point error and discrete ties.
New parameter shapes require training a new checkpoint.
"""
from __future__ import annotations

import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

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

_WARNED = set()


def warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def divide_or_zero(x, denominator):
    valid = denominator > 0
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    return x / safe * valid.to(x.dtype)


def safe_norm(x, dim=-1, keepdim=False):
    return torch.linalg.vector_norm(x.float(), dim=dim, keepdim=keepdim)


def run_checkpoint(fn, enabled, *args):
    if enabled and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class ExactLayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        y = x.float() - x.float().mean(-1, keepdim=True)
        std = safe_norm(y, keepdim=True) / math.sqrt(x.shape[-1])
        y = divide_or_zero(y, std)
        return (y * self.weight.float() + self.bias.float()).to(x.dtype)


class VLinear(nn.Module):
    """Channel mixing without a spatial bias: [N,Cin,3] -> [N,Cout,3]."""
    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin))
        nn.init.normal_(self.weight, std=cin ** -0.5)

    def forward(self, v):
        return torch.einsum('oc,nci->noi', self.weight.to(v.dtype), v)


class VNRMSNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))

    def forward(self, v):
        vf = v.float()
        rms = safe_norm(vf.flatten(-2), keepdim=True) / math.sqrt(v.shape[-2])
        return (divide_or_zero(vf, rms[..., None])
                * self.gain.float()[..., None]).to(v.dtype)


class VNActivation(nn.Module):
    def __init__(self, channels, leak=0.2):
        super().__init__()
        self.direction = VLinear(channels, channels)
        self.leak = leak

    def forward(self, v):
        vf = v.float()
        d = self.direction(vf)
        unit = divide_or_zero(d, safe_norm(d, keepdim=True))
        dot = (vf * unit).sum(-1, keepdim=True)
        return (vf - (1 - self.leak) * dot.clamp_max(0) * unit).to(v.dtype)


class VNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = VLinear(channels, channels)
        self.norm = VNRMSNorm(channels)
        self.activation = VNActivation(channels)

    def forward(self, v):
        return self.activation(self.norm(self.linear(v)))


class VectorReadout(nn.Module):
    def __init__(self, channels, proj=4, triple=False):
        super().__init__()
        self.proj = VLinear(channels, proj)
        self.triple = triple
        self.register_buffer('iu', torch.triu_indices(proj, proj), persistent=False)
        self.out_dim = channels + self.iu.shape[1] + (proj if triple else 0)

    def forward(self, v):
        p = self.proj(v).float()
        gram = torch.einsum('nci,ndi->ncd', p, p)
        values = [safe_norm(v), gram[:, self.iu[0], self.iu[1]]]
        if self.triple:
            values.append((p * torch.cross(p.roll(1, 1), p.roll(2, 1), dim=-1)).sum(-1))
        return torch.cat(values, -1).to(v.dtype)


class VNMaxPool(nn.Module):
    """Select whole vectors using invariant scores; average exact score ties."""
    def forward(self, v, mask):
        vf = v.float()
        direction = divide_or_zero(
            (vf * mask[..., None, None]).sum(1),
            mask.sum(1)[:, None, None].float())
        scores = (vf * direction[:, None]).sum(-1)
        scores = scores.masked_fill(~mask[..., None], -torch.inf)
        chosen = (scores == scores.amax(1, keepdim=True)) & mask[..., None]
        return divide_or_zero((vf * chosen[..., None]).sum(1),
                              chosen.sum(1)[..., None].float()).to(v.dtype)


def counts(offset):
    return torch.diff(F.pad(offset.long(), (1, 0), value=0))


def ranges(offset):
    start = 0
    for end in offset.long().tolist():
        yield start, end
        start = end


@torch.no_grad()
def knn_query(xyz, offset, k, query=None, query_offset=None, chunk_size=2048):
    """Pointops on CUDA; per-cloud brute-force fallback for unsupported inputs.

    The fallback is for debugging/small clouds, not scalable training.
    Returned padding is subsequently masked by graph().
    """
    query = xyz if query is None else query
    query_offset = offset if query_offset is None else query_offset
    if k < 1:
        raise ValueError('k must be positive')
    if xyz.is_cuda and pointops is not None and bool((counts(offset) >= k).all()):
        x, q = xyz.contiguous(), query.contiguous()
        o, qo = offset.int().contiguous(), query_offset.int().contiguous()
        if hasattr(pointops, 'knn_query'):
            # Current Pointcept API. Adapt here if your version differs.
            idx, _ = pointops.knn_query(k, x, o, q, qo)
            return idx.long()
        if hasattr(pointops, 'knnquery'):
            idx, _ = pointops.knnquery(k, x, q, o, qo)
            return idx.long()
    warn_once('knn_fallback', 'Using quadratic torch.cdist kNN fallback. '
              'Compile compatible Pointcept pointops for large-cloud training.')
    result = []
    for (s, e), (qs, qe) in zip(ranges(offset), ranges(query_offset)):
        kk = min(k, e - s)
        for start in range(qs, qe, chunk_size):
            d = torch.cdist(query[start:min(start + chunk_size, qe)].float(),
                            xyz[s:e].float(), compute_mode='donot_use_mm_for_euclid_dist')
            idx = d.topk(kk, largest=False, sorted=True).indices + s
            if kk < k:
                idx = torch.cat([idx, idx[:, -1:].expand(-1, k - kk)], 1)
            result.append(idx)
    return torch.cat(result)


@torch.no_grad()
def graph(source, offset, k, query=None, query_offset=None):
    query = source if query is None else query
    query_offset = offset if query_offset is None else query_offset
    with torch.autocast(device_type=source.device.type, enabled=False):
        idx = knn_query(source.float(), offset, k, query.float(), query_offset)
        if bool((idx < 0).any()):
            raise RuntimeError('Neighbor search returned invalid indices')
        rel = source[idx].float() - query.float()[:, None]
        dist = safe_norm(rel)
        order = dist.argsort(dim=1, stable=True)
        idx, dist = idx.gather(1, order), dist.gather(1, order)
        rel = rel.gather(1, order[..., None].expand(-1, -1, 3))
        mask = torch.ones_like(idx, dtype=torch.bool)
        for j in range(1, k):
            mask[:, j] = (idx[:, j:j + 1] != idx[:, :j]).all(1)
    return idx, rel, dist, mask


@torch.no_grad()
def farthest_point_sampling(xyz, offset, new_offset):
    if xyz.is_cuda and pointops is not None:
        for name in ('farthest_point_sampling', 'furthestsampling'):
            if hasattr(pointops, name):
                return getattr(pointops, name)(xyz.contiguous(), offset.int().contiguous(),
                                               new_offset.int().contiguous()).long()
    warn_once('fps_fallback', 'Using Python-loop FPS fallback; use CUDA pointops for training.')
    sampled = []
    for (s, e), m in zip(ranges(offset), counts(new_offset).tolist()):
        p = xyz[s:e].float()
        selected = torch.empty(m, device=xyz.device, dtype=torch.long)
        selected[0] = 0  # Matches the original fallback; order-dependent seed.
        nearest = torch.full((len(p),), torch.inf, device=xyz.device)
        used = torch.zeros(len(p), dtype=torch.bool, device=xyz.device)
        for t in range(1, m):
            used[selected[t - 1]] = True
            nearest = torch.minimum(nearest, ((p - p[selected[t - 1]]) ** 2).sum(-1))
            selected[t] = nearest.masked_fill(used, -torch.inf).argmax()
        sampled.append(selected + s)
    return torch.cat(sampled)


@torch.no_grad()
def vector_geometry(rel, dist, mask, chunk_size=4096):
    result = {key: [] for key in ('z', 'delta', 'rho', 'ell')}
    with torch.autocast(device_type=rel.device.type, enabled=False):
        for start in range(0, len(rel), chunk_size):
            r, d, valid = rel[start:start + chunk_size].float(), dist[start:start + chunk_size].float(), mask[start:start + chunk_size]
            count = valid.sum(1).float()
            rho = d.masked_fill(~valid, 0).amax(1, keepdim=True)
            z = divide_or_zero(r, rho[..., None]) * valid[..., None]
            mean = divide_or_zero(z.sum(1), count[:, None])
            centered = (z - mean[:, None]) * valid[..., None]
            cov = divide_or_zero(torch.einsum('nki,nkj->nij', centered, centered), count[:, None, None])
            cov = (0.5 * (cov + cov.transpose(-1, -2))).contiguous()
            if not bool(torch.isfinite(cov).all()):
                raise ValueError('Nonfinite geometry covariance')
            try:
                lam = torch.linalg.eigvalsh(cov)
            except RuntimeError:
                if not cov.is_cuda:
                    raise
                warn_once('eig_cpu', 'CUDA eigensolver failed; using CPU for this geometry chunk.')
                lam = torch.linalg.eigvalsh(cov.cpu()).to(cov.device)
            lam = lam.flip(-1).clamp_min(0)
            ell = divide_or_zero(lam, lam.sum(-1, keepdim=True))
            for key, value in (('z', z), ('delta', safe_norm(z)), ('rho', rho), ('ell', ell)):
                result[key].append(value)
    return {key: torch.cat(value) for key, value in result.items()}


def geometry_features(geo, idx):
    n, k = idx.shape
    return torch.cat([
        geo['delta'][..., None],
        torch.log1p(geo['rho'])[:, None].expand(n, k, 1),
        torch.log1p(geo['delta'] * geo['rho'])[..., None],
        geo['ell'][:, None].expand(n, k, 3),
        geo['ell'][idx],
    ], -1)


class GeometryAttention(nn.Module):
    """Independent feature/geometry softmax attention and geometry residual bias."""
    def __init__(self, channels, groups, pos_width, attention_type, vector_channels,
                 chunk_size=256, checkpoint_edges=True, attn_drop=0.):
        super().__init__()
        if attention_type not in ('grouped_vector', 'scalar'):
            raise ValueError('Invalid attention_type')
        if channels % groups or (vector_channels and vector_channels % groups):
            raise ValueError('Scalar and vector channel counts must divide groups')
        self.channels, self.groups, self.cv = channels, groups, vector_channels
        self.kind, self.chunk_size = attention_type, chunk_size
        self.checkpoint_edges = checkpoint_edges

        def mlp(out):
            return nn.Sequential(nn.Linear(9, pos_width), nn.SiLU(), nn.Linear(pos_width, out))

        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        if attention_type == 'grouped_vector':
            self.group_weight = nn.Parameter(torch.randn(groups, channels // groups) / math.sqrt(channels // groups))
            self.weight_encoder = nn.Sequential(
                ExactLayerNorm(groups) if groups > 1 else nn.Identity(),
                nn.SiLU(), nn.Linear(groups, groups))
        self.geometry_logits = mlp(groups)
        self.geometry_values = mlp(channels)
        self.geometry_bias = mlp(channels)
        self.out = nn.Linear(2 * channels, channels)
        self.dropout = nn.Dropout(attn_drop)
        if vector_channels:
            self.vv = VLinear(vector_channels, vector_channels)
            self.geometry_vector_coef = mlp(vector_channels)
            self.vout = VLinear(2 * vector_channels, vector_channels)

    def _attention(self, logits, mask):
        # Every neighborhood must contain at least one valid point.
        return self.dropout(logits.float().masked_fill(~mask[..., None], -torch.inf).softmax(1))

    def _chunk(self, q, k, v, vv, idx, geo, mask, z):
        n, neighbors = idx.shape
        g, width = self.groups, self.channels // self.groups
        qi = q.reshape(n, g, width)[:, None]
        kj = k[idx].reshape(n, neighbors, g, width)
        if self.kind == 'grouped_vector':
            logits = self.weight_encoder(((qi - kj) * self.group_weight).sum(-1))
        else:
            logits = (qi * kj).sum(-1) / math.sqrt(width)
        wf = self._attention(logits, mask)
        wg = self._attention(self.geometry_logits(geo), mask)
        vf = v[idx].reshape(n, neighbors, g, width)
        vg = self.geometry_values(geo).reshape(n, neighbors, g, width)
        of = (wf[..., None].to(vf.dtype) * vf).sum(1).reshape(n, self.channels)
        og = (wg[..., None].to(vg.dtype) * vg).sum(1).reshape(n, self.channels)
        summary = divide_or_zero((geo.float() * mask[..., None]).sum(1), mask.sum(1, keepdim=True).float())
        scalar = self.out(torch.cat([of, og], -1)) + self.geometry_bias(summary.to(geo.dtype))
        if not self.cv:
            return scalar, None
        wvf = wf.repeat_interleave(self.cv // g, -1)
        wvg = wg.repeat_interleave(self.cv // g, -1)
        ovf = (wvf[..., None] * vv[idx].float()).sum(1)
        geometry_vectors = self.geometry_vector_coef(geo).float()[..., None] * z.float()[:, :, None]
        ovg = (wvg[..., None] * geometry_vectors).sum(1)
        return scalar, self.vout(torch.cat([ovf, ovg], 1).to(vv.dtype))

    def forward(self, feat, vec, idx, pos, mask, z):
        q, k, v = self.qkv(feat).chunk(3, -1)
        vv = self.vv(vec) if self.cv else None
        scalars, vectors = [], []
        for s in range(0, len(feat), self.chunk_size):
            e = s + self.chunk_size
            so, vo = run_checkpoint(self._chunk, self.training and self.checkpoint_edges,
                                    q[s:e], k, v, vv, idx[s:e], pos[s:e], mask[s:e], z[s:e])
            scalars.append(so)
            if vo is not None:
                vectors.append(vo)
        return torch.cat(scalars), torch.cat(vectors) if self.cv else None


class Block(nn.Module):
    def __init__(self, channels, groups, pos_width, vector_channels, mlp_ratio=2., proj_drop=0., **attention):
        super().__init__()
        self.cv = vector_channels
        self.norm1, self.norm2 = ExactLayerNorm(channels), ExactLayerNorm(channels)
        self.attn = GeometryAttention(channels, groups, pos_width, vector_channels=vector_channels, **attention)
        if self.cv:
            self.vnorm, self.vblock = VNRMSNorm(self.cv), VNBlock(self.cv)
            self.readout = VectorReadout(self.cv)
            self.v2s = nn.Sequential(ExactLayerNorm(self.readout.out_dim), nn.Linear(self.readout.out_dim, channels))
        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(proj_drop),
                                 nn.Linear(hidden, channels), nn.Dropout(proj_drop))

    def forward(self, feat, vec, idx, pos, mask, z):
        so, vo = self.attn(self.norm1(feat), self.vnorm(vec) if self.cv else None, idx, pos, mask, z)
        feat = feat + so
        if self.cv:
            vec = vec + self.vblock(vo)
            feat = feat + self.v2s(self.readout(vec))
        return feat + self.ffn(self.norm2(feat)), vec


@dataclass
class VState:
    coord: torch.Tensor
    feat: torch.Tensor
    vec: Optional[torch.Tensor]
    offset: torch.Tensor
    sample_index: torch.Tensor


class Stage(nn.Module):
    def __init__(self, channels, depth, groups, k, pos_width, vector_channels,
                 initialize_vectors=False, **options):
        super().__init__()
        self.k, self.cv = k, vector_channels
        self.chunk_size = options['chunk_size']
        self.checkpoint_edges = options['checkpoint_edges']
        if initialize_vectors and self.cv:
            self.seed_key = nn.Linear(channels, pos_width)
            self.seed_geometry = nn.Sequential(nn.Linear(9, pos_width), nn.SiLU(),
                                                nn.Linear(pos_width, pos_width), ExactLayerNorm(pos_width))
            self.seed_coef = nn.Sequential(nn.Linear(3 * pos_width, pos_width), nn.SiLU(),
                                           nn.Linear(pos_width, self.cv))
            self.seed_block = VNBlock(self.cv)
        self.blocks = nn.ModuleList([Block(channels, groups, pos_width, self.cv, **options) for _ in range(depth)])

    def _seed_chunk(self, hi, hall, idx, pos, z, mask):
        embedding = self.seed_geometry(pos)
        edge = torch.cat([hi[:, None].expand(-1, idx.shape[1], -1), hall[idx], embedding], -1)
        message = self.seed_coef(edge).float()[..., None] * z[:, :, None].float()
        pooled = divide_or_zero((message * mask[..., None, None]).sum(1), mask.sum(1)[:, None, None].float())
        return self.seed_block(pooled.to(hi.dtype))

    def forward(self, state, cache=None):
        if cache is None:
            idx, rel, dist, mask = graph(state.coord, state.offset, self.k)
            geo = vector_geometry(rel, dist, mask)
            cache = idx, mask, geo
        idx, mask, geo = cache
        pos = geometry_features(geo, idx)
        feat, vec = state.feat, state.vec
        if vec is None and self.cv:
            if not hasattr(self, 'seed_key'):
                raise RuntimeError('Vectors must be initialized in the first stage')
            h = self.seed_key(feat)
            vec = torch.cat([run_checkpoint(self._seed_chunk, self.training and self.checkpoint_edges,
                                            h[s:s + self.chunk_size], h, idx[s:s + self.chunk_size],
                                            pos[s:s + self.chunk_size], geo['z'][s:s + self.chunk_size],
                                            mask[s:s + self.chunk_size])
                             for s in range(0, len(feat), self.chunk_size)])
        for block in self.blocks:
            feat, vec = block(feat, vec, idx, pos, mask, geo['z'])
        return VState(state.coord, feat, vec, state.offset, state.sample_index), cache


class Downsample(nn.Module):
    def __init__(self, cin, cout, ratio, k, vector_channels, chunk_size, vector_pooling='max'):
        super().__init__()
        self.ratio, self.k, self.cv, self.chunk_size = ratio, k, vector_channels, chunk_size
        self.proj = nn.Sequential(ExactLayerNorm(cin), nn.Linear(cin, cout), nn.GELU())
        self.vector_pooling = vector_pooling
        if self.cv:
            self.vpool, self.vproj = VNMaxPool(), VNBlock(self.cv)

    def forward(self, state):
        n = (counts(state.offset).float() * self.ratio).ceil().long().clamp_min(1)
        offset = n.cumsum(0).to(state.offset.dtype)
        selected = farthest_point_sampling(state.coord, state.offset, offset)
        idx, _, _, mask = graph(state.coord, state.offset, self.k, state.coord[selected], offset)
        with torch.no_grad():
            coord = divide_or_zero((state.coord[idx].float() * mask[..., None]).sum(1),
                                   mask.sum(1, keepdim=True).float())
        feat, pooled, vectors = self.proj(state.feat), [], []
        for s in range(0, len(idx), self.chunk_size):
            ii, mm = idx[s:s + self.chunk_size], mask[s:s + self.chunk_size]
            pooled.append(feat[ii].masked_fill(~mm[..., None], -torch.inf).amax(1))
            if self.cv:
                v = state.vec[ii]
                vectors.append(self.vpool(v, mm) if self.vector_pooling == 'max' else
                               divide_or_zero((v.float() * mm[..., None, None]).sum(1),
                                              mm.sum(1)[:, None, None].float()).to(v.dtype))
        vec = self.vproj(torch.cat(vectors)) if self.cv else None
        return VState(coord, torch.cat(pooled), vec, offset, state.sample_index[selected])


@torch.no_grad()
def interp_weights(coarse, fine, k=3):
    idx, _, dist, mask = graph(coarse.coord, coarse.offset, k, fine.coord, fine.offset)
    zero = (dist == 0) & mask
    inverse = divide_or_zero(mask.float(), dist)
    ordinary = divide_or_zero(inverse, inverse.sum(1, keepdim=True))
    exact = divide_or_zero(zero.float(), zero.sum(1, keepdim=True).float())
    return idx, torch.where(zero.any(1, keepdim=True), exact, ordinary)


def interpolate(feat, weights, chunk_size):
    idx, weight = weights
    extra = (1,) * (feat.ndim - 1)
    return torch.cat([(weight[s:s + chunk_size].to(feat.dtype).reshape(
        *weight[s:s + chunk_size].shape, *extra) * feat[idx[s:s + chunk_size]]).sum(1)
        for s in range(0, len(idx), chunk_size)])


class Upsample(nn.Module):
    def __init__(self, coarse, skip, out, k, chunk_size, vector_channels):
        super().__init__()
        self.k, self.chunk_size, self.cv = k, chunk_size, vector_channels
        self.coarse, self.skip = nn.Linear(coarse, out), nn.Linear(skip, out)
        self.fuse = nn.Sequential(nn.Linear(2 * out, out), ExactLayerNorm(out), nn.GELU())
        if self.cv:
            self.vfuse = VLinear(2 * self.cv, self.cv)

    def forward(self, coarse, fine):
        weights = interp_weights(coarse, fine, self.k)
        up = interpolate(self.coarse(coarse.feat), weights, self.chunk_size)
        feat = self.fuse(torch.cat([up, self.skip(fine.feat)], -1))
        vec = None
        if self.cv:
            vup = interpolate(coarse.vec, weights, self.chunk_size)
            vec = self.vfuse(torch.cat([vup, fine.vec], 1))
        return VState(fine.coord, feat, vec, fine.offset, fine.sample_index)


@MODELS.register_module('SO3-VN-PTv2-Independent-v1')
class SO3VectorNeuronNet(nn.Module):
    def __init__(self, in_channels=6, scalar_feature_indices=None,
                 channels=(32, 64, 128, 256, 384), depths=(2, 2, 2, 4, 2),
                 heads=(2, 4, 8, 16, 16), neighbors=(8, 8, 12, 12, 16),
                 sample_ratios=(0.25, 0.25, 0.25, 0.25), head_channels=64,
                 attention_type='grouped_vector', pos_channels=16, vector_channels=16,
                 readout_triple=False, use_decoder=True, decoder_depths=None,
                 mlp_ratio=2., attn_drop=0., proj_drop=0., interp_k=3,
                 edge_chunk_size=16384, checkpoint_edges=True,
                 share_geometry=True, vector_pooling='max'):
        super().__init__()
        n = len(channels)
        if n < 1 or any(len(x) != n for x in (depths, heads, neighbors)):
            raise ValueError('channels/depths/heads/neighbors must have equal nonzero length')
        if any(c < 1 for c in channels) or any(d < 1 for d in depths) or any(h < 1 for h in heads) or any(k < 1 for k in neighbors):
            raise ValueError('Stage sizes must be positive')
        if len(sample_ratios) != n - 1 or any(not 0 < r <= 1 for r in sample_ratios):
            raise ValueError('sample_ratios must contain nstages-1 ratios in (0,1]')
        if vector_channels < 0 or any(c % h or vector_channels % h for c, h in zip(channels, heads)):
            raise ValueError('Scalar/vector channels must divide every corresponding group count')
        if vector_pooling not in ('max', 'mean'):
            raise ValueError('vector_pooling must be max or mean')
        if edge_chunk_size < 1 or interp_k < 1:
            raise ValueError('Chunk size and interpolation k must be positive')
        if scalar_feature_indices is None:
            raise ValueError('Set invariant scalar_feature_indices: RGB (0,1,2), XYZRGB (3,4,5)')
        indices = list(scalar_feature_indices)
        if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= in_channels for i in indices):
            raise ValueError('Invalid scalar_feature_indices')
        self.register_buffer('_scalar_indices', torch.tensor(indices, dtype=torch.long), persistent=False)
        self.in_channels, self.head_channels = in_channels, head_channels
        self.channels, self.cv, self.use_decoder = tuple(channels), vector_channels, use_decoder
        self.chunk_size, self.interp_k, self.share_geometry = edge_chunk_size, interp_k, share_geometry
        self.stem = nn.Sequential(nn.Linear(len(indices), channels[0]), ExactLayerNorm(channels[0]), nn.GELU())
        options = dict(attention_type=attention_type, chunk_size=edge_chunk_size,
                       checkpoint_edges=checkpoint_edges, attn_drop=attn_drop,
                       proj_drop=proj_drop, mlp_ratio=mlp_ratio)
        self.stages = nn.ModuleList([Stage(channels[i], depths[i], heads[i], neighbors[i],
            pos_channels, self.cv, initialize_vectors=(i == 0), **options) for i in range(n)])
        self.down = nn.ModuleList([Downsample(channels[i], channels[i + 1], sample_ratios[i],
            neighbors[i], self.cv, edge_chunk_size, vector_pooling) for i in range(n - 1)])
        if use_decoder:
            dd = list(decoder_depths) if decoder_depths is not None else [1] * (n - 1)
            if len(dd) != n - 1 or any(d < 1 for d in dd):
                raise ValueError('decoder_depths must contain nstages-1 positive depths')
            self.up, self.decoder = nn.ModuleList(), nn.ModuleList()
            for i in reversed(range(n - 1)):
                self.up.append(Upsample(channels[i + 1], channels[i], channels[i], interp_k, edge_chunk_size, self.cv))
                self.decoder.append(Stage(channels[i], dd[i], heads[i], neighbors[i], pos_channels, self.cv, **options))
            out_channels = channels[0]
        else:
            self.scale_proj = nn.ModuleList([nn.Linear(c, head_channels) for c in channels])
            self.scale_logits = nn.Parameter(torch.zeros(n))
            out_channels = head_channels
        if self.cv:
            self.final_readout = VectorReadout(self.cv, triple=readout_triple)
            out_channels += self.final_readout.out_dim
        self.output = nn.Sequential(nn.Linear(out_channels, head_channels), ExactLayerNorm(head_channels), nn.GELU())

    def forward(self, data_dict, return_stages=False):
        point = Point(dict(data_dict))
        coord, feat, offset = point.coord.float(), point.feat, point.offset
        if coord.ndim != 2 or coord.shape[1] != 3 or not bool(torch.isfinite(coord).all()):
            raise ValueError('coord must be finite [N,3]')
        if feat.shape != (len(coord), self.in_channels) or not feat.is_floating_point():
            raise ValueError(f'feat must be floating [N,{self.in_channels}]')
        if offset.ndim != 1 or not len(offset) or int(offset[-1]) != len(coord) or bool((counts(offset) <= 0).any()):
            raise ValueError('offset must contain cumulative positive cloud sizes ending at N')
        state = VState(coord, self.stem(feat.index_select(1, self._scalar_indices)), None,
                       offset, torch.arange(len(coord), device=coord.device))
        stages, caches = [], []
        for i, stage in enumerate(self.stages):
            state, cache = stage(state)
            stages.append(state)
            caches.append(cache if self.share_geometry else None)
            if i < len(self.down):
                state = self.down[i](state)
        if self.use_decoder:
            for lvl, up, stage in zip(reversed(range(len(stages) - 1)), self.up, self.decoder):
                state = up(state, stages[lvl])
                state, _ = stage(state, caches[lvl])
            out, vout = state.feat, state.vec
        else:
            out, vout = 0, 0
            for i, (st, weight) in enumerate(zip(stages, self.scale_logits.softmax(0))):
                value, vector = self.scale_proj[i](st.feat), st.vec
                if i:
                    weights = interp_weights(st, stages[0], self.interp_k)
                    value = interpolate(value, weights, self.chunk_size)
                    if self.cv:
                        vector = interpolate(vector, weights, self.chunk_size)
                out = out + weight * value
                if self.cv:
                    vout = vout + weight * vector
        if self.cv:
            out = torch.cat([out, self.final_readout(vout)], -1)
            point['vector_feat'] = vout
        point.feat = self.output(out)
        return (point, stages) if return_stages else point


@MODELS.register_module('VNPTv2IndependentSegmentor')
class VNPTv2Segmentor(nn.Module):
    def __init__(self, num_classes, backbone_out_channels, backbone, criteria,
                 ignore_index=-1, deep_supervision=False, aux_loss_weight=0.2,
                 aux_stage_weights=None, freeze_backbone=False):
        super().__init__()
        self.backbone = build_model(backbone)
        if self.backbone.head_channels != backbone_out_channels:
            raise ValueError('backbone_out_channels must equal backbone.head_channels')
        if not math.isfinite(aux_loss_weight) or aux_loss_weight < 0:
            raise ValueError('aux_loss_weight must be finite and nonnegative')
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.criteria = build_criteria(criteria)
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.freeze_backbone = freeze_backbone
        self.aux_loss_weight = aux_loss_weight
        n = len(self.backbone.channels)
        self.aux_weights = tuple(aux_stage_weights) if aux_stage_weights is not None else (1.,) * n
        if len(self.aux_weights) != n or any(not math.isfinite(w) or w < 0 for w in self.aux_weights):
            raise ValueError('Invalid auxiliary stage weights')
        self.aux_heads = nn.ModuleDict()
        if deep_supervision and aux_loss_weight > 0:
            for i, (c, w) in enumerate(zip(self.backbone.channels, self.aux_weights)):
                if w > 0:
                    self.aux_heads[str(i)] = nn.Linear(c, num_classes)
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def _loss(self, logits, target):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            if not bool((target != self.ignore_index).any()):
                return logits.float().sum() * 0
            return self.criteria(logits.float(), target)

    def forward(self, input_dict, return_point=False):
        target = input_dict.get('segment')
        if self.training and target is None:
            raise ValueError('Training requires segment labels')
        if target is not None:
            if target.ndim != 1 or target.dtype != torch.long:
                raise ValueError('segment must be a one-dimensional long tensor')
            invalid = (target != self.ignore_index) & ((target < 0) | (target >= self.num_classes))
            if bool(invalid.any()):
                raise ValueError('Invalid segment labels')
        need = self.training and bool(len(self.aux_heads))
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            result = self.backbone(input_dict, return_stages=need)
        point, stages = result if need else (result, None)
        logits = self.seg_head(point.feat)
        if target is not None and len(target) != len(logits):
            raise ValueError('segment and output point counts differ')
        if self.training:
            final = self._loss(logits, target)
            out = {'loss': final}
            if need:
                aux = final * 0
                for key, head in self.aux_heads.items():
                    i = int(key)
                    loss = self._loss(head(stages[i].feat), target[stages[i].sample_index])
                    aux = aux + self.aux_weights[i] * loss
                    out[f'loss_aux_stage_{i}'] = loss.detach()
                out.update(loss=final + self.aux_loss_weight * aux,
                           loss_final=final.detach(), loss_aux=aux.detach())
        else:
            out = {'seg_logits': logits}
            if target is not None:
                out['loss'] = self._loss(logits, target)
        if return_point:
            out['point'] = point
        return out
