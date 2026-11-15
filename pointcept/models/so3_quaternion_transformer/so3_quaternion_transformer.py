"""
Joint Feature-Geometry SO(3) Quaternion Transport Network (replacement v2).

Install as pointcept/models/so3_quaternion/so3_quaternion_transformer.py,
replacing the old file. Keep its package __init__.py import. Both registry
names SO3-QuatTransport-v1 and SO3-QuatTransport-v2 resolve to this new design;
DefaultSegmentorV3SO3 retains deep supervision and paired consistency losses.
Do not import the old and new files together (duplicate registry entries).

TRAIN FROM SCRATCH: old checkpoints are not architecture compatible.

Input contract: scalar_feature_indices must select ONLY invariant attributes,
e.g. RGB. For feat=[x,y,z,r,g,b], use in_channels=6 and indices=(3,4,5).
For feat=[r,g,b], use in_channels=3 and indices=(0,1,2). Vector feature groups
may contain normals, which must rotate with coord. Never treat XYZ or fixed
world-z height as invariant scalars. The model does not infer channel semantics.

New operations:
* metric + relative distance RBFs and covariance contractions;
* feature-conditioned geometric stem, pooling, quaternion angles and values;
* direct invariant edge messages into scalar values;
* transported vector values plus direction and covariance-direction messages;
* learned vector-channel dot products in invariant readout;
* geometry cached per stage and query-chunked edge attention.

geometry_scales are positive distance scales in the SAME UNITS as coord.
Default scales assume meter coordinates: (0.1,0.2,0.4,0.8,1.6).
edge_chunk_size controls temporary edge activations (default 1024); autograd
still retains training activations across chunks. This is not a hard memory cap.
Decoder defaults to enabled. Explicit use_decoder=False remains supported.
Full SO(3) behavior applies to fixed input point sets; coordinate-axis voxel
preprocessing, sampling ties and numerical roundoff can change neighborhoods.
CUDA pointops is required for practical large-cloud training. CPU fallback is
for small correctness tests. No S3DIS accuracy improvement has been measured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import nullcontext
from functools import partial
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point

try:
    import pointops
except ImportError:
    pointops = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

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


def _safe_unit(x: torch.Tensor, eps: float = 1e-8):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / norm.clamp_min(eps), norm.squeeze(-1)


def _local_normalized_distance(dist: torch.Tensor, eps: float = 1e-6):
    # kNN distance ordering is normally ascending. max() is robust to backends
    # that do not guarantee ordering.
    scale = dist.max(dim=-1, keepdim=True).values.clamp_min(eps)
    return dist / scale


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
            q = qry[q0:q0 + chunk_size]
            d = torch.cdist(q.float(), src.float(), compute_mode="donot_use_mm_for_euclid_dist")
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

    if pointops is None or not xyz.is_cuda or bool((_offset_to_counts(offset) < k).any()):
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
            idx, dist = pointops.knnquery(
                k, xyz, new_xyz, offset_i, new_offset_i
            )
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
        return _fps_fallback(xyz, offset, new_offset)

    names = ("farthest_point_sampling", "furthestsampling")
    errors = []
    for name in names:
        if hasattr(pointops, name):
            try:
                return getattr(pointops, name)(
                    xyz, offset_i, new_offset_i
                ).long()
            except (TypeError, RuntimeError) as exc:
                errors.append(exc)

    raise RuntimeError(
        "A pointops module was found, but no supported FPS API worked. "
        "Edit farthest_point_sampling() for the installed pointops version. "
        f"Errors: {[str(e) for e in errors[-2:]]}"
    )


# ---------------------------------------------------------------------------
# SO(3) feature primitives
# ---------------------------------------------------------------------------

class EquivariantLinear(nn.Module):
    """
    Linear map over vector CHANNELS only.

    Input:  [..., Cin, 3]
    Output: [..., Cout, 3]

    No vector bias is allowed because a fixed learned 3-vector would break
    rotation equivariance.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oi,...ic->...oc", self.weight, v)


class VectorRMSNorm(nn.Module):
    """SO(3)-equivariant normalization using invariant vector norms."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # RMS across vector channels, invariant to global 3D rotation.
        rms = torch.sqrt(
            v.square().sum(-1).mean(-1, keepdim=True) + self.eps
        )  # [..., 1]
        return (
            v / rms.unsqueeze(-1)
            * self.weight.view(*([1] * (v.ndim - 2)), -1, 1)
        )


class VectorGatedSiLU(nn.Module):
    """
    Equivariant vector activation:
        v' = gate(invariant quantities) * v
    """
    def __init__(
        self,
        vector_channels: int,
        scalar_channels: Optional[int] = None,
        hidden_ratio: float = 1.0,
    ):
        super().__init__()
        in_dim = vector_channels + (scalar_channels or 0)
        hidden = max(vector_channels, int(vector_channels * hidden_ratio))
        self.scalar_channels = scalar_channels
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, vector_channels),
        )

    def forward(
        self,
        v: torch.Tensor,
        scalar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = torch.linalg.vector_norm(v, dim=-1)
        x = n if scalar is None else torch.cat([n, scalar], dim=-1)
        g = F.silu(self.gate(x))
        return v * g.unsqueeze(-1)


class ScalarFFN(nn.Module):
    def __init__(self, channels: int, ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(channels * ratio)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, channels),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class VectorFFN(nn.Module):
    def __init__(
        self,
        vector_channels: int,
        scalar_channels: int,
        ratio: float = 2.0,
    ):
        super().__init__()
        hidden = max(vector_channels, int(vector_channels * ratio))
        self.fc1 = EquivariantLinear(vector_channels, hidden)
        self.act = VectorGatedSiLU(hidden, scalar_channels)
        self.fc2 = EquivariantLinear(hidden, vector_channels)

    def forward(self, v, scalar):
        x = self.fc1(v)
        x = self.act(x, scalar)
        return self.fc2(x)


class GaussianRBF(nn.Module):
    """
    Gaussian RBF on locally normalized distance. Since normalization uses only
    Euclidean distances, the encoding is SO(3)-invariant.
    """
    def __init__(self, num_basis: int = 16, max_value: float = 1.0):
        super().__init__()
        centers = torch.linspace(0.0, max_value, num_basis)
        self.register_buffer("centers", centers)
        delta = max(max_value / max(num_basis - 1, 1), 1e-3)
        self.gamma = 1.0 / (delta * delta)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.gamma * (d.unsqueeze(-1) - self.centers) ** 2
        )


def quaternion_rotate(
    v: torch.Tensor,
    qw: torch.Tensor,
    qv: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate vectors with a unit quaternion.

    v:  [..., Dv, 3]
    qw: [..., 1]     or broadcastable to [..., Dv, 1]
    qv: [..., 3]     or broadcastable to [..., Dv, 3]

    Formula:
        v' = v + 2 q_vec x (q_vec x v + q_w v)
    """
    while qw.ndim < v.ndim:
        qw = qw.unsqueeze(-2)
    while qv.ndim < v.ndim:
        qv = qv.unsqueeze(-2)

    # Geometry is float32 while channel projections may be fp16/bfloat16 under
    # autocast. torch.cross requires matching dtypes and does not promote them.
    dtype = torch.promote_types(torch.promote_types(v.dtype, qv.dtype), qw.dtype)
    v, qw, qv = v.to(dtype), qw.to(dtype), qv.to(dtype)
    t = 2.0 * torch.cross(qv.expand_as(v), v, dim=-1)
    return v + qw * t + torch.cross(qv.expand_as(v), t, dim=-1)


# ---------------------------------------------------------------------------
# Feature state
# ---------------------------------------------------------------------------

@dataclass
class SO3State:
    coord: torch.Tensor       # [N, 3]
    scalar: torch.Tensor      # [N, Cs]
    vector: torch.Tensor      # [N, Cv, 3]
    offset: torch.Tensor      # [B]
    sample_index: Optional[torch.Tensor] = None  # rows in ORIGINAL input batch
    pool_index: Optional[torch.Tensor] = None    # rows in preceding stage
    geometry: Optional[dict] = None  # cached only for the current local graph


# ---------------------------------------------------------------------------
# Input embedding
# ---------------------------------------------------------------------------

def local_geometry(coord, idx, rbf, metric_scale, query=None, source_cov=None):
    """Invariant descriptors and equivariant covariance, computed in FP32.

    Uses second moments about each query (not a signed normal/eigenframe).
    A zero-length/self edge is retained in attention but gives zero direction.
    """
    with torch.autocast(device_type=coord.device.type, enabled=False):
        coord = coord.float()
        query = coord if query is None else query.float()
        rel = coord[idx] - query[:, None, :]
        unit, dist = _safe_unit(rel)
        radius = dist.amax(1, keepdim=True).clamp_min(1e-6)
        cov = torch.einsum("nki,nkj->nij", rel, rel) / idx.shape[1]
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
        cov = cov / trace.clamp_min(1e-8)[:, None, None]
        cov2 = cov @ cov
        t2 = cov2.diagonal(dim1=-2, dim2=-1).sum(-1)
        t3 = (cov2 @ cov).diagonal(dim1=-2, dim2=-1).sum(-1)
        cu = torch.einsum("nij,nkj->nki", cov, unit)
        di = (unit * cu).sum(-1)
        # For self graphs, cov is the source covariance as well.
        neighbor_cov = cov[idx] if source_cov is None else source_cov[idx]
        dj = torch.einsum("nki,nkij,nkj->nk", unit, neighbor_cov, unit)
        extra = torch.stack([
            torch.log1p(dist), torch.log1p(radius).expand_as(dist),
            t2[:, None].expand_as(dist), t3[:, None].expand_as(dist), di, dj,
        ], -1)
        # Compress unbounded metric distances to [0,1), avoiding RBF extinction.
        metric = dist / (dist + float(metric_scale))
        edge = torch.cat([rbf(metric), rbf(dist / radius), extra], -1)
    return dict(edge=edge, unit=unit, cov=cov, cu=cu, dist=dist)


class LocalEquivariantStem(nn.Module):
    """Joint feature/shape stem with invariant scalar and equivariant vector output."""
    def __init__(self, scalar_in, scalar_out, vector_out, k=16, rbf_dim=16,
                 input_vector_groups=0):
        super().__init__()
        self.k, self.metric_scale, self.chunk_size = k, 0.1, 1024
        self.rbf = GaussianRBF(rbf_dim)
        width = min(64, scalar_out)
        self.scalar = nn.Linear(scalar_in, scalar_out)
        self.edge = nn.Sequential(nn.Linear(2 * scalar_in + 2 * rbf_dim + 6, width),
                                  nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.score = nn.Linear(width, 1)
        self.scalar_value = nn.Linear(width, scalar_out)
        self.vector_value = nn.Linear(width, 2 * vector_out)
        self.snorm = nn.LayerNorm(scalar_out)
        self.vnorm = VectorRMSNorm(vector_out)
        self.vact = VectorGatedSiLU(vector_out, scalar_out)
        self.input_vector_proj = (EquivariantLinear(input_vector_groups, vector_out)
                                  if input_vector_groups else None)

    def forward(self, coord, scalar_feat, offset, vector_feat=None):
        idx, _ = knn_query(coord, offset, self.k)
        geo = local_geometry(coord, idx, self.rbf, self.metric_scale)
        ss, vv = [], []
        for start in range(0, len(coord), self.chunk_size):
            end = min(start + self.chunk_size, len(coord))
            ids = idx[start:end]
            center = scalar_feat[start:end, None].expand(-1, ids.shape[1], -1)
            edge = self.edge(torch.cat([center, scalar_feat[ids] - center,
                                       geo['edge'][start:end].to(scalar_feat.dtype)], -1))
            weight = self.score(edge).float().softmax(1)
            ss.append((weight * self.scalar_value(edge)).sum(1))
            a, b = self.vector_value(edge).chunk(2, -1)
            value = (a[..., None] * geo['unit'][start:end, :, None]
                     + b[..., None] * geo['cu'][start:end, :, None])
            vv.append((weight[..., None] * value).sum(1))
        s = F.gelu(self.snorm(self.scalar(scalar_feat) + torch.cat(ss)))
        v = torch.cat(vv)
        if vector_feat is not None and self.input_vector_proj is not None:
            v = v + self.input_vector_proj(vector_feat)
        v = self.vact(self.vnorm(v), s)
        return SO3State(coord, s, v, offset,
                        sample_index=torch.arange(len(coord), device=coord.device))


class RelativeQuaternionTransportAttention(nn.Module):
    """Joint invariant edge attention with geometric scalar/vector VALUES.

    No component-wise nonlinearities or biases are applied to 3-vectors.
    Query chunking reduces peak eager temporary tensors; training still saves
    activations for all chunks. Each head normalizes over its neighbors.
    """
    def __init__(self, scalar_channels, vector_channels, num_heads, rbf_dim=16,
                 attn_drop=0.0, max_angle=math.pi):
        super().__init__()
        if scalar_channels % num_heads or vector_channels % num_heads:
            raise ValueError('Scalar/vector channels must be divisible by heads')
        self.scalar_channels, self.vector_channels = scalar_channels, vector_channels
        self.num_heads = num_heads
        self.ds, self.dv = scalar_channels // num_heads, vector_channels // num_heads
        self.max_angle, self.metric_scale, self.chunk_size = max_angle, 0.1, 1024
        width, geom_v = min(64, scalar_channels), min(16, vector_channels)
        self.edge_scalar = nn.Linear(scalar_channels, width)
        self.edge_vector = EquivariantLinear(vector_channels, geom_v)
        self.edge = nn.Sequential(nn.Linear(2*width + 3*geom_v + 2*rbf_dim + 6, width),
                                  nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.qs = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.ks = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.vs = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.qv = EquivariantLinear(vector_channels, vector_channels)
        self.kv = EquivariantLinear(vector_channels, vector_channels)
        self.vv = EquivariantLinear(vector_channels, vector_channels)
        self.edge_key = nn.Linear(width, scalar_channels, bias=False)
        self.edge_value = nn.Linear(width, scalar_channels, bias=False)
        self.angle_mlp = nn.Linear(width, num_heads)
        self.bias_mlp = nn.Linear(width, num_heads)
        self.vector_coeff = nn.Linear(width, 3 * vector_channels)
        self.vector_score_scale = nn.Parameter(torch.ones(num_heads))
        self.scalar_out = nn.Linear(scalar_channels, scalar_channels)
        self.vector_out = EquivariantLinear(vector_channels, vector_channels)
        self.attn_drop = nn.Dropout(attn_drop)
        self.rbf = GaussianRBF(rbf_dim)

    def forward(self, state, idx, dist):
        s, v = state.scalar, state.vector
        n, k = idx.shape
        h, ds, dv = self.num_heads, self.ds, self.dv
        geo = state.geometry
        if geo is None:
            geo = local_geometry(state.coord, idx, self.rbf, self.metric_scale)
        fs, fv = self.edge_scalar(s), self.edge_vector(v).float()
        qs = self.qs(s).reshape(n, h, ds)
        ks = self.ks(s).reshape(n, h, ds)
        vs = self.vs(s).reshape(n, h, ds)
        # Normalize vector Q/K to bound score magnitudes independently of channels.
        qv = F.normalize(self.qv(v).float().reshape(n, h, dv*3), dim=-1).reshape(n,h,dv,3)
        kv = F.normalize(self.kv(v).float().reshape(n, h, dv*3), dim=-1).reshape(n,h,dv,3)
        vv = self.vv(v).reshape(n, h, dv, 3)
        output_s, output_v = [], []
        for start in range(0, n, self.chunk_size):
            end = min(start+self.chunk_size, n)
            ids, unit = idx[start:end], geo['unit'][start:end]
            center = fs[start:end, None].expand(-1, k, -1)
            vi, vj = fv[start:end, None], fv[ids]
            invariant = torch.cat([(vi*unit[:,:,None]).sum(-1),
                                   (vj*unit[:,:,None]).sum(-1), (vi*vj).sum(-1)], -1)
            edge = self.edge(torch.cat([center, fs[ids]-center,
                                       geo['edge'][start:end].to(center.dtype),
                                       invariant.to(center.dtype)], -1))
            theta = self.max_angle * torch.tanh(self.angle_mlp(edge).float())
            theta = theta * (geo['dist'][start:end] > 1e-8)[..., None]
            qw = torch.cos(theta/2)[..., None]
            qvec = unit[:,:,None] * torch.sin(theta/2)[...,None]
            kt = quaternion_rotate(kv[ids], qw, qvec)
            ek = self.edge_key(edge).reshape(end-start,k,h,ds)
            logits = (qs[start:end,None].float() * (ks[ids]+ek).float()).sum(-1) / math.sqrt(ds)
            logits = logits + (qv[start:end,None]*kt).sum((-1,-2)) * self.vector_score_scale
            logits = logits + self.bias_mlp(edge).float()
            weight = self.attn_drop(logits.softmax(1))
            value_s = vs[ids] + self.edge_value(edge).reshape(end-start,k,h,ds)
            output_s.append((weight[...,None]*value_s).sum(1).reshape(end-start,-1))
            coeff = self.vector_coeff(edge).reshape(end-start,k,3,h,dv).float()
            a, b, c = coeff.unbind(2)
            value_v = ((1 + torch.tanh(a))[...,None] * quaternion_rotate(vv[ids],qw,qvec)
                       + b[...,None]*unit[:,:,None,None]
                       + c[...,None]*geo['cu'][start:end,:,None,None])
            output_v.append((weight[...,None,None]*value_v).sum(1).reshape(end-start,-1,3))
        return self.scalar_out(torch.cat(output_s)), self.vector_out(torch.cat(output_v))


class SO3QuaternionBlock(nn.Module):
    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        vector_mlp_ratio: float = 2.0,
        rbf_dim: int = 16,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.snorm1 = nn.LayerNorm(scalar_channels)
        self.vnorm1 = VectorRMSNorm(vector_channels)
        self.attn = RelativeQuaternionTransportAttention(
            scalar_channels,
            vector_channels,
            num_heads,
            rbf_dim=rbf_dim,
            attn_drop=attn_drop,
        )

        self.snorm2 = nn.LayerNorm(scalar_channels)
        self.vnorm2 = VectorRMSNorm(vector_channels)
        self.sffn = ScalarFFN(
            scalar_channels, ratio=mlp_ratio, drop=proj_drop
        )
        self.vffn = VectorFFN(
            vector_channels,
            scalar_channels,
            ratio=vector_mlp_ratio,
        )

    def forward(self, state: SO3State, idx, dist):
        norm_state = SO3State(
            coord=state.coord,
            scalar=self.snorm1(state.scalar),
            vector=self.vnorm1(state.vector),
            offset=state.offset,
            geometry=state.geometry,
        )
        ds, dv = self.attn(norm_state, idx, dist)
        s = state.scalar + ds
        v = state.vector + dv

        sn = self.snorm2(s)
        vn = self.vnorm2(v)
        s = s + self.sffn(sn)
        v = v + self.vffn(vn, sn)

        return replace(state, scalar=s, vector=v)


class SO3LocalStage(nn.Module):
    """
    Reuses one k-NN graph across all blocks in the stage.
    """
    def __init__(
        self,
        depth: int,
        scalar_channels: int,
        vector_channels: int,
        num_heads: int,
        k: int,
        mlp_ratio: float = 4.0,
        vector_mlp_ratio: float = 2.0,
        rbf_dim: int = 16,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.metric_scale = 0.1
        self.rbf = GaussianRBF(rbf_dim)
        self.blocks = nn.ModuleList([
            SO3QuaternionBlock(
                scalar_channels,
                vector_channels,
                num_heads,
                mlp_ratio=mlp_ratio,
                vector_mlp_ratio=vector_mlp_ratio,
                rbf_dim=rbf_dim,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for _ in range(depth)
        ])

    def forward(self, state: SO3State):
        idx, dist = knn_query(
            state.coord, state.offset, self.k
        )
        state = replace(state, geometry=local_geometry(
            state.coord, idx, self.rbf, self.metric_scale))
        for block in self.blocks:
            state = block(state, idx, dist)
        return state


# ---------------------------------------------------------------------------
# Hierarchical down/up sampling
# ---------------------------------------------------------------------------

class SO3FPSDownsample(nn.Module):
    """FPS with feature-conditioned channel-wise scalar pooling and vector pooling."""
    def __init__(self, scalar_in, scalar_out, vector_in, vector_out,
                 ratio=0.25, k=16, rbf_dim=16):
        super().__init__()
        if not 0 < ratio <= 1:
            raise ValueError('ratio must lie in (0,1]')
        self.ratio, self.k = ratio, k
        self.metric_scale, self.chunk_size = 0.1, 1024
        self.rbf = GaussianRBF(rbf_dim)
        width = min(64, scalar_out)
        self.feature = nn.Linear(scalar_in, width)
        self.edge = nn.Sequential(nn.Linear(2*width + 2*rbf_dim+6, width),
                                  nn.SiLU(), nn.Linear(width,width), nn.SiLU())
        self.scalar_proj = nn.Linear(scalar_in, scalar_out)
        self.vector_proj = EquivariantLinear(vector_in, vector_out)
        self.scalar_weight = nn.Linear(width, scalar_out)
        self.scalar_value = nn.Linear(width, scalar_out)
        self.vector_weight = nn.Linear(width, vector_out)
        self.vector_geo = nn.Linear(width, 2*vector_out)
        self.snorm = nn.LayerNorm(scalar_out)
        self.vnorm = VectorRMSNorm(vector_out)
        self.vact = VectorGatedSiLU(vector_out, scalar_out)

    def forward(self, state, sample_idx=None):
        counts = _offset_to_counts(state.offset)
        new_counts = torch.ceil(counts.float()*self.ratio).long().clamp_min(1)
        new_offset = _counts_to_offset(new_counts, dtype=state.offset.dtype)
        if sample_idx is None:
            sample_idx = farthest_point_sampling(state.coord, state.offset, new_offset)
        coord = state.coord[sample_idx]
        idx, _ = knn_query(state.coord,state.offset,self.k,coord,new_offset)
        if state.geometry is None:
            src_idx, _ = knn_query(state.coord,state.offset,self.k)
            source_cov = local_geometry(state.coord,src_idx,self.rbf,self.metric_scale)['cov']
        else:
            source_cov = state.geometry['cov']
        geo = local_geometry(state.coord,idx,self.rbf,self.metric_scale,
                             query=coord,source_cov=source_cov)
        feat = self.feature(state.scalar)
        sp, vp = self.scalar_proj(state.scalar), self.vector_proj(state.vector)
        ss, vv = [], []
        for start in range(0,len(coord),self.chunk_size):
            end = min(start+self.chunk_size,len(coord))
            ids = idx[start:end]
            center = feat[sample_idx[start:end],None].expand(-1,ids.shape[1],-1)
            e = self.edge(torch.cat([center,feat[ids]-center,
                                     geo['edge'][start:end].to(center.dtype)],-1))
            ws = self.scalar_weight(e).float().softmax(1)
            wv = self.vector_weight(e).float().softmax(1)
            ss.append((ws*(sp[ids]+self.scalar_value(e))).sum(1))
            a,b = self.vector_geo(e).chunk(2,-1)
            value = vp[ids] + a[...,None]*geo['unit'][start:end,:,None] + b[...,None]*geo['cu'][start:end,:,None]
            vv.append((wv[...,None]*value).sum(1))
        s = F.gelu(self.snorm(torch.cat(ss)))
        v = self.vact(self.vnorm(torch.cat(vv)),s)
        if state.sample_index is None:
            raise ValueError('Downsampling requires sample_index')
        return SO3State(coord,s,v,new_offset,
                        sample_index=state.sample_index[sample_idx],pool_index=sample_idx)


class SO3UpsampleFusion(nn.Module):
    """
    Invariant-distance interpolation from coarse to fine, then scalar/vector
    skip fusion.
    """
    def __init__(
        self,
        coarse_scalar: int,
        coarse_vector: int,
        skip_scalar: int,
        skip_vector: int,
        out_scalar: int,
        out_vector: int,
        k: int = 3,
    ):
        super().__init__()
        self.k = k
        self.coarse_s = nn.Linear(coarse_scalar, out_scalar)
        self.skip_s = nn.Linear(skip_scalar, out_scalar)
        self.scalar_fuse = nn.Sequential(
            nn.Linear(2 * out_scalar, out_scalar),
            nn.LayerNorm(out_scalar),
            nn.GELU(),
        )

        self.coarse_v = EquivariantLinear(coarse_vector, out_vector)
        self.skip_v = EquivariantLinear(skip_vector, out_vector)
        self.vector_fuse = EquivariantLinear(2 * out_vector, out_vector)
        self.vnorm = VectorRMSNorm(out_vector)
        self.vact = VectorGatedSiLU(out_vector, out_scalar)

    def forward(self, coarse: SO3State, skip: SO3State):
        idx, dist = knn_query(
            coarse.coord,
            coarse.offset,
            self.k,
            new_xyz=skip.coord,
            new_offset=skip.offset,
        )
        inv = 1.0 / dist.clamp_min(1e-8)
        w = inv / inv.sum(-1, keepdim=True)

        cs = self.coarse_s(coarse.scalar)
        cv = self.coarse_v(coarse.vector)

        s_interp = (w.unsqueeze(-1) * cs[idx]).sum(1)
        v_interp = (
            w.unsqueeze(-1).unsqueeze(-1) * cv[idx]
        ).sum(1)

        s = self.scalar_fuse(
            torch.cat([s_interp, self.skip_s(skip.scalar)], dim=-1)
        )

        v_skip = self.skip_v(skip.vector)
        v = self.vector_fuse(
            torch.cat([v_interp, v_skip], dim=-2)
        )
        v = self.vact(self.vnorm(v), s)

        return replace(skip, scalar=s, vector=v, geometry=None)


# ---------------------------------------------------------------------------
# Invariant output heads
# ---------------------------------------------------------------------------

class SO3InvariantProjection(nn.Module):
    """Norms plus learned bilinear vector invariants retain channel relationships."""
    def __init__(self, scalar_channels, vector_channels, out_channels):
        super().__init__()
        self.left = EquivariantLinear(vector_channels, vector_channels)
        self.right = EquivariantLinear(vector_channels, vector_channels)
        self.net = nn.Sequential(nn.Linear(scalar_channels+2*vector_channels,out_channels),
                                 nn.LayerNorm(out_channels), nn.GELU(),
                                 nn.Linear(out_channels,out_channels),
                                 nn.LayerNorm(out_channels), nn.GELU())

    def forward(self,state):
        norm = torch.linalg.vector_norm(state.vector.float(),dim=-1)
        dot = (self.left(state.vector).float()*self.right(state.vector).float()).sum(-1)
        # Smooth compression for large bilinear features.
        dot = torch.asinh(dot)
        return self.net(torch.cat([state.scalar,norm,dot],-1))


class SO3EncoderOnlyHead(nn.Module):
    """
    Project every encoder scale to invariant features and interpolate to the
    original point set. This mirrors the lightweight encoder-only option in
    ptwno.py but uses distance-based interpolation instead of serialization
    ancestry.
    """
    def __init__(
        self,
        scalar_channels: Sequence[int],
        vector_channels: Sequence[int],
        out_channels: int,
        interp_k: int = 3,
    ):
        super().__init__()
        self.interp_k = interp_k
        self.proj = nn.ModuleList([
            SO3InvariantProjection(s, v, out_channels)
            for s, v in zip(scalar_channels, vector_channels)
        ])
        self.scale_weights = nn.Parameter(
            torch.zeros(len(scalar_channels))
        )
        self.out = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, stages: Sequence[SO3State]):
        target = stages[0]
        scale_w = F.softmax(self.scale_weights, dim=0)
        values = []

        for i, state in enumerate(stages):
            feat = self.proj[i](state)
            if i == 0:
                up = feat
            else:
                idx, dist = knn_query(
                    state.coord,
                    state.offset,
                    self.interp_k,
                    new_xyz=target.coord,
                    new_offset=target.offset,
                )
                inv = 1.0 / dist.clamp_min(1e-8)
                w = inv / inv.sum(-1, keepdim=True)
                up = (w.unsqueeze(-1) * feat[idx]).sum(1)

            values.append(scale_w[i] * up)

        return self.out(torch.stack(values, dim=0).sum(0))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

@MODELS.register_module("SO3-QuatTransport-v2")
@MODELS.register_module("SO3-QuatTransport-v1")
class SO3QuaternionTransportNet(nn.Module):
    """
    Hierarchical SO(3)-equivariant quaternion transport backbone.

    Joint feature/geometry edge embeddings control attention keys, values,
    quaternion angles and pooling. Decoder is enabled by default.
    Registry v1 is a configuration alias, NOT checkpoint compatibility.
    """
    def __init__(
        self,
        in_channels: int = 6,

        # Input feature semantics
        scalar_feature_indices: Optional[Sequence[int]] = None,
        vector_feature_groups: Sequence[Sequence[int]] = (),

        # Encoder
        enc_depths: Sequence[int] = (2, 2, 2, 4, 2),
        scalar_channels: Sequence[int] = (32, 64, 128, 256, 512),
        vector_channels: Sequence[int] = (16, 32, 64, 128, 256),
        num_heads: Sequence[int] = (2, 4, 8, 16, 16),
        neighbors: Sequence[int] = (16, 16, 24, 24, 32),
        sample_ratios: Sequence[float] = (0.25, 0.25, 0.25, 0.25),

        # Local operator
        rbf_dim: int = 16,
        mlp_ratio: float = 4.0,
        vector_mlp_ratio: float = 2.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,

        # Decoder / output
        use_decoder: bool = True,
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_scalar_channels: Sequence[int] = (64, 64, 128, 256),
        decoder_vector_channels: Sequence[int] = (32, 32, 64, 128),
        decoder_heads: Sequence[int] = (4, 4, 8, 16),
        decoder_neighbors: Sequence[int] = (16, 16, 24, 24),
        head_channels: int = 64,
        interp_k: int = 3,
        geometry_scales: Optional[Sequence[float]] = None,
        edge_chunk_size: int = 1024,
    ):
        super().__init__()

        nstages = len(enc_depths)
        if not (
            nstages
            == len(scalar_channels)
            == len(vector_channels)
            == len(num_heads)
            == len(neighbors)
        ):
            raise ValueError("Encoder stage configuration lengths must match.")
        if len(sample_ratios) != nstages - 1:
            raise ValueError("sample_ratios must have len(enc_depths)-1 entries.")

        self.in_channels = in_channels
        self.use_decoder = use_decoder
        self.head_channels = head_channels
        self.scalar_channels = tuple(scalar_channels)
        self.vector_channels = tuple(vector_channels)

        # Validate feature layout.
        vector_groups = [tuple(map(int, g)) for g in vector_feature_groups]
        for g in vector_groups:
            if len(g) != 3:
                raise ValueError(
                    "Each vector_feature_group must contain exactly 3 indices."
                )
            if min(g) < 0 or max(g) >= in_channels:
                raise ValueError(
                    f"Invalid vector feature group {g} for in_channels={in_channels}"
                )

        vector_flat = {j for g in vector_groups for j in g}
        if len(vector_flat) != 3 * len(vector_groups):
            raise ValueError("vector_feature_groups must not overlap.")

        if scalar_feature_indices is None:
            scalar_indices = [
                i for i in range(in_channels) if i not in vector_flat
            ]
        else:
            scalar_indices = list(map(int, scalar_feature_indices))
            if any(i < 0 or i >= in_channels for i in scalar_indices):
                raise ValueError("scalar_feature_indices out of range.")
            if any(i in vector_flat for i in scalar_indices):
                raise ValueError(
                    "A feature channel cannot be both scalar and vector."
                )

        if not scalar_indices:
            raise ValueError(
                "At least one invariant scalar input channel is required."
            )

        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long),
            persistent=False,
        )
        if vector_groups:
            self.register_buffer(
                "_vector_groups",
                torch.tensor(vector_groups, dtype=torch.long),
                persistent=False,
            )
        else:
            self._vector_groups = None

        self.stem = LocalEquivariantStem(
            scalar_in=len(scalar_indices),
            scalar_out=scalar_channels[0],
            vector_out=vector_channels[0],
            k=neighbors[0],
            rbf_dim=rbf_dim,
            input_vector_groups=len(vector_groups),
        )

        self.stages = nn.ModuleList()
        self.down = nn.ModuleList()

        for s in range(nstages):
            self.stages.append(
                SO3LocalStage(
                    depth=enc_depths[s],
                    scalar_channels=scalar_channels[s],
                    vector_channels=vector_channels[s],
                    num_heads=num_heads[s],
                    k=neighbors[s],
                    mlp_ratio=mlp_ratio,
                    vector_mlp_ratio=vector_mlp_ratio,
                    rbf_dim=rbf_dim,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
            )

            if s < nstages - 1:
                self.down.append(
                    SO3FPSDownsample(
                        scalar_in=scalar_channels[s],
                        scalar_out=scalar_channels[s + 1],
                        vector_in=vector_channels[s],
                        vector_out=vector_channels[s + 1],
                        ratio=sample_ratios[s],
                        k=neighbors[s],
                        rbf_dim=rbf_dim,
                    )
                )

        if not use_decoder:
            self.head = SO3EncoderOnlyHead(
                scalar_channels,
                vector_channels,
                head_channels,
                interp_k=interp_k,
            )
        else:
            if not (
                len(decoder_depths)
                == len(decoder_scalar_channels)
                == len(decoder_vector_channels)
                == len(decoder_heads)
                == len(decoder_neighbors)
                == nstages - 1
            ):
                raise ValueError(
                    "All decoder configs must have nstages-1 entries."
                )

            self.up = nn.ModuleList()
            self.dec_stages = nn.ModuleList()

            current_s = scalar_channels[-1]
            current_v = vector_channels[-1]

            # Decoder configuration is indexed by fine encoder stage s.
            for s in reversed(range(nstages - 1)):
                out_s = decoder_scalar_channels[s]
                out_v = decoder_vector_channels[s]

                self.up.append(
                    SO3UpsampleFusion(
                        coarse_scalar=current_s,
                        coarse_vector=current_v,
                        skip_scalar=scalar_channels[s],
                        skip_vector=vector_channels[s],
                        out_scalar=out_s,
                        out_vector=out_v,
                        k=interp_k,
                    )
                )
                self.dec_stages.append(
                    SO3LocalStage(
                        depth=decoder_depths[s],
                        scalar_channels=out_s,
                        vector_channels=out_v,
                        num_heads=decoder_heads[s],
                        k=decoder_neighbors[s],
                        mlp_ratio=mlp_ratio,
                        vector_mlp_ratio=vector_mlp_ratio,
                        rbf_dim=rbf_dim,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                    )
                )
                current_s, current_v = out_s, out_v

            self.final_invariant = SO3InvariantProjection(
                current_s, current_v, head_channels
            )

        scales = ([0.1 * 2**i for i in range(nstages)] if geometry_scales is None
                  else list(geometry_scales))
        if len(scales) != nstages or any(not math.isfinite(x) or x <= 0 for x in scales):
            raise ValueError('geometry_scales must have one finite positive value per encoder stage')
        if not isinstance(edge_chunk_size, int) or edge_chunk_size < 1:
            raise ValueError('edge_chunk_size must be a positive integer')
        self.stem.metric_scale = scales[0]
        for i, stage in enumerate(self.stages):
            stage.metric_scale = scales[i]
            for block in stage.blocks:
                block.attn.metric_scale = scales[i]
        for i, down in enumerate(self.down):
            down.metric_scale = scales[i+1]
        if self.use_decoder:
            for stage, i in zip(self.dec_stages, reversed(range(nstages-1))):
                stage.metric_scale = scales[i]
                for block in stage.blocks:
                    block.attn.metric_scale = scales[i]
        for module in self.modules():
            if hasattr(module, 'chunk_size'):
                module.chunk_size = edge_chunk_size

    def _split_input_features(self, feat: torch.Tensor):
        s = feat.index_select(1, self._scalar_indices.to(feat.device))

        v = None
        if self._vector_groups is not None:
            groups = self._vector_groups.to(feat.device)
            # feat[:, groups] -> [N, G, 3]
            v = feat[:, groups]
        return s, v

    def forward_encoder(self, data_dict, sampling_plan=None):
        """Return encoder states; optionally replay local FPS indices per level."""
        point = data_dict if isinstance(data_dict, Point) else Point(data_dict)

        coord = point.coord.float()
        feat = point.feat
        offset = point.offset

        if coord.ndim != 2 or coord.shape[-1] != 3:
            raise ValueError("coord must have shape [N, 3].")
        if feat.ndim != 2 or feat.shape[-1] != self.in_channels:
            raise ValueError(
                f"feat must have shape [N, {self.in_channels}], "
                f"got {tuple(feat.shape)}."
            )

        scalar_input, vector_input = self._split_input_features(feat)

        state = self.stem(
            coord,
            scalar_input,
            offset,
            vector_feat=vector_input,
        )

        stages = []
        if sampling_plan is not None and len(sampling_plan) != len(self.down):
            raise ValueError("sampling_plan must have one entry per downsample.")
        for s, stage in enumerate(self.stages):
            state = stage(state)
            stages.append(state)
            if s < len(self.down):
                state = self.down[s](
                    state, None if sampling_plan is None else sampling_plan[s]
                )
        return stages

    def forward(self, data_dict, return_stages=False):
        # Copy the container so a second view never sees the output as its input.
        point = Point(dict(data_dict))
        stages = self.forward_encoder(point)

        if not self.use_decoder:
            invariant_feat = self.head(stages)
        else:
            state = stages[-1]
            # self.up / self.dec_stages are ordered coarse -> fine.
            for module_index, s in enumerate(reversed(range(len(stages) - 1))):
                state = self.up[module_index](state, stages[s])
                state = self.dec_stages[module_index](state)
            invariant_feat = self.final_invariant(state)

        # Return a Point structure with invariant features at the
        # original point resolution, so existing Pointcept segmentors/heads can
        # consume it exactly like ptwno.py.
        point.feat = invariant_feat
        return (point, stages) if return_stages else point


# ---------------------------------------------------------------------------
# Segmentor wrapper, analogous to ptwno.py
# ---------------------------------------------------------------------------

def _stage_weights(value, count, name):
    weights = (1.0,) * count if value is None else tuple(float(w) for w in value)
    if len(weights) != count:
        raise ValueError(f"{name} must have {count} entries (fine to coarse).")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError(f"{name} must contain finite, nonnegative weights.")
    return weights


def _invariant_features(state):
    return torch.cat([
        state.scalar.float(),
        torch.linalg.vector_norm(state.vector.float(), dim=-1),
    ], dim=-1)


def _random_so3(device):
    """Haar-uniform proper rotation from a normalized Gaussian quaternion."""
    q = F.normalize(torch.randn(4, device=device, dtype=torch.float32), dim=0)
    w, x, y, z = q.unbind()
    return torch.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ]).reshape(3, 3)


@MODELS.register_module("DefaultSegmentorV3SO3")
class DefaultSegmentorV3SO3(nn.Module):
    """Optional training-only encoder supervision and paired SO(3) losses.

    Total = final + aux_loss_weight * sum(aux_stage_weights * stage_criteria)
                  + equivariance_weight * sum(consistency_stage_weights * vector_MSE)
                  + invariance_weight * sum(consistency_stage_weights * invariant_MSE).

    MSE averages over points AND channels/components; stage weights are not
    normalized. Consistency includes unlabeled points. The rotated encoder uses
    the original FPS indices and matching dropout RNG, with kNN recomputed.
    By default gradients flow through both views. No extra passes run at eval.
    """
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        ignore_index=-1,
        deep_supervision=False,
        aux_loss_weight=1.0,
        aux_stage_weights=None,
        aux_criteria=None,
        equivariance_regularization=False,
        equivariance_weight=0.01,
        invariance_regularization=False,
        invariance_weight=0.01,
        consistency_stage_weights=None,
        consistency_detach_target=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.freeze_backbone = freeze_backbone

        for name, value in (
            ("aux_loss_weight", aux_loss_weight),
            ("equivariance_weight", equivariance_weight),
            ("invariance_weight", invariance_weight),
        ):
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        nstages = len(self.backbone.stages)
        self.aux_stage_weights = _stage_weights(
            aux_stage_weights, nstages, "aux_stage_weights"
        )
        self.consistency_stage_weights = _stage_weights(
            consistency_stage_weights, nstages, "consistency_stage_weights"
        )
        self.deep_supervision = bool(
            deep_supervision and aux_loss_weight > 0 and any(self.aux_stage_weights)
        )
        self.equivariance_regularization = bool(
            equivariance_regularization and equivariance_weight > 0
            and any(self.consistency_stage_weights)
        )
        self.invariance_regularization = bool(
            invariance_regularization and invariance_weight > 0
            and any(self.consistency_stage_weights)
        )
        self.use_consistency = (
            self.equivariance_regularization or self.invariance_regularization
        )
        if self.use_consistency and freeze_backbone:
            raise ValueError("Consistency regularization requires an unfrozen backbone.")
        self.aux_loss_weight = float(aux_loss_weight)
        self.equivariance_weight = float(equivariance_weight)
        self.invariance_weight = float(invariance_weight)
        self.consistency_detach_target = bool(consistency_detach_target)
        self.aux_criteria = build_criteria(aux_criteria) if aux_criteria is not None else None
        # No new parameters when disabled, preserving baseline checkpoint keys.
        # Zero-weight stages get no head, avoiding unused parameters under DDP.
        self.aux_heads = nn.ModuleDict()
        if self.deep_supervision:
            for i, (s, v, weight) in enumerate(zip(
                self.backbone.scalar_channels, self.backbone.vector_channels,
                self.aux_stage_weights,
            )):
                if weight > 0:
                    self.aux_heads[str(i)] = nn.Linear(s + v, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _compute_loss(self, logits, target, criteria=None):
        with torch.autocast(
            device_type=logits.device.type,
            enabled=False,
        ):
            logits_fp32 = logits.float()
            if not (target != self.ignore_index).any().item():
                return logits_fp32.sum() * 0.0
            return (self.criteria if criteria is None else criteria)(logits_fp32, target)

    def _rotated_input(self, input_dict, rotation):
        rotated = dict(input_dict)
        with torch.autocast(device_type=rotation.device.type, enabled=False):
            rotated["coord"] = input_dict["coord"].float() @ rotation.T
            feat = input_dict["feat"].clone()
            groups = self.backbone._vector_groups
            if groups is not None:
                feat[:, groups] = (feat[:, groups].float() @ rotation.T).to(feat.dtype)
            rotated["feat"] = feat
        return rotated

    def _consistency_losses(self, input_dict, stages, cpu_rng, cuda_rng):
        device = stages[0].coord.device
        with torch.autocast(device_type=device.type, enabled=False):
            rotation = _random_so3(device)
        rotated_input = self._rotated_input(input_dict, rotation)
        devices = [device.index] if device.type == "cuda" else []
        # Replay the encoder's dropout draws, then restore the caller's RNG.
        # Only CPU and this rank's CUDA device are touched (DDP-safe).
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            rotated_stages = self.backbone.forward_encoder(
                rotated_input, sampling_plan=[s.pool_index for s in stages[1:]]
            )

        eq = stages[0].vector.float().sum() * 0.0
        inv = stages[0].scalar.float().sum() * 0.0
        with torch.autocast(device_type=device.type, enabled=False):
            for weight, reference, rotated in zip(
                self.consistency_stage_weights, stages, rotated_stages
            ):
                if weight == 0:
                    continue
                if self.equivariance_regularization:
                    expected = reference.vector.float() @ rotation.T
                    if self.consistency_detach_target:
                        expected = expected.detach()
                    eq = eq + weight * F.mse_loss(rotated.vector.float(), expected)
                if self.invariance_regularization:
                    expected = _invariant_features(reference)
                    if self.consistency_detach_target:
                        expected = expected.detach()
                    inv = inv + weight * F.mse_loss(_invariant_features(rotated), expected)
        return eq, inv

    def forward(self, input_dict, return_point=False):
        target = input_dict.get("segment")

        if target is not None:
            if target.dtype != torch.long or target.ndim != 1:
                raise ValueError(
                    "segment must be a one-dimensional torch.long tensor"
                )
            invalid = (
                (target != self.ignore_index)
                & ((target < 0) | (target >= self.num_classes))
            )
            if invalid.any().item():
                raise ValueError(
                    f"Invalid labels {target[invalid].unique().tolist()}; "
                    f"expected 0..{self.num_classes - 1} or {self.ignore_index}."
                )
        elif self.training:
            raise ValueError("Training requires segment labels.")

        need_stages = self.training and (self.deep_supervision or self.use_consistency)
        cpu_rng = cuda_rng = None
        if self.training and self.use_consistency:
            cpu_rng = torch.get_rng_state()
            if input_dict["coord"].is_cuda:
                cuda_rng = torch.cuda.get_rng_state(input_dict["coord"].device)

        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            if need_stages:
                point, stages = self.backbone(input_dict, return_stages=True)
            else:
                point = self.backbone(input_dict)

        logits = self.seg_head(point.feat)

        if target is not None and target.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Logit/target point counts differ: "
                f"{logits.shape[0]} vs {target.shape[0]}"
            )

        if self.training:
            loss = self._compute_loss(logits, target)
            output = {"loss": loss}
            if need_stages:
                output["loss_final"] = loss.detach()
            if self.deep_supervision:
                aux = logits.float().sum() * 0.0
                for key, head in self.aux_heads.items():
                    i = int(key)
                    state = stages[i]
                    stage_logits = head(_invariant_features(state).to(state.scalar.dtype))
                    stage_loss = self._compute_loss(
                        stage_logits, target[state.sample_index], self.aux_criteria
                    )
                    aux = aux + self.aux_stage_weights[i] * stage_loss
                    output[f"loss_aux_stage_{i}"] = stage_loss.detach()
                output["loss_aux"] = aux.detach()
                output["loss"] = output["loss"] + self.aux_loss_weight * aux
            if self.use_consistency:
                eq, inv = self._consistency_losses(input_dict, stages, cpu_rng, cuda_rng)
                if self.equivariance_regularization:
                    output["loss_equivariance"] = eq.detach()
                    output["loss"] = output["loss"] + self.equivariance_weight * eq
                if self.invariance_regularization:
                    output["loss_invariance"] = inv.detach()
                    output["loss"] = output["loss"] + self.invariance_weight * inv
        else:
            output = {"seg_logits": logits}
            if target is not None:
                output["loss"] = self._compute_loss(logits, target)

        if return_point:
            output["point"] = point
        return output


def validate_so3_model(device="cpu"):
    """Small forward/backward and rotation check using actual installed backends.

    Call after importing this module within your Pointcept environment:
        print(validate_so3_model("cuda"))
    CPU uses the debug kNN/FPS fallback. CUDA uses installed pointops where
    neighborhood sizes permit. Tests both decoder and encoder-only modes.
    This is a correctness smoke test, not a segmentation benchmark.
    """
    device = torch.device(device)
    torch.manual_seed(1234)
    coord = torch.randn(80, 3, device=device)
    feat = torch.randn(80, 6, device=device)  # RGB followed by a true vector
    offset = torch.tensor([40, 80], dtype=torch.int32, device=device)
    data = dict(coord=coord, feat=feat, offset=offset)
    rotation = _random_so3(device)
    rotated = dict(coord=coord @ rotation.T,
                   feat=torch.cat([feat[:, :3], feat[:, 3:] @ rotation.T], -1),
                   offset=offset)
    report = {}
    for decoder in (False, True):
        model = SO3QuaternionTransportNet(
            in_channels=6, scalar_feature_indices=(0, 1, 2),
            vector_feature_groups=((3, 4, 5),),
            enc_depths=(1, 1), scalar_channels=(16, 32),
            vector_channels=(8, 16), num_heads=(2, 4), neighbors=(8, 8),
            sample_ratios=(0.5,), use_decoder=decoder,
            decoder_depths=(1,), decoder_scalar_channels=(16,),
            decoder_vector_channels=(8,), decoder_heads=(2,),
            decoder_neighbors=(8,), head_channels=16, edge_chunk_size=17,
        ).to(device).eval()
        with torch.no_grad():
            original, states = model(data, return_stages=True)
            transformed, rotated_states = model(rotated, return_stages=True)
            torch.testing.assert_close(original.feat, transformed.feat,
                                       atol=5e-4, rtol=5e-4)
            for state, rotated_state in zip(states, rotated_states):
                torch.testing.assert_close(state.scalar, rotated_state.scalar,
                                           atol=5e-4, rtol=5e-4)
                torch.testing.assert_close(state.vector @ rotation.T,
                                           rotated_state.vector,
                                           atol=5e-4, rtol=5e-4)
            error = (original.feat - transformed.feat).abs().max().item()
        model.train()
        output = model(data).feat
        assert torch.isfinite(output).all(), "Nonfinite output"
        output.square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
        report["decoder" if decoder else "encoder_only"] = {
            "max_rotation_error": error, "backward": "passed",
        }
    return report
