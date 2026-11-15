"""Quaternion-frame Point Transformer for Pointcept (local equivariant / global invariant).

Registry: SO3-QuatFrame-v1 (backbone), QuatFrameSegmentor (wrapper).
Per neighborhood: covariance -> PCA axes (third-moment sign) -> E_i -> q_i.
Edge descriptor: t_ij = q_i* (x) [0,(x_j-x_i)/rho_i] (x) q_i, used PTv2-style
(multiplier + bias). Q/K/V vector channel groups are transported between
neighborhood frames with q_ij = q_i* (x) q_j (sandwich product). Vector stream v[N,Cv,3] is SO(3)-equivariant; it is read out
to invariants (norms, Gram, optional triple products) for the scalar/global path.
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
import pointops


import os, time, warnings

_PROFILE = os.environ.get("SO3_PROFILE", "0") == "1"
_PROFILE_STATS = {}
_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg, RuntimeWarning)
        print(f"[SO3-QuatFrame WARNING] {msg}", flush=True)


class _timer:
    """SO3_PROFILE=1 -> accumulate CUDA-synchronized wall time per block name."""

    def __init__(self, name, device):
        self.name, self.device = name, device

    def __enter__(self):
        if _PROFILE:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.t = time.perf_counter()

    def __exit__(self, *a):
        if _PROFILE:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            _PROFILE_STATS[self.name] = (
                _PROFILE_STATS.get(self.name, 0.0) + time.perf_counter() - self.t
            )


def profile_report(reset=True):
    rows = sorted(_PROFILE_STATS.items(), key=lambda x: -x[1])
    total = sum(v for _, v in rows) or 1.0
    lines = [f"{k:<28s} {v:9.3f}s {100*v/total:5.1f}%" for k, v in rows]
    if reset:
        _PROFILE_STATS.clear()
    return "\n".join(lines)


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


# =====================================================================
# Quaternion helpers (scalar-first, Hamilton convention: q = [w, x, y, z])
# =====================================================================
def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quat_rotate(q, v):
    """q (x) [0,v] (x) q*  ==  v + 2w(u x v) + 2u x (u x v). Sign of q irrelevant."""
    w, u = q[..., :1], q[..., 1:]
    uv = torch.cross(u, v, dim=-1)
    return v + 2 * (w * uv + torch.cross(u, uv, dim=-1))


def quat_inv_rotate(q, v):
    """q* (x) [0,v] (x) q  ==  E^T v (world -> local)."""
    return quat_rotate(quat_conj(q), v)


def rot_to_quat(E):
    """Proper rotation matrices [...,3,3] -> unit quaternions [...,4].
    Largest-component branch selection (stable near 180 deg)."""
    m00, m01, m02 = E[..., 0, 0], E[..., 0, 1], E[..., 0, 2]
    m10, m11, m12 = E[..., 1, 0], E[..., 1, 1], E[..., 1, 2]
    m20, m21, m22 = E[..., 2, 0], E[..., 2, 1], E[..., 2, 2]
    tw = 1 + m00 + m11 + m22
    tx = 1 + m00 - m11 - m22
    ty = 1 - m00 + m11 - m22
    tz = 1 - m00 - m11 + m22
    cand = torch.stack(
        [
            torch.stack([tw, m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, tx, m01 + m10, m02 + m20], -1),
            torch.stack([m02 - m20, m01 + m10, ty, m12 + m21], -1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, tz], -1),
        ],
        -2,
    )
    t = torch.stack([tw, tx, ty, tz], -1)
    best = t.argmax(-1)
    n = cand.gather(-2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = n / (2 * t.gather(-1, best[..., None]).clamp_min(1e-12).sqrt())
    return F.normalize(q, dim=-1)


# =====================================================================
# Covariance / PCA quaternion frames
# =====================================================================
@torch.no_grad()
def neighborhood_frames(
    rel, dist, mask, gap_eps=0.05, skew_eps=0.05, chunk_size=262144
):
    """One quaternion frame per query neighborhood.

    rel  [N,K,3] = x_j - x_i (world), dist [N,K], mask [N,K] distinct valid.
    Returns dict:
      z     [N,K,3] normalized world displacement (x_j-x_i)/rho_i (equivariant)
      delta [N,K]   ||z||
      q     [N,4]   frame quaternion (local -> world)
      ell   [N,3]   normalized eigenvalues (invariant shape)
      valid [N]     frame validity nu_i
    """
    out = {k: [] for k in ("z", "delta", "q", "ell", "valid")}
    with torch.autocast(device_type=rel.device.type, enabled=False):
        for s in range(0, len(dist), chunk_size):
            r = rel[s : s + chunk_size].float()
            d = dist[s : s + chunk_size].float()
            keep = mask[s : s + chunk_size]
            w = keep.float()[..., None]
            rho = d.masked_fill(~keep, 0).amax(1, keepdim=True)
            pos_r = rho.squeeze(1) > 1e-8
            z = r / rho.clamp_min(1e-8)[..., None] * pos_r[:, None, None]
            delta = torch.linalg.vector_norm(z, dim=-1)
            cnt = w.sum(1).clamp_min(1)  # [n,1]
            mean = (z * w).sum(1) / cnt  # [n,3]
            c = (z - mean[:, None]) * w  # centered, masked
            C = torch.einsum("nki,nkj->nij", c, c) / cnt[..., None]
            lam, U = torch.linalg.eigh(C)  # ascending
            lam, U = lam.flip(-1).clamp_min(0), U.flip(-1)  # descending
            tr = lam.sum(-1, keepdim=True)
            ell = torch.where(
                tr > 1e-12, lam / tr.clamp_min(1e-12), torch.zeros_like(lam)
            )
            # Third-moment sign disambiguation for axes 1,2 (scale-free skewness).
            proj = torch.einsum("nki,nij->nkj", c, U[..., :2])  # [n,K,2]
            h = (proj**3).sum(1) / cnt  # [n,2]
            skew = h / lam[:, :2].clamp_min(1e-12).pow(1.5)
            sgn = torch.where(h >= 0, 1.0, -1.0)
            e1 = U[..., 0] * sgn[:, :1]
            e2 = U[..., 1] * sgn[:, 1:2]
            e3 = torch.cross(e1, e2, dim=-1)
            E = torch.stack([e1, e2, e3], -1)  # columns = local axes
            q = rot_to_quat(E)
            g12 = (lam[:, 0] - lam[:, 1]) / tr.squeeze(-1).clamp_min(1e-12)
            g23 = (lam[:, 1] - lam[:, 2]) / tr.squeeze(-1).clamp_min(1e-12)
            valid = (
                pos_r
                & (tr.squeeze(-1) > 1e-10)
                & (keep.sum(1) >= 3)
                & (g12 > gap_eps)
                & (g23 > gap_eps)
                & (skew.abs() > skew_eps).all(-1)
                & torch.isfinite(q).all(-1)
            )
            q = torch.where(
                valid[:, None], q, torch.tensor([1.0, 0, 0, 0], device=q.device)
            )
            for k, v in zip(out, (z, delta, q, ell, valid)):
                out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


# =====================================================================
# Quaternion-frame positional encoding (PTv2-style input = canonical t_ij)
# =====================================================================
class QuaternionFrameEncoding(nn.Module):
    """p_ij = nu_ij * phi_F([delta, t_ij, (q_ij), (ell_i, ell_j)])
            + (1-nu_ij) * phi_0([delta, ell_i, ell_j])

    t_ij = q_i* (x) [0, z_ij] (x) q_i  (invariant local displacement).
    q_ij = q_i* (x) q_j optional ablation, sign-symmetrized MLP.
    No world-coordinate quantity enters any MLP.
    """

    def __init__(
        self,
        width=16,
        use_relative_quat=False,
        use_shape=False,
        chunk_size=256,
        checkpoint_edges=True,
    ):
        super().__init__()
        self.use_relative_quat, self.use_shape = use_relative_quat, use_shape
        self.chunk_size, self.checkpoint_edges = chunk_size, checkpoint_edges
        fin = 4 + (4 if use_relative_quat else 0) + (6 if use_shape else 0)
        self.frame_mlp = nn.Sequential(
            nn.Linear(fin, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.fallback_mlp = nn.Sequential(
            nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.norm = nn.LayerNorm(width)

    def _embed(self, frame_in, frame_in_neg, fallback_in, nu):
        pf = self.frame_mlp(frame_in)
        if frame_in_neg is not None:
            pf = 0.5 * (pf + self.frame_mlp(frame_in_neg))
        p0 = self.fallback_mlp(fallback_in)
        nu = nu.to(pf.dtype)[..., None]
        return self.norm(nu * pf + (1 - nu) * p0)

    def forward(self, geo, idx, mask, query_is_source=True):
        z, delta, q, ell, valid = (
            geo["z"],
            geo["delta"],
            geo["q"],
            geo["ell"],
            geo["valid"],
        )
        outs = []
        for s in range(0, len(idx), self.chunk_size):
            e = min(s + self.chunk_size, len(idx))
            with torch.no_grad(), torch.autocast(
                device_type=z.device.type, enabled=False
            ):
                ii, kk = idx[s:e], mask[s:e]
                qi = q[s:e, None].expand(-1, ii.shape[1], -1)
                t = quat_inv_rotate(qi, z[s:e])  # [n,K,3]
                dl = delta[s:e, :, None]
                ell_i = ell[s:e, None].expand(-1, ii.shape[1], -1)
                ell_j = ell[ii] if query_is_source else ell_i
                nu = valid[s:e, None] & kk
                if query_is_source:
                    nu = nu & valid[ii]
                parts = [dl, t]
                neg = None
                if self.use_relative_quat:
                    qij = quat_mul(quat_conj(qi), q[ii])
                    neg_parts = parts + [-qij]
                    parts = parts + [qij]
                if self.use_shape:
                    parts = parts + [ell_i, ell_j]
                    if self.use_relative_quat:
                        neg_parts = neg_parts + [ell_i, ell_j]
                frame_in = torch.cat(parts, -1)
                if self.use_relative_quat:
                    neg = torch.cat(neg_parts, -1)
                fallback_in = torch.cat([dl, ell_i, ell_j], -1)
            outs.append(
                _run(
                    self._embed,
                    self.training and self.checkpoint_edges,
                    frame_in,
                    neg,
                    fallback_in,
                    nu,
                )
            )
        return torch.cat(outs)


# =====================================================================
# Equivariant vector-stream primitives
# =====================================================================
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


# =====================================================================
# Attention with scalar + vector messages
# =====================================================================
class GeometryAttention(nn.Module):
    """PTv2-style conditioning on invariant p_ij (= canonical displacement embedding).

    GVA: omega((1+tanh(m(p))) * (q_i-k_j)) + bias(p)
    MSA: sum((1+tanh(m(p))) * q_i*k_j)/sqrt(C/G) + bias(p)
    Vector message (equivariant):
        m_i,c = sum_j a_ij [ A_ij,c * v_j,c + b_ij,c * z_ij ]
    with invariant A, b predicted from [p_ij, s_j]; a_ij = group-mean attention.

    Quaternion Q/K/V transport (transport_qkv=True):
    The first 3*nv channels of every group are read as nv 3-vectors expressed
    in the point's own frame (gauge). Q_i is lifted by q_i, K_j and V_j by q_j,
    and the relation/values are read back in frame i:
        k_{j->i} = q_ij (x) [0,k_j] (x) q_ij*,   q_ij = q_i* (x) q_j
        v_{j->i} = q_ij (x) [0,v_j] (x) q_ij*
    (identical to E_i^T (E_j k_j); Q needs no edge rotation because it is
    already in frame i). Sandwich products make q/-q irrelevant. q_ij is
    invariant, so outputs stay invariant. Pairs with an invalid frame use
    the untransported (still invariant) channels.
    """

    def __init__(
        self,
        channels,
        groups,
        pos_width,
        attention_type,
        vector_channels,
        position_multiplier=True,
        position_bias=True,
        geometry_in_values=False,
        chunk_size=256,
        checkpoint_edges=True,
        attn_drop=0.0,
        transport_qkv=True,
        transport_ratio=0.75,
    ):
        super().__init__()
        if attention_type not in ("grouped_vector", "scalar"):
            raise ValueError("attention_type must be grouped_vector or scalar")
        if channels % groups:
            raise ValueError(
                f"channels={channels} must be divisible by heads/groups={groups}"
            )
        self.channels, self.groups, self.group_width = (
            channels,
            groups,
            channels // groups,
        )
        self.kind, self.cv = attention_type, vector_channels
        if not 0 <= transport_ratio <= 1:
            raise ValueError("transport_ratio must be in [0,1]")
        self.nv = int(self.group_width * transport_ratio) // 3 if transport_qkv else 0
        self.chunk_size, self.checkpoint_edges = chunk_size, checkpoint_edges
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.multiplier = (
            nn.Linear(pos_width, channels) if position_multiplier else None
        )
        self.bias = nn.Linear(pos_width, groups) if position_bias else None
        self.pos_value = (
            nn.Linear(pos_width, channels, bias=False) if geometry_in_values else None
        )
        if attention_type == "grouped_vector":
            self.group_weight = nn.Parameter(torch.empty(groups, self.group_width))
            nn.init.normal_(self.group_weight, std=self.group_width**-0.5)
            self.weight_encoder = nn.Sequential(
                nn.LayerNorm(groups) if groups > 1 else nn.Identity(),
                nn.SiLU(),
                nn.Linear(groups, groups),
            )
        self.dropout = nn.Dropout(attn_drop)
        self.out = nn.Linear(channels, channels)
        if vector_channels > 0:
            self.vkey = nn.Linear(channels, pos_width, bias=False)
            self.vcoef = nn.Sequential(
                nn.Linear(pos_width, pos_width),
                nn.SiLU(),
                nn.Linear(pos_width, 2 * vector_channels),
            )
            self.vout = VLinear(vector_channels, vector_channels)

    def _transport(self, x, qij, nu):
        """Rotate the vector part of x [n,K,g,d] from frame j into frame i."""
        if self.nv == 0:
            return x
        n, kc, g, d = x.shape
        m = 3 * self.nv
        vec = x[..., :m].reshape(n, kc, g, self.nv, 3)
        rot = quat_rotate(qij.to(x.dtype)[:, :, None, None, :], vec)
        nu = nu.to(x.dtype)[:, :, None, None, None]
        vec = nu * rot + (1 - nu) * vec
        return torch.cat([vec.reshape(n, kc, g, m), x[..., m:]], -1)

    def _chunk(self, q, k, v, sk, vec, idx, pos, mask, z, qi, qframe, valid_i, valid):
        n, kc = idx.shape
        g, d = self.groups, self.group_width
        qh = q.reshape(n, g, d)
        kj = k[idx].reshape(n, kc, g, d)
        if self.nv > 0:
            with torch.no_grad():
                qij = quat_mul(quat_conj(qi[:, None].expand(-1, kc, -1)), qframe[idx])
                nu = valid_i[:, None] & valid[idx]
            kj = self._transport(kj, qij, nu)
        relation = (
            qh[:, None] - kj if self.kind == "grouped_vector" else qh[:, None] * kj
        )
        if self.multiplier is not None:
            relation = relation * (
                1 + torch.tanh(self.multiplier(pos)).reshape(n, kc, g, d)
            )
        if self.kind == "grouped_vector":
            logits = self.weight_encoder((relation * self.group_weight).sum(-1))
        else:
            logits = relation.sum(-1) / math.sqrt(d)
        if self.bias is not None:
            logits = logits + self.bias(pos)
        logits = logits.float().masked_fill(~mask[..., None], float("-inf"))
        weight = self.dropout(logits.softmax(dim=1))
        values = v[idx].reshape(n, kc, g, d)
        if self.nv > 0:
            values = self._transport(values, qij, nu)
        if self.pos_value is not None:
            values = values + self.pos_value(pos).reshape(n, kc, g, d)
        s_out = (
            (weight[..., None].to(values.dtype) * values)
            .sum(1)
            .reshape(n, self.channels)
        )
        if self.cv == 0:
            return s_out, None
        a = weight.mean(-1).to(pos.dtype)  # [n,K] invariant
        coef = self.vcoef(pos + sk[idx])  # invariant scalars
        A, b = coef.chunk(2, -1)  # [n,K,Cv] each
        vj = vec[idx]  # [n,K,Cv,3]
        msg = A[..., None] * vj + b[..., None] * z.to(vj.dtype)[:, :, None, :]
        v_out = (a[..., None, None] * msg).sum(1)  # [n,Cv,3]
        return s_out, v_out

    def forward(self, feat, vec, idx, pos, mask, z, qframe, valid):
        q, k, v = self.qkv(feat).chunk(3, -1)
        sk = self.vkey(feat) if self.cv > 0 else None
        s_outs, v_outs = [], []
        for s in range(0, len(feat), self.chunk_size):
            e = min(s + self.chunk_size, len(feat))
            so, vo = _run(
                self._chunk,
                self.training and self.checkpoint_edges,
                q[s:e],
                k,
                v,
                sk,
                vec,
                idx[s:e],
                pos[s:e],
                mask[s:e],
                z[s:e],
                qframe[s:e],
                qframe,
                valid[s:e],
                valid,
            )
            s_outs.append(so)
            v_outs.append(vo)
        s_out = self.out(torch.cat(s_outs))
        v_out = self.vout(torch.cat(v_outs)) if self.cv > 0 else None
        return s_out, v_out


class Block(nn.Module):
    """Scalar pre-norm Transformer block + equivariant vector residual.

    Vector nonlinearity: invariant sigmoid gate from scalars.
    Scalar update receives invariant readout of vectors (local equiv -> invariant).
    """

    def __init__(
        self,
        channels,
        groups,
        pos_width,
        vector_channels,
        mlp_ratio=2.0,
        proj_drop=0.0,
        **attention,
    ):
        super().__init__()
        self.cv = vector_channels
        self.norm1 = nn.LayerNorm(channels)
        self.attn = GeometryAttention(
            channels, groups, pos_width, vector_channels=vector_channels, **attention
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = max(1, int(channels * mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, channels),
            nn.Dropout(proj_drop),
        )
        if vector_channels > 0:
            self.readout = VectorReadout(vector_channels)
            self.v2s = nn.Sequential(
                nn.LayerNorm(self.readout.out_dim),
                nn.Linear(self.readout.out_dim, channels),
            )
            self.gate = nn.Linear(channels, vector_channels)

    def forward(self, feat, vec, idx, pos, mask, z, qframe, valid):
        s_msg, v_msg = self.attn(
            self.norm1(feat), vec, idx, pos, mask, z, qframe, valid
        )
        feat = feat + s_msg
        if self.cv > 0:
            vec = vec + v_msg
            feat = feat + self.v2s(self.readout(vec))
            vec = vec * torch.sigmoid(self.gate(feat))[..., None]
        feat = feat + self.ffn(self.norm2(feat))
        return feat, vec


@dataclass
class VState:
    coord: torch.Tensor
    feat: torch.Tensor
    vec: Optional[torch.Tensor]
    offset: torch.Tensor
    sample_index: torch.Tensor


class Stage(nn.Module):
    def __init__(
        self,
        channels,
        depth,
        groups,
        k,
        pos_width,
        vector_channels,
        use_relative_quat=False,
        use_shape=False,
        gap_eps=0.05,
        skew_eps=0.05,
        **block_options,
    ):
        super().__init__()
        self.k, self.gap_eps, self.skew_eps = k, gap_eps, skew_eps
        self.geometry = QuaternionFrameEncoding(
            pos_width,
            use_relative_quat,
            use_shape,
            block_options["chunk_size"],
            block_options["checkpoint_edges"],
        )
        self.blocks = nn.ModuleList(
            [
                Block(channels, groups, pos_width, vector_channels, **block_options)
                for _ in range(depth)
            ]
        )
        self.last_valid_ratio = None

    def forward(self, state, cache=None, name="stage"):
        """cache = (idx, mask, geo) from the encoder stage at the same resolution.
        Decoder stages reuse it: same coords and same k -> identical graph/frames."""
        dev = state.coord.device
        if cache is None or cache[0].shape[1] != self.k:
            with _timer(f"{name}.knn", dev):
                idx, rel, dist, mask = graph(state.coord, state.offset, self.k)
            with _timer(f"{name}.frames", dev):
                geo = neighborhood_frames(rel, dist, mask, self.gap_eps, self.skew_eps)
            del rel, dist
            cache = (idx, mask, geo)
        idx, mask, geo = cache
        self.last_valid_ratio = geo["valid"].float().mean().detach()
        with _timer(f"{name}.pos_enc", dev):
            pos = self.geometry(geo, idx, mask)
        feat, vec = state.feat, state.vec
        for b, block in enumerate(self.blocks):
            with _timer(f"{name}.block{b}", dev):
                feat, vec = block(
                    feat, vec, idx, pos, mask, geo["z"], geo["q"], geo["valid"]
                )
        return VState(state.coord, feat, vec, state.offset, state.sample_index), cache


class Downsample(nn.Module):
    """FPS + scalar max pool + vector MEAN pool (max is not equivariant) + centroid coord."""

    def __init__(
        self, cin, cout, ratio, k, chunk_size, checkpoint_edges, vector_channels
    ):
        super().__init__()
        self.ratio, self.k, self.chunk_size = ratio, k, chunk_size
        self.checkpoint_edges, self.cv = checkpoint_edges, vector_channels
        self.proj = nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, cout), nn.GELU())
        if vector_channels > 0:
            self.vproj = VLinear(vector_channels, vector_channels)

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
            coord = (state.coord[idx].float() * valid).sum(1) / valid.sum(1).clamp_min(
                1
            )
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
            w = mask.to(state.vec.dtype)
            vec = torch.cat(
                [
                    (
                        state.vec[idx[s : s + self.chunk_size]]
                        * w[s : s + self.chunk_size, :, None, None]
                    ).sum(1)
                    / w[s : s + self.chunk_size].sum(1).clamp_min(1)[:, None, None]
                    for s in range(0, len(idx), self.chunk_size)
                ]
            )
            vec = self.vproj(vec)
        return VState(
            coord, torch.cat(pooled), vec, offset, state.sample_index[selected]
        )


def interp_weights(coarse, fine, k=3):
    idx, _, dist, mask = graph(coarse.coord, coarse.offset, k, fine.coord, fine.offset)
    inverse = dist.clamp_min(1e-8).reciprocal() * mask
    return idx, inverse / inverse.sum(1, keepdim=True)


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
        self.fuse = nn.Sequential(nn.Linear(2 * out, out), nn.LayerNorm(out), nn.GELU())
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


@MODELS.register_module("SO3-QuatFrame-v1")
class SO3QuatFrameNet(nn.Module):
    """Local equivariant (vector stream) / global invariant (scalar stream) backbone.

    Geometry per stage: PCA covariance frame -> quaternion q_i -> canonical
    displacement t_ij -> PTv2-style multiplier + bias. Invalid frames use an
    invariant [delta, ell_i, ell_j] fallback (never world axes).
    Output point.feat is invariant: scalar features + vector invariants.
    """

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
        use_relative_quat=False,
        use_shape=False,
        frame_gap_eps=0.05,
        frame_skew_eps=0.05,
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
        transport_qkv=True,
        transport_ratio=0.75,
        share_geometry=True,
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
        if vector_channels < 0:
            raise ValueError("vector_channels must be >= 0")
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
            nn.Linear(len(indices), channels[0]), nn.LayerNorm(channels[0]), nn.GELU()
        )
        block = dict(
            attention_type=attention_type,
            position_multiplier=position_multiplier,
            position_bias=position_bias,
            geometry_in_values=geometry_in_values,
            chunk_size=edge_chunk_size,
            checkpoint_edges=checkpoint_edges,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
            transport_qkv=transport_qkv,
            transport_ratio=transport_ratio,
        )
        geo = dict(
            use_relative_quat=use_relative_quat,
            use_shape=use_shape,
            gap_eps=frame_gap_eps,
            skew_eps=frame_skew_eps,
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
                    **geo,
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
                        **geo,
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
            nn.LayerNorm(head_channels),
            nn.GELU(),
        )

    def frame_valid_ratios(self):
        mods = list(self.stages) + (list(self.decoder) if self.use_decoder else [])
        return [m.last_valid_ratio for m in mods]

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
        # Vector stream starts at zero; first vector messages come from b_ij * z_ij.
        vec = s.new_zeros(len(coord), self.cv, 3) if self.cv > 0 else None
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


@MODELS.register_module("QuatFrameSegmentor")
class QuatFrameSegmentor(nn.Module):
    """Pointcept segmentor. Optional auxiliary encoder supervision; no paired pass.

    Logits are built from invariant features only. Old DistanceAngle
    checkpoints are incompatible.
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
