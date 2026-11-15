"""Gravity-free SO(3)-invariant segmentation for Pointcept.

Register by importing this module in pointcept/models/__init__.py. Dependencies:
PyTorch >= 2.1 and Pointcept. No pointops, torch_cluster or torch_scatter needed.

Scalar patch attention + independent quaternion radius attention. Quaternion
features transform by conjugation (scalar + physical vector), NOT by arbitrary
four-channel rotations. No QuatRoPE, learned reference frame, kNN or FPS.

The tensor implementation is a correctness-first research baseline. Hash radius
search and parallel radius-cover selection have not been tuned as CUDA kernels.
Coverage uses the complete radius graph, streaming above the memory threshold;
only neural messages are degree-capped.
Scalar and vector normalization use stabilizing epsilon denominators.
Geometry is detached. Rotation invariance is mathematical, subject to floating
point error, radius boundaries, and reductions. Preprocessing must respect it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point


def divide_or_zero(x, denominator):
    valid = denominator > 0
    return (
        x
        / torch.where(valid, denominator, torch.ones_like(denominator))
        * valid.to(x.dtype)
    )


def safe_norm(x, dim=-1, keepdim=False):
    return torch.linalg.vector_norm(x.float(), dim=dim, keepdim=keepdim)


def counts(offset):
    return torch.diff(F.pad(offset.long(), (1, 0)))


def segment_sum(x, index, size):
    return x.new_zeros((size,) + x.shape[1:]).index_add(0, index, x)


def segment_mean(x, index, size):
    total = segment_sum(x.float(), index, size)
    count = torch.bincount(index, minlength=size).float()
    return divide_or_zero(total, count.reshape(size, *([1] * (x.ndim - 1)))).to(x.dtype)


def segment_max(x, index, size):
    expanded = index.reshape(-1, *([1] * (x.ndim - 1))).expand_as(x)
    return x.new_full((size,) + x.shape[1:], -torch.inf).scatter_reduce(
        0, expanded, x, reduce="amax", include_self=True
    )


def segment_softmax(logits, dst, size):
    values = logits.float()
    maximum = segment_max(values.detach(), dst, size)
    exp = (values - maximum[dst]).exp()
    return divide_or_zero(exp, segment_sum(exp, dst, size)[dst])


def mlp(cin, hidden, cout):
    return nn.Sequential(nn.Linear(cin, hidden), nn.SiLU(), nn.Linear(hidden, cout))


class ExactLayerNorm(nn.Module):
    """Scalar normalization with a stabilized float32 denominator."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        centered = x.float() - x.float().mean(-1, keepdim=True)
        scale = (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()
        return (
            divide_or_zero(centered, scale) * self.weight.float() + self.bias.float()
        ).to(x.dtype)


class VLinear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin) / math.sqrt(cin))

    def forward(self, v):
        return torch.einsum("oc,nci->noi", self.weight.to(v.dtype), v)


class VNRMSNorm(nn.Module):
    """SO(3)-equivariant piecewise RMS normalization without additive epsilon.

    Leave RMS <= 1 unchanged before learned gain; normalize larger vectors.
    Clamp squared RMS before sqrt so zero inputs have a safe backward path.
    """

    def __init__(self, channels):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))

    def forward(self, v):
        squared_rms = v.float().square().sum(-1).mean(-1, keepdim=True)
        scale = squared_rms.clamp_min(1.0).sqrt()
        return (
            divide_or_zero(v.float(), scale[..., None]) * self.gain.float()[..., None]
        ).to(v.dtype)


class VNActivation(nn.Module):
    def __init__(self, channels, leak=0.2):
        super().__init__()
        self.direction = VLinear(channels, channels)
        self.leak = leak

    def forward(self, v):
        d = self.direction(v.float())
        unit = d / safe_norm(d, keepdim=True).clamp_min(1e-6)
        dot = (v.float() * unit).sum(-1, keepdim=True)
        return (v.float() - (1 - self.leak) * dot.clamp_max(0) * unit).to(v.dtype)


class VNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = VLinear(channels, channels)
        self.norm = VNRMSNorm(channels)
        self.activation = VNActivation(channels)

    def forward(self, v):
        return self.activation(self.norm(self.linear(v)))


class VectorReadout(nn.Module):
    def __init__(self, channels, proj=4, triple=True):
        super().__init__()
        self.proj = VLinear(channels, proj)
        self.triple = triple
        self.register_buffer("iu", torch.triu_indices(proj, proj), persistent=False)
        self.out_dim = channels + proj * (proj + 1) // 2 + (proj if triple else 0)

    def forward(self, v):
        p = self.proj(v).float()
        gram = torch.einsum("nci,ndi->ncd", p, p)
        out = [safe_norm(v), gram[:, self.iu[0], self.iu[1]]]
        if self.triple:
            out.append((p * torch.cross(p.roll(1, 1), p.roll(2, 1), dim=-1)).sum(-1))
        return torch.cat(out, -1).to(v.dtype)


@dataclass
class RadiusGraph:
    dst: torch.Tensor
    src: torch.Tensor
    radius: float


@torch.no_grad()
def priority_ranks(point_id):
    """Fixed pseudo-random priorities attached to identities, never coordinates.

    Tie break by identity. IDs must be unique within each cloud (global uniqueness
    is recommended). With omitted IDs the input order is the identity: rotation
    consistency holds, but arbitrary input-permutation consistency is not claimed.
    """
    base = torch.argsort(point_id, stable=True)
    hashed = point_id[base].long().remainder(2**32)
    for _ in range(2):
        hashed = ((hashed ^ (hashed >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    hashed = hashed ^ (hashed >> 16)
    order = base[torch.argsort(hashed, stable=True)]
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order), device=order.device)
    return rank


def iter_radius_edges(
    coord, offset, radius, priority, backend="hash", query_chunk=256,
    candidate_limit=2_000_000,
):
    """Yield complete, destination-disjoint query chunks in priority order.

    The spatial index is rebuilt on each pass; only one query chunk is retained.
    candidate_limit bounds hash candidates, including recursive query splitting.
    """
    if backend not in ("hash", "brute"):
        raise ValueError("radius_backend must be hash or brute")
    shifts = torch.tensor(
        list(itertools.product((-1, 0, 1), repeat=3)), device=coord.device
    )
    start = 0
    for end in offset.long().tolist():
        x = coord[start:end].float()
        n = len(x)
        if backend == "hash":
            cell = torch.floor((x - x.amin(0)) / radius).long() + 1
            widths = cell.amax(0) + 2
            wx, wy, wz = [int(v) for v in widths.tolist()]
            if wx * wy * wz >= 2**62:
                raise ValueError(
                    "Spatial hash key overflow: reduce scene extent or increase radius"
                )

            def encode(c):
                return (c[..., 0] * wy + c[..., 1]) * wz + c[..., 2]

            keys = encode(cell)
            sorted_key, permutation = keys.sort()

        def query(a, b):
            if backend == "brute":
                distance2 = ((x[a:b, None] - x[None]) ** 2).sum(-1)
                dd, ss = (distance2 <= radius * radius).nonzero(as_tuple=True)
                return dd + a, ss
            codes = encode(cell[a:b, None] + shifts[None]).flatten()
            left = torch.searchsorted(sorted_key, codes)
            right = torch.searchsorted(sorted_key, codes, right=True)
            lengths = right - left
            ncandidate = int(lengths.sum())
            if ncandidate > candidate_limit:
                if b - a <= 1:
                    raise RuntimeError(
                        "One radius query exceeds candidate_limit; increase it explicitly"
                    )
                middle = (a + b) // 2
                d1, s1 = query(a, middle)
                d2, s2 = query(middle, b)
                return torch.cat([d1, d2]), torch.cat([s1, s2])
            owner = torch.repeat_interleave(
                torch.arange(len(codes), device=x.device), lengths
            )
            local = torch.arange(ncandidate, device=x.device) - torch.repeat_interleave(
                lengths.cumsum(0) - lengths, lengths
            )
            ss = permutation[left[owner] + local]
            dd = owner // 27 + a
            valid = ((x[ss] - x[dd]) ** 2).sum(-1) <= radius * radius
            return dd[valid], ss[valid]

        for a in range(0, n, query_chunk):
            dd, ss = query(a, min(a + query_chunk, n))
            dd, ss = dd + start, ss + start
            order = torch.argsort(
                dd * (len(coord) + 1) + priority[ss], stable=True
            )
            yield dd[order], ss[order]
        start = end


@dataclass
class StreamedRadiusGraph:
    coord: torch.Tensor
    offset: torch.Tensor
    radius: float
    priority: torch.Tensor
    backend: str
    query_chunk: int
    candidate_limit: int

    def edges(self):
        return iter_radius_edges(
            self.coord, self.offset, self.radius, self.priority,
            self.backend, self.query_chunk, self.candidate_limit,
        )


def graph_edges(graph):
    if isinstance(graph, StreamedRadiusGraph):
        return graph.edges()
    return iter(((graph.dst, graph.src),))


@torch.no_grad()
def radius_graph(
    coord, offset, radius, priority, backend="hash", query_chunk=256,
    candidate_limit=2_000_000, max_edges=8_000_000,
    stream_on_overflow=False,
):
    """Exact directed radius graph including self, independently per cloud.

    With stream_on_overflow, max_edges limits materialization, not coverage.
    Larger graphs are replayed in bounded query chunks for exact reductions.
    """
    streamed = StreamedRadiusGraph(
        coord, offset, radius, priority, backend, query_chunk, candidate_limit
    )
    all_dst, all_src, total = [], [], 0
    for dst, src in streamed.edges():
        total += len(dst)
        if total > max_edges:
            if stream_on_overflow:
                return streamed
            raise RuntimeError(
                "Complete radius graph exceeds max_radius_edges. "
                "Enable stream_on_overflow or reduce radius/density; "
                "graph was not truncated."
            )
        all_dst.append(dst)
        all_src.append(src)
    return RadiusGraph(torch.cat(all_dst), torch.cat(all_src), radius)


@torch.no_grad()
def cap_messages(graph, priority, size, max_neighbors):
    """Keep self plus highest-priority nonself edges; coverage stays uncapped."""
    if isinstance(graph, StreamedRadiusGraph):
        chunks = [
            cap_messages(RadiusGraph(dst, src, graph.radius), priority,
                         size, max_neighbors)
            for dst, src in graph.edges()
        ]
        return RadiusGraph(
            torch.cat([chunk.dst for chunk in chunks]),
            torch.cat([chunk.src for chunk in chunks]), graph.radius,
        )
    dst, src = graph.dst, graph.src
    self_edge = dst == src
    dd, ss = dst[~self_edge], src[~self_edge]
    edge_count = torch.bincount(dd, minlength=size)
    begin = edge_count.cumsum(0) - edge_count
    local_rank = torch.arange(len(dd), device=dst.device) - begin[dd]
    keep = local_rank < max_neighbors - 1
    dd = torch.cat([dst[self_edge], dd[keep]])
    ss = torch.cat([src[self_edge], ss[keep]])
    order = torch.argsort(dd * (size + 1) + priority[ss], stable=True)
    return RadiusGraph(dd[order], ss[order], graph.radius)


@torch.no_grad()
def radius_cover(graph, batch, priority):
    """Parallel local-minimum MIS, then highest-priority eligible-center parent.

    All radius edges are used. No fixed iteration cap: continue to maximality.
    Self is forced to own its center cluster, so every coarse cluster is nonempty.
    """
    n, device = len(priority), priority.device
    active = torch.ones(n, device=device, dtype=torch.bool)
    selected = torch.zeros_like(active)
    while bool(active.any()):
        best = torch.full((n,), n, device=device, dtype=torch.long)
        for dst, src in graph_edges(graph):
            eligible = active[dst] & active[src]
            best.scatter_reduce_(0, dst[eligible], priority[src[eligible]], reduce="amin")
        chosen = active & (priority == best)
        selected |= chosen
        suppressed = torch.zeros(n, device=device, dtype=torch.long)
        for dst, src in graph_edges(graph):
            suppressed.scatter_reduce_(0, dst, chosen[src].long(), reduce="amax")
        active &= ~suppressed.bool()
    centers = selected.nonzero(as_tuple=True)[0]
    centers = centers[
        torch.argsort(batch[centers] * (n + 1) + priority[centers], stable=True)
    ]
    best = torch.full((n,), n, device=device, dtype=torch.long)
    for dst, src in graph_edges(graph):
        eligible = selected[src]
        best.scatter_reduce_(0, dst[eligible], priority[src[eligible]], reduce="amin")
    rank_to_parent = torch.full((n,), -1, device=device, dtype=torch.long)
    rank_to_parent[priority[centers]] = torch.arange(len(centers), device=device)
    parent = rank_to_parent[best]
    parent[centers] = torch.arange(len(centers), device=device)
    return parent, centers


@dataclass
class PatchLayout:
    index: torch.Tensor
    mask: torch.Tensor
    patch_of: torch.Tensor
    slot_of: torch.Tensor
    cluster_of_patch: torch.Tensor


@torch.no_grad()
def patch_layout(parent, priority, patch_size):
    n, m = len(parent), int(parent.max()) + 1
    order = torch.argsort(parent * (n + 1) + priority, stable=True)
    cluster_count = torch.bincount(parent, minlength=m)
    begin = cluster_count.cumsum(0) - cluster_count
    position = torch.arange(n, device=parent.device) - begin[parent[order]]
    width = min(patch_size, int(cluster_count.max()))
    nchunk = (cluster_count + width - 1) // width
    chunk_begin = nchunk.cumsum(0) - nchunk
    patches = chunk_begin[parent[order]] + position // width
    slots = position % width
    index = torch.full(
        (int(nchunk.sum()), width), n, device=parent.device, dtype=torch.long
    )
    index[patches, slots] = order
    patch_of, slot_of = torch.empty_like(parent), torch.empty_like(parent)
    patch_of[order], slot_of[order] = patches, slots
    return PatchLayout(
        index,
        index != n,
        patch_of,
        slot_of,
        torch.repeat_interleave(torch.arange(m, device=parent.device), nchunk),
    )


@dataclass
class StageCache:
    graph: RadiusGraph
    parent: torch.Tensor
    centers: torch.Tensor
    layout: PatchLayout


@dataclass
class VState:
    coord: torch.Tensor
    feat: torch.Tensor
    vec: Optional[torch.Tensor]
    offset: torch.Tensor
    point_id: torch.Tensor
    sample_index: torch.Tensor
    pool_parent: Optional[torch.Tensor] = None


class ScalarPatchAttention(nn.Module):
    """Invariant scalar SDPA; no coordinate projection and no rotary embedding."""

    def __init__(self, channels, heads, attn_drop=0.0, chunk_size=128):
        super().__init__()
        self.heads, self.width = heads, channels // heads
        self.chunk_size, self.attn_drop = chunk_size, attn_drop
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.out = nn.Linear(channels, channels)
        self.summary = nn.Linear(channels, channels)

    def forward(self, feat, cache):
        layout, parent = cache.layout, cache.parent
        m = len(cache.centers)
        # Shared cluster summary connects chunks without dense all-cluster attention.
        feat = feat + self.summary(segment_mean(feat, parent, m))[parent]
        qkv = self.qkv(feat)
        padded = torch.cat([qkv, qkv.new_zeros(1, qkv.shape[-1])])
        outputs = []
        for start in range(0, len(layout.index), self.chunk_size):
            ii = layout.index[start : start + self.chunk_size]
            mm = layout.mask[start : start + self.chunk_size]
            q, k, v = (
                padded[ii]
                .reshape(len(ii), ii.shape[1], 3, self.heads, self.width)
                .permute(2, 0, 3, 1, 4)
            )
            value = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mm[:, None, None, :],
                dropout_p=self.attn_drop if self.training else 0.0,
            )
            outputs.append(value.transpose(1, 2).reshape(len(ii), ii.shape[1], -1))
        patches = torch.cat(outputs)
        return self.out(patches[layout.patch_of, layout.slot_of])


class QuaternionRadiusAttention(nn.Module):
    """Independent attention from Q_i * conjugate(K_j), with equivariant values.

    Q=(a,u), K=(b,w): real=a*b+dot(u,w), imag=-a*w+b*u-cross(u,w).
    Real, norm(imag), dot(imag, relative displacement), and distance form logits.
    No unrestricted quaternion weight matrix or componentwise spatial activation.
    """

    def __init__(
        self,
        channels,
        vector_channels,
        quaternion_channels=4,
        groups=4,
        geometry_width=32,
        chunk_size=65536,
        checkpoint_edges=True,
        attn_drop=0.0,
    ):
        super().__init__()
        cv, cq = vector_channels, quaternion_channels
        self.cv, self.cq, self.groups = cv, cq, groups
        self.chunk_size, self.checkpoint_edges = chunk_size, checkpoint_edges
        self.attn_drop = attn_drop
        self.real_qk = nn.Linear(channels, 2 * cq)
        self.vector_q, self.vector_k = VLinear(cv, cq), VLinear(cv, cq)
        self.vector_value = VLinear(cv, cv)
        self.relation_vector = VLinear(cq, cv)
        self.logits = mlp(3 * cq + 1, geometry_width, groups)
        self.values = mlp(3 * cq + 1, geometry_width, 3 * cv + channels)
        self.scalar_out, self.vector_out = nn.Linear(channels, channels), VLinear(
            cv, cv
        )

    def _relation(self, aq, bk, uq, wk, dst, src, rel):
        a, b = aq[dst].float(), bk[src].float()
        u, w = uq[dst].float(), wk[src].float()
        real = a * b + (u * w).sum(-1)
        imag = -a[..., None] * w + b[..., None] * u - torch.cross(u, w, dim=-1)
        invariants = torch.cat(
            [
                real,
                safe_norm(imag),
                (imag * rel[:, None]).sum(-1),
                safe_norm(rel, keepdim=True),
            ],
            -1,
        )
        return invariants, imag

    def _logit_chunk(self, aq, bk, uq, wk, dst, src, rel):
        invariant, _ = self._relation(aq, bk, uq, wk, dst, src, rel)
        return self.logits(invariant.to(aq.dtype))

    def _message_chunk(self, aq, bk, uq, wk, value, dst, src, rel, weights):
        invariant, imag = self._relation(aq, bk, uq, wk, dst, src, rel)
        coefficients = self.values(invariant.to(aq.dtype))
        av, bv, cv, scalar = torch.split(
            coefficients,
            [self.cv, self.cv, self.cv, coefficients.shape[-1] - 3 * self.cv],
            -1,
        )
        relation = self.relation_vector(imag.to(value.dtype)).float()
        message = (
            av.float()[..., None] * value[src].float()
            + bv.float()[..., None] * relation
            + cv.float()[..., None] * rel[:, None]
        )
        expanded = weights.repeat_interleave(self.cv // self.groups, -1)
        vector = message * expanded[..., None]
        scalar = scalar.float() * weights.mean(-1, keepdim=True)
        return scalar, vector

    def forward(self, feat, vec, coord, graph):
        aq, bk = self.real_qk(feat).chunk(2, -1)
        uq, wk = self.vector_q(vec), self.vector_k(vec)
        value = self.vector_value(vec)
        dst, src, n = graph.dst, graph.src, len(feat)
        rel = ((coord[src] - coord[dst]) / graph.radius).detach()
        enabled = self.training and self.checkpoint_edges and torch.is_grad_enabled()

        def run(fn, *args):
            return checkpoint(fn, *args, use_reentrant=False) if enabled else fn(*args)

        logits = torch.cat(
            [
                run(
                    self._logit_chunk,
                    aq,
                    bk,
                    uq,
                    wk,
                    dst[a : a + self.chunk_size],
                    src[a : a + self.chunk_size],
                    rel[a : a + self.chunk_size],
                )
                for a in range(0, len(dst), self.chunk_size)
            ]
        )
        weights = segment_softmax(logits, dst, n)
        weights = F.dropout(weights, self.attn_drop, self.training)
        so = torch.zeros(n, feat.shape[-1], device=feat.device, dtype=torch.float32)
        vo = torch.zeros(n, self.cv, 3, device=feat.device, dtype=torch.float32)
        for a in range(0, len(dst), self.chunk_size):
            dd, ss, rr = (
                dst[a : a + self.chunk_size],
                src[a : a + self.chunk_size],
                rel[a : a + self.chunk_size],
            )
            si, vi = run(
                self._message_chunk,
                aq,
                bk,
                uq,
                wk,
                value,
                dd,
                ss,
                rr,
                weights[a : a + self.chunk_size],
            )
            so = so.index_add(0, dd, si)
            vo = vo.index_add(0, dd, vi)
        return self.scalar_out(so.to(feat.dtype)), self.vector_out(vo.to(vec.dtype))


class DualAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        vector_channels,
        heads,
        quaternion_channels,
        geometry_groups,
        geometry_width,
        mlp_ratio=2.0,
        proj_drop=0.0,
        attn_drop=0.0,
        patch_chunk_size=128,
        edge_chunk_size=65536,
        checkpoint_edges=True,
        use_quaternion_attention=True,
        use_scalar_attention=True,
    ):
        super().__init__()
        self.use_scalar, self.use_quaternion = (
            use_scalar_attention,
            use_quaternion_attention,
        )
        self.norm1, self.norm2 = ExactLayerNorm(channels), ExactLayerNorm(channels)
        if self.use_scalar:
            self.scalar_attention = ScalarPatchAttention(
                channels, heads, attn_drop, patch_chunk_size
            )
        if self.use_quaternion:
            self.vector_norm = VNRMSNorm(vector_channels)
            self.quaternion_attention = QuaternionRadiusAttention(
                channels,
                vector_channels,
                quaternion_channels,
                geometry_groups,
                geometry_width,
                edge_chunk_size,
                checkpoint_edges,
                attn_drop,
            )
            self.vector_update = VNBlock(vector_channels)
            self.geometry_gate = nn.Linear(channels, channels)
        self.readout = VectorReadout(vector_channels)
        self.v2s = nn.Sequential(
            ExactLayerNorm(self.readout.out_dim),
            nn.Linear(self.readout.out_dim, channels),
        )
        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, channels),
            nn.Dropout(proj_drop),
        )

    def forward(self, feat, vec, coord, cache):
        normalized = self.norm1(feat)
        scalar_delta = (
            self.scalar_attention(normalized, cache) if self.use_scalar else 0
        )
        if self.use_quaternion:
            geom, vector = self.quaternion_attention(
                normalized, self.vector_norm(vec), coord, cache.graph
            )
            scalar_delta = (
                scalar_delta + self.geometry_gate(normalized).sigmoid() * geom
            )
            # Keep the initial vector residual branch close to identity.
            vec = vec + 0.1 * self.vector_update(vector)
        feat = feat + scalar_delta + self.v2s(self.readout(vec))
        return feat + self.ffn(self.norm2(feat)), vec


class VectorSeed(nn.Module):
    def __init__(self, scalar_channels, vector_channels, hidden=16, chunk_size=65536):
        super().__init__()
        self.key = nn.Linear(scalar_channels, hidden)
        self.coefficient = mlp(2 * hidden + 1, hidden, vector_channels)
        self.chunk_size, self.cv = chunk_size, vector_channels

    def forward(self, feat, coord, graph):
        key, n = self.key(feat), len(feat)
        out = torch.zeros(n, self.cv, 3, device=feat.device, dtype=torch.float32)
        for a in range(0, len(graph.dst), self.chunk_size):
            dst, src = (
                graph.dst[a : a + self.chunk_size],
                graph.src[a : a + self.chunk_size],
            )
            rel = ((coord[src] - coord[dst]) / graph.radius).detach()
            coef = self.coefficient(
                torch.cat(
                    [key[dst], key[src], safe_norm(rel, keepdim=True).to(key.dtype)], -1
                )
            )
            out = out.index_add(0, dst, coef.float()[..., None] * rel[:, None])
        denom = torch.bincount(graph.dst, minlength=n).float()[:, None, None]
        return divide_or_zero(out, denom).to(feat.dtype)


class Stage(nn.Module):
    def __init__(self, channels, vector_channels, depth, heads, **options):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DualAttentionBlock(channels, vector_channels, heads, **options)
                for _ in range(depth)
            ]
        )

    def forward(self, state, cache):
        feat, vec = state.feat, state.vec
        for block in self.blocks:
            feat, vec = block(feat, vec, state.coord, cache)
        return VState(
            state.coord, feat, vec, state.offset, state.point_id, state.sample_index
        )


class Downsample(nn.Module):
    def __init__(self, cin, cout, vin, vout):
        super().__init__()
        self.scalar = nn.Sequential(
            nn.Linear(2 * cin, cout), ExactLayerNorm(cout), nn.GELU()
        )
        self.vector_weight = nn.Linear(cin, vin)
        self.vector = VLinear(vin, vout)

    def forward(self, state, cache):
        parent, center = cache.parent, cache.centers
        m = len(center)
        feat = self.scalar(
            torch.cat(
                [
                    segment_mean(state.feat, parent, m),
                    segment_max(state.feat, parent, m),
                ],
                -1,
            )
        )
        weight = segment_softmax(self.vector_weight(state.feat), parent, m)
        vector = segment_sum(weight[..., None] * state.vec.float(), parent, m)
        vector = self.vector(vector.to(state.vec.dtype))
        with torch.no_grad():
            coord = segment_mean(state.coord, parent, m)
            old_batch = torch.repeat_interleave(
                torch.arange(len(state.offset), device=parent.device),
                counts(state.offset),
            )
            new_count = torch.bincount(old_batch[center], minlength=len(state.offset))
            offset = new_count.cumsum(0).to(state.offset.dtype)
        return VState(
            coord,
            feat,
            vector,
            offset,
            state.point_id[center],
            state.sample_index[center],
        )


class Upsample(nn.Module):
    def __init__(self, coarse, fine, vcoarse, vfine):
        super().__init__()
        self.scalar = nn.Sequential(
            nn.Linear(coarse + fine, fine), ExactLayerNorm(fine), nn.GELU()
        )
        self.vector = VLinear(vcoarse + vfine, vfine)

    def forward(self, coarse, fine, parent):
        feat = self.scalar(torch.cat([fine.feat, coarse.feat[parent]], -1))
        vector = self.vector(torch.cat([fine.vec, coarse.vec[parent]], 1))
        return VState(
            fine.coord, feat, vector, fine.offset, fine.point_id, fine.sample_index
        )


class GlobalScalarContext(nn.Module):
    """Small per-cloud learned latent-token cross-attention at the bottleneck."""

    def __init__(self, channels, heads, tokens=8):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(tokens, channels) / math.sqrt(channels))
        self.norm = ExactLayerNorm(channels)
        self.gather = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.broadcast = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, feat, offset):
        outputs, start = [], 0
        for end in offset.long().tolist():
            x = self.norm(feat[start:end])[None]
            token, _ = self.gather(
                self.tokens[None].to(x.dtype), x, x, need_weights=False
            )
            out, _ = self.broadcast(x, token, token, need_weights=False)
            outputs.append(out[0])
            start = end
        return feat + torch.cat(outputs)


@MODELS.register_module("SO3-Quaternion-Dual-v1")
class SO3QuaternionDualNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        scalar_feature_indices=(0, 1, 2),
        vector_feature_indices=None,
        channels=(48, 96, 192, 384),
        vector_channels=(8, 8, 16, 16),
        depths=(2, 2, 4, 4),
        heads=(3, 6, 8, 12),
        radii=(0.4, 0.8, 1.6, 3.2),
        patch_sizes=(32, 64, 64, 128),
        max_neighbors=(24, 24, 32, 32),
        quaternion_channels=4,
        geometry_groups=4,
        geometry_width=32,
        decoder_depths=(1, 1, 1),
        decoder_scalar_attention=False,
        use_scalar_attention=True,
        use_quaternion_attention=True,
        head_channels=64,
        mlp_ratio=2.0,
        attn_drop=0.0,
        proj_drop=0.0,
        edge_chunk_size=65536,
        patch_chunk_size=128,
        checkpoint_edges=True,
        radius_backend="hash",
        radius_query_chunk=256,
        radius_candidate_limit=2_000_000,
        max_radius_edges=8_000_000,
        global_context_tokens=8,
        readout_triple=True,
    ):
        super().__init__()
        n = len(channels)
        sequences = (vector_channels, depths, heads, radii, patch_sizes, max_neighbors)
        if n < 1 or any(len(v) != n for v in sequences):
            raise ValueError(
                "All encoder stage sequences must have equal nonzero length"
            )
        if any(c < 2 or h < 1 or c % h for c, h in zip(channels, heads)):
            raise ValueError("Scalar widths >=2 must divide their head counts")
        if geometry_groups < 1 or quaternion_channels < 1 or geometry_width < 2:
            raise ValueError("Invalid quaternion geometry dimensions")
        if any(v < 1 or v % geometry_groups for v in vector_channels):
            raise ValueError(
                "Vector widths must be positive multiples of geometry_groups"
            )
        if any(d < 1 for d in depths) or any(
            k < 1 for k in tuple(patch_sizes) + tuple(max_neighbors)
        ):
            raise ValueError("Depths, patch sizes and neighbor limits must be positive")
        if any(not math.isfinite(r) or r <= 0 for r in radii):
            raise ValueError("radii must be finite and positive in coordinate units")
        if len(decoder_depths) != n - 1 or any(d < 1 for d in decoder_depths):
            raise ValueError("decoder_depths must contain nstages-1 positive depths")
        if (
            min(
                edge_chunk_size,
                patch_chunk_size,
                radius_query_chunk,
                radius_candidate_limit,
                max_radius_edges,
            )
            < 1
        ):
            raise ValueError("Chunk and memory limits must be positive")
        if not 0 <= attn_drop < 1 or not 0 <= proj_drop < 1 or mlp_ratio <= 0:
            raise ValueError("Invalid dropout or mlp_ratio")
        if global_context_tokens < 0 or head_channels < 2 or in_channels < 1:
            raise ValueError("Invalid context/head/input dimensions")
        indices = [] if scalar_feature_indices is None else list(scalar_feature_indices)
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= in_channels for i in indices
        ):
            raise ValueError("Invalid invariant scalar feature indices")
        triples = (
            []
            if vector_feature_indices is None
            else [tuple(t) for t in vector_feature_indices]
        )
        if any(
            len(t) != 3 or len(set(t)) != 3 or any(i < 0 or i >= in_channels for i in t)
            for t in triples
        ):
            raise ValueError("vector_feature_indices must contain triples")
        if set(indices) & {i for t in triples for i in t}:
            raise ValueError("Do not feed vector components into the scalar stem")
        self.register_buffer(
            "_scalar_indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_vector_indices",
            torch.tensor(triples, dtype=torch.long).reshape(-1, 3),
            persistent=False,
        )
        self.in_channels, self.head_channels = in_channels, head_channels
        self.channels, self.vector_channels = tuple(channels), tuple(vector_channels)
        self.radii, self.patch_sizes, self.max_neighbors = (
            tuple(radii),
            tuple(patch_sizes),
            tuple(max_neighbors),
        )
        self.search_options = dict(
            backend=radius_backend,
            query_chunk=radius_query_chunk,
            candidate_limit=radius_candidate_limit,
            max_edges=max_radius_edges,
        )
        self.stem = nn.Sequential(
            nn.Linear(max(1, len(indices)), channels[0]),
            ExactLayerNorm(channels[0]),
            nn.GELU(),
        )
        self.seed = VectorSeed(
            channels[0], vector_channels[0], chunk_size=edge_chunk_size
        )
        if triples:
            self.vector_input = VLinear(len(triples), vector_channels[0])
        options = dict(
            quaternion_channels=quaternion_channels,
            geometry_groups=geometry_groups,
            geometry_width=geometry_width,
            mlp_ratio=mlp_ratio,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            patch_chunk_size=patch_chunk_size,
            edge_chunk_size=edge_chunk_size,
            checkpoint_edges=checkpoint_edges,
            use_quaternion_attention=use_quaternion_attention,
        )
        self.stages = nn.ModuleList(
            [
                Stage(
                    channels[i],
                    vector_channels[i],
                    depths[i],
                    heads[i],
                    use_scalar_attention=use_scalar_attention,
                    **options,
                )
                for i in range(n)
            ]
        )
        self.down = nn.ModuleList(
            [
                Downsample(
                    channels[i],
                    channels[i + 1],
                    vector_channels[i],
                    vector_channels[i + 1],
                )
                for i in range(n - 1)
            ]
        )
        self.up = nn.ModuleList(
            [
                Upsample(
                    channels[i + 1],
                    channels[i],
                    vector_channels[i + 1],
                    vector_channels[i],
                )
                for i in reversed(range(n - 1))
            ]
        )
        self.decoder = nn.ModuleList(
            [
                Stage(
                    channels[i],
                    vector_channels[i],
                    decoder_depths[i],
                    heads[i],
                    use_scalar_attention=decoder_scalar_attention,
                    **options,
                )
                for i in reversed(range(n - 1))
            ]
        )
        self.context = (
            GlobalScalarContext(channels[-1], heads[-1], global_context_tokens)
            if global_context_tokens
            else None
        )
        self.final_readout = VectorReadout(vector_channels[0], triple=readout_triple)
        self.output = nn.Sequential(
            nn.Linear(channels[0] + self.final_readout.out_dim, head_channels),
            ExactLayerNorm(head_channels),
            nn.GELU(),
        )

    @torch.no_grad()
    def _cache(self, state, level):
        priority = priority_ranks(state.point_id)
        full_graph = radius_graph(
            state.coord,
            state.offset,
            self.radii[level],
            priority,
            stream_on_overflow=True,
            **self.search_options,
        )
        batch = torch.repeat_interleave(
            torch.arange(len(state.offset), device=state.coord.device),
            counts(state.offset),
        )
        parent, centers = radius_cover(full_graph, batch, priority)
        graph = cap_messages(
            full_graph, priority, len(state.coord), self.max_neighbors[level]
        )
        layout = patch_layout(parent, priority, self.patch_sizes[level])
        return StageCache(graph, parent, centers, layout)

    def forward(self, data_dict, return_stages=False):
        point = Point(dict(data_dict))
        coord, feat, offset = point.coord.detach().float(), point.feat, point.offset
        if (
            coord.ndim != 2
            or coord.shape[-1] != 3
            or not len(coord)
            or not bool(torch.isfinite(coord).all())
        ):
            raise ValueError("coord must be a nonempty finite [N,3] tensor")
        if (
            feat.shape != (len(coord), self.in_channels)
            or not feat.is_floating_point()
            or not bool(torch.isfinite(feat).all())
        ):
            raise ValueError(f"feat must be finite floating [N,{self.in_channels}]")
        if (
            offset.dtype not in (torch.int32, torch.int64)
            or offset.ndim != 1
            or not len(offset)
        ):
            raise ValueError("offset must be an integer cumulative cloud-size vector")
        if int(offset[-1]) != len(coord) or bool((counts(offset) <= 0).any()):
            raise ValueError("offset must end at N and have positive cloud sizes")
        if feat.device != coord.device or offset.device != coord.device:
            raise ValueError("coord, feat and offset must share a device")
        identity = point.get("point_id", torch.arange(len(coord), device=coord.device))
        if (
            identity.shape != (len(coord),)
            or identity.dtype not in (torch.int32, torch.int64)
            or identity.device != coord.device
        ):
            raise ValueError("point_id must be integer [N] on the coordinate device")
        start = 0
        for end in offset.long().tolist():
            if len(identity[start:end].unique()) != end - start:
                raise ValueError("point_id must be unique within each cloud")
            start = end
        scalars = (
            feat.index_select(1, self._scalar_indices)
            if self._scalar_indices.numel()
            else feat.new_ones(len(coord), 1)
        )
        state = VState(
            coord,
            self.stem(scalars),
            None,
            offset,
            identity,
            torch.arange(len(coord), device=coord.device),
        )
        stages, caches = [], []
        for i, stage in enumerate(self.stages):
            cache = self._cache(state, i)
            if i == 0:
                state.vec = self.seed(state.feat, state.coord, cache.graph)
                if self._vector_indices.numel():
                    state.vec = state.vec + self.vector_input(
                        feat[:, self._vector_indices]
                    )
            state = stage(state, cache)
            state.pool_parent = cache.parent
            stages.append(state)
            caches.append(cache)
            if i < len(self.down):
                state = self.down[i](state, cache)
        if self.context is not None:
            state = VState(
                state.coord,
                self.context(state.feat, state.offset),
                state.vec,
                state.offset,
                state.point_id,
                state.sample_index,
            )
        for level, up, stage in zip(
            reversed(range(len(stages) - 1)), self.up, self.decoder
        ):
            state = up(state, stages[level], caches[level].parent)
            state = stage(state, caches[level])
        point.feat = self.output(
            torch.cat([state.feat, self.final_readout(state.vec)], -1)
        )
        point["vector_feat"] = state.vec
        return (point, stages) if return_stages else point


@MODELS.register_module("SO3QuaternionDualSegmentor")
class SO3QuaternionDualSegmentor(nn.Module):
    """Same train/eval return convention as the supplied segmentor.

    Auxiliary supervision uses pooled class distributions, rather than one FPS
    representative's label. CE is used for auxiliary heads only; final criteria
    remain configurable through Pointcept (e.g. CE + Lovasz).
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
        debug_numerics=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        if self.backbone.head_channels != backbone_out_channels:
            raise ValueError("backbone_out_channels must match backbone.head_channels")
        if num_classes < 1 or not math.isfinite(aux_loss_weight) or aux_loss_weight < 0:
            raise ValueError("Invalid class count or auxiliary weight")
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.criteria = build_criteria(criteria)
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.freeze_backbone, self.aux_loss_weight = freeze_backbone, aux_loss_weight
        n = len(self.backbone.channels)
        self.aux_weights = (
            tuple(aux_stage_weights) if aux_stage_weights is not None else (1.0,) * n
        )
        if len(self.aux_weights) != n or any(
            not math.isfinite(w) or w < 0 for w in self.aux_weights
        ):
            raise ValueError("Invalid auxiliary stage weights")
        self.aux_heads = nn.ModuleDict(
            {
                str(i): nn.Linear(c, num_classes)
                for i, (c, w) in enumerate(
                    zip(self.backbone.channels, self.aux_weights)
                )
                if deep_supervision and aux_loss_weight > 0 and w > 0
            }
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        if debug_numerics:
            # Keep anomaly tracing active through the trainer's loss.backward().
            # It records forward stacks and names backward ops returning NaNs.
            # Diagnostic process-wide setting; restart with debug_numerics=False
            # after diagnosis to restore normal training performance.
            torch.autograd.set_detect_anomaly(True, check_nan=True)
            # Diagnostic only: these checks synchronize CUDA and slow training.
            # Hooks raise before an optimizer step can consume invalid gradients.
            def check_tensor(value, location):
                if isinstance(value, torch.Tensor):
                    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"Non-finite tensor at {location}; shape={tuple(value.shape)}")
                elif isinstance(value, (tuple, list)):
                    for i, item in enumerate(value):
                        check_tensor(item, f"{location}[{i}]")
                elif isinstance(value, dict):
                    for key, item in value.items():
                        check_tensor(item, f"{location}.{key}")

            def watch_gradient(value, location):
                if isinstance(value, torch.Tensor):
                    if value.requires_grad:
                        value.register_hook(gradient_hook(f"activation:{location}"))
                elif isinstance(value, (tuple, list)):
                    for i, item in enumerate(value):
                        watch_gradient(item, f"{location}[{i}]")
                elif isinstance(value, dict):
                    for key, item in value.items():
                        watch_gradient(item, f"{location}.{key}")

            def output_hook(name):
                def hook(module, inputs, output):
                    check_tensor(output, f"forward:{name}")
                    watch_gradient(output, name)
                return hook

            def gradient_hook(name):
                def hook(gradient):
                    check_tensor(gradient, f"gradient:{name}")
                    return gradient
                return hook

            for name, module in self.named_modules():
                if name:
                    module.register_forward_hook(output_hook(name))
            for name, parameter in self.named_parameters():
                if parameter.requires_grad:
                    parameter.register_hook(gradient_hook(name))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _loss(self, logits, target):
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite segmentation logits before loss")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            if not bool((target != self.ignore_index).any()):
                return logits.float().sum() * 0
            loss = self.criteria(logits.float(), target)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Non-finite segmentation loss with finite logits")
            return loss

    def forward(self, input_dict, return_point=False):
        target = input_dict.get("segment")
        if self.training and target is None:
            raise ValueError("Training requires segment labels")
        if target is not None:
            if (
                target.ndim != 1
                or target.dtype != torch.long
                or len(target) != len(input_dict["coord"])
            ):
                raise ValueError("segment must be long [N]")
            invalid = (target != self.ignore_index) & (
                (target < 0) | (target >= self.num_classes)
            )
            if bool(invalid.any()):
                raise ValueError("Invalid segment labels")
        need = self.training and bool(len(self.aux_heads))
        if self.freeze_backbone:
            with torch.no_grad():
                result = self.backbone(input_dict, return_stages=need)
        else:
            result = self.backbone(input_dict, return_stages=need)
        point, stages = result if need else (result, None)
        logits = self.seg_head(point.feat)
        out = {"seg_logits": logits} if not self.training else {}
        if target is not None:
            final = self._loss(logits, target)
            out["loss"] = final
            if need:
                # Reuse saved fine-to-coarse maps, without rebuilding any graph.
                valid = target != self.ignore_index
                histogram = (
                    F.one_hot(target.masked_fill(~valid, 0), self.num_classes).float()
                    * valid[:, None]
                )
                aux, weight_total = final * 0, 0.0
                for i, st in enumerate(stages):
                    key = str(i)
                    if key in self.aux_heads:
                        prediction = self.aux_heads[key](st.feat).float()
                        count = histogram.sum(-1)
                        distribution = divide_or_zero(histogram, count[:, None])
                        per_point = -(distribution * prediction.log_softmax(-1)).sum(-1)
                        loss = divide_or_zero(
                            (per_point * (count > 0)).sum(), (count > 0).sum().float()
                        )
                        aux = aux + self.aux_weights[i] * loss
                        weight_total += self.aux_weights[i]
                        out[f"loss_aux_stage_{i}"] = loss.detach()
                    if i < len(stages) - 1:
                        histogram = segment_sum(
                            histogram, st.pool_parent, len(stages[i + 1].feat)
                        )
                aux = aux / weight_total
                out.update(
                    loss=final + self.aux_loss_weight * aux,
                    loss_final=final.detach(),
                    loss_aux=aux.detach(),
                )
        if return_point:
            out["point"] = point
        return out
