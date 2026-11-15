"""Distance + relative-angle Point Transformer for Pointcept.

Registry: SO3-DistanceAngle-v1 (backbone), DistanceAngleSegmentor (wrapper).
No quaternion operations, geometric vector feature stream, or Q/K/V transport.
Attention: attention_type="grouped_vector" or "scalar" (multi-head scalar).
See README.md and example_model_config.py shipped with this module.
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


class DistanceAngleEncoding(nn.Module):
    """Embed edges by distances and angles to M nearest nonzero references.

    Pair: [log1p(d/sigma), d/rho, log1p(rho/sigma)].
    Triplet: [log1p(d/sigma), log1p(d_ref/sigma),
              d/rho, d_ref/rho, cos(theta), sin(theta)].
    A shared triplet MLP + masked mean treats reference order symmetrically.
    Self/coincident edges carry valid pair encoding but no defined angle.
    'distance' mode allocates no angle MLP, for a clean ablation.
    No covariance, RBF, absolute XYZ, or learned Cartesian frames are used.
    """

    def __init__(
        self,
        width=16,
        refs=3,
        mode="distance_angle",
        chunk_size=256,
        checkpoint_edges=True,
    ):
        super().__init__()
        if mode not in ("distance", "distance_angle"):
            raise ValueError("geometry_mode must be distance or distance_angle")
        self.width, self.refs = width, refs
        self.mode, self.chunk_size = mode, chunk_size
        self.checkpoint_edges = checkpoint_edges
        self.pair = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        if mode == "distance_angle":
            self.triplet = nn.Sequential(
                nn.Linear(4, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
        self.norm = nn.LayerNorm(width)

    def _embed(self, pair, triplet, valid):
        result = self.pair(pair)
        if self.mode == "distance_angle":
            z = self.triplet(triplet)
            weight = valid.to(z.dtype)[..., None]
            result = result + (z * weight).sum(2) / weight.sum(2).clamp_min(1)
        return self.norm(result)

    def forward(self, rel, dist, mask):
        outputs = []
        for start in range(0, len(dist), self.chunk_size):
            stop = min(start + self.chunk_size, len(dist))
            # Explicit FP32 geometric calculations, including under AMP.
            with torch.no_grad(), torch.autocast(
                device_type=dist.device.type, enabled=False
            ):
                r, d, keep = (
                    rel[start:stop].float(),
                    dist[start:stop].float(),
                    mask[start:stop],
                )
                radius = d.masked_fill(~keep, 0).amax(1, keepdim=True)
                safe_radius = torch.where(radius > 0, radius, torch.ones_like(radius))
                relative = d / safe_radius
                pair = relative[..., None]
                if self.mode == "distance_angle":
                    nonzero = keep & (d > 1e-8)
                    m = min(self.refs, d.shape[1])
                    anchor = d.masked_fill(~nonzero, float("inf")).argsort(
                        dim=1, stable=True
                    )[:, :m]
                    ad = d.gather(1, anchor)
                    ar = r.gather(1, anchor[..., None].expand(-1, -1, 3))
                    av = nonzero.gather(1, anchor)
                    unit = r / d.clamp_min(1e-8)[..., None]
                    au = ar / ad.clamp_min(1e-8)[..., None]
                    cosine = torch.einsum("nki,nmi->nkm", unit, au).clamp(-1, 1)
                    # Cross norm is more stable than sqrt(1-cos^2) at 0/pi.
                    sine = torch.linalg.vector_norm(
                        torch.cross(
                            unit[:, :, None].expand(-1, -1, m, -1),
                            au[:, None].expand(-1, d.shape[1], -1, -1),
                            dim=-1,
                        ),
                        dim=-1,
                    )
                    valid = nonzero[:, :, None] & av[:, None, :]
                    shape = cosine.shape
                    reference_relative = ad / safe_radius
                    triplet = torch.stack(
                        [
                            relative[:, :, None].expand(shape),
                            reference_relative[:, None, :].expand(shape),
                            cosine,
                            sine,
                        ],
                        dim=-1,
                    )
                else:
                    triplet, valid = pair.new_empty(0), keep.new_empty(0)
            outputs.append(
                _run(
                    self._embed,
                    self.training and self.checkpoint_edges,
                    pair,
                    triplet,
                    valid,
                )
            )
        return torch.cat(outputs)


class GeometryAttention(nn.Module):
    """GVA or MSA, sharing the SAME geometric descriptor and bias policy.

    GVA: omega((1+tanh(m(p))) * (q_i-k_j)) + bias(p), omega maps C->G.
    MSA: sum((1+tanh(m(p))) * q_i*k_j)/sqrt(C/G) + bias(p).
    The explicit per-group/head bias is computed from distance-angle encoding.
    This is PTv2-inspired conditioning, not an exact reproduction of PTv2.
    Values optionally add a projection of p; no value rotation is performed.
    """

    def __init__(
        self,
        channels,
        groups,
        pos_width,
        attention_type,
        position_multiplier=True,
        position_bias=True,
        geometry_in_values=False,
        chunk_size=256,
        checkpoint_edges=True,
        attn_drop=0.0,
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
        self.kind = attention_type
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
            # LayerNorm would erase the signal for G=1; Identity supports that case.
            self.weight_encoder = nn.Sequential(
                nn.LayerNorm(groups) if groups > 1 else nn.Identity(),
                nn.SiLU(),
                nn.Linear(groups, groups),
            )
        self.dropout = nn.Dropout(attn_drop)
        self.out = nn.Linear(channels, channels)

    def _chunk(self, q, k, v, idx, pos, mask):
        n, kcount = idx.shape
        g, d = self.groups, self.group_width
        q = q.reshape(n, g, d)
        kj = k[idx].reshape(n, kcount, g, d)
        if self.kind == "grouped_vector":
            relation = q[:, None] - kj
        else:
            relation = q[:, None] * kj
        if self.multiplier is not None:
            mult = 1 + torch.tanh(self.multiplier(pos)).reshape(n, kcount, g, d)
            relation = relation * mult
        if self.kind == "grouped_vector":
            logits = self.weight_encoder((relation * self.group_weight).sum(-1))
        else:
            logits = relation.sum(-1) / math.sqrt(d)
        if self.bias is not None:
            logits = logits + self.bias(pos)
        logits = logits.float().masked_fill(~mask[..., None], float("-inf"))
        weight = self.dropout(logits.softmax(dim=1))
        values = v[idx].reshape(n, kcount, g, d)
        if self.pos_value is not None:
            values = values + self.pos_value(pos).reshape(n, kcount, g, d)
        return (weight[..., None] * values).sum(1).reshape(n, self.channels)

    def forward(self, feat, idx, pos, mask):
        q, k, v = self.qkv(feat).chunk(3, -1)
        outputs = []
        for start in range(0, len(feat), self.chunk_size):
            stop = min(start + self.chunk_size, len(feat))
            outputs.append(
                _run(
                    self._chunk,
                    self.training and self.checkpoint_edges,
                    q[start:stop],
                    k,
                    v,
                    idx[start:stop],
                    pos[start:stop],
                    mask[start:stop],
                )
            )
        return self.out(torch.cat(outputs))


class Block(nn.Module):
    def __init__(
        self, channels, groups, pos_width, mlp_ratio=2.0, proj_drop=0.0, **attention
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = GeometryAttention(channels, groups, pos_width, **attention)
        self.norm2 = nn.LayerNorm(channels)
        hidden = max(1, int(channels * mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, channels),
            nn.Dropout(proj_drop),
        )

    def forward(self, feat, idx, pos, mask):
        feat = feat + self.attn(self.norm1(feat), idx, pos, mask)
        return feat + self.ffn(self.norm2(feat))


class Stage(nn.Module):
    def __init__(
        self,
        channels,
        depth,
        groups,
        k,
        pos_width,
        angle_refs,
        geometry_mode,
        geometry_in_stem=False,
        **block_options,
    ):
        super().__init__()
        self.k = k
        self.geometry = DistanceAngleEncoding(
            pos_width,
            angle_refs,
            geometry_mode,
            block_options["chunk_size"],
            block_options["checkpoint_edges"],
        )
        self.stem_geometry = (
            nn.Linear(pos_width, channels, bias=False) if geometry_in_stem else None
        )
        self.blocks = nn.ModuleList(
            [Block(channels, groups, pos_width, **block_options) for _ in range(depth)]
        )

    def forward(self, state):
        idx, rel, dist, mask = graph(state.coord, state.offset, self.k)
        pos = self.geometry(rel, dist, mask)
        feat = state.feat
        if self.stem_geometry is not None:
            pooled = (pos * mask[..., None]).sum(1) / mask.sum(
                1, keepdim=True
            ).clamp_min(1)
            feat = feat + self.stem_geometry(pooled)
        for block in self.blocks:
            feat = block(feat, idx, pos, mask)
        return State(state.coord, feat, state.offset, state.sample_index)


class Downsample(nn.Module):
    """FPS grouping + feature max pooling + coordinate mean pooling.

    No learned geometry here, to keep descriptor comparisons within attention.
    Source features are projected once before gathering; groups never cross clouds.
    FPS points only define query neighborhoods. Output locations are centroids
    of their distinct valid source points (neighborhoods can overlap).
    sample_index tracks the FPS anchor for optional auxiliary label supervision;
    the centroid itself need not coincide with an original labeled point.
    """

    def __init__(self, cin, cout, ratio, k, chunk_size, checkpoint_edges):
        super().__init__()
        self.ratio, self.k, self.chunk_size = ratio, k, chunk_size
        self.checkpoint_edges = checkpoint_edges
        self.proj = nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, cout), nn.GELU())

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
        for start in range(0, len(idx), self.chunk_size):
            stop = min(start + self.chunk_size, len(idx))
            pooled.append(
                _run(
                    self._pool,
                    self.training and self.checkpoint_edges,
                    feat,
                    idx[start:stop],
                    mask[start:stop],
                )
            )
        return State(coord, torch.cat(pooled), offset, state.sample_index[selected])


def interpolate(coarse, fine, feat, k=3, chunk_size=256):
    idx, _, dist, mask = graph(coarse.coord, coarse.offset, k, fine.coord, fine.offset)
    inverse = dist.clamp_min(1e-8).reciprocal() * mask
    weight = inverse / inverse.sum(1, keepdim=True)
    return torch.cat(
        [
            (weight[s : s + chunk_size, :, None] * feat[idx[s : s + chunk_size]]).sum(1)
            for s in range(0, len(idx), chunk_size)
        ]
    )


class Upsample(nn.Module):
    def __init__(self, coarse, skip, out, k, chunk_size):
        super().__init__()
        self.k, self.chunk_size = k, chunk_size
        self.coarse = nn.Linear(coarse, out)
        self.skip = nn.Linear(skip, out)
        self.fuse = nn.Sequential(nn.Linear(2 * out, out), nn.LayerNorm(out), nn.GELU())

    def forward(self, coarse, fine):
        up = interpolate(
            coarse, fine, self.coarse(coarse.feat), self.k, self.chunk_size
        )
        feat = self.fuse(torch.cat([up, self.skip(fine.feat)], -1))
        return State(fine.coord, feat, fine.offset, fine.sample_index)


@MODELS.register_module("SO3-DistanceAngle-v1")
class SO3DistanceAngleNet(nn.Module):
    """Invariant scalar-feature backbone with distance and relative-angle geometry.

    heads means groups for GVA and heads for MSA. Positional bias is explicit in
    BOTH modes. Existing RGB+XYZ inputs require scalar_feature_indices selecting
    RGB only. Input normals cannot enter the ordinary feature MLP as components.
    All input features must be floating point. Coordinates are treated as data.
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
        geometry_mode="distance_angle",
        pos_channels=16,
        angle_references=3,
        geometry_scales=None,
        position_multiplier=True,
        position_bias=True,
        geometry_in_values=False,
        geometry_in_stem=False,
        use_decoder=True,
        decoder_depths=None,
        mlp_ratio=2.0,
        attn_drop=0.0,
        proj_drop=0.0,
        interp_k=3,
        edge_chunk_size=256,
        checkpoint_edges=True,
    ):
        super().__init__()
        n = len(channels)
        if n < 1 or any(len(x) != n for x in (depths, heads, neighbors)):
            raise ValueError(
                "channels, depths, heads, neighbors must have equal nonzero length"
            )
        if len(sample_ratios) != n - 1 or any(not 0 < r <= 1 for r in sample_ratios):
            raise ValueError("sample_ratios must have nstages-1 entries in (0,1]")
        if any(
            not isinstance(v, int) or v < 1
            for seq in (channels, depths, heads, neighbors)
            for v in seq
        ):
            raise ValueError(
                "stage channels/depths/heads/neighbors must be positive integers"
            )
        if any(c % h for c, h in zip(channels, heads)):
            raise ValueError("every channels[i] must be divisible by heads[i]")
        if (
            min(
                pos_channels, angle_references, edge_chunk_size, head_channels, interp_k
            )
            < 1
        ):
            raise ValueError(
                "encoding widths, references, chunks, output channels and interp_k must be positive"
            )
        if not position_bias and not position_multiplier and not geometry_in_values:
            raise ValueError(
                "Enable position_bias, position_multiplier, or geometry_in_values; each stage needs a geometry path"
            )
        # Force explicit semantics to prevent silently treating raw XYZ as scalars.
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
        # scales = (
        #     list(geometry_scales)
        #     if geometry_scales is not None
        #     else [0.1 * 2**i for i in range(n)]
        # )
        # if len(scales) != n or any(not math.isfinite(s) or s <= 0 for s in scales):
        #     raise ValueError(
        #         "geometry_scales must contain one positive finite scale per stage"
        #     )
        self.register_buffer(
            "_scalar_indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.in_channels, self.head_channels = in_channels, head_channels
        self.channels, self.use_decoder = tuple(channels), use_decoder
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
        )
        self.stages = nn.ModuleList(
            [
                Stage(
                    channels[i],
                    depths[i],
                    heads[i],
                    neighbors[i],
                    # scales[i],
                    pos_channels,
                    angle_references,
                    geometry_mode,
                    geometry_in_stem=(geometry_in_stem and i == 0),
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
                )
                for i in range(n - 1)
            ]
        )
        if use_decoder:
            dd = list(decoder_depths) if decoder_depths is not None else [1] * (n - 1)
            if len(dd) != n - 1 or any(not isinstance(x, int) or x < 1 for x in dd):
                raise ValueError(
                    "decoder_depths must have nstages-1 positive entries (fine to coarse)"
                )
            self.up = nn.ModuleList()
            self.decoder = nn.ModuleList()
            for i in reversed(range(n - 1)):
                self.up.append(
                    Upsample(
                        channels[i + 1],
                        channels[i],
                        channels[i],
                        interp_k,
                        edge_chunk_size,
                    )
                )
                self.decoder.append(
                    Stage(
                        channels[i],
                        dd[i],
                        heads[i],
                        neighbors[i],
                        # scales[i],
                        pos_channels,
                        angle_references,
                        geometry_mode,
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
        self.output = nn.Sequential(
            nn.Linear(out_channels, head_channels),
            nn.LayerNorm(head_channels),
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
        state = State(coord, s, offset, torch.arange(len(coord), device=coord.device))
        stages = []
        for i, stage in enumerate(self.stages):
            state = stage(state)
            stages.append(state)
            if i < len(self.down):
                state = self.down[i](state)
        if self.use_decoder:
            for up, stage, skip in zip(self.up, self.decoder, reversed(stages[:-1])):
                state = stage(up(state, skip))
            out = state.feat
        else:
            weights = self.scale_logits.softmax(0)
            out = 0
            for i, stage in enumerate(stages):
                value = self.scale_proj[i](stage.feat)
                if i:
                    value = interpolate(
                        stage, stages[0], value, self.interp_k, self.chunk_size
                    )
                out = out + weights[i] * value
        point.feat = self.output(out)
        return (point, stages) if return_stages else point


@MODELS.register_module("DistanceAngleSegmentor")
class DistanceAngleSegmentor(nn.Module):
    """Pointcept segmentor. Optional auxiliary encoder supervision; no paired pass.

    Existing SO3 vector consistency options do not apply: this model has no
    vector stream. Use the new example config; old checkpoints are incompatible.
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
