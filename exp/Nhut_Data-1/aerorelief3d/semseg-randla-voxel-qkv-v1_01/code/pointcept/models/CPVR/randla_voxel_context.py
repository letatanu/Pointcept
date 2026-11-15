"""Pointcept backbone: centroid-aware RandLA geometry + explicit multi-head QKV.

Install this standalone file in pointcept/models/ and import RandLAVoxelQKVNetV3
from pointcept/models/__init__.py. Accepts coord/feat/offset and returns Point
with [N_original, out_channels] features in the original input order.

Points attend to all retained points or exact KNN in their own voxel, including self.
Selectable sparse CNN or neighboring-voxel attention conditions Q and K; learned 10D pair geometry augments K and V.
No point-neighbor search, FlashAttention, SO3/quaternion code or torch_scatter.
spconv is the deployment backend; torch is an exact slow reference backend.
Chunking bounds temporary edge embeddings, not total graph scores/indices.
Optional whole-block checkpointing also recomputes context, QKV and softmax.
"""

from dataclasses import dataclass
import itertools
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point


def counts(offset):
    return torch.diff(offset, prepend=offset.new_zeros(1)).long()


def segment_mean(value, index, size):
    # Accumulate in fp32 under AMP; preserve fp64 for numerical checks.
    dtype = (
        torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
    )
    total = torch.zeros((size, value.shape[-1]), device=value.device, dtype=dtype)
    total = total.index_add(0, index, value.to(dtype))
    denom = torch.bincount(index, minlength=size).clamp_min(1).to(dtype)
    return (total / denom[:, None]).to(value.dtype)


def segment_softmax(logits, dst, size):
    score = logits.float()
    index = dst[:, None].expand_as(score)
    maximum = score.new_full((size, score.shape[-1]), -torch.inf)
    # Detaching the stabilizing maximum preserves the softmax derivative.
    maximum.scatter_reduce_(0, index, score.detach(), reduce="amax", include_self=True)
    numer = (score - maximum[dst]).exp()
    denom = score.new_zeros((size, score.shape[-1])).index_add(0, dst, numer)
    return numer / denom[dst].clamp_min(torch.finfo(score.dtype).tiny)


@dataclass
class VoxelLayout:
    keys: torch.Tensor  # [V,4]: batch, grid x, grid y, grid z
    parent: torch.Tensor  # [N]: point -> occupied voxel
    centroid: torch.Tensor  # [V,3], retained-point centroids
    occupancy: torch.Tensor  # [V]
    batch: torch.Tensor  # [N]
    batch_size: int


@torch.no_grad()
def voxel_partition(coord, offset, size):
    batch = torch.repeat_interleave(
        torch.arange(len(offset), device=coord.device), counts(offset)
    )
    cells = torch.floor(coord.detach().float() / size).long()
    keys, parent = torch.unique(
        torch.cat([batch[:, None], cells], dim=1),
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    n = len(keys)
    return VoxelLayout(
        keys,
        parent,
        segment_mean(coord.detach().float(), parent, n),
        torch.bincount(parent, minlength=n),
        batch,
        len(offset),
    )


@torch.no_grad()
def within_voxel_graph(layout, max_edges=None):
    """All ordered pairs within each voxel, including self; batch-safe, no K cap."""
    parent, occupancy = layout.parent, layout.occupancy
    n = len(parent)
    point_ids = torch.arange(n, device=parent.device)
    edge_count = int(occupancy.square().sum())
    if max_edges is not None and edge_count > max_edges:
        raise RuntimeError(
            f"All-pairs voxel attention needs {edge_count:,} edges, exceeding "
            f"max_edges={max_edges:,}. Reduce voxel size, crop size, batch size, "
            "or sampling ratios. No points or edges were silently dropped."
        )
    order = point_ids[torch.argsort(parent, stable=True)]
    starts = occupancy.cumsum(0) - occupancy
    degree = occupancy[parent]
    dst = torch.repeat_interleave(point_ids, degree)
    edge_starts = degree.cumsum(0) - degree
    local = torch.arange(len(dst), device=parent.device) - edge_starts[dst]
    position = starts[parent[dst]] + local
    return dst, order[position]


@torch.no_grad()
def within_voxel_knn_graph(coord, layout, k, max_edges=None):
    """Exact Euclidean KNN within each batch-safe voxel; K includes self.

    Groups similarly sized voxels and tiles query/key distances. Never builds
    the full all-pairs edge graph or an N-by-N distance matrix. Search remains
    quadratic in each voxel's occupancy; retained edges are at most N*K.
    Equidistant non-self ties may be resolved differently across devices.
    """
    parent, occupancy = layout.parent, layout.occupancy
    degree = occupancy[parent].clamp_max(k)
    edge_count = int(degree.sum())
    if max_edges is not None and edge_count > max_edges:
        raise RuntimeError(
            f"Voxel KNN attention needs {edge_count:,} edges, exceeding "
            f"max_edges={max_edges:,}. Reduce K, crop size or batch size."
        )
    ids = torch.arange(len(parent), device=parent.device)
    dst = torch.repeat_interleave(ids, degree)
    src = torch.empty_like(dst)
    edge_start = degree.cumsum(0) - degree
    order = torch.argsort(parent, stable=True)
    start = occupancy.cumsum(0) - occupancy
    # Powers-of-two buckets keep padding below 2x actual voxel occupancy.
    bucket = torch.ceil(torch.log2(occupancy.double())).long()
    for b in range(int(bucket.max()) + 1):
        voxels = (bucket == b).nonzero(as_tuple=True)[0]
        width = 1 << b
        if not len(voxels):
            continue
        qtile, stile = min(width, 128), min(width, 512)
        group_size = max(1, min(65536, 1_048_576 // (qtile * stile)))
        take = min(k, width)
        for a in range(0, len(voxels), group_size):
            group = voxels[a : a + group_size]
            lengths = occupancy[group]
            for qa in range(0, width, qtile):
                qr = torch.arange(qa, min(qa + qtile, width), device=coord.device)
                qi = order[(start[group, None] + qr[None, :]).clamp_max(len(order) - 1)]
                qcoord = coord[qi].float()
                best = coord.new_full(
                    (len(group), len(qr), take), torch.inf, dtype=torch.float32
                )
                best_id = torch.zeros_like(best, dtype=torch.long)
                for sa in range(0, width, stile):
                    sr = torch.arange(sa, min(sa + stile, width), device=coord.device)
                    si = order[
                        (start[group, None] + sr[None, :]).clamp_max(len(order) - 1)
                    ]
                    distance = (
                        (qcoord[:, :, None, :] - coord[si].float()[:, None, :, :])
                        .square()
                        .sum(-1)
                    )
                    distance.masked_fill_(
                        sr[None, None, :] >= lengths[:, None, None], torch.inf
                    )
                    # Guarantee self inclusion, including coincident-point ties.
                    is_self = (qi[:, :, None] == si[:, None, :]) & (
                        sr[None, None, :] < lengths[:, None, None]
                    )
                    distance.masked_fill_(is_self, -1.0)
                    candidate = torch.cat([best, distance], dim=-1)
                    candidates_id = torch.cat(
                        [best_id, si[:, None, :].expand(-1, len(qr), -1)], dim=-1
                    )
                    best, choice = candidate.topk(
                        take, dim=-1, largest=False, sorted=True
                    )
                    best_id = candidates_id.gather(-1, choice)
                rank = torch.arange(take, device=coord.device)
                valid = (qr[None, :, None] < lengths[:, None, None]) & (
                    rank[None, None, :] < lengths.clamp_max(k)[:, None, None]
                )
                slots = edge_start[qi, None] + rank
                src[slots[valid]] = best_id[valid]
    return dst, src


@torch.no_grad()
def sparse_stencil(keys, kernel_size=3):
    """Reference occupied-site convolution edges: dst, src, kernel offset."""
    radius = kernel_size // 2
    cells = keys[:, 1:] - keys[:, 1:].amin(0) + radius
    widths = cells.amax(0) + radius + 1
    wx, wy, wz = [int(v) for v in widths]
    if (int(keys[:, 0].max()) + 2) * wx * wy * wz >= 2**62:
        raise ValueError("Sparse lookup key overflow: increase voxel size")

    def encode(batch, xyz):
        return ((batch * wx + xyz[:, 0]) * wy + xyz[:, 1]) * wz + xyz[:, 2]

    codes, order = encode(keys[:, 0], cells).sort()
    dsts, srcs, kernels = [], [], []
    ids = torch.arange(len(keys), device=keys.device)
    for kernel, shift in enumerate(
        itertools.product(range(-radius, radius + 1), repeat=3)
    ):
        candidate = cells + cells.new_tensor(shift)
        code = encode(keys[:, 0], candidate)
        pos = torch.searchsorted(codes, code).clamp_max(len(keys) - 1)
        src = order[pos]
        valid = (codes[pos] == code) & (keys[src, 0] == keys[:, 0])
        valid &= (cells[src] == candidate).all(-1)
        dsts.append(ids[valid])
        srcs.append(src[valid])
        kernels.append(torch.full_like(ids[valid], kernel))
    return torch.cat(dsts), torch.cat(srcs), torch.cat(kernels)


class ReferenceSubMConv(nn.Module):
    """Pure torch 3^3 convolution at occupied sites; CPU/reference backend."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(27, channels, channels))
        nn.init.uniform_(
            self.weight, -1 / math.sqrt(27 * channels), 1 / math.sqrt(27 * channels)
        )

    def forward(self, feat, stencil):
        dst, src, kernel = stencil
        out = torch.zeros_like(feat)
        for k in range(27):
            selected = kernel == k
            message = nn.functional.linear(feat[src[selected]], self.weight[k])
            out.index_add_(0, dst[selected], message.to(out.dtype))
        return out


class SparseVoxelContext(nn.Module):
    def __init__(self, channels, backend):
        super().__init__()
        self.backend = backend
        self.norm1 = nn.LayerNorm(channels)
        if backend == "spconv":
            try:
                import spconv.pytorch as spconv
            except ImportError as exc:
                raise ImportError(
                    "sparse_backend='spconv' requires a compatible spconv installation; "
                    "use sparse_backend='torch' for reference checks"
                ) from exc
            self.conv1 = spconv.SubMConv3d(
                channels, channels, 3, padding=1, bias=False, indice_key="context"
            )
            self.conv2 = spconv.SubMConv3d(
                channels, channels, 3, padding=1, bias=False, indice_key="context"
            )
        elif backend == "torch":
            self.conv1 = ReferenceSubMConv(channels)
            self.conv2 = ReferenceSubMConv(channels)
        else:
            raise ValueError("sparse_backend must be 'spconv' or 'torch'")

    def forward(self, feat, layout, stencil=None):
        if self.backend == "torch":
            x = nn.functional.gelu(self.norm1(self.conv1(feat, stencil)))
            return feat + self.conv2(x, stencil)
        import spconv.pytorch as spconv

        # Translate the grid by an integer without altering relative offsets.
        grid = layout.keys[:, 1:] - layout.keys[:, 1:].amin(0)
        shape = (grid.amax(0) + 1).clamp_min(3).tolist()
        if int(grid.max()) >= torch.iinfo(torch.int32).max:
            raise ValueError("Voxel grid exceeds spconv int32 coordinate range")
        indices = torch.cat([layout.keys[:, :1], grid], dim=1).int().contiguous()
        sparse = spconv.SparseConvTensor(
            feat.contiguous(), indices, shape, layout.batch_size
        )
        x = self.conv1(sparse)
        x = x.replace_feature(nn.functional.gelu(self.norm1(x.features)))
        x = self.conv2(x)
        return feat + x.features


class NeighborVoxelAttention(nn.Module):
    """Attention over occupied sites in a configurable cubic stencil, including self.

    Complexity is at most voxel_kernel_size**3 edges per occupied voxel. Empty sites contribute
    nothing, matching the occupied-site support of a SubMConv3d layer.
    """

    def __init__(
        self,
        channels,
        num_heads,
        edge_chunk_size=32768,
        checkpoint_edges=True,
        voxel_kernel_size=3,
    ):
        super().__init__()
        self.chunk_size = edge_chunk_size
        self.checkpoint_edges = checkpoint_edges
        self.voxel_kernel_size = voxel_kernel_size
        self.heads = num_heads
        self.dim = channels // num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.position = nn.Sequential(
            nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.bias = nn.Linear(channels, num_heads, bias=False)
        self.proj = nn.Linear(channels, channels)

    def forward(self, feat, layout, stencil=None):
        if stencil is None:
            stencil = sparse_stencil(layout.keys, self.voxel_kernel_size)
        dst, src, kernel = stencil
        q, k, v = (
            self.qkv(self.norm(feat))
            .reshape(len(feat), 3, self.heads, self.dim)
            .unbind(1)
        )
        # sparse_stencil kernel IDs follow this exact product order. Geometry
        # depends only on the discrete offset: encode K**3 offsets, NOT E rows.
        radius = self.voxel_kernel_size // 2
        offsets = feat.new_tensor(
            list(itertools.product(range(-radius, radius + 1), repeat=3))
        )
        position = self.position(offsets)
        position_bias = self.bias(position)

        def score_chunk(query, key, bias, dd, ss, kk):
            return (query[dd].float() * key[ss].float()).sum(
                -1
            ) * self.dim**-0.5 + bias[kk].float()

        logits = torch.cat(
            [
                edge_run(
                    self,
                    score_chunk,
                    q,
                    k,
                    position_bias,
                    dst[a : a + self.chunk_size],
                    src[a : a + self.chunk_size],
                    kernel[a : a + self.chunk_size],
                )
                for a in range(0, len(dst), self.chunk_size)
            ]
        )
        weights = segment_softmax(logits, dst, len(feat))

        def value_chunk(value, pos, ss, kk, alpha):
            message = (
                value[ss].float() + pos[kk].reshape(-1, self.heads, self.dim).float()
            )
            return (alpha[..., None] * message).flatten(1)

        out = torch.zeros_like(feat, dtype=torch.float32)
        for a in range(0, len(dst), self.chunk_size):
            dd, ss = dst[a : a + self.chunk_size], src[a : a + self.chunk_size]
            message = edge_run(
                self,
                value_chunk,
                v,
                position,
                ss,
                kernel[a : a + self.chunk_size],
                weights[a : a + self.chunk_size],
            )
            out.index_add_(0, dd, message)
        return feat + self.proj(out.to(feat.dtype))


def make_voxel_context(
    channels, backend, num_heads, edge_chunk_size, checkpoint_edges, voxel_kernel_size=3
):
    if backend == "neighbor_attention":
        return NeighborVoxelAttention(
            channels, num_heads, edge_chunk_size, checkpoint_edges, voxel_kernel_size
        )
    return SparseVoxelContext(channels, backend)


class PairGeometry(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, width), nn.GELU(), nn.Linear(width, width)
        )

    def forward(self, coord, relative, dst, src, size):
        displacement = (coord[dst].float() - coord[src].float()) / size
        geom = torch.cat(
            [
                relative[dst],
                relative[src],
                displacement,
                torch.linalg.vector_norm(displacement, dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        return self.encoder(geom.to(self.encoder[0].weight.dtype))


def edge_run(module, fn, *args):
    enabled = module.training and module.checkpoint_edges and torch.is_grad_enabled()
    return checkpoint(fn, *args, use_reentrant=False) if enabled else fn(*args)


class ContextQKVAttention(nn.Module):
    """Per-head scaled dot products with pair-specific geometric K and V.

    Q = Wq LN(h + Wt T_v), K = Wk LN(h + Wt T_v), V = Wv h.
    logits_ij = Q_i . (K_j + WrK r_ij) / sqrt(head_dim).
    Softmax is over every j in the query's voxel, independently per head.
    """

    def __init__(
        self,
        channels,
        geometry_width,
        num_heads,
        backend,
        edge_chunk_size,
        checkpoint_edges,
        attn_drop=0.0,
        ffn_ratio=2.0,
        context_residual=False,
        voxel_kernel_size=3,
    ):
        super().__init__()
        if num_heads < 1 or channels % num_heads:
            raise ValueError("num_heads must divide channels")
        self.channels, self.num_heads = channels, num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.chunk_size = edge_chunk_size
        self.checkpoint_edges, self.attn_drop = checkpoint_edges, attn_drop
        self.voxel_context = make_voxel_context(
            channels,
            backend,
            num_heads,
            edge_chunk_size,
            checkpoint_edges,
            voxel_kernel_size,
        )
        self.context_projection = nn.Linear(channels, channels, bias=False)
        self.condition_norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.position_key = nn.Linear(geometry_width, channels, bias=False)
        self.position_value = nn.Linear(geometry_width, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels)
        self.context_residual = context_residual
        if context_residual:
            self.context_out = nn.Linear(channels, channels, bias=False)
            self.context_gain = nn.Parameter(torch.tensor(0.1))
        if ffn_ratio > 0:
            hidden = max(1, round(channels * ffn_ratio))
            self.ffn_norm = nn.LayerNorm(channels)
            self.ffn = nn.Sequential(
                nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
            )
        else:
            self.ffn = None

    def project_qkv(self, feat, voxel_context, parent):
        """Project once per node, then gather for edges; [N, heads, head_dim]."""
        conditioned = self.condition_norm(
            feat + self.context_projection(voxel_context)[parent]
        )
        shape = (len(feat), self.num_heads, self.head_dim)
        return (
            self.q_proj(conditioned).reshape(shape),
            self.k_proj(conditioned).reshape(shape),
            self.v_proj(feat).reshape(shape),
        )

    def forward(self, feat, coord, relative, layout, graph, geometry, size, stencil):
        dst, src = graph
        pooled = segment_mean(feat, layout.parent, len(layout.keys))
        voxel_context = self.voxel_context(pooled, layout, stencil)
        query, key, value = self.project_qkv(feat, voxel_context, layout.parent)

        def score_chunk(q, k, p, u, dd, ss):
            r = geometry(p, u, dd, ss, size)
            pos_key = self.position_key(r).reshape(-1, self.num_heads, self.head_dim)
            # Accumulate scores in fp32 even under fp16 autocast.
            return (q[dd].float() * (k[ss].float() + pos_key.float())).sum(
                -1
            ) * self.scale

        logits = torch.cat(
            [
                edge_run(
                    self,
                    score_chunk,
                    query,
                    key,
                    coord,
                    relative,
                    dst[a : a + self.chunk_size],
                    src[a : a + self.chunk_size],
                )
                for a in range(0, len(dst), self.chunk_size)
            ]
        )
        weights = segment_softmax(logits, dst, len(feat))
        weights = nn.functional.dropout(weights, self.attn_drop, self.training)

        def value_chunk(v, p, u, dd, ss, alpha):
            r = geometry(p, u, dd, ss, size)
            pos_value = self.position_value(r).reshape(
                -1, self.num_heads, self.head_dim
            )
            return ((v[ss].float() + pos_value.float()) * alpha[..., None]).flatten(1)

        messages = torch.zeros(feat.shape, device=feat.device, dtype=torch.float32)
        for a in range(0, len(dst), self.chunk_size):
            dd, ss = dst[a : a + self.chunk_size], src[a : a + self.chunk_size]
            message = edge_run(
                self,
                value_chunk,
                value,
                coord,
                relative,
                dd,
                ss,
                weights[a : a + self.chunk_size],
            )
            messages.index_add_(0, dd, message)
        out = feat + self.out_proj(messages.to(feat.dtype))
        if self.context_residual:
            # Optional direct context path: singleton voxels otherwise have alpha=1
            # and cannot receive neighboring information through scores alone.
            out = (
                out + self.context_gain * self.context_out(voxel_context)[layout.parent]
            )
        if self.ffn is not None:
            # FFN/LayerNorm are independent per point. Bound dense GEMM rows as
            # well as edge operations; normalization belongs inside the chunk.
            # Checkpointing avoids saving every chunk's expanded hidden features.
            def ffn_chunk(x):
                return x + self.ffn(self.ffn_norm(x))

            out = torch.cat(
                [
                    edge_run(self, ffn_chunk, out[a : a + self.chunk_size])
                    for a in range(0, len(out), self.chunk_size)
                ],
                dim=0,
            )
        return out


class RandLAVoxelQKVStage(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        depth,
        size,
        geometry_width,
        num_heads,
        backend,
        edge_chunk_size,
        checkpoint_edges,
        attn_drop,
        ffn_ratio,
        context_residual,
        max_edges,
        voxel_kernel_size=3,
        point_attention_mode="all",
        point_knn_k=16,
        checkpoint_blocks=False,
    ):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.size, self.backend, self.max_edges = size, backend, max_edges
        self.point_attention_mode, self.point_knn_k = point_attention_mode, point_knn_k
        self.voxel_kernel_size = voxel_kernel_size
        self.geometry_width = geometry_width
        self.chunk_size, self.checkpoint_edges = edge_chunk_size, checkpoint_edges
        self.geometry = PairGeometry(geometry_width)
        self.initialize = nn.Sequential(
            nn.Linear(in_channels + geometry_width, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.blocks = nn.ModuleList(
            [
                ContextQKVAttention(
                    channels,
                    geometry_width,
                    num_heads,
                    backend,
                    edge_chunk_size,
                    # The outer checkpoint covers context, QKV, softmax and FFN.
                    # Avoid redundant nested checkpoints inside attention blocks.
                    checkpoint_edges and not checkpoint_blocks,
                    attn_drop,
                    ffn_ratio,
                    context_residual,
                    voxel_kernel_size,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, coord, feat, offset):
        layout = voxel_partition(coord, offset, self.size)
        graph = (
            within_voxel_graph(layout, self.max_edges)
            if self.point_attention_mode == "all"
            else within_voxel_knn_graph(coord, layout, self.point_knn_k, self.max_edges)
        )
        relative = (coord.detach().float() - layout.centroid[layout.parent]) / self.size
        dst, src = graph
        summary = torch.zeros(
            (len(coord), self.geometry_width), device=coord.device, dtype=torch.float32
        )
        for a in range(0, len(dst), self.chunk_size):
            dd, ss = dst[a : a + self.chunk_size], src[a : a + self.chunk_size]
            r = edge_run(
                self, self.geometry, coord.detach(), relative, dd, ss, self.size
            )
            summary.index_add_(0, dd, r.float())
        summary = (
            summary / torch.bincount(dst, minlength=len(coord)).clamp_min(1)[:, None]
        ).to(feat.dtype)
        feat = self.initialize(torch.cat([feat, summary], dim=-1))
        stencil = (
            sparse_stencil(
                layout.keys,
                self.voxel_kernel_size if self.backend == "neighbor_attention" else 3,
            )
            if self.backend in ("torch", "neighbor_attention")
            else None
        )
        for block in self.blocks:
            args = (
                feat,
                coord.detach(),
                relative,
                layout,
                graph,
                self.geometry,
                self.size,
                stencil,
            )
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                # Pass the block directly: a loop closure can bind the wrong block
                # when backward recomputes it after the loop has advanced.
                feat = checkpoint(block, *args, use_reentrant=False)
            else:
                feat = block(*args)
        return feat


@torch.no_grad()
def random_voxel_sample(coord, offset, size, ratio, training):
    """Stratified sampling: retain ceil(ratio*n_v), at least one per voxel.

    Returns selected point indices and a link for voxel-based decoder unpooling.
    Random training choices; deterministic first-in-input-order choices in eval.
    """
    layout = voxel_partition(coord, offset, size)
    parent, occupancy = layout.parent, layout.occupancy
    order = (
        torch.randperm(len(coord), device=coord.device)
        if training
        else torch.arange(len(coord), device=coord.device)
    )
    order = order[torch.argsort(parent[order], stable=True)]
    start = occupancy.cumsum(0) - occupancy
    rank = torch.arange(len(coord), device=coord.device) - start[parent[order]]
    keep = (occupancy.float() * ratio).ceil().long().clamp_min(1)
    selected = order[rank < keep[parent[order]]].sort().values
    new_offset = (
        torch.bincount(layout.batch[selected], minlength=len(offset))
        .cumsum(0)
        .to(offset.dtype)
    )
    return selected, new_offset, parent, parent[selected], len(occupancy)


def stage_options(value, n, name):
    if isinstance(value, (tuple, list)):
        if len(value) != n:
            raise ValueError(f"{name} must have one entry per stage")
        return tuple(value)
    return (value,) * n


class CentroidAwareDecoderFusion(nn.Module):
    """Decode pooled coarse context using each fine point's centroid offset.

    Centroids use ALL fine points before sampling, not only retained points.
    Parent indices are batch-safe mappings from the encoder's sampling step.
    Geometry is normalized by the coarse voxel size used for that step.
    This is FPT-inspired reconstruction, not distance interpolation or an
    implementation of FPT's lightweight attention.
    """

    def __init__(self, fine_channels, coarse_channels, geometry_width):
        super().__init__()
        self.position = nn.Sequential(
            nn.Linear(4, geometry_width),
            nn.GELU(),
            nn.Linear(geometry_width, geometry_width),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fine_channels + coarse_channels + geometry_width, fine_channels),
            nn.LayerNorm(fine_channels),
            nn.GELU(),
            nn.Linear(fine_channels, fine_channels),
        )

    def forward(self, skip, coarse, fine_coord, fine_parent, coarse_parent, nv, size):
        context = segment_mean(coarse, coarse_parent, nv)[fine_parent]
        with torch.no_grad():
            centroid = segment_mean(fine_coord.detach().float(), fine_parent, nv)
            relative = (fine_coord.detach().float() - centroid[fine_parent]) / size
            geometry = torch.cat(
                [relative, torch.linalg.vector_norm(relative, dim=-1, keepdim=True)],
                dim=-1,
            )
        position = self.position(geometry.to(skip.dtype))
        return skip + self.fusion(torch.cat([skip, context, position], dim=-1))


@MODELS.register_module("RandLA-Voxel-QKV-v3")
class RandLAVoxelQKVNetV3(nn.Module):
    """Multiscale point QKV backbone returning features for original input rows.

    Select all within-voxel pairs or exact bounded-K neighbors, including self.
    Sampling is voxel-stratified: random in train, deterministic in eval, and
    keeps at least one point per voxel. The decoder uses centroid-aware coarse-to-fine fusion
    and fine point skips; initially discarded points get a separate refinement.
    Set input_sample_ratio=1 and all sample_ratios=1 for no point sampling.
    """

    def __init__(
        self,
        in_channels=6,
        channels=(32, 64, 128),
        depths=None,  # Legacy alias for enc_depths.
        voxel_sizes=(0.1, 0.2, 0.4),
        sample_ratios=(0.5, 0.5),
        input_sample_ratio=1.0,
        geometry_width=16,
        num_heads=(2, 4, 8),
        ffn_ratio=2.0,
        context_residual=True,
        decoder_refine_depth=None,  # Legacy uniform decoder-depth alias.
        decoder_geometry=True,
        max_edges=8_000_000,
        sparse_backend="spconv",
        voxel_kernel_size=3,
        edge_chunk_size=32768,
        checkpoint_edges=True,
        attn_drop=0.0,
        out_channels=64,
        point_attention_mode="all",
        point_knn_k=16,
        enc_depths=None,
        dec_depths=None,
        checkpoint_blocks=False,
    ):
        super().__init__()
        n = len(channels)
        if enc_depths is not None and depths is not None:
            raise ValueError("Specify enc_depths or legacy depths, not both")
        enc_depths = enc_depths if enc_depths is not None else depths
        enc_depths = (1,) * n if enc_depths is None else enc_depths
        if not isinstance(enc_depths, (tuple, list)):
            raise ValueError(
                "enc_depths must be a tuple/list with one entry per encoder stage"
            )
        if dec_depths is not None and decoder_refine_depth is not None:
            raise ValueError(
                "Specify dec_depths or legacy decoder_refine_depth, not both"
            )
        if dec_depths is None:
            uniform = 1 if decoder_refine_depth is None else decoder_refine_depth
            if isinstance(uniform, bool) or not isinstance(uniform, int) or uniform < 0:
                raise ValueError("decoder_refine_depth must be a nonnegative integer")
            dec_depths = (uniform,) * max(0, n - 1)
        if not isinstance(dec_depths, (tuple, list)) or len(dec_depths) != max(
            0, n - 1
        ):
            raise ValueError("dec_depths must have S-1 entries ordered fine to coarse")
        if any(
            isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in dec_depths
        ):
            raise ValueError("dec_depths entries must be nonnegative integers")
        if not isinstance(checkpoint_blocks, bool):
            raise ValueError("checkpoint_blocks must be a bool")
        self.enc_depths, self.dec_depths = tuple(enc_depths), tuple(dec_depths)
        if (
            n < 1
            or len(enc_depths) != n
            or len(voxel_sizes) != n
            or len(sample_ratios) != n - 1
        ):
            raise ValueError(
                "channels/enc_depths/voxel_sizes need S entries; sample_ratios needs S-1"
            )
        for name, seq in (("channels", channels), ("enc_depths", enc_depths)):
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in seq):
                raise ValueError(f"{name} entries must be positive integers")
        if not isinstance(in_channels, int) or in_channels < 0:
            raise ValueError("in_channels must be a nonnegative integer")
        if any(not math.isfinite(s) or s <= 0 for s in voxel_sizes):
            raise ValueError("voxel_sizes must be finite and positive")
        if any(b <= a for a, b in zip(voxel_sizes, voxel_sizes[1:])):
            raise ValueError("voxel_sizes must increase with depth")
        if any(not math.isfinite(r) or not 0 < r <= 1 for r in sample_ratios):
            raise ValueError("sample_ratios must be in (0,1]")
        if not math.isfinite(input_sample_ratio) or not 0 < input_sample_ratio <= 1:
            raise ValueError("input_sample_ratio must be in (0,1]")
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in (geometry_width, edge_chunk_size, out_channels)
        ):
            raise ValueError(
                "widths, edge_chunk_size and out_channels must be positive integers"
            )
        if not 0 <= attn_drop < 1:
            raise ValueError("attn_drop must be in [0,1)")
        heads = stage_options(num_heads, n, "num_heads")
        if any(
            isinstance(h, bool) or not isinstance(h, int) or h < 1 or channels[i] % h
            for i, h in enumerate(heads)
        ):
            raise ValueError(
                "num_heads must give a positive divisor of each stage width"
            )
        if not math.isfinite(ffn_ratio) or ffn_ratio < 0:
            raise ValueError("ffn_ratio must be finite and nonnegative; 0 disables FFN")
        if not isinstance(context_residual, bool):
            raise ValueError("context_residual must be a bool")
        if max_edges is not None and (
            isinstance(max_edges, bool)
            or not isinstance(max_edges, int)
            or max_edges < 1
        ):
            raise ValueError("max_edges must be a positive integer or None")
        if (
            isinstance(voxel_kernel_size, bool)
            or not isinstance(voxel_kernel_size, int)
            or voxel_kernel_size < 1
            or voxel_kernel_size % 2 == 0
        ):
            raise ValueError("voxel_kernel_size must be a positive odd integer")
        if sparse_backend not in ("spconv", "torch", "neighbor_attention"):
            raise ValueError(
                "sparse_backend must be spconv, torch, or neighbor_attention"
            )
        if not isinstance(decoder_geometry, bool):
            raise ValueError("decoder_geometry must be a bool")
        if point_attention_mode not in ("all", "knn"):
            raise ValueError("point_attention_mode must be all or knn")
        if (
            isinstance(point_knn_k, bool)
            or not isinstance(point_knn_k, int)
            or point_knn_k < 1
        ):
            raise ValueError("point_knn_k must be a positive integer")
        self.decoder_geometry = decoder_geometry
        self.in_channels, self.out_channels = in_channels, out_channels
        self.head_channels = out_channels
        self.input_sample_ratio = input_sample_ratio
        self.voxel_sizes, self.sample_ratios = tuple(voxel_sizes), tuple(sample_ratios)
        self.stages = nn.ModuleList(
            [
                RandLAVoxelQKVStage(
                    in_channels if level == 0 else channels[level - 1],
                    width,
                    enc_depths[level],
                    voxel_sizes[level],
                    geometry_width,
                    heads[level],
                    sparse_backend,
                    edge_chunk_size,
                    checkpoint_edges,
                    attn_drop,
                    ffn_ratio,
                    context_residual,
                    max_edges,
                    voxel_kernel_size,
                    point_attention_mode,
                    point_knn_k,
                    checkpoint_blocks,
                )
                for level, width in enumerate(channels)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                (
                    CentroidAwareDecoderFusion(
                        channels[i], channels[i + 1], geometry_width
                    )
                    if decoder_geometry
                    else nn.Sequential(
                        nn.Linear(channels[i] + channels[i + 1], channels[i]),
                        nn.LayerNorm(channels[i]),
                        nn.GELU(),
                        nn.Linear(channels[i], channels[i]),
                    )
                )
                for i in range(n - 1)
            ]
        )
        # A fresh point-attention stage after each coarse/fine fusion. Its direct
        # context path transfers features across occupied voxel boundaries.
        self.decoder_refine = nn.ModuleList(
            [
                (
                    RandLAVoxelQKVStage(
                        channels[i],
                        channels[i],
                        dec_depths[i],
                        voxel_sizes[i],
                        geometry_width,
                        heads[i],
                        sparse_backend,
                        edge_chunk_size,
                        checkpoint_edges,
                        attn_drop,
                        ffn_ratio,
                        True,
                        max_edges,
                        voxel_kernel_size,
                        point_attention_mode,
                        point_knn_k,
                        checkpoint_blocks,
                    )
                    if dec_depths[i]
                    else nn.Identity()
                )
                for i in range(n - 1)
            ]
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(channels[0]), nn.Linear(channels[0], out_channels)
        )
        self.input_restore = (
            nn.Sequential(
                nn.Linear(out_channels + in_channels + 4, out_channels),
                nn.GELU(),
                nn.Linear(out_channels, out_channels),
            )
            if input_sample_ratio < 1
            else None
        )

    def _inputs(self, data):
        coord, offset = data["coord"], data["offset"]
        if (
            coord.ndim != 2
            or coord.shape[1] != 3
            or not len(coord)
            or not coord.is_floating_point()
            or not bool(torch.isfinite(coord).all())
        ):
            raise ValueError("coord must be finite floating-point nonempty [N,3]")
        if (
            offset.ndim != 1
            or offset.dtype not in (torch.int32, torch.int64)
            or not len(offset)
            or int(offset[-1]) != len(coord)
            or bool((counts(offset) <= 0).any())
        ):
            raise ValueError(
                "offset must contain positive cumulative cloud sizes ending at N"
            )
        feat = data.get("feat")
        if feat is None and self.in_channels == 0:
            feat = coord.new_empty((len(coord), 0))
        if (
            feat is None
            or feat.shape != (len(coord), self.in_channels)
            or not feat.is_floating_point()
            or not bool(torch.isfinite(feat).all())
        ):
            raise ValueError(
                "feat must be finite [N,in_channels]; may be absent only when in_channels=0"
            )
        if coord.device != feat.device or coord.device != offset.device:
            raise ValueError("coord, feat and offset must share a device")
        return coord, feat, offset

    def forward(self, data_dict, return_stages=False):
        if return_stages:
            raise ValueError(
                "return_stages/deep supervision is unsupported; intermediate labels require sampling maps"
            )
        coord, feat, offset = self._inputs(data_dict)
        original_coord, original_feat = coord, feat
        input_link = None
        if self.input_sample_ratio < 1:
            selected, offset, fine_parent, coarse_parent, nv = random_voxel_sample(
                coord,
                offset,
                self.voxel_sizes[0],
                self.input_sample_ratio,
                self.training,
            )
            coord, feat = coord[selected], feat[selected]
            input_link = (selected, fine_parent, coarse_parent, nv)
        saved = []
        for level, stage in enumerate(self.stages):
            feat = stage(coord, feat, offset)
            if level + 1 < len(self.stages):
                selected, new_offset, fine_parent, coarse_parent, nv = (
                    random_voxel_sample(
                        coord,
                        offset,
                        self.voxel_sizes[level + 1],
                        self.sample_ratios[level],
                        self.training,
                    )
                )
                saved.append((feat, fine_parent, coarse_parent, nv, coord, offset))
                coord, feat, offset = coord[selected], feat[selected], new_offset
        for level in reversed(range(len(saved))):
            skip, fine_parent, coarse_parent, nv, fine_coord, fine_offset = saved[level]
            if self.decoder_geometry:
                feat = self.decoder[level](
                    skip,
                    feat,
                    fine_coord,
                    fine_parent,
                    coarse_parent,
                    nv,
                    self.voxel_sizes[level + 1],
                )
            else:
                context = segment_mean(feat, coarse_parent, nv)[fine_parent]
                feat = skip + self.decoder[level](torch.cat([skip, context], dim=-1))
            if self.dec_depths[level]:
                feat = feat + self.decoder_refine[level](fine_coord, feat, fine_offset)
        feat = self.projection(feat)
        if input_link is not None:
            selected, fine_parent, coarse_parent, nv = input_link
            context = segment_mean(feat, coarse_parent, nv)[fine_parent]
            with torch.no_grad():
                centroid = segment_mean(
                    original_coord[selected].float(), coarse_parent, nv
                )
                u = (original_coord.float() - centroid[fine_parent]) / self.voxel_sizes[
                    0
                ]
                geom = torch.cat(
                    [u, torch.linalg.vector_norm(u, dim=-1, keepdim=True)], -1
                )
            restored = context + self.input_restore(
                torch.cat(
                    [context, original_feat.to(context.dtype), geom.to(context.dtype)],
                    -1,
                )
            )
            # Retained points keep their exact point-attention outputs. Only
            # discarded points use voxel broadcast + their own attributes/offset.
            feat = restored.index_copy(0, selected, feat)
        point = Point(dict(data_dict))
        point.feat = feat
        return point
