"""Pointcept backbone: centroid-aware RandLA geometry + explicit multi-head QKV.

Install this standalone file in pointcept/models/ and import RandLAVoxelQKVNet
from pointcept/models/__init__.py. Accepts coord/feat/offset and returns Point
with [N_original, out_channels] features in the original input order.

All retained points inside a voxel attend to every other point (including self).
Sparse CNN context conditions Q and K; learned 10D pair geometry augments K and V.
No point-neighbor search, FlashAttention, SO3/quaternion code or torch_scatter.
spconv is the deployment backend; torch is an exact slow reference backend.
Chunking bounds edge embeddings, not total E=sum_v(n_v**2) scores/indices.
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
    dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
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
        torch.cat([batch[:, None], cells], dim=1), dim=0, sorted=True,
        return_inverse=True,
    )
    n = len(keys)
    return VoxelLayout(
        keys, parent, segment_mean(coord.detach().float(), parent, n),
        torch.bincount(parent, minlength=n), batch, len(offset),
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
def sparse_stencil(keys):
    """Reference occupied-site convolution edges: dst, src, kernel offset."""
    cells = keys[:, 1:] - keys[:, 1:].amin(0) + 1
    widths = cells.amax(0) + 2
    wx, wy, wz = [int(v) for v in widths]
    if (int(keys[:, 0].max()) + 2) * wx * wy * wz >= 2**62:
        raise ValueError("Sparse lookup key overflow: increase voxel size")

    def encode(batch, xyz):
        return ((batch * wx + xyz[:, 0]) * wy + xyz[:, 1]) * wz + xyz[:, 2]

    codes, order = encode(keys[:, 0], cells).sort()
    dsts, srcs, kernels = [], [], []
    ids = torch.arange(len(keys), device=keys.device)
    for kernel, shift in enumerate(itertools.product((-1, 0, 1), repeat=3)):
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
        nn.init.uniform_(self.weight, -1 / math.sqrt(27 * channels), 1 / math.sqrt(27 * channels))

    def forward(self, feat, stencil):
        dst, src, kernel = stencil
        out = torch.zeros_like(feat)
        for k in range(27):
            selected = kernel == k
            message = nn.functional.linear(feat[src[selected]], self.weight[k])
            out = out.index_add(0, dst[selected], message.to(out.dtype))
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
            self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False, indice_key="context")
            self.conv2 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False, indice_key="context")
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
        sparse = spconv.SparseConvTensor(feat.contiguous(), indices, shape, layout.batch_size)
        x = self.conv1(sparse)
        x = x.replace_feature(nn.functional.gelu(self.norm1(x.features)))
        x = self.conv2(x)
        return feat + x.features


class PairGeometry(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, coord, relative, dst, src, size):
        displacement = (coord[dst].float() - coord[src].float()) / size
        geom = torch.cat(
            [relative[dst], relative[src], displacement,
             torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)], dim=-1,
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

    def __init__(self, channels, geometry_width, num_heads, backend,
                 edge_chunk_size, checkpoint_edges, attn_drop=0.0,
                 ffn_ratio=2.0, context_residual=False):
        super().__init__()
        if num_heads < 1 or channels % num_heads:
            raise ValueError("num_heads must divide channels")
        self.channels, self.num_heads = channels, num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.chunk_size = edge_chunk_size
        self.checkpoint_edges, self.attn_drop = checkpoint_edges, attn_drop
        self.voxel_context = SparseVoxelContext(channels, backend)
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
        return (self.q_proj(conditioned).reshape(shape),
                self.k_proj(conditioned).reshape(shape),
                self.v_proj(feat).reshape(shape))

    def forward(self, feat, coord, relative, layout, graph, geometry, size, stencil):
        dst, src = graph
        pooled = segment_mean(feat, layout.parent, len(layout.keys))
        voxel_context = self.voxel_context(pooled, layout, stencil)
        query, key, value = self.project_qkv(feat, voxel_context, layout.parent)

        def score_chunk(q, k, p, u, dd, ss):
            r = geometry(p, u, dd, ss, size)
            pos_key = self.position_key(r).reshape(-1, self.num_heads, self.head_dim)
            # Accumulate scores in fp32 even under fp16 autocast.
            return (q[dd].float() * (k[ss].float() + pos_key.float())).sum(-1) * self.scale

        logits = torch.cat([
            edge_run(self, score_chunk, query, key, coord, relative,
                     dst[a:a + self.chunk_size], src[a:a + self.chunk_size])
            for a in range(0, len(dst), self.chunk_size)
        ])
        weights = segment_softmax(logits, dst, len(feat))
        weights = nn.functional.dropout(weights, self.attn_drop, self.training)

        def value_chunk(v, p, u, dd, ss, alpha):
            r = geometry(p, u, dd, ss, size)
            pos_value = self.position_value(r).reshape(-1, self.num_heads, self.head_dim)
            return ((v[ss].float() + pos_value.float()) * alpha[..., None]).flatten(1)

        messages = torch.zeros(feat.shape, device=feat.device, dtype=torch.float32)
        for a in range(0, len(dst), self.chunk_size):
            dd, ss = dst[a:a + self.chunk_size], src[a:a + self.chunk_size]
            message = edge_run(self, value_chunk, value, coord, relative, dd, ss,
                               weights[a:a + self.chunk_size])
            messages = messages.index_add(0, dd, message)
        out = feat + self.out_proj(messages.to(feat.dtype))
        if self.context_residual:
            # Optional direct context path: singleton voxels otherwise have alpha=1
            # and cannot receive neighboring information through scores alone.
            out = out + self.context_gain * self.context_out(voxel_context)[layout.parent]
        if self.ffn is not None:
            out = out + self.ffn(self.ffn_norm(out))
        return out


class RandLAVoxelQKVStage(nn.Module):
    def __init__(self, in_channels, channels, depth, size, geometry_width,
                 num_heads, backend, edge_chunk_size, checkpoint_edges,
                 attn_drop, ffn_ratio, context_residual, max_edges):
        super().__init__()
        self.size, self.backend, self.max_edges = size, backend, max_edges
        self.geometry_width = geometry_width
        self.chunk_size, self.checkpoint_edges = edge_chunk_size, checkpoint_edges
        self.geometry = PairGeometry(geometry_width)
        self.initialize = nn.Sequential(
            nn.Linear(in_channels + geometry_width, channels), nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.blocks = nn.ModuleList([
            ContextQKVAttention(channels, geometry_width, num_heads, backend,
                                edge_chunk_size, checkpoint_edges, attn_drop,
                                ffn_ratio, context_residual)
            for _ in range(depth)
        ])

    def forward(self, coord, feat, offset):
        layout = voxel_partition(coord, offset, self.size)
        graph = within_voxel_graph(layout, self.max_edges)
        relative = (coord.detach().float() - layout.centroid[layout.parent]) / self.size
        dst, src = graph
        summary = torch.zeros((len(coord), self.geometry_width), device=coord.device,
                              dtype=torch.float32)
        for a in range(0, len(dst), self.chunk_size):
            dd, ss = dst[a:a + self.chunk_size], src[a:a + self.chunk_size]
            r = edge_run(self, self.geometry, coord.detach(), relative, dd, ss, self.size)
            summary = summary.index_add(0, dd, r.float())
        summary = (summary / layout.occupancy[layout.parent, None]).to(feat.dtype)
        feat = self.initialize(torch.cat([feat, summary], dim=-1))
        stencil = sparse_stencil(layout.keys) if self.backend == "torch" else None
        for block in self.blocks:
            feat = block(feat, coord.detach(), relative, layout, graph,
                         self.geometry, self.size, stencil)
        return feat




@torch.no_grad()
def random_voxel_sample(coord, offset, size, ratio, training):
    """Stratified sampling: retain ceil(ratio*n_v), at least one per voxel.

    Returns selected point indices and a link for voxel-based decoder unpooling.
    Random training choices; deterministic first-in-input-order choices in eval.
    """
    layout = voxel_partition(coord, offset, size)
    parent, occupancy = layout.parent, layout.occupancy
    order = torch.randperm(len(coord), device=coord.device) if training else torch.arange(len(coord), device=coord.device)
    order = order[torch.argsort(parent[order], stable=True)]
    start = occupancy.cumsum(0) - occupancy
    rank = torch.arange(len(coord), device=coord.device) - start[parent[order]]
    keep = (occupancy.float() * ratio).ceil().long().clamp_min(1)
    selected = order[rank < keep[parent[order]]].sort().values
    new_offset = torch.bincount(layout.batch[selected], minlength=len(offset)).cumsum(0).to(offset.dtype)
    return selected, new_offset, parent, parent[selected], len(occupancy)


def stage_options(value, n, name):
    if isinstance(value, (tuple, list)):
        if len(value) != n:
            raise ValueError(f"{name} must have one entry per stage")
        return tuple(value)
    return (value,) * n


@MODELS.register_module("RandLA-Voxel-QKV-v1")
class RandLAVoxelQKVNet(nn.Module):
    """Multiscale point QKV backbone returning features for original input rows.

    All within-voxel ordered point pairs are used. No K cap or FlashAttention.
    Sampling is voxel-stratified: random in train, deterministic in eval, and
    keeps at least one point per voxel. The decoder uses coarse voxel broadcast
    and fine point skips; initially discarded points get a separate refinement.
    Set input_sample_ratio=1 and all sample_ratios=1 for no point sampling.
    """

    def __init__(
        self, in_channels=6, channels=(32, 64, 128), depths=(1, 1, 1),
        voxel_sizes=(0.1, 0.2, 0.4), sample_ratios=(0.5, 0.5),
        input_sample_ratio=1.0,
        geometry_width=16, num_heads=(2, 4, 8), ffn_ratio=2.0,
        context_residual=False, max_edges=8_000_000,
        sparse_backend="spconv", edge_chunk_size=32768, checkpoint_edges=True,
        attn_drop=0.0, out_channels=64,
    ):
        super().__init__()
        n = len(channels)
        if n < 1 or len(depths) != n or len(voxel_sizes) != n or len(sample_ratios) != n - 1:
            raise ValueError("channels/depths/voxel_sizes need S entries; sample_ratios needs S-1")
        for name, seq in (("channels", channels), ("depths", depths)):
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
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (geometry_width, edge_chunk_size, out_channels)):
            raise ValueError("widths, edge_chunk_size and out_channels must be positive integers")
        if not 0 <= attn_drop < 1:
            raise ValueError("attn_drop must be in [0,1)")
        heads = stage_options(num_heads, n, "num_heads")
        if any(isinstance(h, bool) or not isinstance(h, int) or h < 1
               or channels[i] % h for i, h in enumerate(heads)):
            raise ValueError("num_heads must give a positive divisor of each stage width")
        if not math.isfinite(ffn_ratio) or ffn_ratio < 0:
            raise ValueError("ffn_ratio must be finite and nonnegative; 0 disables FFN")
        if not isinstance(context_residual, bool):
            raise ValueError("context_residual must be a bool")
        if max_edges is not None and (isinstance(max_edges, bool)
                                     or not isinstance(max_edges, int) or max_edges < 1):
            raise ValueError("max_edges must be a positive integer or None")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.head_channels = out_channels
        self.input_sample_ratio = input_sample_ratio
        self.voxel_sizes, self.sample_ratios = tuple(voxel_sizes), tuple(sample_ratios)
        self.stages = nn.ModuleList([
            RandLAVoxelQKVStage(
                in_channels if level == 0 else channels[level - 1], width, depths[level],
                voxel_sizes[level], geometry_width, heads[level], sparse_backend,
                edge_chunk_size, checkpoint_edges, attn_drop, ffn_ratio,
                context_residual, max_edges,
            ) for level, width in enumerate(channels)
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Linear(channels[i] + channels[i + 1], channels[i]),
                          nn.LayerNorm(channels[i]), nn.GELU(), nn.Linear(channels[i], channels[i]))
            for i in range(n - 1)
        ])
        self.projection = nn.Sequential(nn.LayerNorm(channels[0]), nn.Linear(channels[0], out_channels))
        self.input_restore = (
            nn.Sequential(nn.Linear(out_channels + in_channels + 4, out_channels),
                          nn.GELU(), nn.Linear(out_channels, out_channels))
            if input_sample_ratio < 1 else None
        )

    def _inputs(self, data):
        coord, offset = data["coord"], data["offset"]
        if coord.ndim != 2 or coord.shape[1] != 3 or not len(coord) or not coord.is_floating_point() or not bool(torch.isfinite(coord).all()):
            raise ValueError("coord must be finite floating-point nonempty [N,3]")
        if offset.ndim != 1 or offset.dtype not in (torch.int32, torch.int64) or not len(offset) or int(offset[-1]) != len(coord) or bool((counts(offset) <= 0).any()):
            raise ValueError("offset must contain positive cumulative cloud sizes ending at N")
        feat = data.get("feat")
        if feat is None and self.in_channels == 0:
            feat = coord.new_empty((len(coord), 0))
        if feat is None or feat.shape != (len(coord), self.in_channels) or not feat.is_floating_point() or not bool(torch.isfinite(feat).all()):
            raise ValueError("feat must be finite [N,in_channels]; may be absent only when in_channels=0")
        if coord.device != feat.device or coord.device != offset.device:
            raise ValueError("coord, feat and offset must share a device")
        return coord, feat, offset

    def forward(self, data_dict, return_stages=False):
        if return_stages:
            raise ValueError("return_stages/deep supervision is unsupported; intermediate labels require sampling maps")
        coord, feat, offset = self._inputs(data_dict)
        original_coord, original_feat = coord, feat
        input_link = None
        if self.input_sample_ratio < 1:
            selected, offset, fine_parent, coarse_parent, nv = random_voxel_sample(
                coord, offset, self.voxel_sizes[0], self.input_sample_ratio, self.training,
            )
            coord, feat = coord[selected], feat[selected]
            input_link = (selected, fine_parent, coarse_parent, nv)
        saved = []
        for level, stage in enumerate(self.stages):
            feat = stage(coord, feat, offset)
            if level + 1 < len(self.stages):
                selected, new_offset, fine_parent, coarse_parent, nv = random_voxel_sample(
                    coord, offset, self.voxel_sizes[level + 1], self.sample_ratios[level], self.training,
                )
                saved.append((feat, fine_parent, coarse_parent, nv))
                coord, feat, offset = coord[selected], feat[selected], new_offset
        for level in reversed(range(len(saved))):
            skip, fine_parent, coarse_parent, nv = saved[level]
            coarse_voxel = segment_mean(feat, coarse_parent, nv)
            context = coarse_voxel[fine_parent]
            feat = skip + self.decoder[level](torch.cat([skip, context], dim=-1))
        feat = self.projection(feat)
        if input_link is not None:
            selected, fine_parent, coarse_parent, nv = input_link
            context = segment_mean(feat, coarse_parent, nv)[fine_parent]
            with torch.no_grad():
                centroid = segment_mean(original_coord[selected].float(), coarse_parent, nv)
                u = (original_coord.float() - centroid[fine_parent]) / self.voxel_sizes[0]
                geom = torch.cat([u, torch.linalg.vector_norm(u, dim=-1, keepdim=True)], -1)
            restored = context + self.input_restore(
                torch.cat([context, original_feat.to(context.dtype), geom.to(context.dtype)], -1)
            )
            # Retained points keep their exact point-attention outputs. Only
            # discarded points use voxel broadcast + their own attributes/offset.
            feat = restored.index_copy(0, selected, feat)
        point = Point(dict(data_dict))
        point.feat = feat
        return point
