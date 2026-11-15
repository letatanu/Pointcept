"""Minimal Pointcept B0/B1 invariant scalar attention encoder-decoder.

Install this file beside your existing so3_quaternion_transformer.py and add:
    from .so3_scalar_baseline import SO3ScalarBaseline, ScalarBaselineSegmentor
in that package's __init__.py. The package must be imported by pointcept.models.
No old model imports or registry names need to be removed.

Copy MODEL_CONFIG below into your dataset config as `model`. Set num_classes
for your dataset (13 S3DIS; 5 3DAeroRelief). Keep the dataset pipeline, optimizer,
schedule, augmentation and evaluation identical between runs. Disable old
auxiliary/consistency model arguments by replacing the WHOLE model dictionary.

B0: geometry_in_values=False. B1: geometry_in_values=True. Train separately
from scratch. All other architecture settings are identical. Both instantiate
the same parameters (the B0 value projection is frozen and unused), preserving
common initialization under the same seed. Report trainable parameters too.

Only RGB/invariant attributes may be selected as scalar features. XYZ enters
via distances, FPS, kNN and inverse-distance interpolation. No vectors,
quaternions, RBF, covariance, learned pooling or auxiliary losses are used.
Distances retain their input units; there is no neighborhood normalization.
The architecture is also reflection-invariant, and cannot detect chirality.
Invariance applies to fixed point sets, up to numerical errors and sampling
or neighbor ties; axis-aligned voxel preprocessing is NOT rotation-invariant.

CUDA pointops is required for practical large clouds. CPU fallbacks are small
correctness-test implementations. Chunking limits temporary activations, not
all autograd memory. Optional checkpointing reduces saved attention memory.
"""
from __future__ import annotations
from typing import Optional
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
try:
    import pointops
except ImportError:
    pointops = None
try:
    from pointcept.models.builder import MODELS, build_model
    from pointcept.models.losses import build_criteria
except ModuleNotFoundError as exc:
    if not (exc.name == 'pointcept' or exc.name.startswith('pointcept.')):
        raise
    MODELS = None  # Standalone backbone smoke tests only.


def register(name):
    return MODELS.register_module(name) if MODELS is not None else lambda cls: cls

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

class ScalarAttention(nn.Module):
    def __init__(self, channels, heads, geometry_in_values, chunk_size,
                 checkpoint_attention=False):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        self.heads, self.dim = heads, channels // heads
        self.geometry_in_values = geometry_in_values
        self.chunk_size = chunk_size
        self.checkpoint_attention = checkpoint_attention
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.distance = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16))
        self.bias = nn.Linear(16, heads)
        self.geometry_value = nn.Linear(16, channels, bias=False)
        if not geometry_in_values:
            self.geometry_value.requires_grad_(False)
        self.proj = nn.Linear(channels, channels)
        self.ffn = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels * 4),
                                 nn.GELU(), nn.Linear(channels * 4, channels))

    def edge_attention(self, q, k, v, idx, distance):
        # q: [queries,H,D]; gathered k/v: [queries,neighbors,H,D].
        e = self.distance(distance[..., None].to(q.dtype))
        score = (q[:, None] * k[idx]).sum(-1) * self.dim ** -0.5
        score = score + self.bias(e)
        weights = score.float().softmax(dim=1).to(v.dtype)
        value = v[idx]
        if self.geometry_in_values:
            value = value + self.geometry_value(e).reshape(*idx.shape, self.heads, self.dim)
        return (weights[..., None] * value).sum(1).flatten(1)

    def forward(self, x, idx, distance):
        q, k, v = self.qkv(self.norm(x)).reshape(len(x), 3, self.heads, self.dim).unbind(1)
        pieces = []
        for start in range(0, len(x), self.chunk_size):
            args = (q[start:start+self.chunk_size], k, v,
                    idx[start:start+self.chunk_size], distance[start:start+self.chunk_size])
            if self.checkpoint_attention and self.training and torch.is_grad_enabled():
                out = checkpoint(self.edge_attention, *args, use_reentrant=False)
            else:
                out = self.edge_attention(*args)
            pieces.append(out)
        x = x + self.proj(torch.cat(pieces))
        return x + self.ffn(x)


def mean_pool(features, idx, chunk):
    return torch.cat([features[i].mean(1) for i in idx.split(chunk)])


def interpolate(features, source_xyz, source_offset, target_xyz, target_offset, chunk):
    with torch.no_grad():
        idx, dist = knn_query(source_xyz, source_offset, 3, target_xyz, target_offset)
        exact = dist < 1e-8
        inverse = dist.clamp_min(1e-8).reciprocal()
        weights = torch.where(exact.any(1, keepdim=True), exact.float(), inverse)
        weights = weights / weights.sum(1, keepdim=True)
    return torch.cat([(features[i] * w[..., None].to(features.dtype)).sum(1)
                      for i, w in zip(idx.split(chunk), weights.split(chunk))])


@register('SO3-ScalarBaseline')
class SO3ScalarBaseline(nn.Module):
    def __init__(self, 
                 in_channels=6, 
                 scalar_feature_indices=(3, 4, 5),
                 channels=(32, 64, 128, 256), 
                 depths=(1, 1, 2, 2),
                 heads=(2, 4, 8, 8), 
                 neighbors=(16, 16, 16, 16),
                 sample_ratios=(0.25, 0.25, 0.25), 
                 head_channels=64,
                 geometry_in_values=False, 
                 edge_chunk_size=256,
                 checkpoint_attention=False):
        super().__init__()
        n = len(channels)
        if n < 1 or not (len(depths) == len(heads) == len(neighbors) == n):
            raise ValueError('channels/depths/heads/neighbors must have equal nonzero length')
        if len(sample_ratios) != n - 1 or any(not 0 < r <= 1 for r in sample_ratios):
            raise ValueError('one sampling ratio in (0,1] per transition is required')
        if edge_chunk_size < 1 or any(k < 1 for k in neighbors) or any(d < 1 for d in depths):
            raise ValueError('chunk size, neighbors and depths must be positive')
        indices = tuple(scalar_feature_indices)
        if not indices or min(indices) < 0 or max(indices) >= in_channels:
            raise ValueError('explicit valid invariant scalar feature indices are required')
        self.indices, self.in_channels = indices, in_channels
        self.ratios, self.neighbors = tuple(sample_ratios), tuple(neighbors)
        self.chunk, self.out_channels = edge_chunk_size, head_channels
        self.stem = nn.Sequential(nn.Linear(len(indices), channels[0]), nn.LayerNorm(channels[0]), nn.GELU())
        def block(i):
            return ScalarAttention(channels[i], heads[i], geometry_in_values,
                                   edge_chunk_size, checkpoint_attention)
        self.encoder = nn.ModuleList([nn.ModuleList([block(i) for _ in range(depths[i])]) for i in range(n)])
        self.down = nn.ModuleList([nn.Sequential(nn.Linear(channels[i], channels[i+1]),
                                  nn.LayerNorm(channels[i+1]), nn.GELU()) for i in range(n-1)])
        self.fuse = nn.ModuleList([nn.Sequential(nn.Linear(channels[i]+channels[i+1], channels[i]),
                                  nn.LayerNorm(channels[i]), nn.GELU()) for i in range(n-1)])
        self.decoder = nn.ModuleList([block(i) for i in range(n-1)])
        self.output = nn.Sequential(nn.LayerNorm(channels[0]), nn.Linear(channels[0], head_channels), nn.GELU())

    def forward(self, data):
        xyz, feat, offset = data['coord'].float(), data['feat'], data['offset']
        if feat.ndim != 2 or feat.shape != (len(xyz), self.in_channels):
            raise ValueError('feat shape must be [N,in_channels]')
        counts = _offset_to_counts(offset)
        if len(offset) == 0 or int(offset[-1]) != len(xyz) or bool((counts <= 0).any()):
            raise ValueError('offset must contain positive, cumulative cloud sizes ending at N')
        xyz, offset = xyz.contiguous(), offset.to(device=xyz.device, dtype=torch.int32)
        x = self.stem(feat[:, self.indices])
        levels = []
        for stage, blocks in enumerate(self.encoder):
            with torch.no_grad():
                idx, dist = knn_query(xyz, offset, self.neighbors[stage])
            for block in blocks:
                x = block(x, idx, dist)
            levels.append((xyz, offset, x, idx, dist))
            if stage < len(self.down):
                with torch.no_grad():
                    counts = (_offset_to_counts(offset).float() * self.ratios[stage]).floor().long().clamp_min(1)
                    new_offset = _counts_to_offset(counts)
                    sampled = farthest_point_sampling(xyz, offset, new_offset)
                    new_xyz = xyz[sampled]
                    pool_idx, _ = knn_query(xyz, offset, self.neighbors[stage], new_xyz, new_offset)
                x = self.down[stage](mean_pool(x, pool_idx, self.chunk))
                xyz, offset = new_xyz, new_offset
        for stage in reversed(range(len(self.decoder))):
            fine_xyz, fine_offset, skip, idx, dist = levels[stage]
            up = interpolate(x, xyz, offset, fine_xyz, fine_offset, self.chunk)
            x = self.fuse[stage](torch.cat((skip, up), -1))
            x = self.decoder[stage](x, idx, dist)
            xyz, offset = fine_xyz, fine_offset
        return self.output(x)


@register('ScalarBaselineSegmentor')
class ScalarBaselineSegmentor(nn.Module):
    """Pointcept loss/logit contract; deliberately no auxiliary losses."""
    def __init__(self, num_classes, backbone, criteria):
        super().__init__()
        if MODELS is None:
            raise ImportError('Segmentor requires Pointcept; backbone works standalone')
        self.backbone = build_model(backbone)
        self.head = nn.Linear(self.backbone.out_channels, num_classes)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        logits = self.head(self.backbone(input_dict))
        if self.training:
            return dict(loss=self.criteria(logits, input_dict['segment']))
        result = dict(seg_logits=logits)
        if 'segment' in input_dict:
            result['loss'] = self.criteria(logits, input_dict['segment'])
        return result


# Replace your WHOLE model dictionary with a deepcopy of this dictionary.
# No scheduler, dataset split or label mapping is silently changed here.
MODEL_CONFIG = dict(
    type='ScalarBaselineSegmentor', num_classes=13,
    criteria=[dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
              dict(type='LovaszLoss', mode='multiclass', loss_weight=1.0, ignore_index=-1)],
    backbone=dict(type='SO3-ScalarBaseline', in_channels=6,
                  scalar_feature_indices=(3, 4, 5),
                  channels=(32, 64, 128, 256), depths=(1, 1, 2, 2),
                  heads=(2, 4, 8, 8), neighbors=(16, 16, 16, 16),
                  sample_ratios=(0.25, 0.25, 0.25), head_channels=64,
                  geometry_in_values=False, edge_chunk_size=256,
                  checkpoint_attention=False))


def smoke_test():
    """Run `python so3_scalar_baseline.py`; CPU only, no Pointcept needed."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    coord = torch.randn(37, 3)
    rgb = torch.randn(37, 3)
    data = dict(coord=coord, feat=torch.cat((coord, rgb), 1), offset=torch.tensor([23, 37]))
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= torch.linalg.det(rotation)
    moved = coord @ rotation.T + torch.tensor([0.5, -0.7, 1.1])
    transformed = dict(coord=moved, feat=torch.cat((moved, rgb), 1), offset=data['offset'])
    configs = dict(channels=(8, 16), depths=(1, 1), heads=(2, 4),
                   neighbors=(4, 4), sample_ratios=(0.5,), head_channels=8,
                   edge_chunk_size=7)
    models = []
    for enabled in (False, True):
        torch.manual_seed(10)
        model = SO3ScalarBaseline(**configs, geometry_in_values=enabled)
        model.eval()
        with torch.no_grad():
            y, yr = model(data), model(transformed)
        assert y.shape == (37, 8)
        torch.testing.assert_close(y, yr, atol=2e-5, rtol=2e-5)
        # Packed cloud isolation.
        single = dict(coord=coord[:23], feat=data['feat'][:23], offset=torch.tensor([23]))
        with torch.no_grad():
            torch.testing.assert_close(y[:23], model(single), atol=2e-5, rtol=2e-5)
        model.train()
        model(data).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        print(f'B{int(enabled)} forward/backward/batch-isolation/rotation OK; error={(y-yr).abs().max().item():.3g}')
        models.append(model)
    # B1 with zero value projection must be numerically the B0 computation.
    with torch.no_grad():
        for module in models[1].modules():
            if isinstance(module, ScalarAttention):
                module.geometry_value.weight.zero_()
        torch.testing.assert_close(models[0](data), models[1](data), atol=0, rtol=0)
    # Check checkpointed attention uses exactly the same forward/backward.
    a = models[0]
    b = copy.deepcopy(a)
    for module in b.modules():
        if isinstance(module, ScalarAttention):
            module.checkpoint_attention = True
    a.zero_grad(); b.zero_grad()
    a(data).sum().backward(); b(data).sum().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.requires_grad:
            torch.testing.assert_close(pa.grad, pb.grad)
    print('B0/B1 zero-projection equivalence and checkpoint gradients OK')


if __name__ == '__main__':
    smoke_test()
