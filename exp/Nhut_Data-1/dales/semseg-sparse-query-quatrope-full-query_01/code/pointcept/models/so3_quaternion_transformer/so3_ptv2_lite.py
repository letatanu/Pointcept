"""SO(3)-invariant segmentation with PTv2-style scalar/vector attention.

Install in pointcept/models/so3_ptv2_lite.py and import it from models/__init__.py.
Use the companion transforms/config: they build exact Euclidean kNN graphs in
CPU data-loader workers. There is no FPS, voxel pooling, eigensolver, or CUDA
pointops dependency in this model. No coordinate gradients are supported.

Spatial vectors are [N, Cv, 3]. Learned channel mixing has NO spatial bias.
Attention weights and gates are invariant scalars and are broadcast over XYZ.
All final logits are built from invariant scalar features/readouts. In exact
arithmetic this design is O(3)-invariant (therefore also SO(3)-invariant), subject
to consistent point identities, sampling randomness and neighborhood ties.
It is a research implementation, not a benchmarked claim of PTv3 accuracy/speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from pointcept.models.builder import MODELS
except ModuleNotFoundError as exc:
    if exc.name != "pointcept":
        raise
    MODELS = None  # Standalone numerical tests do not need Pointcept.


def register(name):
    return MODELS.register_module(name) if MODELS is not None else lambda cls: cls


def gather(x, index):
    """One differentiable gather per full edge tensor, not many tiny gathers."""
    return x.index_select(0, index.reshape(-1)).reshape(*index.shape, *x.shape[1:])


class VectorLinear(nn.Module):
    """Mix channels, never spatial axes: [N, Cin, 3] -> [N, Cout, 3]."""
    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x.transpose(-1, -2), self.weight).transpose(-1, -2)


class VectorRMSNorm(nn.Module):
    """One invariant RMS denominator per point, plus channel-wise gains.

    A scalar epsilon preserves rotation equivariance. Independent XYZ gains,
    XYZ biases, or coordinate-wise LayerNorm would not.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        inv_rms = (y.square().sum(-1).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (y * inv_rms[..., None] * self.gain.float()[None, :, None]).to(x.dtype)


class VectorUpdate(nn.Module):
    """Cheap equivariant nonlinearity: invariant gates scale whole vectors."""
    def __init__(self, channels):
        super().__init__()
        self.linear = VectorLinear(channels, channels)
        self.norm = VectorRMSNorm(channels)
        self.gate = nn.Linear(channels, channels)

    def forward(self, x):
        x = self.norm(self.linear(x))
        norms = torch.linalg.vector_norm(x.float(), dim=-1).to(x.dtype)
        return x * torch.sigmoid(self.gate(norms))[..., None]


class VectorReadout(nn.Module):
    """Invariant per-channel norms and the upper triangle of a small Gram matrix."""
    def __init__(self, channels, rank=4):
        super().__init__()
        self.norm = VectorRMSNorm(channels)
        self.proj = VectorLinear(channels, rank)
        self.register_buffer("upper", torch.triu_indices(rank, rank), persistent=False)
        self.out_channels = channels + rank * (rank + 1) // 2

    def forward(self, x):
        z = self.norm(x)
        norms = torch.linalg.vector_norm(z.float(), dim=-1)
        p = self.proj(z).float()
        gram = p @ p.transpose(-1, -2)
        return torch.cat((norms, gram[:, self.upper[0], self.upper[1]]), -1).to(x.dtype)


@dataclass
class Graph:
    index: torch.Tensor
    mask: torch.Tensor
    unit: torch.Tensor       # (p_j - p_i) / (distance + eps)
    relative: torch.Tensor   # (p_j - p_i) / (local radius + eps)
    features: torch.Tensor   # four invariant scalar geometry channels


@torch.no_grad()
def geometry(coord, index, mask, eps=1e-6):
    # All spatial reductions remain FP32 even under BF16 training autocast.
    with torch.autocast(device_type=coord.device.type, enabled=False):
        rel = gather(coord.float(), index) - coord.float()[:, None]
        dist = torch.linalg.vector_norm(rel, dim=-1)
        radius = dist.masked_fill(~mask, 0).amax(-1, keepdim=True)
        unit = rel / (dist[..., None] + eps)
        relative = rel / (radius[..., None] + eps)
        features = torch.stack((
            torch.log1p(dist),
            dist / (radius + eps),
            torch.log1p(radius).expand_as(dist),
            torch.log1p(gather(radius, index).squeeze(-1)),
        ), -1)
        return Graph(index, mask, unit, relative, features)


class VectorSeed(nn.Module):
    """Initialize vectors from invariant coefficients times relative vectors.

    This creates geometric vectors, not unconstrained MLP(XYZ) embeddings.
    Symmetric neighborhoods can have zero first moments; do not interpret the
    vector bank as a guaranteed complete frame or complete shape descriptor.
    """
    def __init__(self, channels, vector_channels, hidden=8):
        super().__init__()
        self.key = nn.Linear(channels, hidden)
        self.coeff = nn.Sequential(
            nn.Linear(2 * hidden + 4, hidden), nn.SiLU(),
            nn.Linear(hidden, vector_channels), nn.Tanh(),
        )
        self.update = VectorUpdate(vector_channels)

    def forward(self, scalar, graph):
        h = self.key(scalar)
        edge = torch.cat((h[:, None].expand(-1, graph.index.shape[1], -1),
                          gather(h, graph.index), graph.features.to(h.dtype)), -1)
        coeff = self.coeff(edge).float() * graph.mask[..., None]
        vector = (coeff[..., None] * graph.relative[:, :, None]).sum(1)
        vector = vector / graph.mask.sum(1).clamp_min(1)[:, None, None]
        return self.update(vector.to(scalar.dtype))


class GroupWeightEncoder(nn.Module):
    """PTv2-style grouped linear reduction C -> G, then mixing between groups."""
    def __init__(self, channels, groups):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(groups, channels // groups))
        nn.init.normal_(self.weight, std=(channels // groups) ** -0.5)
        self.mix = nn.Sequential(
            nn.LayerNorm(groups) if groups > 1 else nn.Identity(),
            nn.SiLU(), nn.Linear(groups, groups),
        )

    def forward(self, relation):
        shape = (*relation.shape[:-1], self.groups, -1)
        reduced = (relation.reshape(shape) * self.weight.to(relation.dtype)).sum(-1)
        return self.mix(reduced)


class ScalarVectorAttention(nn.Module):
    """PTv2 ordering: omega(delta_mul(e) * gamma(q,k) + delta_bias(e)).

    e includes distance/radius and signed projections onto learned equivariant
    query/key vectors. Both delta MLPs output C scalars BEFORE C -> G reduction.
    Softmax runs over neighbors. Each group weight is shared by whole vector
    channels, including all three XYZ components. 'Vector attention' in PTv2
    refers to channel-wise weights; it is different from a spatial XYZ vector.
    """
    def __init__(self, channels, vector_channels, groups, pos_hidden=16,
                 direction_channels=4, relation_mode="scalar"):
        super().__init__()
        if channels % groups or vector_channels % groups:
            raise ValueError("Scalar and vector channels must both divide into groups")
        if relation_mode not in ("scalar", "vector"):
            raise ValueError("relation_mode must be 'scalar' or 'vector'")
        self.c, self.cv, self.groups = channels, vector_channels, groups
        self.relation_mode = relation_mode
        self.scalar_qk = nn.Linear(channels, 2 * channels, bias=False) if relation_mode == "scalar" else None
        self.vector_relation = nn.Linear(direction_channels, channels) if relation_mode == "vector" else None
        self.scalar_value = nn.Linear(channels, channels, bias=False)
        self.vector_q = VectorLinear(vector_channels, direction_channels)
        self.vector_k = VectorLinear(vector_channels, direction_channels)
        self.vector_value = VectorLinear(vector_channels, vector_channels)
        edge_channels = 4 + 3 * direction_channels

        def positional_mlp():
            return nn.Sequential(nn.Linear(edge_channels, pos_hidden), nn.SiLU(),
                                 nn.Linear(pos_hidden, channels))

        self.multiplier = positional_mlp()
        self.bias = positional_mlp()
        self.omega = GroupWeightEncoder(channels, groups)
        # Small invariant coefficients replace the old wide [h_i,h_j,e] edge MLP.
        self.vector_coeff = nn.Linear(edge_channels, 2 * vector_channels)
        self.scalar_out = nn.Linear(channels, channels)
        self.vector_out = VectorLinear(vector_channels, vector_channels)

    def forward(self, scalar, vector, graph):
        n, k = graph.index.shape
        g = self.groups
        qv = self.vector_q(vector)
        kv = gather(self.vector_k(vector), graph.index)
        # Dot products are invariant and computed with FP32 accumulation.
        unit = graph.unit[:, :, None]
        dq = (qv.float()[:, None] * unit).sum(-1)
        dk = (kv.float() * unit).sum(-1)
        qk = (qv.float()[:, None] * kv.float()).sum(-1)
        edge = torch.cat((graph.features, dq, dk, qk), -1).to(scalar.dtype)
        if self.scalar_qk is not None:
            q, key = self.scalar_qk(scalar).chunk(2, -1)
            relation = q[:, None] - gather(key, graph.index)
        else:
            # An invariant relation from vector queries/keys: optional ablation.
            difference = (qv.float()[:, None] - kv.float()).square().sum(-1)
            relation = self.vector_relation(difference.to(scalar.dtype))
        relation = self.multiplier(edge) * relation + self.bias(edge)
        logits = self.omega(relation)
        alpha = logits.float().masked_fill(~graph.mask[..., None], -torch.inf).softmax(1)

        values = gather(self.scalar_value(scalar), graph.index).reshape(n, k, g, self.c // g)
        scalar_out = (alpha.to(values.dtype)[..., None] * values).sum(1).reshape(n, self.c)

        vv = gather(self.vector_value(vector), graph.index)
        a, b = self.vector_coeff(edge).chunk(2, -1)
        # a,b are scalars per VECTOR channel. Never learn separate XYZ gates.
        message = (torch.sigmoid(a)[..., None] * vv +
                   torch.tanh(b)[..., None] * graph.relative.to(vv.dtype)[:, :, None])
        message = message.reshape(n, k, g, self.cv // g, 3)
        vector_out = (alpha.to(message.dtype)[..., None, None] * message).sum(1)
        return self.scalar_out(scalar_out), self.vector_out(vector_out.reshape(n, self.cv, 3))


class AttentionBlock(nn.Module):
    def __init__(self, channels, vector_channels, groups, pos_hidden,
                 direction_channels, relation_mode, mlp_ratio=2):
        super().__init__()
        self.scalar_norm = nn.LayerNorm(channels)
        self.vector_norm = VectorRMSNorm(vector_channels)
        self.attention = ScalarVectorAttention(channels, vector_channels, groups,
                                              pos_hidden, direction_channels, relation_mode)
        self.vector_update = VectorUpdate(vector_channels)
        self.readout = VectorReadout(vector_channels)
        self.vector_to_scalar = nn.Sequential(nn.LayerNorm(self.readout.out_channels),
                                             nn.Linear(self.readout.out_channels, channels))
        self.ffn = nn.Sequential(nn.LayerNorm(channels),
                                 nn.Linear(channels, int(channels * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(channels * mlp_ratio), channels))
        self.scalar_scale = nn.Parameter(torch.full((channels,), 0.1))
        self.vector_scale = nn.Parameter(torch.full((vector_channels,), 0.1))

    def forward(self, scalar, vector, graph):
        ds, dv = self.attention(self.scalar_norm(scalar), self.vector_norm(vector), graph)
        # Cast channel scales to activation dtype: FP32 parameters must not
        # silently promote all BF16 residual activations back to FP32.
        scalar = scalar + ds * self.scalar_scale.to(ds.dtype)
        update = self.vector_update(dv)
        vector = vector + update * self.vector_scale.to(update.dtype)[None, :, None]
        scalar = scalar + self.vector_to_scalar(self.readout(vector))
        return scalar + self.ffn(scalar), vector


class Downsample(nn.Module):
    """Pool at randomly retained anchors using rows of the existing fine graph."""
    def __init__(self, cin, cout, vin, vout):
        super().__init__()
        self.scalar = nn.Sequential(nn.Linear(2 * cin, cout), nn.LayerNorm(cout), nn.GELU())
        self.vector = nn.Sequential(VectorLinear(vin, vout), VectorRMSNorm(vout))

    def forward(self, scalar, vector, graph, take):
        index = graph.index.index_select(0, take)
        mask = graph.mask.index_select(0, take)
        denom = mask.sum(1).clamp_min(1)
        s = gather(scalar, index)
        mean = (s * mask[..., None]).sum(1) / denom[:, None]
        maximum = s.masked_fill(~mask[..., None], -torch.inf).amax(1)
        v = (gather(vector, index) * mask[..., None, None]).sum(1) / denom[:, None, None]
        return self.scalar(torch.cat((mean, maximum), -1)), self.vector(v)


class Upsample(nn.Module):
    """Cached 3-NN interpolation + skip fusion. No decoder edge attention."""
    def __init__(self, coarse, fine, vcoarse, vfine):
        super().__init__()
        self.scalar_proj = nn.Linear(coarse, fine)
        self.scalar_fuse = nn.Sequential(nn.Linear(2 * fine, fine), nn.LayerNorm(fine), nn.GELU())
        self.vector_proj = VectorLinear(vcoarse, vfine)
        self.vector_fuse = VectorLinear(2 * vfine, vfine)
        self.vector_update = VectorUpdate(vfine)
        self.readout = VectorReadout(vfine)
        self.readout_proj = nn.Linear(self.readout.out_channels, fine)

    def forward(self, scalar, vector, skip_scalar, skip_vector, index, weight):
        s = self.scalar_proj(scalar)
        v = self.vector_proj(vector)
        s = (gather(s, index) * weight.to(s.dtype)[..., None]).sum(1)
        v = (gather(v, index) * weight.to(v.dtype)[..., None, None]).sum(1)
        s = self.scalar_fuse(torch.cat((s, skip_scalar), -1))
        v = self.vector_update(self.vector_fuse(torch.cat((v, skip_vector), 1)))
        return s + self.readout_proj(self.readout(v)), v


def local_to_packed(index, query_counts, source_counts):
    """Translate each cloud's local indices after standard Pointcept collation."""
    source_starts = F.pad(source_counts.cumsum(0)[:-1], (1, 0))
    base = torch.repeat_interleave(source_starts, query_counts, output_size=index.shape[0])
    return index.long() + base.reshape(-1, *([1] * (index.ndim - 1)))


@register("SO3-PTv2-Lite")
class SO3PTv2Lite(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 96, 160, 256),
                 vector_channels=(4, 8, 8, 8, 8), depths=(1, 1, 2, 2, 2),
                 groups=(2, 4, 4, 8, 8), neighbors=(12, 16, 16, 24, 24),
                 pos_hidden=16, direction_channels=4, relation_mode="scalar",
                 head_channels=64, mlp_ratio=2, checkpoint_blocks=False):
        super().__init__()
        n = len(channels)
        if n < 2 or any(len(v) != n for v in (vector_channels, depths, groups, neighbors)):
            raise ValueError("All stage lists must have equal length >= 2")
        if any(c < 1 for seq in (channels, vector_channels, depths, groups, neighbors) for c in seq):
            raise ValueError("Stage sizes must be positive")
        self.neighbors = tuple(neighbors)
        self.in_channels = in_channels
        self.checkpoint_blocks = checkpoint_blocks
        self.stem = nn.Sequential(nn.Linear(in_channels, channels[0]), nn.LayerNorm(channels[0]), nn.GELU())
        self.seed = VectorSeed(channels[0], vector_channels[0])
        self.encoder = nn.ModuleList([
            nn.ModuleList([AttentionBlock(c, v, g, pos_hidden, direction_channels, relation_mode, mlp_ratio)
                           for _ in range(d)])
            for c, v, g, d in zip(channels, vector_channels, groups, depths)
        ])
        self.down = nn.ModuleList([Downsample(channels[i], channels[i+1], vector_channels[i], vector_channels[i+1])
                                   for i in range(n-1)])
        self.up = nn.ModuleList([Upsample(channels[i+1], channels[i], vector_channels[i+1], vector_channels[i])
                                 for i in range(n-1)])
        self.readout = VectorReadout(vector_channels[0])
        self.output = nn.Sequential(nn.Linear(channels[0] + self.readout.out_channels, head_channels),
                                    nn.LayerNorm(head_channels), nn.GELU())
        self.head_channels = head_channels

    def forward(self, data):
        if "so3_sizes" not in data:
            raise ValueError("Missing cached hierarchy. Use SO3Pack from the companion transforms.")
        sizes = data["so3_sizes"].long()
        if sizes.ndim != 2 or sizes.shape[1] != len(self.encoder):
            raise ValueError("Hierarchy stage count does not match the model")
        # A single early check catches mix_prob != 0 and incompatible collation.
        expected = sizes[:, 0].cumsum(0)
        if not torch.equal(expected, data["offset"].long()):
            raise ValueError("Cloud boundaries changed after graph creation: set mix_prob=0.0")
        coord = data["coord"].float()
        feat = data["feat"]
        if feat.shape != (len(coord), self.in_channels):
            raise ValueError("feat must contain invariant scalar channels only (RGB by default)")
        scalar = self.stem(feat)
        vector = None
        skips = []
        for level, blocks in enumerate(self.encoder):
            index = local_to_packed(data[f"so3_knn_{level}"], sizes[:, level], sizes[:, level])
            if index.shape[1] != self.neighbors[level]:
                raise ValueError("Hierarchy neighbor count does not match the model")
            graph = geometry(coord, index, data[f"so3_mask_{level}"].bool())
            if vector is None:
                vector = self.seed(scalar, graph)
            for block in blocks:
                if self.training and self.checkpoint_blocks:
                    # Checkpoint whole blocks, never 256-point edge fragments.
                    scalar, vector = checkpoint(block, scalar, vector, graph, use_reentrant=False)
                else:
                    scalar, vector = block(scalar, vector, graph)
            skips.append((scalar, vector))
            if level < len(self.down):
                take = local_to_packed(data[f"so3_take_{level+1}"], sizes[:, level+1], sizes[:, level])
                scalar, vector = self.down[level](scalar, vector, graph, take)
                coord = coord.index_select(0, take)
        for level in reversed(range(len(self.up))):
            index = local_to_packed(data[f"so3_up_{level+1}"], sizes[:, level], sizes[:, level+1])
            scalar, vector = self.up[level](scalar, vector, *skips[level], index,
                                           data[f"so3_up_weight_{level+1}"])
        scalar = self.output(torch.cat((scalar, self.readout(vector).to(scalar.dtype)), -1))
        return {"feat": scalar, "vector_feat": vector}


def lovasz_softmax_flat(logits, labels, ignore_index=-1):
    """Multiclass Lovasz-Softmax, averaged over ground-truth-present classes.

    Defined here to keep the standalone tests and Pointcept training on the same
    loss implementation. Ignore labels are removed before softmax and sorting.
    """
    valid = labels != ignore_index
    logits, labels = logits[valid].float(), labels[valid]
    if labels.numel() == 0:
        return logits.sum() * 0
    prob = logits.softmax(-1)
    losses = []
    for c in range(prob.shape[1]):
        foreground = (labels == c).float()
        total = foreground.sum()
        if total.item() == 0:
            continue
        errors, order = (foreground - prob[:, c]).abs().sort(descending=True)
        fg_sorted = foreground[order]
        intersection = total - fg_sorted.cumsum(0)
        union = total + (1 - fg_sorted).cumsum(0)
        jaccard = 1 - intersection / union
        gradient = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
        losses.append(torch.dot(errors, gradient))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


@register("SO3PTv2LiteSegmentor")
class SO3PTv2LiteSegmentor(nn.Module):
    """Pointcept DefaultTrainer / SemSegEvaluator / SemSegTester interface."""
    def __init__(self, num_classes=5, backbone=None, ignore_index=-1,
                 ce_weight=1.0, lovasz_weight=1.0, class_weights=None):
        super().__init__()
        options = dict(backbone or {})
        kind = options.pop("type", "SO3-PTv2-Lite")
        if kind != "SO3-PTv2-Lite":
            raise ValueError("This segmentor expects the SO3-PTv2-Lite backbone")
        if num_classes < 2 or min(ce_weight, lovasz_weight) < 0 or ce_weight + lovasz_weight <= 0:
            raise ValueError("Invalid class count or loss weights")
        self.backbone = SO3PTv2Lite(**options)
        self.classifier = nn.Linear(self.backbone.head_channels, num_classes)
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.ce_weight, self.lovasz_weight = ce_weight, lovasz_weight
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        if weights is not None and (weights.shape != (num_classes,) or not torch.all(weights > 0)):
            raise ValueError("class_weights must be positive and have length num_classes")
        self.register_buffer("class_weights", weights)

    def forward(self, data, return_features=False):
        features = self.backbone(data)
        logits = self.classifier(features["feat"])
        result = {} if self.training else {"seg_logits": logits}
        labels = data.get("segment")
        if self.training and labels is None:
            raise ValueError("Training requires segment labels")
        if labels is not None:
            if labels.shape != (len(logits),) or labels.dtype != torch.long:
                raise ValueError("segment must be int64 [N]")
            valid = labels != self.ignore_index
            if ((labels[valid] < 0) | (labels[valid] >= self.num_classes)).any():
                raise ValueError("segment contains a class outside the configured range")
            with torch.autocast(device_type=logits.device.type, enabled=False):
                if valid.any():
                    loss = logits.float().sum() * 0
                    if self.ce_weight:
                        loss = loss + self.ce_weight * F.cross_entropy(
                            logits.float(), labels, weight=self.class_weights, ignore_index=self.ignore_index)
                    if self.lovasz_weight:
                        loss = loss + self.lovasz_weight * lovasz_softmax_flat(logits, labels, self.ignore_index)
                else:
                    loss = logits.float().sum() * 0
            result["loss"] = loss
        if return_features:
            result.update(features)
            result["seg_logits"] = logits
        return result
