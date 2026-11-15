"""Native Pointcept reimplementation of Fast Point Transformer.

This independently implements FPT using PyTorch, without importing the
upstream model, MinkowskiEngine, or its CUDA extension. The memory-bounded
aggregation is an internal execution detail, not a model hyperparameter.
It preserves the signed cosine aggregation and first-order gradients.
Floating-point reduction order and sparse backend performance differ.
Singleton BatchNorm uses running statistics, unlike ordinary training BN.
Whole-network numerical parity against MinkowskiEngine is not established.

INSTALL
    Copy to pointcept/models/FPT/fast_point_transformer.py.
    In pointcept/models/FPT/__init__.py add:
        from .fast_point_transformer import FastPointTransformer, FastPointTransformerSegmentor
    In pointcept/models/__init__.py add:
        from .FPT import FastPointTransformer, FastPointTransformerSegmentor
    Import only one copy: model registry names must be unique.

MODEL CONFIG (insert into your complete dataset/runtime config)
    model = dict(
        type="FastPointTransformerSegmentor", num_classes=13, ignore_index=-1,
        backbone=dict(type="FastPointTransformer-v1", in_channels=3,
                      voxel_size=0.04),
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1., ignore_index=-1)],
    )
    Collect feat_keys=("color",) for 3 input channels, or set in_channels to
    match your concatenated features. Pass coord, feat, offset, and segment
    for training; omit segment for inference. No grid_coord is required.

Backbone returns Point with per-input-point features in original order.
Segmentor returns loss during training, seg_logits at inference, both in
labeled evaluation; return_point=True also returns the feature-bearing Point.
The backbone can also use DefaultSegmentorV2 with backbone_out_channels=128
for the default channel layout.

Requires PyTorch >=2.1 and Pointcept. No additional project helper module is
required. This native implementation does not reproduce upstream CUDA speed
and is not compatible with upstream pretrained checkpoints. Previous native
checkpoints retain identical parameter names/shapes for the same architecture.
Based on POSTECH-CVLab/FastPointTransformer (MIT), commit
9b2cec39bc05f4921e4856d32fc563a6b43087cf; preserve the upstream MIT license.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point


class SafeBatchNorm(nn.BatchNorm1d):
    """Use running statistics for a singleton sparse level during training."""

    def forward(self, x):
        if self.training and len(x) == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                0.0,
                self.eps,
            )
        return super().forward(x)


def position_mlp(cin, hidden, cout, activation=nn.ReLU):
    return nn.Sequential(
        nn.Linear(cin, hidden, bias=False),
        SafeBatchNorm(hidden),
        activation(),
        nn.Linear(hidden, cout, bias=False),
        SafeBatchNorm(cout),
        activation(),
    )


def reduce_sum(x, parent, size):
    return x.new_zeros((size,) + x.shape[1:]).index_add(0, parent, x)


def reduce_mean(x, parent, size):
    count = torch.bincount(parent, minlength=size).to(x.dtype)
    return reduce_sum(x, parent, size) / count[:, None]


@dataclass
class VoxelLevel:
    grid: torch.Tensor  # [M,4], batch and integer coordinates at this level
    centroid: torch.Tensor  # in base-voxel units
    count: torch.Tensor  # number of original points, not occupied child voxels
    stride: int
    maps: dict = field(default_factory=dict)

    @property
    def normalized_centroid(self):
        return self.centroid / self.stride - self.grid[:, 1:].float() - 0.5

    @torch.no_grad()
    def kernel_map(self, kernel_size):
        """Occupied cubic neighbors, exact batch separation, no all-pairs matrix.

        Unique-row lookup avoids integer hash overflow. Maps are cached for the
        current forward only; costs are O(K M log M), with O(M) lookup workspace.
        """
        if kernel_size not in self.maps:
            ids = torch.arange(len(self.grid), device=self.grid.device)
            maps = []
            radius = kernel_size // 2
            for shift in itertools.product(range(-radius, radius + 1), repeat=3):
                query = self.grid.clone()
                query[:, 1:] += query.new_tensor(shift)
                unique, inverse = torch.unique(
                    torch.cat((self.grid, query)), dim=0, return_inverse=True
                )
                lookup = ids.new_full((len(unique),), -1)
                lookup[inverse[: len(ids)]] = ids
                src = lookup[inverse[len(ids) :]]
                valid = src >= 0
                maps.append((ids[valid], src[valid]))
            self.maps[kernel_size] = maps
        return self.maps[kernel_size]


@torch.no_grad()
def make_level(grid, xyz, count=None, stride=1):
    grid, inverse = torch.unique(grid, dim=0, return_inverse=True)
    if count is None:
        count = xyz.new_ones((len(xyz), 1))
    total = reduce_sum(count, inverse, len(grid))
    centroid = reduce_sum(xyz * count, inverse, len(grid)) / total
    return VoxelLevel(grid, centroid, total, stride), inverse


class SparseCosineAggregation(torch.autograd.Function):
    """Recompute edge messages in bounded chunks instead of saving them.

    Inputs q and pos are already normalized. Save O(NC + KC) features and
    integer maps, not O(EC) gathered edge features. First derivatives only.
    """

    @staticmethod
    def forward(ctx, q, values, pos, maps, chunk_size):
        ctx.save_for_backward(q, values, pos)
        ctx.maps, ctx.chunk_size = maps, chunk_size
        result = torch.zeros_like(values)
        for k, (dst, src) in enumerate(maps):
            for a in range(0, len(dst), chunk_size):
                d, s = dst[a : a + chunk_size], src[a : a + chunk_size]
                weight = (q[d] * pos[k]).sum(-1)
                result.index_add_(0, d, weight[..., None] * values[s])
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        q, values, pos = ctx.saved_tensors
        dq, dv, dp = (
            torch.zeros_like(q),
            torch.zeros_like(values),
            torch.zeros_like(pos),
        )
        with torch.autocast(device_type=q.device.type, enabled=False):
            for k, (dst, src) in enumerate(ctx.maps):
                for a in range(0, len(dst), ctx.chunk_size):
                    d = dst[a : a + ctx.chunk_size]
                    s = src[a : a + ctx.chunk_size]
                    g, query, value = grad_output[d], q[d], values[s]
                    weight = (query * pos[k]).sum(-1)
                    dw = (g * value).sum(-1)
                    dv.index_add_(0, s, weight[..., None] * g)
                    dq.index_add_(0, d, dw[..., None] * pos[k])
                    dp[k].add_((dw[..., None] * query).sum(0))
        return dq, dv, dp, None, None


class LightweightSelfAttention(nn.Module):
    """Upstream LSA: sum_j cosine(q_i, inter_pos[offset]) * value_j.

    Signed weights are neither softmaxed nor divided by neighbor count.
    Continuous centroid encoding is added before query/value projections.
    """

    def __init__(self, cin, cout, heads=8, kernel_size=3):
        super().__init__()
        if heads < 1 or cout % heads or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("cout must be divisible by heads; kernel_size must be odd")
        self.heads, self.width, self.kernel_size = heads, cout // heads, kernel_size
        self.intra = nn.Sequential(position_mlp(3, 3, cin), nn.Linear(cin, cin))
        self.query, self.value = nn.Linear(cin, cout), nn.Linear(cin, cout)
        self.inter = nn.Parameter(torch.randn(kernel_size**3, heads, self.width))
        self.out = nn.Linear(cout, cout)

    def forward(self, x, level):
        x = x + self.intra(level.normalized_centroid.to(x.dtype))
        q = self.query(x).reshape(-1, self.heads, self.width)
        v = self.value(x).reshape(-1, self.heads, self.width)
        # Autocast may promote normalization to float32 while value stays half.
        # Normalize and accumulate explicitly in float32, also avoiding the
        # tiny default normalize epsilon underflowing in float16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1)
            pos = F.normalize(self.inter.float(), dim=-1)
            values = v.float()
            result = SparseCosineAggregation.apply(
                q, values, pos, level.kernel_map(self.kernel_size), 4096
            )
        return self.out(result.flatten(1).to(v.dtype))


class AttentionNorm(nn.Module):
    def __init__(self, cin, cout, heads, kernel_size=3):
        super().__init__()
        self.attention = LightweightSelfAttention(cin, cout, heads, kernel_size)
        self.norm = SafeBatchNorm(cout)

    def forward(self, x, level):
        return F.relu(self.norm(self.attention(x, level)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.first = AttentionNorm(channels, channels, heads)
        self.second = LightweightSelfAttention(channels, channels, heads)
        self.norm = SafeBatchNorm(channels)

    def forward(self, x, level):
        return F.relu(x + self.norm(self.second(self.first(x, level), level)))


@MODELS.register_module("FastPointTransformer-v1")
class FastPointTransformer(nn.Module):
    """Pointcept backbone returning a Point with original point order/count.

    Input: coord float [N,3], feat float [N,in_channels], offset cumulative [B]
    or batch integer [N]. Internal voxelization uses coord / voxel_size, not
    grid_coord; retain continuous coord in the Pointcept data pipeline.
    """

    def __init__(
        self,
        in_channels=6,
        voxel_size=0.04,
        init_channels=32,
        enc_channels=32,
        channels=(64, 128, 384, 640, 384, 384, 256, 128),
        depths=(2, 3, 4, 6, 2, 2, 2, 2),
        heads=8,
    ):
        super().__init__()
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        if len(channels) != 8 or len(depths) != 8 or any(d < 0 for d in depths):
            raise ValueError("channels and depths must each contain eight entries")
        self.in_channels, self.voxel_size = in_channels, voxel_size
        self.out_channels = channels[-1]  # final per-point backbone feature width
        self.enc = position_mlp(3, enc_channels, enc_channels, nn.Tanh)
        self.stem = AttentionNorm(in_channels + enc_channels, init_channels, heads, 5)
        self.first = AttentionNorm(init_channels, channels[0], heads)
        self.enc_transitions = nn.ModuleList(
            AttentionNorm(channels[i], channels[i + 1], heads) for i in range(3)
        )
        self.blocks = nn.ModuleList(
            nn.ModuleList(ResidualBlock(channels[i], heads) for _ in range(depths[i]))
            for i in range(8)
        )
        decoder_inputs = (
            channels[3] * 2,
            channels[4] + channels[2],
            channels[5] + channels[1],
            channels[6] + channels[0],
        )
        self.dec_transitions = nn.ModuleList(
            AttentionNorm(cin, channels[i + 4], heads)
            for i, cin in enumerate(decoder_inputs)
        )
        # Upstream final classifier is split into this feature MLP and the
        # Pointcept segmentor's linear classifier.
        self.final = nn.Sequential(
            nn.Linear(channels[-1] + enc_channels, channels[-1], bias=False),
            SafeBatchNorm(channels[-1]),
            nn.ReLU(),
        )

    def forward(self, input_dict):
        point = Point(dict(input_dict))
        coord, feat = point.coord, point.feat
        if coord.ndim != 2 or coord.shape[1] != 3 or len(coord) == 0:
            raise ValueError("coord must be nonempty [N,3]")
        if feat.shape != (len(coord), self.in_channels) or not feat.is_floating_point():
            raise ValueError("feat must be floating [N,in_channels]")
        if not bool(torch.isfinite(coord).all()) or not bool(
            torch.isfinite(feat).all()
        ):
            raise ValueError("coord and feat must be finite")
        if "offset" in point:
            offset = point.offset.long()
            if offset.ndim != 1 or not len(offset) or int(offset[-1]) != len(coord):
                raise ValueError("offset must end at N")
            counts = torch.diff(F.pad(offset, (1, 0)))
            if bool((counts < 0).any()):
                raise ValueError("offset must be nondecreasing")
            batch = torch.repeat_interleave(
                torch.arange(len(offset), device=coord.device), counts
            )
            if "batch" in point and not torch.equal(batch, point.batch.long()):
                raise ValueError("batch and offset disagree")
        elif "batch" in point:
            batch = point.batch.long()
        else:
            raise ValueError("provide offset or batch")
        if batch.shape != (len(coord),) or bool((batch < 0).any()):
            raise ValueError("batch must be nonnegative [N]")
        with torch.no_grad():
            xyz = coord.detach().float() / self.voxel_size
            grid = torch.cat((batch[:, None], torch.floor(xyz).long()), dim=1)
            level, inverse = make_level(grid, xyz)
            residual = xyz - level.centroid[inverse]
        embeddings = self.enc(residual.to(feat.dtype))
        x = torch.cat(
            (
                reduce_mean(feat, inverse, len(level.grid)),
                reduce_mean(embeddings, inverse, len(level.grid)),
            ),
            dim=-1,
        )
        x = self.first(self.stem(x, level), level)
        skips, levels, parents = [x], [level], []
        # Upstream blocks run AFTER each max pool and BEFORE channel transition.
        for i in range(4):
            grid = level.grid.clone()
            grid[:, 1:] = torch.div(grid[:, 1:], 2, rounding_mode="floor")
            coarse, parent = make_level(
                grid, level.centroid, count=level.count, stride=level.stride * 2
            )
            expanded = parent[:, None].expand_as(x)
            x = x.new_full((len(coarse.grid), x.shape[1]), -torch.inf).scatter_reduce(
                0, expanded, x, reduce="amax", include_self=True
            )
            parents.append(parent)
            level = coarse
            levels.append(level)
            for block in self.blocks[i]:
                x = block(x, level)
            if i < 3:
                x = self.enc_transitions[i](x, level)
                skips.append(x)
        for i in range(4):
            level = levels[3 - i]
            # Non-overlapping kernel-2 transpose pooling broadcasts parent values.
            x = torch.cat((x[parents[3 - i]], skips[3 - i]), dim=-1)
            x = self.dec_transitions[i](x, level)
            for block in self.blocks[i + 4]:
                x = block(x, level)
        point.feat = self.final(torch.cat((x[inverse], embeddings), dim=-1))
        return point


@MODELS.register_module("FastPointTransformerSegmentor")
class FastPointTransformerSegmentor(nn.Module):
    """Same train/eval/test dictionary contract as Pointcept DefaultSegmentorV2."""

    def __init__(
        self,
        num_classes,
        backbone,
        criteria=None,
        ignore_index=-1,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(self.backbone.out_channels, num_classes)
        self.criteria = build_criteria(
            criteria
            if criteria is not None
            else [
                dict(
                    type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index
                )
            ]
        )
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, input_dict, return_point=False):
        target = input_dict.get("segment")
        if self.training and target is None:
            raise ValueError("training requires segment labels")
        point = self.backbone(input_dict)
        logits = self.seg_head(point.feat)
        output = {} if self.training else {"seg_logits": logits}
        if target is not None:
            if target.shape != (len(logits),) or target.dtype != torch.long:
                raise ValueError("segment must be long [N]")
            valid = target != self.ignore_index
            if bool(((target[valid] < 0) | (target[valid] >= self.num_classes)).any()):
                raise ValueError(
                    f"segment contains invalid IDs {torch.unique(target[valid][(target[valid] < 0) | (target[valid] >= self.num_classes)]).tolist()}; "
                    f"expected 0..{self.num_classes - 1} or ignore_index={self.ignore_index}"
                )
            with torch.autocast(device_type=logits.device.type, enabled=False):
                output["loss"] = (
                    self.criteria(logits.float(), target)
                    if bool(valid.any())
                    else logits.float().sum() * 0
                )
        if return_point:
            output["point"] = point
        return output
