"""PTv3 extension using the existing Pointcept PT-v3m1 implementation.

Original PTv3 modules remain in Pointcept; this file replaces only attention,
pooling and unpooling, and inserts sampled radius enhancement at chosen stages.
Please cite PTv3, RandLA-Net and Fast Point Transformer when using this design.
No accuracy, novelty, speed or memory improvement has been established.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils.structure import Point
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
    SerializedAttention,
    flash_attn,
)

from .geometry import (
    PolarQuaternionMessage,
    capped_members,
    radius_edges,
    rotary_qkv,
    sample_queries,
    segment_softmax,
    segment_sum,
)


def sync_feature(point):
    if "sparse_conv_feat" in point:
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
    return point


class RotarySerializedAttention(SerializedAttention):
    """Existing PTv3 packing/unpacking with continuous Q/K RoPE."""

    def __init__(self, original, scale):
        super().__init__(
            channels=original.channels,
            num_heads=original.num_heads,
            patch_size=(
                original.patch_size
                if original.enable_flash
                else original.patch_size_max
            ),
            qkv_bias=original.qkv.bias is not None,
            qk_scale=original.scale,
            attn_drop=(
                original.attn_drop if original.enable_flash else original.attn_drop.p
            ),
            proj_drop=original.proj_drop.p,
            order_index=original.order_index,
            enable_rpe=False,
            enable_flash=original.enable_flash,
            upcast_attention=original.upcast_attention,
            upcast_softmax=original.upcast_softmax,
        )
        self.load_state_dict(original.state_dict())
        self.metric_scale = scale

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                int(offset2bincount(point.offset).min()), self.patch_size_max
            )
        h, k, c = self.num_heads, self.patch_size, self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat).reshape(-1, 3, h, c // h)
        qkv = rotary_qkv(qkv, point.coord, self.metric_scale)[order]
        if self.enable_flash:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            feature = (
                flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv.to(dtype),
                    cu_seqlens,
                    max_seqlen=k,
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale,
                )
                .reshape(-1, c)
                .to(qkv.dtype)
            )
        else:
            q, key, value = (
                qkv.reshape(-1, k, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
            )
            if self.upcast_attention:
                q, key = q.float(), key.float()
            score = (q * self.scale) @ key.transpose(-2, -1)
            if self.upcast_softmax:
                score = score.float()
            weight = self.attn_drop(score.softmax(-1)).to(value.dtype)
            feature = (weight @ value).transpose(1, 2).reshape(-1, c)
        point.feat = self.proj_drop(self.proj(feature[inverse]))
        return point


class SampledRadiusEnhancement(PointModule):
    def __init__(
        self,
        channels,
        scale,
        radius,
        ratio,
        max_neighbors,
        hidden_channels,
        embedding_mode,
        use_quaternion,
        query_chunk_size,
        max_candidate_edges,
    ):
        super().__init__()
        self.scale, self.radius, self.ratio = scale, radius, ratio
        self.max_neighbors = max_neighbors
        self.query_chunk_size = query_chunk_size
        self.max_candidate_edges = max_candidate_edges
        self.norm = nn.LayerNorm(channels)
        self.value = nn.Linear(channels, hidden_channels)
        self.geometry = PolarQuaternionMessage(
            hidden_channels, embedding_mode, use_quaternion
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.output = nn.Linear(hidden_channels, channels)
        self.gate = nn.Linear(channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, point):
        query = sample_queries(point.batch, self.ratio, self.training)
        row, support = radius_edges(
            point.coord,
            point.batch,
            query,
            self.radius,
            self.max_neighbors,
            self.training,
            self.query_chunk_size,
            self.max_candidate_edges,
        )
        value = self.value(self.norm(point.feat))
        delta = (point.coord[support] - point.coord[query[row]]) / self.scale
        message = self.geometry(delta, value[support] - value[query[row]])
        weight = segment_softmax(self.score(message).squeeze(-1), row, query.numel())
        aggregate = segment_sum(
            message * weight[:, None].to(message.dtype), row, query.numel()
        )
        update = self.output(aggregate)
        update = update * self.gate(point.feat[query]).sigmoid() * self.gamma
        point.feat = point.feat.index_add(0, query, update.to(point.feat.dtype))
        return sync_feature(point)


class GeometryPooling(PointModule):
    """Canonical Cartesian grouping, sampled learned feature aggregation.

    Coordinates are mass-weighted centroids; all children determine centroids
    and maps, only up to max_members determine learned pooled features.
    """

    def __init__(
        self,
        original,
        scale,
        orders,
        max_members,
        embedding_mode,
        use_quaternion,
        shuffle_orders,
    ):
        super().__init__()
        self.stride = original.stride
        self.proj, self.norm, self.act = original.proj, original.norm, original.act
        self.scale, self.orders = scale, orders
        self.max_members, self.shuffle_orders = max_members, shuffle_orders
        channels = original.out_channels
        self.geometry = PolarQuaternionMessage(channels, embedding_mode, use_quaternion)
        self.score = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.GELU(), nn.Linear(channels // 2, 1)
        )
        self.fuse = nn.Linear(channels * 2, channels)

    def forward(self, point):
        depth = int(math.log2(self.stride))
        if depth > point.serialized_depth:
            depth = 0
        key = torch.cat(
            (point.batch[:, None].long(), point.grid_coord.long() >> depth), dim=1
        )
        cell, inverse = torch.unique(key, dim=0, sorted=True, return_inverse=True)
        size = cell.shape[0]
        with torch.no_grad():
            mass = point.get(
                "geometry_mass", torch.ones_like(point.batch, dtype=torch.float32)
            )
            total_mass = segment_sum(mass, inverse, size)
            centroid = segment_sum(point.coord.float() * mass[:, None], inverse, size)
            centroid = centroid / total_mass[:, None].clamp_min(1)
            selected = capped_members(inverse, size, self.max_members, self.training)
        group = inverse[selected]
        feature = self.proj(point.feat[selected])
        delta = (point.coord[selected] - centroid[group]) / self.scale
        message = self.geometry(delta, feature)
        weight = segment_softmax(self.score(message).squeeze(-1), group, size)
        mean = segment_sum(message * weight[:, None].to(message.dtype), group, size)
        maximum = message.new_full((size, message.shape[-1]), -torch.inf)
        maximum.scatter_reduce_(
            0,
            group[:, None].expand_as(message),
            message,
            reduce="amax",
            include_self=True,
        )
        pooled = Point(
            dict(
                feat=self.fuse(torch.cat((mean, maximum), dim=-1)),
                coord=centroid,
                grid_coord=cell[:, 1:].int(),
                batch=cell[:, 0],
                geometry_mass=total_mass,
                pooling_inverse=inverse,
                pooling_parent=point,
            )
        )
        for name in ("condition", "context"):
            if name in point:
                pooled[name] = point[name]
        # Grouping is independent of attention order; recompute canonical codes.
        pooled.serialization(order=self.orders, shuffle_orders=self.shuffle_orders)
        if self.norm is not None:
            pooled = self.norm(pooled)
        if self.act is not None:
            pooled = self.act(pooled)
        pooled.sparsify()
        return pooled


class GeometryUnpooling(PointModule):
    def __init__(self, original, channels, scale, embedding_mode, use_quaternion):
        super().__init__()
        self.proj, self.proj_skip = original.proj, original.proj_skip
        self.scale = scale
        self.geometry = PolarQuaternionMessage(channels, embedding_mode, use_quaternion)
        self.gate = nn.Linear(channels * 3, channels)
        self.output = nn.Linear(channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, point):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        coarse_coord = point.coord
        point, parent = self.proj(point), self.proj_skip(parent)
        coarse, skip = point.feat[inverse], parent.feat
        delta = (parent.coord - coarse_coord[inverse]) / self.scale
        geometry = self.geometry(delta, skip)
        gate = self.gate(torch.cat((skip, coarse, geometry), -1)).sigmoid()
        parent.feat = skip + coarse + self.gamma * gate * self.output(geometry)
        return sync_feature(parent)


class SynchronizedUnpooling(PointModule):
    """Keep additive-skip ablations synchronized with their sparse tensor."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, point):
        return sync_feature(self.original(point))


@MODELS.register_module("PT-v3m1-PQ")
class PolarQuaternionPTv3(PointTransformerV3):
    def __init__(
        self,
        embedding_mode="cylindrical",
        use_quaternion=True,
        attention_position="rope",
        base_grid_size=0.5,
        local_stages=(0, 1),
        query_ratio=0.125,
        radius_multiplier=2.0,
        max_neighbors=16,
        geometry_channels=32,
        radius_query_chunk_size=128,
        radius_max_candidate_edges=262144,
        learned_pool=True,
        pool_max_members=8,
        enhanced_skip=True,
        deterministic_eval=True,
        **kwargs,
    ):
        if base_grid_size <= 0 or radius_multiplier <= 0:
            raise ValueError("base_grid_size and radius_multiplier must be positive")
        if embedding_mode not in ("spherical", "cylindrical"):
            raise ValueError("embedding_mode must be spherical or cylindrical")
        if attention_position not in ("rope", "none"):
            raise ValueError("attention_position must be rope or none")
        if kwargs.get("enable_rpe", False):
            raise ValueError(
                "Set enable_rpe=False; pairwise RPE is not used by this model"
            )
        if not 0 < query_ratio <= 1 or max_neighbors < 1 or pool_max_members < 1:
            raise ValueError("Invalid query ratio or neighbor/member cap")
        if geometry_channels < 4 or geometry_channels % 4:
            raise ValueError("geometry_channels must be a positive multiple of four")
        super().__init__(**kwargs)
        if any(stage < 0 or stage >= self.num_stages for stage in local_stages):
            raise ValueError("local_stages contains an invalid encoder stage")
        self.embedding_mode = embedding_mode
        self.deterministic_eval = deterministic_eval
        self.input_grid_size = base_grid_size
        strides = kwargs.get("stride", (2, 2, 2, 2))
        scales = [base_grid_size]
        for stride in strides:
            scales.append(scales[-1] * stride)
        self.stage_scales = scales
        enc_channels = kwargs.get("enc_channels", (32, 64, 128, 256, 512))
        dec_channels = kwargs.get("dec_channels", (64, 64, 128, 256))
        for stage, encoder in enumerate(self.enc._modules.values()):
            replacement = PointSequential()
            if "down" in encoder._modules:
                down = encoder._modules["down"]
                if learned_pool:
                    down = GeometryPooling(
                        down,
                        scales[stage],
                        self.order,
                        pool_max_members,
                        embedding_mode,
                        use_quaternion,
                        self.shuffle_orders,
                    )
                replacement.add(down, name="down")
            if stage in local_stages:
                replacement.add(
                    SampledRadiusEnhancement(
                        enc_channels[stage],
                        scales[stage],
                        scales[stage] * radius_multiplier,
                        query_ratio,
                        max_neighbors,
                        geometry_channels,
                        embedding_mode,
                        use_quaternion,
                        radius_query_chunk_size,
                        radius_max_candidate_edges,
                    ),
                    name="radius_geometry",
                )
            for name, block in encoder._modules.items():
                if name == "down":
                    continue
                if attention_position == "rope":
                    block.attn = RotarySerializedAttention(block.attn, scales[stage])
                replacement.add(block, name=name)
            self.enc._modules[f"enc{stage}"] = replacement
        if not self.enc_mode:
            for name, decoder in self.dec._modules.items():
                stage = int(name.removeprefix("dec"))
                if enhanced_skip:
                    decoder._modules["up"] = GeometryUnpooling(
                        decoder._modules["up"],
                        dec_channels[stage],
                        scales[stage + 1],
                        embedding_mode,
                        use_quaternion,
                    )
                else:
                    decoder._modules["up"] = SynchronizedUnpooling(
                        decoder._modules["up"]
                    )
                if attention_position == "rope":
                    for block_name, block in decoder._modules.items():
                        if block_name != "up":
                            block.attn = RotarySerializedAttention(
                                block.attn, scales[stage]
                            )

    def train(self, mode=True):
        super().train(mode)
        # Also control unchanged stock pooling in learned_pool=False ablations.
        if hasattr(self, "enc") and hasattr(self, "deterministic_eval"):
            shuffle = self.shuffle_orders and (mode or not self.deterministic_eval)
            for encoder in self.enc._modules.values():
                down = encoder._modules.get("down")
                if down is not None and hasattr(down, "shuffle_orders"):
                    down.shuffle_orders = shuffle
        return self

    def forward(self, data_dict):
        point = Point(data_dict)
        # Hash/centroid geometry remains FP32 under AMP.
        point.coord = point.coord.float()
        shuffle = self.shuffle_orders and (self.training or not self.deterministic_eval)
        point.serialization(order=self.order, shuffle_orders=shuffle)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point
