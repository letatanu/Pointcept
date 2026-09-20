"""
PT-QWNO redesigned

Features
--------
* ``context_operator='haar_wno'`` is a genuine fixed-Haar wavelet neural
  operator. ``'haar_wno_cnn'`` selects a learned multiscale 3-D CNN baseline
  for a controlled ablation. ``'fno'`` is also provided; ``'none'`` disables
  volumetric context.
* Quaternion position encoding is pairwise: q_ij is built from w_i - w_j
  and is used as an attention bias in every non-flash attention block.
* Scalar invariant context and quaternion feature channels are kept separate.
  WNO context modulates quaternion features through scalar gates.
* ``use_decoder`` selects either the lightweight encoder-only upsample head or
  ordinary PTv3-style unpooling decoder stages.

This file is intended to be placed in the Pointcept model package. It uses
the Pointcept PTv3 mode-1 base classes supplied with the original submission.
"""

from contextlib import nullcontext
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from addict import Dict

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Block,
    Embedding,
    SerializedPooling,
    SerializedUnpooling,
)
import weakref


def _unit_quaternion(delta, eps=1e-6):
    """Return unit quaternions [w, x, y, z] for delta (..., 3)."""
    radius = delta.norm(dim=-1, keepdim=True)
    axis = delta / radius.clamp_min(eps)

    theta = torch.sigmoid(radius) * math.pi
    half_theta = 0.5 * theta
    quat = torch.cat(
        [torch.cos(half_theta), torch.sin(half_theta) * axis],
        dim=-1,
    )

    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where((radius > eps).expand_as(quat), quat, identity)


class QuaternionPointEmbedding(nn.Module):
    """
    Linear-memory quaternion positional embedding.

    Builds one quaternion q_i for each point relative to the centroid of
    its own scene, then projects quaternion and radial-frequency features
    from (N, 5 + 2F) to (N, C).

    Unlike QuaternionRelativeBias, this never creates (W, H, K, K) tensors.
    """

    def __init__(self, channels, num_frequencies=8):
        super().__init__()
        self.register_buffer(
            "freqs",
            2.0 ** torch.arange(num_frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.proj = nn.Sequential(
            nn.Linear(5 + 2 * num_frequencies, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    @staticmethod
    def _batch_mean(coord, batch):
        num_batches = int(batch.max().item()) + 1
        center = coord.new_zeros(num_batches, coord.shape[-1])
        center.index_add_(0, batch, coord)

        count = torch.bincount(
            batch,
            minlength=num_batches,
        ).to(coord.dtype).clamp_min_(1)

        return center / count[:, None]

    @staticmethod
    def _batch_max(values, batch, num_batches):
        maximum = values.new_zeros(num_batches, values.shape[-1])
        maximum.scatter_reduce_(
            0,
            batch[:, None],
            values,
            reduce="amax",
            include_self=False,
        )
        return maximum

    def forward(self, point):
        coord = point.coord.float()
        batch = point.batch.long()

        centers = self._batch_mean(coord, batch)
        delta = coord - centers[batch]

        radius = delta.norm(dim=-1, keepdim=True)
        quat = _unit_quaternion(delta)

        num_batches = centers.shape[0]
        radius_max = self._batch_max(radius, batch, num_batches)
        normalized_radius = radius / radius_max[batch].clamp_min(1e-6)

        freqs = self.freqs.to(device=coord.device, dtype=coord.dtype)
        phase = normalized_radius * freqs.unsqueeze(0)

        features = torch.cat(
            [
                quat,
                normalized_radius,
                torch.sin(phase),
                torch.cos(phase),
            ],
            dim=-1,
        )
        return self.proj(features)


class QuaternionPointEmbeddingBlock(PointModule):
    """
    Adds a per-point quaternion embedding before PTv3 attention.

    This retains quaternion positional conditioning but permits FlashAttention,
    because it does not use the base attention module's explicit RPE bias path.
    """

    def __init__(self, block, channels, num_frequencies=8):
        super().__init__()
        self.block = block
        self.quaternion_embedding = QuaternionPointEmbedding(
            channels=channels,
            num_frequencies=num_frequencies,
        )

    def forward(self, point):
        position_embedding = self.quaternion_embedding(point).to(point.feat.dtype)
        point.feat = point.feat + position_embedding
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return self.block(point)


def make_quaternion_block(
    channels,
    heads,
    patch_size,
    drop_path,
    order_index,
    stage,
    mlp_ratio=4,
    enable_quaternion_rpe=True,
    enable_flash=True,
):
    """
    Construct a PTv3 block with linear-memory quaternion position embedding.
    """

    block = Block(
        channels=channels,
        num_heads=heads,
        patch_size=patch_size,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=order_index,
        cpe_indice_key=f"ptqwno_stage{stage}",
        enable_rpe=False,
        enable_flash=enable_flash,
        upcast_attention=False,
        upcast_softmax=False,
    )

    if enable_quaternion_rpe:
        return QuaternionPointEmbeddingBlock(
            block=block,
            channels=channels,
        )
    return block


def _haar_filters(device, dtype):
    """Eight separable 3-D Haar analysis filters."""
    h = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    g = torch.tensor([1.0, -1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    filters = []
    for a in (h, g):
        for b in (h, g):
            for c in (h, g):
                filters.append(a[:, None, None] * b[None, :, None] * c[None, None, :])
    return torch.stack(filters, dim=0).unsqueeze(1)  # (8, 1, 2, 2, 2)


class HaarWNO(nn.Module):
    """Fixed-Haar multilevel wavelet neural operator.

    No learned spatial convolution is used here. ``F.conv3d`` and
    ``F.conv_transpose3d`` implement fixed Haar analysis/synthesis filters;
    all learned operators are channel-space linear maps on wavelet
    coefficients.
    """

    def __init__(self, channels, levels=2):
        super().__init__()
        self.channels = channels
        self.levels = levels
        self.channel_mix = nn.ModuleList(
            [nn.Linear(channels, channels) for _ in range(levels)]
        )
        self.subband_mix = nn.ParameterList(
            [nn.Parameter(torch.eye(8)) for _ in range(levels)]
        )
        self.low_mix = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)

    @staticmethod
    def _channel_map(x, layer):
        """Apply a learned channel operator without a spatial CNN."""
        # (..., C) is the operator domain; spatial locations are independent.
        x = x.movedim(1, -1)
        x = layer(x)
        return x.movedim(-1, 1)

    def _analysis(self, x):
        filt = _haar_filters(x.device, x.dtype).repeat(self.channels, 1, 1, 1, 1)
        y = F.conv3d(x, filt, stride=2, groups=self.channels)
        b, _, h, w, d = y.shape
        return y.view(b, self.channels, 8, h, w, d)

    def _synthesis(self, bands):
        b, c, _, h, w, d = bands.shape
        filt = _haar_filters(bands.device, bands.dtype).repeat(self.channels, 1, 1, 1, 1)
        y = bands.reshape(b, 8 * c, h, w, d)
        return F.conv_transpose3d(y, filt, stride=2, groups=self.channels)

    def forward(self, x):
        original_shape = x.shape[-3:]
        divisor = 2 ** self.levels
        pad = [0, (divisor - original_shape[2] % divisor) % divisor,
               0, (divisor - original_shape[1] % divisor) % divisor,
               0, (divisor - original_shape[0] % divisor) % divisor]
        if any(pad):
            x = F.pad(x, pad)

        details = []
        low = x
        for level in range(self.levels):
            bands = self._analysis(low)
            mixed = self._channel_map(bands, self.channel_mix[level])
            mixed = torch.einsum("bcshwd,st->bcthwd", mixed, self.subband_mix[level])
            details.append(mixed[:, :, 1:])
            low = mixed[:, :, 0]

        low = self._channel_map(low, self.low_mix)
        for level in reversed(range(self.levels)):
            bands = torch.cat([low.unsqueeze(2), details[level]], dim=2)
            low = self._synthesis(bands)
        low = low[..., :original_shape[0], :original_shape[1], :original_shape[2]]
        return self._channel_map(low, self.out)


class HaarWNOCNNAlternative(nn.Module):
    """Learned multiscale CNN alternative for the Haar-WNO ablation.

    This is intentionally *not* called a WNO: it uses learned depthwise
    strided/transposed 3-D convolutions and therefore has ordinary CNN
    receptive fields rather than a fixed wavelet basis.
    """

    def __init__(self, channels, levels=2):
        super().__init__()
        self.levels = levels
        self.analysis = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 3, stride=2, padding=1,
                          groups=channels, bias=False),
                nn.Conv3d(channels, channels, 1),
                nn.GELU(),
            ) for _ in range(levels)
        ])
        self.synthesis = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(channels, channels, 2, stride=2,
                                   groups=channels, bias=False),
                nn.Conv3d(channels, channels, 1),
                nn.GELU(),
            ) for _ in range(levels)
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, 1),
            nn.GELU(),
        )
        self.skip = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        shape = x.shape[-3:]
        divisor = 2 ** self.levels
        pad = [0, (divisor - shape[2] % divisor) % divisor,
               0, (divisor - shape[1] % divisor) % divisor,
               0, (divisor - shape[0] % divisor) % divisor]
        if any(pad):
            x = F.pad(x, pad)
        skip = self.skip(x)
        y = x
        for layer in self.analysis:
            y = layer(y)
        y = self.bottleneck(y)
        for layer in reversed(self.synthesis):
            y = layer(y)
        y = y[..., :skip.shape[-3], :skip.shape[-2], :skip.shape[-1]]
        return y + skip


class FNO3d(nn.Module):
    """Optional Fourier operator retained for controlled comparisons."""

    def __init__(self, channels, modes=8):
        super().__init__()
        self.channels, self.modes = channels, modes
        scale = 1.0 / max(1, channels * channels)
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes, modes, 2)
        )
        self.bypass = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape
        xf = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
        out = torch.zeros_like(xf)
        mh, mw, md = min(self.modes, h), min(self.modes, w), min(self.modes, xf.shape[-1])
        weight = torch.view_as_complex(self.weight[:, :, :mh, :mw, :md].float())
        out[:, :, :mh, :mw, :md] = torch.einsum(
            "bixyz,ioxyz->boxyz", xf[:, :, :mh, :mw, :md], weight
        )
        return torch.fft.irfftn(out, s=(h, w, d), dim=(-3, -2, -1)).to(x.dtype) + self.bypass(x)


class BatchedVolumetricContext(nn.Module):
    """Voxelize each scene independently, apply an operator, gather to points."""

    def __init__(
        self,
        in_channels,
        context_channels=64,
        grid_size=(64, 64, 64),
        operator="haar_wno",
        levels=2,
        modes=8,
        shared_operator=None,
    ):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.context_channels = context_channels
        self.in_proj = nn.Linear(in_channels, context_channels)
        self.out_proj = nn.Linear(context_channels, context_channels)

        if shared_operator is not None:
            if operator != "haar_wno":
                raise ValueError(
                    "shared_operator is only supported when operator='haar_wno'."
                )
            if not isinstance(shared_operator, HaarWNO):
                raise TypeError("shared_operator must be an instance of HaarWNO.")

            # Keep a non-owning reference: the backbone registers and owns the
            # shared HaarWNO once, rather than registering aliases per transition.
            self.operator = None
            self._shared_operator = weakref.ref(shared_operator)
        else:
            self._shared_operator = None
            if operator == "haar_wno":
                self.operator = HaarWNO(context_channels, levels)
            elif operator == "haar_wno_cnn":
                self.operator = HaarWNOCNNAlternative(context_channels, levels)
            elif operator == "fno":
                self.operator = FNO3d(context_channels, modes)
            elif operator == "none":
                self.operator = nn.Identity()
            else:
                raise ValueError(
                    "operator must be haar_wno, haar_wno_cnn, fno, or none"
                )

        self.norm = nn.LayerNorm(context_channels)
        
    def _get_operator(self):
        if self.operator is not None:
            return self.operator

        operator = self._shared_operator() # type: ignore
        if operator is None:
            raise RuntimeError(
                "The shared HaarWNO operator is no longer available. "
                "It must be owned by the parent backbone."
            )
        return operator

    def forward(self, point):
        feat = self.in_proj(point.feat)
        result = torch.zeros_like(feat)
        batch_ids = point.batch.long()
        for batch_id in batch_ids.unique(sorted=True):
            ids = torch.nonzero(batch_ids == batch_id, as_tuple=False).flatten()
            coord = point.coord[ids].float()
            lo = coord.min(0).values
            hi = coord.max(0).values
            scale = (hi - lo).clamp_min(1e-6)
            grid_coord = ((coord - lo) / scale * (torch.tensor(self.grid_size, device=coord.device) - 1)).long()
            grid_coord = torch.maximum(grid_coord, torch.zeros_like(grid_coord))
            grid_coord = torch.minimum(grid_coord, torch.tensor(self.grid_size, device=coord.device) - 1)
            gx, gy, gz = self.grid_size
            flat = grid_coord[:, 0] * gy * gz + grid_coord[:, 1] * gz + grid_coord[:, 2]
            grid = feat.new_zeros(gx * gy * gz, feat.shape[-1])
            count = feat.new_zeros(gx * gy * gz, 1)
            grid.index_add_(0, flat, feat[ids])
            count.index_add_(0, flat, feat.new_ones(ids.numel(), 1))
            grid = (
                (grid / count.clamp_min(1))
                .view(1, gx, gy, gz, -1)
                .permute(0, 4, 1, 2, 3)
                .contiguous(memory_format=torch.contiguous_format)
            )
            grid = self._get_operator()(grid)
            gathered = grid[0].permute(1, 2, 3, 0).reshape(-1, feat.shape[-1])[flat]
            result[ids] = gathered
        return self.norm(self.out_proj(result))


class GatedResidualFusion(nn.Module):
    """Fuse volumetric context into the encoder feature and form a skip."""

    def __init__(self, channels, context_channels, quaternion_channels=0):
        super().__init__()
        self.context_proj = nn.Linear(context_channels, channels)
        self.gate = nn.Sequential(nn.Linear(context_channels, channels), nn.Sigmoid())
        self.quaternion_channels = quaternion_channels

    def forward(self, feat, context):
        gate = self.gate(context)
        context = self.context_proj(context)
        return feat + gate * context


class HaarWNOTransition(PointModule):
    """Haar-WNO transition producing an explicit enhanced skip feature.

    The feature named ``enhanced_skip`` is used twice:
    1. it is pooled into the next encoder stage;
    2. it is retained as the decoder skip feature.

    ``context_operator='haar_wno_cnn'`` selects the CNN ablation while
    retaining this same transition and skip-connection structure.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        context_channels,
        operator,
        grid_size=(64, 64, 64),
        haar_levels=2,
        fno_modes=8,
        stride=2,
        patch_size=64,
        enable_context=True,
        shared_operator=None,
    ):
        super().__init__()
        self.enable_context = enable_context
        self.input_projection = BatchedVolumetricContext(
            in_channels,
            context_channels,
            grid_size=grid_size,
            operator=operator,
            levels=haar_levels,
            modes=fno_modes,
            shared_operator=shared_operator,
        ) if enable_context else None
        
        self.output_projection = GatedResidualFusion(in_channels, context_channels) if enable_context else None
        self.pool = SerializedPooling(
            in_channels, out_channels, stride=stride,
            norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            act_layer=nn.GELU, traceable=True,
        )

    def forward(self, point):
        if self.enable_context:
            context = self.input_projection(point)
            enhanced_skip = self.output_projection(point.feat, context)
            point.feat = enhanced_skip
            # Keep the semantic name visible in the point structure. The
            # SerializedPooling parent stores this same feature for decoder use.
            point["enhanced_skip_feat"] = enhanced_skip
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return self.pool(point)


class EncoderOnlyHead(PointModule):
    def __init__(self, channels, out_channels=64, fusion="sum"):
        super().__init__()
        self.fusion = fusion
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(c, out_channels), nn.LayerNorm(out_channels)) for c in channels])
        self.weights = nn.Parameter(torch.zeros(len(channels)))
        self.out = nn.Sequential(nn.Linear(out_channels if fusion == "sum" else len(channels) * out_channels, out_channels), nn.GELU())

    def forward(self, stages):
        weights = self.weights.softmax(0)
        values = []
        for s, point in enumerate(stages):
            feat = self.proj[s](point.feat)
            for t in range(s, 0, -1):
                feat = feat[stages[t].pooling_inverse]
            values.append(weights[s] * feat)
        feat = torch.stack(values).sum(0) if self.fusion == "sum" else torch.cat(values, -1)
        point = stages[0]
        point.feat = self.out(feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


@MODELS.register_module("PT-v3m1-QWNO-Redesigned")
class PTQWNOEncoderDecoder(PointModule):
    def __init__(
        self, in_channels=6, enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512), enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024), stride=(2, 2, 2, 2),
        drop_path=0.3, order=("z", "z-trans", "hilbert", "hilbert-trans"),
        use_quaternion_rpe=True, 
        enable_flash=False,
        context_operator="haar_wno", context_stages=(True, True, True, True),
        context_channels=64, context_grid_size=(64, 64, 64), haar_levels=2, shared_haar_wno=True,
        fno_modes=8,
        use_decoder=False, dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256), dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024), head_channels=64,
    ):
        super().__init__()
        self.order = [order] if isinstance(order, str) else list(order)
        self.use_decoder = use_decoder
        self.shared_haar_wno = shared_haar_wno

        if shared_haar_wno:
            if context_operator != "haar_wno":
                raise ValueError(
                    "shared_haar_wno=True requires context_operator='haar_wno'."
                )
            if not any(context_stages):
                raise ValueError(
                    "shared_haar_wno=True requires at least one enabled context stage."
                )

            # This is deliberately owned by the backbone exactly once.
            self.shared_haar_operator = HaarWNO(
                channels=context_channels,
                levels=haar_levels,
            )
        else:
            self.shared_haar_operator = None
            
        nstages = len(enc_channels)
        assert len(enc_depths) == nstages == len(enc_num_head)
        assert len(context_stages) == nstages - 1
        if use_quaternion_rpe and not enable_flash:
            print(
                "Warning: quaternion embedding is linear-memory, but non-Flash "
                "attention still materializes quadratic attention logits."
            )
        self.embedding = Embedding(in_channels, enc_channels[0],
                                   partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), nn.GELU)
        paths = torch.linspace(0, drop_path, sum(enc_depths)).tolist()
        self.enc = nn.ModuleList()
        cursor = 0
        for s in range(nstages):
            layers = []
            if s > 0:
                layers.append(
                    HaarWNOTransition(
                        enc_channels[s - 1],
                        enc_channels[s],
                        context_channels,
                        context_operator,
                        grid_size=context_grid_size,
                        haar_levels=haar_levels,
                        fno_modes=fno_modes,
                        stride=stride[s - 1],
                        enable_context=context_stages[s - 1],
                        shared_operator=self.shared_haar_operator,
                    )
                )
            for i in range(enc_depths[s]):
                layers.append(make_quaternion_block(
                    enc_channels[s],
                    enc_num_head[s],
                    enc_patch_size[s],
                    paths[cursor],
                    i % len(self.order),
                    s,
                    enable_quaternion_rpe=use_quaternion_rpe,
                    enable_flash=enable_flash,
                ))
                cursor += 1
            self.enc.append(PointSequential(*layers))

        self.head = None
        if use_decoder:
            self.dec = nn.ModuleList()
            dec_paths = torch.linspace(0, drop_path, sum(dec_depths)).tolist()
            cursor = 0
            current = enc_channels[-1]
            for s in reversed(range(nstages - 1)):
                layers = [SerializedUnpooling(current, enc_channels[s], dec_channels[s],
                                               norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                                               act_layer=nn.GELU)]
                for i in range(dec_depths[s]):
                    layers.append(make_quaternion_block(
                        dec_channels[s],
                        dec_num_head[s],
                        dec_patch_size[s],
                        dec_paths[cursor],
                        i % len(self.order),
                        s,
                        enable_quaternion_rpe=use_quaternion_rpe,
                        enable_flash=enable_flash,
                    ))
                    cursor += 1
                self.dec.append(PointSequential(*layers))
                current = dec_channels[s]
            self.final_proj = nn.Sequential(nn.Linear(current, head_channels), nn.LayerNorm(head_channels), nn.GELU())
        else:
            self.head = EncoderOnlyHead(enc_channels, head_channels)

    def forward(self, data_dict):
        point = data_dict if isinstance(data_dict, Point) else Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=True)
        point.sparsify()
        point = self.embedding(point)
        stages = []
        for stage in self.enc:
            point = stage(point)
            stages.append(point)
        if self.use_decoder:
            for stage in self.dec:
                point = stage(point)
            point.feat = self.final_proj(point.feat)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
            return point
        return self.head(stages)


@MODELS.register_module()
class DefaultSegmentorV3Redesigned(nn.Module):
    def __init__(self, num_classes, backbone_out_channels, backbone=None, criteria=None,
                 freeze_backbone=False):
        super().__init__()
        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(self, input_dict, return_point=False):
        # Respect the caller's gradient mode during evaluation and training.
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            point = self.backbone(input_dict)
        logits = self.seg_head(point.feat)

        if self.training:
            output = {
                "loss": self.criteria(logits, input_dict["segment"])
            }
        else:
            output = {
                "seg_logits": logits
            }

            if "segment" in input_dict:
                output["loss"] = self.criteria(logits, input_dict["segment"])

        if return_point:
            output["point"] = point

        return output
