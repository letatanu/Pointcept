"""
PT-QWNO redesigned

Features
--------
* ``context_operator='haar_wno'`` is a genuine fixed-Haar wavelet neural
  operator. ``'haar_wno_cnn'`` selects a learned multiscale 3-D CNN baseline
  for a controlled ablation. ``'fno'`` is also provided; ``'none'`` disables
  volumetric context.
* ``shared_haar_wno`` enables weight sharing for either Haar or CNN context;
  the input projections and gated residual fusions remain stage-specific.
* Quaternion Q/K rotations encode pairwise x_i - x_j with no centroid
  or explicit pairwise positional tensor. FlashAttention is enabled by default.
* PTv3 sparse-convolution CPE is retained before attention. HaarWNO context
  is retained at encoder transitions with learned per-channel gates.
* Quaternion axes are shared across points within each group; linear phases
  make this a factorized rotary encoding, replacing the old radial mapping.
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
    SerializedAttention,
    Embedding,
    SerializedPooling,
    SerializedUnpooling,
)
import weakref
from pointcept.models.utils.misc import offset2bincount

try:
    import flash_attn
except ImportError:
    flash_attn = None


class QuaternionRelativePosition(nn.Module):
    """Factorized quaternion encoding of x_i - x_j, without pairwise tensors.

    For each head/group, U(x) = (cos(w.x), axis * sin(w.x)).
    Apply L(U(x)^-1) to Q and K. Their dot product then contains
    L(U(x_i - x_j)). Axes are shared across points within a group.
    This is a rotary encoding, not the original radius/direction mapping.
    """

    def __init__(self, head_dim, num_heads, 
                 base=10000.0,
                 learnable_frequencies=True, learnable_axes=True):
        super().__init__()
        if head_dim <= 0 or head_dim % 4:
            raise ValueError("Quaternion attention requires head_dim divisible by 4.")
        if num_heads <= 0 or base <= 0:
            raise ValueError("num_heads and quaternion_base must be positive.")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.groups = head_dim // 4

        # Each default 16-channel head covers x, y, z and a diagonal.
        directions = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                                   [0., 0., 1.], [1., 1., 1.]])
        directions = F.normalize(directions, dim=-1)
        h = torch.arange(num_heads)[:, None]
        g = torch.arange(self.groups)[None, :]
        direction = directions[(g + h) % 4]
        levels_per_head = math.ceil(self.groups / 4)
        level = h * levels_per_head + g // 4
        levels = num_heads * levels_per_head
        magnitude = base ** (-level.float() / max(levels - 1, 1))
        omega = direction * magnitude[..., None]

        # Two spherical angles give an always-unit quaternion axis; unlike
        # normalizing an unconstrained vector, this cannot collapse to zero.
        axes = torch.eye(3)[(g + h) % 3]
        axis_angles = torch.stack((torch.atan2(axes[..., 1], axes[..., 0]),
                                   torch.asin(axes[..., 2])), dim=-1)
        if learnable_frequencies:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)
        if learnable_axes:
            self.axis_angles = nn.Parameter(axis_angles)
        else:
            self.register_buffer("axis_angles", axis_angles)

    @staticmethod
    def left_multiply(t, scalar, vector):
        """Hamilton product (scalar, vector) * t; last dimension of t is 4."""
        real, imag = t[..., :1], t[..., 1:]
        return torch.cat((scalar * real - (vector * imag).sum(-1, keepdim=True),
                          scalar * imag + real * vector
                          + torch.cross(vector.expand_as(imag), imag, dim=-1)), -1)

    def forward(self, q, k, coord):
        if q.shape != k.shape or q.ndim != 3:
            raise ValueError("Q and K must have equal shape (N, H, D).")
        if q.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("Q/K dimensions do not match this quaternion module.")
        if coord.shape != (q.shape[0], 3):
            raise ValueError("Coordinates must have shape (N, 3), aligned with Q/K.")

        with torch.autocast(device_type=q.device.type, enabled=False):
            phase = (coord.float()[:, None, None, :]
                     * self.omega.float()[None]).sum(-1)
            azimuth, elevation = self.axis_angles.float().unbind(-1)
            axis = torch.stack((elevation.cos() * azimuth.cos(),
                                elevation.cos() * azimuth.sin(),
                                elevation.sin()), -1)
            scalar = phase.cos()[..., None]
            # Conjugate U(x): negative vector yields the x_i - x_j convention.
            vector = -phase.sin()[..., None] * axis[None]

        def rotate(t):
            grouped = t.reshape(-1, self.num_heads, self.groups, 4)
            return self.left_multiply(grouped, scalar.to(t.dtype),
                                      vector.to(t.dtype)).reshape_as(t)
        return rotate(q), rotate(k)


class QuaternionSerializedAttention(SerializedAttention):
    """PTv3 patching/projections, quaternion Q/K, standard FlashAttention.

    The non-Flash branch is a reference implementation, useful for ablation
    and numerical checks; it materializes attention matrices per patch.
    """

    def __init__(self, channels, num_heads, patch_size, quaternion_base=10000.0,
                 quaternion_learnable_frequencies=True,
                 quaternion_learnable_axes=True, **kwargs):
        if kwargs.get("enable_rpe", False):
            raise ValueError("Disable additive RPE when using quaternion attention.")
        super().__init__(channels=channels, num_heads=num_heads,
                         patch_size=patch_size, **kwargs)
        self.quaternion = QuaternionRelativePosition(
            channels // num_heads, num_heads, quaternion_base,
            quaternion_learnable_frequencies, quaternion_learnable_axes)

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(int(offset2bincount(point.offset).min().item()),
                                  self.patch_size_max)
        H, K, C = self.num_heads, self.patch_size, self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order].reshape(-1, 3, H, C // H)
        q, k, v = qkv.unbind(1)
        q, k = self.quaternion(q, k, point.coord[order])

        if self.enable_flash:
            if q.device.type != "cuda":
                raise RuntimeError("FlashAttention requires CUDA; use enable_flash=False on CPU.")
            packed = torch.stack((q, k, v), dim=1)
            # Honor AMP dtype. Use fp16 for fp32 input on the CUDA path.
            flash_dtype = packed.dtype if packed.dtype in (
                torch.float16, torch.bfloat16) else torch.float16
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                packed.to(flash_dtype).contiguous(), cu_seqlens,
                max_seqlen=K,
                dropout_p=self.attn_drop if self.training else 0.0,
                softmax_scale=self.scale,
            ).reshape(-1, C).to(qkv.dtype)
        else:
            q, k, v = [t.reshape(-1, K, H, C // H).transpose(1, 2)
                       for t in (q, k, v)]
            if self.upcast_attention:
                q, k = q.float(), k.float()
            logits = (q * self.scale) @ k.transpose(-2, -1)
            if self.upcast_softmax:
                logits = logits.float()
            weights = self.attn_drop(self.softmax(logits)).to(v.dtype)
            feat = (weights @ v).transpose(1, 2).reshape(-1, C)

        point.feat = self.proj_drop(self.proj(feat[inverse]))
        return point


def make_quaternion_block(
    channels, heads, patch_size, drop_path, order_index, stage,
    mlp_ratio=4, enable_quaternion_rpe=True, enable_flash=True,
    quaternion_base=10000.0, quaternion_learnable_frequencies=True,
    quaternion_learnable_axes=True,
):
    """Retain the base block's sparse CPE, normalization, MLP and residuals."""
    block = Block(
        channels=channels, num_heads=heads, patch_size=patch_size,
        mlp_ratio=mlp_ratio, drop_path=drop_path,
        norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_norm=True,
        order_index=order_index, cpe_indice_key=f"ptqwno_stage{stage}",
        enable_rpe=False, enable_flash=enable_flash,
        upcast_attention=False, upcast_softmax=False,
    )
    if enable_quaternion_rpe:
        attention = QuaternionSerializedAttention(
            channels=channels, num_heads=heads, patch_size=patch_size,
            order_index=order_index, enable_rpe=False, enable_flash=enable_flash,
            upcast_attention=False, upcast_softmax=False,
            quaternion_base=quaternion_base,
            quaternion_learnable_frequencies=quaternion_learnable_frequencies,
            quaternion_learnable_axes=quaternion_learnable_axes,
        )
        # Preserve the base block's initialized projections.
        attention.qkv = block.attn.qkv
        attention.proj = block.attn.proj
        block.attn = attention
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
        self.channels = channels
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
        return (y + skip)[..., :shape[0], :shape[1], :shape[2]]


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
            expected_type = {
                "haar_wno": HaarWNO,
                "haar_wno_cnn": HaarWNOCNNAlternative,
            }.get(operator)
            if expected_type is None:
                raise ValueError(
                    "shared_operator requires operator='haar_wno' or 'haar_wno_cnn'."
                )
            if not isinstance(shared_operator, expected_type):
                raise TypeError(f"shared_operator must be an instance of {expected_type.__name__}.")
            if shared_operator.channels != context_channels or shared_operator.levels != levels:
                raise ValueError("shared_operator channels and levels must match the context configuration.")

            # Keep a non-owning reference: the backbone registers and owns the
            # shared operator once, rather than registering aliases per transition.
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
                "The shared context operator is no longer available. "
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
            # Accumulate sums/counts in float32 even under AMP. Dense voxels
            # otherwise lose precision in bfloat16 (or overflow in float16).
            grid = torch.zeros(gx * gy * gz, feat.shape[-1], device=feat.device, dtype=torch.float32)
            count = torch.zeros(gx * gy * gz, 1, device=feat.device, dtype=torch.float32)
            grid.index_add_(0, flat, feat[ids].float())
            count.index_add_(0, flat, torch.ones(ids.numel(), 1, device=feat.device, dtype=torch.float32))
            grid = (
                (grid / count.clamp_min(1))
                .view(1, gx, gy, gz, -1)
                .permute(0, 4, 1, 2, 3)
                .contiguous(memory_format=torch.contiguous_format)
                .to(feat.dtype)
            )
            grid = self._get_operator()(grid)
            gathered = grid[0].permute(1, 2, 3, 0).reshape(-1, feat.shape[-1])[flat]
            result[ids] = gathered.to(result.dtype)
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
        enc_channels=(32, 64, 128, 256, 512), 
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024), 
        stride=(2, 2, 2, 2),
        drop_path=0.3, order=("z", "z-trans", "hilbert", "hilbert-trans"),
        use_quaternion_rpe=True,
        enable_flash=True,
        context_operator="haar_wno", 
        context_stages=(True, True, True, True),
        context_channels=64, context_grid_size=(64, 64, 64), haar_levels=2, 
        shared_haar_wno=True,
        fno_modes=8,
        use_decoder=False, dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256), dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024), head_channels=64,
        quaternion_base=10000.0,
        quaternion_learnable_frequencies=True,
        quaternion_learnable_axes=True,
    ):
        super().__init__()
        self.order = [order] if isinstance(order, str) else list(order)
        self.use_decoder = use_decoder
        self.shared_haar_wno = shared_haar_wno

        if shared_haar_wno:
            operator_type = {
                "haar_wno": HaarWNO,
                "haar_wno_cnn": HaarWNOCNNAlternative,
            }.get(context_operator)
            if operator_type is None:
                raise ValueError(
                    "shared_haar_wno=True requires context_operator='haar_wno' or 'haar_wno_cnn'."
                )
            if not any(context_stages):
                raise ValueError(
                    "shared_haar_wno=True requires at least one enabled context stage."
                )

            # This is deliberately owned by the backbone exactly once.
            self.shared_haar_operator = operator_type(
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
                    quaternion_base=quaternion_base,
                    quaternion_learnable_frequencies=quaternion_learnable_frequencies,
                    quaternion_learnable_axes=quaternion_learnable_axes,
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
                        quaternion_base=quaternion_base,
                        quaternion_learnable_frequencies=quaternion_learnable_frequencies,
                        quaternion_learnable_axes=quaternion_learnable_axes,
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
