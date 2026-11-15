"""Standalone QMC-3DNet backbone and optional Pointcept registration.

The backbone uses PyTorch, with optional FlashAttention and spconv.
Dyadic voxel pooling follows the geometric rule described by PTv3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F


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
            with torch.autocast(device_type=t.device.type, enabled=False):
                return self.left_multiply(grouped, scalar.to(t.dtype),
                                          vector.to(t.dtype)).reshape_as(t)
        return rotate(q), rotate(k)


def _haar_filters(device, dtype):
    """Eight separable 3-D Haar analysis filters."""
    h = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    g = torch.tensor([1.0, -1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    filters = []
    for a in (h, g):
        for b in (h, g):
            for c in (h, g):
                filters.append(a[:, None, None] * b[None, :, None] * c[None, None, :])
    return torch.stack(filters, dim=0).unsqueeze(1)


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


def _interleave(axes, depth):
    """Encode three nonnegative integer coordinates as a Morton key."""
    x, y, z = axes.unbind(dim=-1)
    code = torch.zeros_like(x)
    for bit in range(depth - 1, -1, -1):
        octant = (((x >> bit) & 1) << 2) | (((y >> bit) & 1) << 1) | ((z >> bit) & 1)
        code = (code << 3) | octant
    return code


def _hilbert_code(grid, depth):
    """Skilling's axes-to-transpose transform followed by bit interleaving."""
    axes = list(grid.unbind(dim=-1))
    for bit in range(depth - 1, 0, -1):
        high = 1 << bit
        low_mask = high - 1
        for dim in range(3):
            invert = (axes[dim] & high) != 0
            if dim == 0:
                axes[0] = axes[0] ^ (invert.long() * low_mask)
            else:
                exchange = (axes[0] ^ axes[dim]) & low_mask
                exchange = torch.where(invert, 0, exchange)
                axes[0] = axes[0] ^ (invert.long() * low_mask) ^ exchange
                axes[dim] = axes[dim] ^ exchange

    axes[1] = axes[1] ^ axes[0]
    axes[2] = axes[2] ^ axes[1]
    correction = torch.zeros_like(axes[2])
    for bit in range(depth - 1, 0, -1):
        high = 1 << bit
        correction ^= ((axes[2] & high) != 0).long() * (high - 1)
    return _interleave(torch.stack([axis ^ correction for axis in axes], dim=-1), depth)


@dataclass
class PatchLayout:
    orders: tuple
    inverses: tuple
    cu_seqlens: torch.Tensor | None
    groups: tuple
    max_length: int

    @classmethod
    def build(cls, grid, counts, names, depth, patch_size, backend):
        positions = torch.arange(len(grid), device=grid.device)
        orders, inverses = [], []
        for name in names:
            axes = grid[:, [1, 0, 2]] if name.endswith("-trans") else grid
            code = (_hilbert_code if name.startswith("hilbert") else _interleave)(axes, depth)
            order = torch.cat([
                idx[key.argsort(stable=True)]
                for idx, key in zip(positions.split(counts), code.split(counts))
            ])
            orders.append(order)
            inverses.append(torch.empty_like(order).scatter_(0, order, positions))

        # A short final patch stays inside its cloud and uses only real tokens.
        lengths = []
        starts_by_length = {}
        start = 0
        for count in counts:
            for size in range(0, count, patch_size):
                length = min(patch_size, count - size)
                lengths.append(length)
                starts_by_length.setdefault(length, []).append(start)
                start += length

        cu_seqlens = None
        groups = ()
        if backend == "flash":
            boundaries = torch.tensor([0, *lengths], dtype=torch.int32, device=grid.device)
            cu_seqlens = boundaries.cumsum(0, dtype=torch.int32)
        else:
            groups = tuple(
                torch.tensor(starts, device=grid.device)[:, None]
                + torch.arange(length, device=grid.device)[None, :]
                for length, starts in starts_by_length.items()
            )
        return cls(tuple(orders), tuple(inverses), cu_seqlens, groups, max(lengths))


class PatchAttention(nn.Module):
    def __init__(self, channels, heads, use_quatrope, backend, quaternion_base):
        super().__init__()
        if heads < 1 or channels % heads:
            raise ValueError("Channels must be divisible by the number of heads.")
        self.heads = heads
        self.head_dim = channels // heads
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.quaternion = (
            QuaternionRelativePosition(self.head_dim, heads, base=quaternion_base)
            if use_quatrope else None
        )
        self.flash = None
        if backend == "flash":
            try:
                from flash_attn import flash_attn_varlen_qkvpacked_func
            except ImportError as exc:
                raise ImportError("Install flash-attn or set attention_backend='sdpa'.") from exc
            self.flash = flash_attn_varlen_qkvpacked_func

    def forward(self, feat, coord, layout, order_index):
        order = layout.orders[order_index]
        qkv = self.qkv(feat)[order].reshape(-1, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        if self.quaternion is not None:
            q, k = self.quaternion(q, k, coord[order])
        packed = torch.stack((q, k, v), dim=1)

        if self.flash is not None:
            if packed.device.type != "cuda":
                raise ValueError("FlashAttention requires CUDA. Use attention_backend='sdpa' on CPU.")
            dtype = packed.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                dtype = torch.float16
            # The logged 48/4 configuration has 12 channels per head.
            # Keep its original softmax scale when padding for the fused kernel.
            padded = F.pad(packed, (0, (-self.head_dim) % 8))
            out = self.flash(
                padded.to(dtype).contiguous(), layout.cu_seqlens, layout.max_length,
                dropout_p=0.0, softmax_scale=self.head_dim ** -0.5,
            )[..., :self.head_dim].to(qkv.dtype)
        else:
            out = torch.empty_like(q, dtype=qkv.dtype)
            for group in layout.groups:
                # Bound the temporary attention size when SDPA uses its math kernel.
                for index in group.split(16):
                    patch = packed[index]
                    qp, kp, vp = (x.transpose(1, 2) for x in patch.unbind(dim=2))
                    value = F.scaled_dot_product_attention(qp, kp, vp)
                    out[index] = value.transpose(1, 2).to(out.dtype)

        return self.proj(out[layout.inverses[order_index]].reshape_as(feat))


@dataclass
class SparseVoxelLayout:
    indices: torch.Tensor
    spatial_shape: list
    batch_size: int
    inverse: torch.Tensor | None
    counts: torch.Tensor | None
    indice_dict: dict = field(default_factory=dict)

    @classmethod
    def build(cls, grid, batch, batch_size, *, already_unique):
        indices = torch.cat((batch[:, None], grid), dim=1)
        inverse = counts = None
        if not already_unique:
            unique, point_to_voxel, voxel_counts = torch.unique(
                indices, dim=0, return_inverse=True, return_counts=True,
            )
            if len(unique) != len(indices):
                indices, inverse, counts = unique, point_to_voxel, voxel_counts
        return cls(indices.int().contiguous(), (grid.amax(dim=0) + 1).tolist(),
                   batch_size, inverse, counts)


class SparseXCPE(nn.Module):
    """Submanifold 3D convolution, pointwise projection and normalization."""

    def __init__(self, channels, spconv):
        super().__init__()
        self.sparse_tensor = spconv.SparseConvTensor
        self.conv = spconv.SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=True, indice_key="xcpe",
        )
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, feat, layout):
        # Use the convolution's parameter dtype, normally FP32, even under BF16 AMP.
        with torch.autocast(device_type=feat.device.type, enabled=False):
            voxels = feat.to(self.conv.weight.dtype)
            if layout.inverse is not None:
                sums = voxels.new_zeros((len(layout.indices), voxels.shape[-1]))
                voxels = sums.index_add_(0, layout.inverse, voxels) / layout.counts[:, None]
            sparse = self.sparse_tensor(
                voxels, layout.indices, layout.spatial_shape, layout.batch_size,
                indice_dict=layout.indice_dict,
            )
            result = self.conv(sparse)
        layout.indice_dict = result.indice_dict
        positional = self.norm(self.proj(result.features))
        if layout.inverse is not None:
            positional = positional[layout.inverse]
        return positional.to(feat.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, channels, heads, use_quatrope, backend, quaternion_base, drop_path):
        super().__init__()
        self.attn_norm = nn.LayerNorm(channels)
        self.attn = PatchAttention(channels, heads, use_quatrope, backend, quaternion_base)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(),
                                 nn.Linear(channels * 4, channels))
        self.drop_path = drop_path
        self.xcpe = None

    def _drop_residual(self, residual, batch, batch_size):
        if not self.training or not self.drop_path:
            return residual
        keep = 1.0 - self.drop_path
        mask = residual.new_empty(batch_size, 1).bernoulli_(keep) / keep
        return residual * mask[batch]

    def forward(self, feat, coord, layout, order_index, batch, batch_size, sparse_layout):
        if self.xcpe is not None:
            feat = feat + self.xcpe(feat, sparse_layout)
        attn = self.attn(self.attn_norm(feat), coord, layout, order_index)
        feat = feat + self._drop_residual(attn, batch, batch_size)
        return feat + self._drop_residual(self.ffn(self.ffn_norm(feat)), batch, batch_size)


class VoxelPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, feat, coord, grid, batch):
        key = torch.cat((batch[:, None], grid >> 1), dim=1)
        parents, inverse, sizes = torch.unique(
            key, sorted=True, dim=0, return_inverse=True, return_counts=True,
        )
        projected = self.proj(feat)
        pooled = projected.new_full((len(parents), projected.shape[-1]), -torch.inf)
        pooled.scatter_reduce_(
            0, inverse[:, None].expand_as(projected), projected,
            reduce="amax", include_self=False,
        )
        centers = coord.new_zeros((len(parents), 3)).index_add_(0, inverse, coord)
        centers = centers / sizes[:, None]
        return F.gelu(self.norm(pooled)), centers, parents[:, 1:], parents[:, 0].contiguous(), inverse


class VoxelUnpool(nn.Module):
    def __init__(self, coarse_channels, skip_channels, out_channels):
        super().__init__()
        self.coarse = nn.Sequential(nn.Linear(coarse_channels, out_channels),
                                    nn.LayerNorm(out_channels), nn.GELU())
        self.skip = nn.Sequential(nn.Linear(skip_channels, out_channels),
                                  nn.LayerNorm(out_channels), nn.GELU())

    def forward(self, feat, skip, inverse):
        return self.skip(skip) + self.coarse(feat)[inverse]


class EarlyHaarContext(nn.Module):
    def __init__(self, channels, context_channels, grid_size, levels):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.in_proj = nn.Linear(channels, context_channels)
        self.operator = HaarWNO(context_channels, levels)
        self.out_proj = nn.Sequential(nn.Linear(context_channels, context_channels),
                                     nn.LayerNorm(context_channels))
        self.context_proj = nn.Linear(context_channels, channels)
        self.gate = nn.Linear(context_channels, channels)

    def forward(self, feat, coord, counts):
        projected = self.in_proj(feat)
        grid_limit = coord.new_tensor(self.grid_size, dtype=torch.long) - 1
        gx, gy, gz = self.grid_size
        context = []
        for points, values in zip(coord.split(counts), projected.split(counts)):
            lower = points.amin(dim=0)
            extent = (points.amax(dim=0) - lower).clamp_min(1e-6)
            voxel = (((points - lower) / extent) * grid_limit).long().clamp_min(0)
            x, y, z = torch.minimum(voxel, grid_limit).unbind(dim=1)
            index = (x * gy + y) * gz + z
            channels = values.shape[-1]
            sums = values.new_zeros((gx * gy * gz, channels), dtype=torch.float32)
            sizes = values.new_zeros((gx * gy * gz, 1), dtype=torch.float32)
            sums.index_add_(0, index, values.float())
            sizes.index_add_(0, index, sizes.new_ones((len(index), 1)))
            grid = (sums / sizes.clamp_min(1)).T.reshape(1, channels, gx, gy, gz)
            result = self.operator(grid.to(values.dtype).contiguous())
            context.append(result.reshape(channels, -1).T[index].to(values.dtype))
        gathered = self.out_proj(torch.cat(context))
        return feat + self.gate(gathered).sigmoid() * self.context_proj(gathered)


class QMC3DNet(nn.Module):
    """Consume Pointcept-style input tensors and return point-aligned features."""

    def __init__(
        self,
        in_channels=6,
        out_channels=64,
        enc_channels=(48, 96, 192, 384, 512),
        enc_depths=(3, 3, 3, 12, 3),
        enc_heads=(4, 8, 12, 24, 32),
        dec_channels=(48, 96, 192, 384),
        dec_depths=(2, 2, 2, 2),
        dec_heads=(4, 8, 12, 24),
        patch_sizes=(1024, 1024, 1024, 1024, 1024),
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        use_quatrope=True,
        use_haar_wno=True,
        quaternion_base=10000.0,
        context_channels=32,
        context_grid_size=(32, 32, 32),
        haar_levels=3,
        drop_path=0.2,
        attention_backend="sdpa",
        use_xcpe=False,
    ):
        super().__init__()
        stages = len(enc_channels)
        if stages < 2 or not len(enc_depths) == len(enc_heads) == len(patch_sizes) == stages:
            raise ValueError("Specify matching encoder settings for at least two stages.")
        if not len(dec_channels) == len(dec_depths) == len(dec_heads) == stages - 1:
            raise ValueError("Specify decoder settings for every non-bottleneck stage.")
        if isinstance(order, str):
            order = (order,)
        if not order or set(order) - {"z", "z-trans", "hilbert", "hilbert-trans"}:
            raise ValueError("Invalid serialization order.")
        if min(patch_sizes) < 1 or min(*enc_depths, *dec_depths) < 0 or not 0 <= drop_path < 1:
            raise ValueError("Invalid patch size, depth or drop path rate.")
        if attention_backend not in ("flash", "sdpa"):
            raise ValueError("attention_backend must be 'flash' or 'sdpa'.")
        if use_haar_wno and (context_channels < 1 or haar_levels < 1
                             or len(context_grid_size) != 3
                             or min(context_grid_size) < 2 ** haar_levels):
            raise ValueError("Invalid Haar channels, grid size or decomposition depth.")
        self.order = tuple(order)
        self.patch_sizes = tuple(patch_sizes)
        self.attention_backend = attention_backend
        self.use_xcpe = use_xcpe
        self.stem = nn.Sequential(nn.Linear(in_channels, enc_channels[0]),
                                  nn.LayerNorm(enc_channels[0]), nn.GELU())
        self.pools = nn.ModuleList(VoxelPool(a, b) for a, b in zip(enc_channels, enc_channels[1:]))

        def make_stages(channels, heads, depths):
            rates = iter(torch.linspace(0, drop_path, sum(depths)).tolist())
            return nn.ModuleList(
                nn.ModuleList(
                    TransformerBlock(c, h, use_quatrope, attention_backend, quaternion_base, next(rates))
                    for _ in range(d)
                )
                for c, h, d in zip(channels, heads, depths)
            )

        self.encoder = make_stages(enc_channels, enc_heads, enc_depths)
        self.decoder = make_stages(dec_channels, dec_heads, dec_depths)
        self.unpools = nn.ModuleList(
            VoxelUnpool(coarse, skip, out)
            for coarse, skip, out in zip(
                (*dec_channels[1:], enc_channels[-1]), enc_channels[:-1], dec_channels,
            )
        )
        self.head = nn.Sequential(nn.Linear(dec_channels[0], out_channels),
                                  nn.LayerNorm(out_channels), nn.GELU())
        self.context = (
            EarlyHaarContext(enc_channels[0], context_channels, context_grid_size, haar_levels)
            if use_haar_wno else None
        )
        if use_xcpe:
            try:
                import spconv.pytorch as spconv
            except ImportError as exc:
                raise ImportError("Install spconv 2.x or set use_xcpe=False.") from exc
            # Build xCPE after the common layers so a paired seed initializes
            # those layers identically whether xCPE is enabled or disabled.
            for blocks, channels in zip((*self.encoder, *self.decoder),
                                         (*enc_channels, *dec_channels)):
                for block in blocks:
                    block.xcpe = SparseXCPE(channels, spconv)

    @staticmethod
    def _run(blocks, feat, coord, layout, batch, batch_size, sparse_layout):
        for i, block in enumerate(blocks):
            feat = block(feat, coord, layout, i % len(layout.orders),
                         batch, batch_size, sparse_layout)
        return feat

    def forward(self, data):
        feat = data["feat"]
        coord = data["coord"].float()
        grid = data["grid_coord"].long()
        if not len(feat) or coord.shape != (len(feat), 3) or grid.shape != coord.shape:
            raise ValueError("Provide nonempty, point-aligned feat, coord and grid_coord tensors.")
        if "offset" in data:
            ends = data["offset"].reshape(-1).long().tolist()
            counts = tuple(end - start for start, end in zip([0, *ends], ends))
            if not counts or min(counts) <= 0 or ends[-1] != len(feat):
                raise ValueError("offset must contain strictly increasing cumulative point counts.")
        else:
            counts = (len(feat),)
        batch_size = len(counts)
        batch = torch.repeat_interleave(
            torch.arange(batch_size, device=grid.device),
            torch.tensor(counts, device=grid.device), output_size=len(feat),
        )
        low, high = torch.stack((grid.amin(), grid.amax())).tolist()
        if low < 0 or high >= 2 ** 21:
            raise ValueError("grid_coord must be nonnegative and fit in 21 bits per axis.")
        depth = max(high.bit_length(), 1)

        feat = self.stem(feat)
        skips, inverses = [], []
        for stage, blocks in enumerate(self.encoder):
            layout = PatchLayout.build(
                grid, counts, self.order, depth, self.patch_sizes[stage], self.attention_backend,
            )
            sparse_layout = None
            if self.use_xcpe:
                sparse_layout = SparseVoxelLayout.build(
                    grid, batch, batch_size, already_unique=stage > 0,
                )
            feat = self._run(blocks, feat, coord, layout, batch, batch_size, sparse_layout)
            if stage == 0 and self.context is not None:
                feat = self.context(feat, coord, counts)
            if stage < len(self.pools):
                skips.append((feat, coord, layout, batch, sparse_layout))
                feat, coord, grid, batch, inverse = self.pools[stage](feat, coord, grid, batch)
                inverses.append(inverse)
                counts = tuple(batch.bincount(minlength=batch_size).tolist())
                depth = max(depth - 1, 1)

        for stage in reversed(range(len(self.decoder))):
            skip, coord, layout, batch, sparse_layout = skips[stage]
            feat = self.unpools[stage](feat, skip, inverses[stage])
            feat = self._run(self.decoder[stage], feat, coord, layout,
                             batch, batch_size, sparse_layout)
        return self.head(feat)


class QMC3DNetSegmentor(nn.Module):
    """Pointcept train/eval adapter for the standalone backbone."""

    def __init__(self, num_classes, backbone, backbone_out_channels, criteria, ignore_index=-1):
        super().__init__()
        from pointcept.models.builder import build_model
        from pointcept.models.losses import build_criteria

        self.classifier = nn.Linear(backbone_out_channels, num_classes)
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.ignore_index = ignore_index

    def forward(self, data):
        logits = self.classifier(self.backbone(data))
        result = {} if self.training else {"seg_logits": logits}
        if "segment" in data:
            target = data["segment"]
            # Lovasz and the all-ignored case need an explicit float32 path.
            with torch.autocast(device_type=logits.device.type, enabled=False):
                logits_float = logits.float()
                result["loss"] = (
                    self.criteria(logits_float, target)
                    if (target != self.ignore_index).any()
                    else logits_float.sum() * 0.0
                )
        elif self.training:
            raise ValueError("Training requires segment labels.")
        return result


def register_pointcept():
    """Call from pointcept/models/__init__.py to expose the config names."""
    from pointcept.models.builder import MODELS

    MODELS.register_module()(QMC3DNet)
    MODELS.register_module()(QMC3DNetSegmentor)
