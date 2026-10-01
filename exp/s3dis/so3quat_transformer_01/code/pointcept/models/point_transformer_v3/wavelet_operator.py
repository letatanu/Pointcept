"""Explicit Haar wavelet-domain operator and batch-isolated point-grid bridge.

Requires PyTorch only. There are NO learned spatial analysis/synthesis filters.
The orthonormal transform retains LLL and all seven detail bands at every level.
Learned channel maps act on coefficients; a separable global kernel mixes all
coarsest LLL locations. This is a new Haar WNO variant, not a reproduction of
Tripura & Chakraborty's exact architecture. It is NOT SO(3)-equivariant.
"""

import math
import torch
from torch import nn


# Character order corresponds to the three tensor spatial axes (D,H,W).
DETAIL_BANDS = ("LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")


def _analysis_axis(x, axis):
    if x.shape[axis] % 2:
        raise ValueError("Haar analysis requires even spatial sizes at every level")
    even = [slice(None)] * x.ndim
    odd = list(even)
    even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
    a, b = x[tuple(even)], x[tuple(odd)]
    return (a + b) / math.sqrt(2), (a - b) / math.sqrt(2)


def _synthesis_axis(low, high, axis):
    if low.shape != high.shape:
        raise ValueError("Low/high coefficient shapes must match")
    a, b = (low + high) / math.sqrt(2), (low - high) / math.sqrt(2)
    axis %= low.ndim
    # Interleave even and odd entries, with autograd preserved.
    return torch.stack((a, b), dim=axis + 1).flatten(axis, axis + 1)


def haar_dwt3d(x):
    """Single-level orthonormal DWT: B,C,D,H,W -> LLL and seven details.

    Uses h_L=[1,1]/sqrt(2), h_H=[1,-1]/sqrt(2) on aligned pairs.
    Sizes must be even. No padding or boundary extension is performed.
    """
    if x.ndim != 5 or any(n < 2 or n % 2 for n in x.shape[-3:]):
        raise ValueError("Expected B,C,D,H,W with positive even spatial sizes")
    bands = {"": x}
    for axis in (-3, -2, -1):
        next_bands = {}
        for name, values in bands.items():
            low, high = _analysis_axis(values, axis)
            next_bands[name + "L"] = low
            next_bands[name + "H"] = high
        bands = next_bands
    return bands["LLL"], {name: bands[name] for name in DETAIL_BANDS}


def haar_idwt3d(low, details):
    """Inverse of haar_dwt3d; all seven detail bands are required."""
    if set(details) != set(DETAIL_BANDS):
        raise ValueError("Expected exactly seven detail bands: " + str(DETAIL_BANDS))
    bands = {"LLL": low, **details}
    for axis in (-1, -2, -3):
        prefixes = sorted({name[:-1] for name in bands})
        bands = {
            name: _synthesis_axis(bands[name + "L"], bands[name + "H"], axis)
            for name in prefixes
        }
    return bands[""]


def haar_wavedec3d(x, levels):
    """Return coarsest LLL and details ordered finest -> coarsest."""
    if not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")
    if x.ndim != 5 or any(n < 2**levels or n % (2**levels) for n in x.shape[-3:]):
        raise ValueError("Each spatial dimension must be divisible by 2**levels")
    details = []
    for _ in range(levels):
        x, detail = haar_dwt3d(x)
        details.append(detail)
    return x, details


def haar_waverec3d(low, details):
    for detail in reversed(details):
        low = haar_idwt3d(low, detail)
    return low


class CoefficientChannelMap(nn.Module):
    """A learned matrix at each wavelet coefficient, NOT a spatial filter."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.einsum("bcxyz,oc->boxyz", x, self.weight)


class GlobalCoarseKernel(nn.Module):
    """Low-rank, full-support kernel on normalized coarse-grid coordinates.

    A[t,r] = softmax_r(q_theta(z_t)); B[r,s] = softmax_s(k_theta(z_s)).
    Output[t] = sum_r A[t,r] sum_s B[r,s] V u[s].
    Thus K[t,s] = sum_r A[t,r] B[r,s] has full spatial support in exact
    arithmetic. The actual learned feature Jacobian can still degenerate.
    This is a learned wavelet-coefficient kernel, not an FFT or spatial CNN.
    Cost O(T*rank*C + T*C*C); no T-by-T attention matrix is allocated.
    """
    def __init__(self, channels, rank=16):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.query = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, rank))
        self.key = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, rank))
        self.value = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        batch, channels, d, h, w = x.shape
        axes = [(torch.arange(n, device=x.device, dtype=x.dtype) + .5) / n
                for n in (d, h, w)]
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        query = self.query(coords).softmax(dim=-1)        # T,R
        key = self.key(coords).transpose(0, 1).softmax(dim=-1)  # R,T
        values = self.value(x.flatten(2).transpose(1, 2))  # B,T,C
        summaries = torch.einsum("rt,btc->brc", key, values)
        out = torch.einsum("tr,brc->btc", query, summaries)
        return out.transpose(1, 2).reshape(batch, channels, d, h, w)


class WNO3dBlock(nn.Module):
    """u -> GELU(W_local u + W_Haar^-1 R_theta W_Haar u).

    R_theta contains independent channel maps for all details and the coarse
    approximation, plus an optional global coarse coefficient kernel. The raw
    DWT/IDWT is perfectly reconstructing; learned processing need not be.
    Parameters are shared over coefficient positions, so admissible grid sizes
    can change. This does not by itself establish resolution-independent accuracy.
    Use autocast with FP32 model weights; explicitly half-cast models are rejected.
    """
    def __init__(self, channels, levels=3, global_rank=16, global_mixing=True,
                 activation=True):
        super().__init__()
        if channels < 1 or not isinstance(levels, int) or levels < 1:
            raise ValueError("channels and levels must be positive")
        self.levels = levels
        self.detail_maps = nn.ModuleList([
            nn.ModuleDict({name: CoefficientChannelMap(channels) for name in DETAIL_BANDS})
            for _ in range(levels)
        ])
        self.low_map = CoefficientChannelMap(channels)
        self.global_kernel = GlobalCoarseKernel(channels, global_rank) if global_mixing else None
        self.local_map = CoefficientChannelMap(channels)
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        if self.low_map.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError("Keep WNO weights in float32; use torch.autocast for mixed precision")
        original_dtype = x.dtype
        # Stable coefficient arithmetic, contractions, and softmax under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            u = x.to(self.low_map.weight.dtype)
            low, details = haar_wavedec3d(u, self.levels)
            transformed_low = self.low_map(low)
            if self.global_kernel is not None:
                transformed_low = transformed_low + self.global_kernel(low)
            transformed_details = [
                {name: maps[name](detail[name]) for name in DETAIL_BANDS}
                for maps, detail in zip(self.detail_maps, details)
            ]
            reconstructed = haar_waverec3d(transformed_low, transformed_details)
            out = self.activation(reconstructed + self.local_map(u))
        return out.to(original_dtype)


