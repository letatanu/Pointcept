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


# Use pairwise neighbor-relative encoding at BOTH injection locations.
if __package__:
    from .neighbor_quatrpe import QuatRPE, apply_quat_rotation_to_features
else:
    from neighbor_quatrpe import QuatRPE, apply_quat_rotation_to_features


class NOGlobalBranch(nn.Module):
    """Per-scene voxel averaging -> Haar WNO -> lookup -> channel LayerNorm.

    Each scene uses a separate grid and bounds. FP32 sums/counts avoid half
    overflow. Unoccupied cells are zero. Axes are normalized independently,
    retaining the original geometric convention (not rotation equivariant).
    The global domain is the supplied scene/crop, not unseen tiles of a survey.
    """
    def __init__(self, channels, grid_size=(64, 64, 64), norm_layer=nn.LayerNorm,
                 levels=3, global_rank=16, use_quatrpe=True, global_mixing=True,
                 quatrpe_k=16, quatrpe_backend="auto", quatrpe_chunk_size=512,
                 quatrpe_edge_chunk_size=2048, quatrpe_distance_scale=1.0):
        super().__init__()
        if len(grid_size) != 3 or any(g < 2**levels or g % (2**levels) for g in grid_size):
            raise ValueError("grid_size dimensions must be positive multiples of 2**levels")
        self.grid_size = tuple(grid_size)
        self.no = WNO3dBlock(channels, levels, global_rank, global_mixing)
        self.norm = norm_layer(channels)
        self.rel_pos_enc = QuatRPE(
            channels, k=quatrpe_k, backend=quatrpe_backend,
            chunk_size=quatrpe_chunk_size, edge_chunk_size=quatrpe_edge_chunk_size,
            distance_scale=quatrpe_distance_scale,
        ) if use_quatrpe else None

    def forward(self, feat, point):
        if feat.shape[0] == 0:
            return feat
        if point.coord.shape != (feat.shape[0], 3) or point.batch.shape != (feat.shape[0],):
            raise ValueError("Expected matching features, coordinates and batch IDs")
        if self.rel_pos_enc is not None:
            feat = feat + self.rel_pos_enc(point).to(feat.dtype)
        gx, gy, gz = self.grid_size
        cells, channels = gx * gy * gz, feat.shape[1]
        indices, values = [], []
        for scene in torch.unique(point.batch, sorted=True):
            idx = torch.nonzero(point.batch == scene, as_tuple=False).flatten()
            # Use metric coordinates directly, avoiding an extra quantization.
            coord = point.coord[idx].float()
            lo, hi = coord.amin(dim=0), coord.amax(dim=0)
            extent = (hi - lo).clamp_min(1e-6)
            limit = coord.new_tensor([gx - 1, gy - 1, gz - 1])
            voxel = (((coord - lo) / extent).clamp(0, 1) * limit).long()
            flat = voxel[:, 0] * (gy * gz) + voxel[:, 1] * gz + voxel[:, 2]
            # index_add is differentiable with respect to feature values.
            grid = torch.zeros(cells, channels, device=feat.device, dtype=torch.float32)
            grid = grid.index_add(0, flat, feat[idx].float())
            count = torch.bincount(flat, minlength=cells).float().unsqueeze(-1)
            grid = grid / count.clamp_min(1)
            grid = grid.reshape(1, gx, gy, gz, channels).permute(0, 4, 1, 2, 3).contiguous()
            output = self.no(grid)
            output = output.permute(0, 2, 3, 4, 1).reshape(cells, channels)[flat]
            values.append(self.norm(output).to(feat.dtype))
            indices.append(idx)
        # Preserve input point order, including interleaved/noncontiguous batch IDs.
        return torch.cat(values)[torch.argsort(torch.cat(indices))]
