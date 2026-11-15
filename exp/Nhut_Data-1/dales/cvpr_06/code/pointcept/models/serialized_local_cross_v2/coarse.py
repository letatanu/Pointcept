"""Bounded occupied-grid context: one global branch after the final encoder stage.

This is spatial pooling, NOT a wavelet operator. No dense 64^3 tensor is built.
Each cloud is independently normalized to a grid solely for assigning cells;
rotary attention uses physical centroid coordinates divided by position_scale.
"""
import math
import torch
from torch import nn
from .attention import PackedAttention
from .geometry import group_mean


@torch.no_grad()
def coarse_cells(coord, batch, grid_size):
    clouds = int(batch.max()) + 1
    low = coord.new_full((clouds, 3), torch.inf)
    high = coord.new_full((clouds, 3), -torch.inf)
    indices = batch[:, None].expand(-1, 3)
    low.scatter_reduce_(0, indices, coord, reduce="amin", include_self=True)
    high.scatter_reduce_(0, indices, coord, reduce="amax", include_self=True)
    span = (high - low).clamp_min(1e-6)
    normalized = (coord - low[batch]) / span[batch]
    grid = torch.tensor(grid_size, device=coord.device, dtype=coord.dtype)
    cell = torch.floor(normalized * grid).long()
    cell = torch.minimum(cell.clamp_min(0), grid.long() - 1)
    cells, inverse = torch.unique(torch.cat((batch[:, None], cell), -1),
                                  dim=0, sorted=True, return_inverse=True)
    return cells, inverse


class CoarseGlobalContext(nn.Module):
    def __init__(self, channels, context_channels=64, heads=8, depth=1,
                 grid_size=(8, 8, 8), rotary="qk", position_scale=32.0,
                 backend="flash_varlen", chunk_size=128,
                 min_frequency=.25, max_frequency=2.0):
        super().__init__()
        if (len(grid_size) != 3 or any(type(v) is not int or v < 1 for v in grid_size)
                or any(type(v) is not int or v < 1 for v in (context_channels, heads, depth))):
            raise ValueError("coarse grid dimensions/channels/heads/depth must be positive integers")
        if context_channels % heads:
            raise ValueError("coarse_channels must be divisible by coarse_heads")
        if backend == "flash_varlen" and (context_channels // heads) % 8:
            raise ValueError("FlashAttention coarse head dimension must be a multiple of 8")
        if not math.isfinite(position_scale) or position_scale <= 0:
            raise ValueError("coarse_position_scale must be finite and positive")
        self.grid_size = tuple(grid_size)
        self.max_tokens = math.prod(grid_size)
        self.position_scale = position_scale
        self.project_in = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, context_channels))
        self.attentions = nn.ModuleList([
            PackedAttention(context_channels, heads, rotary, backend, chunk_size,
                            min_frequency, max_frequency) for _ in range(depth)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(context_channels), nn.Linear(context_channels, context_channels * 2),
                          nn.GELU(), nn.Linear(context_channels * 2, context_channels)) for _ in range(depth)])
        self.ffn_scales = nn.ParameterList([
            nn.Parameter(torch.full((context_channels,), .1)) for _ in range(depth)])
        self.project_out = nn.Sequential(nn.Linear(context_channels, channels), nn.LayerNorm(channels))
        self.gate = nn.Sequential(nn.LayerNorm(channels * 2), nn.Linear(channels * 2, channels), nn.Sigmoid())
        self.scale = nn.Parameter(torch.full((channels,), .1))

    def forward(self, feature, coord, batch, mass, return_debug=False):
        cells, inverse = coarse_cells(coord, batch, self.grid_size)
        count = len(cells)
        pooled, _ = group_mean(feature, inverse, count, mass)
        with torch.no_grad():
            centers, _ = group_mean(coord, inverse, count, mass)
        token = self.project_in(pooled)
        cloud = cells[:, 0]
        counts = torch.bincount(cloud, minlength=int(batch.max()) + 1)
        starts = counts.cumsum(0) - counts
        # Occupied cells only. Empty cells never become keys or queries.
        width = int(counts.max())
        table = torch.full((len(counts), width), -1, device=batch.device, dtype=torch.long)
        ids = torch.arange(count, device=batch.device)
        table[cloud, ids - starts[cloud]] = ids
        for attention, ffn, scale in zip(self.attentions, self.ffns, self.ffn_scales):
            token = attention(token, centers, table, table, width, width, self.position_scale)
            token = token + scale.to(token.dtype) * ffn(token)
        context = self.project_out(token)[inverse]
        gate = self.gate(torch.cat((feature, context), -1))
        result = feature + self.scale.to(feature.dtype) * gate * context
        if return_debug:
            return result, dict(cells=cells.detach(), inverse=inverse.detach(),
                                token_counts=counts.detach(), token_members=table.detach())
        return result
