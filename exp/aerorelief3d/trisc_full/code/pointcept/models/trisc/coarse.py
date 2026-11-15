"""Bounded occupied-grid global context, applied at configured encoder locations.

This is spatial pooling, NOT a wavelet operator. No dense 64^3 tensor is built.
Each cloud is independently normalized to a grid solely for assigning cells.
Rotary attention uses physical positions and the selected rotary mode's scale.
"""
import math
import torch
from torch import nn
from .attention import PackedAttention
from .geometry import group_mean


@torch.no_grad()
def crop_bounds(coord, batch):
    clouds = int(batch.max()) + 1
    low = coord.new_full((clouds, 3), torch.inf)
    high = coord.new_full((clouds, 3), -torch.inf)
    indices = batch[:, None].expand(-1, 3)
    low.scatter_reduce_(0, indices, coord, reduce="amin", include_self=True)
    high.scatter_reduce_(0, indices, coord, reduce="amax", include_self=True)
    return low, (high - low).clamp_min(1e-6)


@torch.no_grad()
def cell_indices(coord, batch, grid_size, bounds):
    low, span = bounds
    normalized = (coord - low[batch]) / span[batch]
    grid = torch.tensor(grid_size, device=coord.device, dtype=coord.dtype)
    cell = torch.floor(normalized * grid).long()
    cell = torch.minimum(cell.clamp_min(0), grid.long() - 1)
    return torch.cat((batch[:, None], cell), -1)


@torch.no_grad()
def coarse_cells(coord, batch, grid_size):
    indices = cell_indices(coord, batch, grid_size, crop_bounds(coord, batch))
    return torch.unique(indices, dim=0, sorted=True, return_inverse=True)


def add_context(feature, token, inverse, project, gate, scale):
    context = project(token)[inverse]
    weight = gate(torch.cat((feature, context), -1))
    return feature + scale.to(feature.dtype) * weight * context


CELL_POOL_MODES = ("mean_ln_project", "ln_mean_project", "ln_max_project")


def pool_cell_features(feature, inverse, count, mass, project, mode):
    norm, linear = project
    if mode != "mean_ln_project":
        feature = norm(feature)
    if mode == "ln_max_project":
        present = torch.bincount(inverse, minlength=count) > 0
        pooled = feature.new_full((count, feature.shape[1]), -torch.inf)
        pooled.scatter_reduce_(0, inverse[:, None].expand_as(feature), feature,
                               reduce="amax", include_self=True)
        pooled = pooled.masked_fill(~present[:, None], 0)
    else:
        pooled, weight = group_mean(feature, inverse, count, mass)
        present = weight > 0
    if mode == "mean_ln_project":
        pooled = norm(pooled)
    return linear(pooled), present


class CoarseGlobalContext(nn.Module):
    def __init__(self, channels, context_channels=64, heads=8, depth=1,
                 grid_size=(8, 8, 8), rotary="qk", position_scale=32.0,
                 backend="flash_varlen", chunk_size=128,
                 min_frequency=.25, max_frequency=2.0, fine_channels=None,
                 use_fine_features=True, cell_pool="mean_ln_project"):
        super().__init__()
        if (len(grid_size) != 3 or any(type(v) is not int or v < 1 for v in grid_size)
                or any(type(v) is not int or v < 1 for v in (context_channels, heads, depth))):
            raise ValueError("coarse grid dimensions/channels/heads/depth must be positive integers")
        if context_channels % heads:
            raise ValueError("coarse_channels must be divisible by coarse_heads")
        if backend == "flash_varlen" and rotary != "utonia_qk" and (context_channels // heads) % 8:
            raise ValueError("FlashAttention coarse head dimension must be a multiple of 8")
        if not math.isfinite(position_scale) or position_scale <= 0:
            raise ValueError("coarse_position_scale must be finite and positive")
        if cell_pool not in CELL_POOL_MODES:
            raise ValueError(f"coarse_cell_pool must be one of {CELL_POOL_MODES}")
        self.grid_size = tuple(grid_size)
        self.max_tokens = math.prod(grid_size)
        self.position_scale = position_scale
        self.use_fine_features = use_fine_features
        self.cell_pool = cell_pool
        input_width = context_channels
        if fine_channels is not None:
            if context_channels % 2:
                raise ValueError("Shared coarse attention needs an even coarse_channels")
            input_width = context_channels // 2
            self.fine_project_in = nn.Sequential(
                nn.LayerNorm(fine_channels), nn.Linear(fine_channels, input_width))
            self.presence = nn.Linear(2, context_channels, bias=False)
            self.fine_project_out = nn.Sequential(
                nn.Linear(context_channels, fine_channels), nn.LayerNorm(fine_channels))
            self.fine_gate = nn.Sequential(nn.LayerNorm(fine_channels * 2),
                                           nn.Linear(fine_channels * 2, fine_channels), nn.Sigmoid())
            self.fine_scale = nn.Parameter(torch.full((fine_channels,), .1))
        self.project_in = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, input_width))
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
        token, _ = pool_cell_features(feature, inverse, count, mass,
                                      self.project_in, self.cell_pool)
        with torch.no_grad():
            centers, _ = group_mean(coord, inverse, count, mass)
        token, counts, table = self._mix_cells(token, centers, cells)
        result = add_context(feature, token, inverse, self.project_out, self.gate, self.scale)
        if return_debug:
            return result, dict(cells=cells.detach(), inverse=inverse.detach(),
                                token_counts=counts.detach(), token_members=table.detach())
        return result

    def forward_shared(self, fine, final, bounds):
        fine_feat, fine_coord, fine_batch, fine_mass = fine
        final_feat, final_coord, final_batch, final_mass = final
        with torch.no_grad():
            fine_cells = cell_indices(fine_coord, fine_batch, self.grid_size, bounds)
            final_cells = cell_indices(final_coord, final_batch, self.grid_size, bounds)
            cells, inverse = torch.unique(torch.cat((fine_cells, final_cells)), dim=0,
                                          sorted=True, return_inverse=True)
            fine_inverse, final_inverse = inverse.split((len(fine_cells), len(final_cells)))
            low, span = bounds
            centers = low[cells[:, 0]] + (
                (cells[:, 1:] + .5) / span.new_tensor(self.grid_size)) * span[cells[:, 0]]
        fine_token, fine_present = pool_cell_features(
            fine_feat, fine_inverse, len(cells), fine_mass, self.fine_project_in, self.cell_pool)
        final_token, final_present = pool_cell_features(
            final_feat, final_inverse, len(cells), final_mass, self.project_in, self.cell_pool)
        # Keep the same union and projections in the control, masking both inputs
        # from the finer stage. Keeping the projection in the graph also lets DDP
        # use the same parameter set with find_unused_parameters=False.
        fine_input = fine_present & self.use_fine_features
        present = torch.stack((fine_input, final_present), dim=-1)
        fine_token = fine_token.masked_fill(~fine_input[:, None], 0)
        final_token = final_token.masked_fill(~final_present[:, None], 0)
        token = torch.cat((fine_token, final_token), dim=-1)
        token, counts, table = self._mix_cells(
            token + self.presence(present.to(token.dtype)), centers, cells)
        fine_feat = add_context(fine_feat, token, fine_inverse, self.fine_project_out,
                                self.fine_gate, self.fine_scale)
        final_feat = add_context(final_feat, token, final_inverse, self.project_out,
                                 self.gate, self.scale)
        return fine_feat, final_feat, dict(cells=cells, fine_inverse=fine_inverse,
                                           inverse=final_inverse, token_counts=counts,
                                           token_members=table, stage_presence=present,
                                           stage_occupancy=torch.stack((fine_present, final_present), dim=-1))

    def _mix_cells(self, token, centers, cells):
        cloud = cells[:, 0]
        counts = torch.bincount(cloud)
        starts = counts.cumsum(0) - counts
        # Occupied cells only. Empty cells never become keys or queries.
        width = int(counts.max())
        table = cloud.new_full((len(counts), width), -1)
        ids = torch.arange(len(cells), device=cloud.device)
        table[cloud, ids - starts[cloud]] = ids
        for attention, ffn, scale in zip(self.attentions, self.ffns, self.ffn_scales):
            token = attention(token, centers, table, table, width, width, self.position_scale)
            token = token + scale.to(token.dtype) * ffn(token)
        return token, counts, table
