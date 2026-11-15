"""Bounded metric neighborhoods for the optional point stem."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layout import make_grid_cross_layout, sample_grid_rows


STEM_MODES = ("pointwise", "relative_xyz", "bezier", "bezier_features")


class NeighborhoodStem(nn.Module):
    def __init__(self, channels, mode, radius, samples_per_cell, neighbors, chunk_size):
        super().__init__()
        if mode not in STEM_MODES[1:]:
            raise ValueError("Unknown neighborhood stem mode")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("stem_radius must be finite and positive")
        if any(type(v) is not int or v < 1 for v in (samples_per_cell, chunk_size)):
            raise ValueError("stem_samples_per_cell and stem_chunk_size must be positive integers")
        if type(neighbors) is not int or not 2 <= neighbors <= 27 * samples_per_cell:
            raise ValueError("stem_neighbors must be between 2 and 27 * stem_samples_per_cell")
        self.mode = mode
        self.radius = radius
        self.samples_per_cell = samples_per_cell
        self.neighbors = neighbors
        self.chunk_size = chunk_size
        axis = torch.arange(-1, 2)
        self.register_buffer("offsets", torch.cartesian_prod(axis, axis, axis), persistent=False)
        self.register_buffer("pairs", torch.triu_indices(neighbors, neighbors, offset=1),
                             persistent=False)
        descriptor_width = 6 + channels if mode == "bezier_features" else 6
        self.encode = nn.Sequential(
            nn.Linear(descriptor_width, 16), nn.GELU(), nn.Linear(16, channels)
        )
        self.scale = nn.Parameter(torch.full((channels,), 0.1))

    def _neighbors(self, coord, grid, batch, layout, members, start, end):
        cells = grid[start:end, None] + self.offsets
        in_bounds = ((cells >= 0) & (cells <= layout.upper)).all(dim=-1)
        cells = torch.minimum(cells.clamp_min(0), layout.upper)
        codes = batch[start:end, None] * layout.strides[0] + (cells * layout.strides[1:]).sum(-1)
        index = torch.searchsorted(layout.codes, codes).clamp_max(len(layout.codes) - 1)
        cell_valid = in_bounds & (layout.codes[index] == codes)
        ids = members[layout.lookup[index]].masked_fill(~cell_valid[..., None], -1).flatten(1)
        delta = (coord[ids.clamp_min(0)] - coord[start:end, None]) / self.radius
        dist2 = delta.square().sum(-1)
        valid = (ids >= 0) & (ids != torch.arange(start, end, device=ids.device)[:, None])
        valid = valid & (dist2 > 0) & (dist2 <= 1)
        dist2, indices = dist2.masked_fill(~valid, torch.inf).topk(
            self.neighbors, dim=1, largest=False
        )
        delta = delta.gather(1, indices[..., None].expand(-1, -1, 3))
        ids = ids.gather(1, indices).masked_fill(~dist2.isfinite(), -1)
        return delta.masked_fill(~dist2.isfinite()[..., None], 0), dist2, ids

    def _bezier_pairs(self, delta, dist2):
        left, right = self.pairs
        a, b = delta[:, left], delta[:, right]
        da, db = a.norm(dim=-1), b.norm(dim=-1)
        cosine = (a * b).sum(-1) / (da * db).clamp_min(1e-12)
        spacing = torch.minimum(da, db) / torch.maximum(da, db).clamp_min(1e-12)
        valid = dist2[:, left].isfinite() & dist2[:, right].isfinite()
        valid = valid & (cosine <= -0.5) & (spacing >= 0.5)
        score = (-cosine + spacing - 0.25 * (da + db)).masked_fill(~valid, -torch.inf)
        first = score.argmax(dim=1, keepdim=True)
        chord = b - a
        direction = F.normalize(chord, dim=-1)
        first_direction = direction.gather(1, first[..., None].expand(-1, -1, 3))
        different_direction = (direction * first_direction).sum(-1).abs() < 0.85
        first_left, first_right = left[first], right[first]
        disjoint = ((left != first_left) & (left != first_right)
                    & (right != first_left) & (right != first_right))
        second = score.masked_fill(~(different_direction & disjoint), -torch.inf)
        selected = torch.cat((first, second.argmax(dim=1, keepdim=True)), dim=1)
        selected_valid = torch.stack(
            (score.gather(1, first).squeeze(1).isfinite(), second.amax(1).isfinite()), dim=1
        )
        descriptors = torch.cat(
            ((a + b) / 4, chord.norm(dim=-1, keepdim=True),
             torch.minimum(da, db)[..., None], torch.maximum(da, db)[..., None]), dim=-1
        )
        descriptor = descriptors.gather(1, selected[..., None].expand(-1, -1, 6))
        return descriptor, selected_valid, left[selected], right[selected]

    def forward(self, feature, coord, batch):
        with torch.no_grad():
            grid = torch.floor(coord / self.radius).long()
            grid = grid - grid.amin(dim=0)
            layout = make_grid_cross_layout(grid, batch)
            members = sample_grid_rows(layout, self.samples_per_cell, training=False)

        residuals = []
        for start in range(0, len(coord), self.chunk_size):
            end = min(start + self.chunk_size, len(coord))
            with torch.no_grad():
                delta, dist2, ids = self._neighbors(
                    coord, grid, batch, layout, members, start, end
                )
                if self.mode in ("bezier", "bezier_features"):
                    descriptor, valid, left, right = self._bezier_pairs(delta, dist2)
                    if self.mode == "bezier_features":
                        left_ids = ids.gather(1, left).clamp_min(0)
                        right_ids = ids.gather(1, right).clamp_min(0)
                else:
                    descriptor, valid = torch.cat((delta, delta.abs()), dim=-1), dist2.isfinite()
            descriptor = descriptor.to(feature.dtype)
            if self.mode == "bezier_features":
                # B_feature(0.5) - center, using pointwise stem features as control values.
                # Keep this outside no_grad so the residual trains the neighboring features.
                center = feature[start:end, None]
                midpoint_delta = 0.25 * ((feature[left_ids] - center)
                                         + (feature[right_ids] - center))
                midpoint_delta = midpoint_delta.masked_fill(~valid[..., None], 0)
                descriptor = torch.cat((descriptor, midpoint_delta), dim=-1)
            encoded = self.encode(descriptor)
            pooled = encoded.masked_fill(~valid[..., None], 0).sum(dim=1)
            residuals.append(pooled / valid.sum(dim=1, keepdim=True).clamp_min(1))
        return feature + self.scale.to(feature.dtype) * torch.cat(residuals).to(feature.dtype)
