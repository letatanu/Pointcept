"""Cartesian Q/K RoPE following Point3DRoPE in Pointcept's Utonia model."""

import torch
from torch import nn


class AxialRotary(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.axis_dim = 2 * (head_dim // 6)
        if not self.axis_dim:
            raise ValueError("Axial 3D RoPE needs at least six channels per head")
        # TriSC heads need not divide evenly into three even-sized axis blocks.
        # Rotate the largest such prefix and leave the other channels unchanged.
        exponent = torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim
        self.register_buffer("inv_freq", base ** -exponent, persistent=False)

    def forward(self, feature, position):
        rotary_dim = 3 * self.axis_dim
        axes = feature[..., :rotary_dim].float().unflatten(-1, (3, self.axis_dim))
        first, second = axes.chunk(2, dim=-1)
        phase = position.float()[:, None, :, None] * self.inv_freq.float()
        cos, sin = phase.cos(), phase.sin()
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        return torch.cat((rotated.flatten(-2).to(feature.dtype), feature[..., rotary_dim:]), dim=-1)
