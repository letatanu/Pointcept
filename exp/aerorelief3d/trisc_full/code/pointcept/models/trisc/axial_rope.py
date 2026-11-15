"""Cartesian Q/K RoPE following Point3DRoPE in Pointcept's Utonia model."""

from math import log
import torch
from torch import nn


UTONIA_HEAD_DIM = 18
UTONIA_ROPE_BASE = 10.0


def utonia_coordinates(coord, training):
    """Apply the RoPE-only jitter and rescale used in the Utonia runs."""
    if not training:
        return coord
    jitter = log(1.1)
    rescale = log(1.2)
    axis_scale = coord.new_empty(3).uniform_(-jitter, jitter).exp()
    common_scale = coord.new_empty(1).uniform_(-rescale, rescale).exp()
    return (coord * axis_scale) * common_scale


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
