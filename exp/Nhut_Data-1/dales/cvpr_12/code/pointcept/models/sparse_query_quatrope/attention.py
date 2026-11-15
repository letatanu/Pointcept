"""Selected-query/all-key attention with fixed-axis, relative QuatRoPE."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import select_queries


def hamilton(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), dim=-1)


class QuatRoPE(nn.Module):
    """Per-head/group shared axis and linear coordinate phase.

    Applies L(r(x)^-1) to Q and K; values are untouched. Consequently the
    affinity contains L(r(x_i) r(x_j)^-1), with phase omega dot (x_i-x_j).
    Latent quaternion rotation does not guarantee physical SO(3) equivariance.
    """
    def __init__(self, heads, head_dim):
        super().__init__()
        if head_dim % 4:
            raise ValueError("QuatRoPE needs head_dim divisible by four")
        groups = head_dim // 4
        direction = F.normalize(torch.randn(heads, groups, 3), dim=-1)
        frequency = torch.logspace(-1, 1, groups)[None, :, None]
        self.omega = nn.Parameter(direction * frequency)
        self.azimuth = nn.Parameter(torch.empty(heads, groups).uniform_(-math.pi, math.pi))
        self.elevation = nn.Parameter(torch.empty(heads, groups).uniform_(-math.pi / 2, math.pi / 2))

    def forward(self, feature, coord):
        """feature [patch,H,L,D], coord [patch,L,3]."""
        dtype = feature.dtype
        # Geometry stays FP32 even when the surrounding model uses AMP.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            phase = torch.einsum("plc,hgc->phlg", coord.float(), self.omega.float())
            a, e = self.azimuth.float(), self.elevation.float()
            axis = torch.stack((e.cos()*a.cos(), e.cos()*a.sin(), e.sin()), dim=-1)
            inverse_rotor = torch.cat((phase.cos()[..., None],
                                       -phase.sin()[..., None] * axis[None, :, None]), dim=-1)
            result = hamilton(inverse_rotor, feature.float().unflatten(-1, (-1, 4))).flatten(-2)
        return result.to(dtype)


class SelectedQueryAttention(nn.Module):
    def __init__(self, channels, heads, query_ratio=.25, position_encoding="quaternion",
                 dropout=0., patch_chunk_size=128, max_distance=None, checkpoint_chunks=False):
        super().__init__()
        if channels % heads or (channels // heads) % 4:
            raise ValueError("channels/heads must be an integer multiple of four")
        if not 0 < query_ratio <= 1 or patch_chunk_size < 1:
            raise ValueError("query_ratio must be in (0,1], patch_chunk_size positive")
        if position_encoding not in ("none", "quaternion"):
            raise ValueError("position_encoding must be none or quaternion")
        if max_distance is not None and max_distance <= 0:
            raise ValueError("max_distance must be positive or None")
        self.channels, self.heads = channels, heads
        self.head_dim, self.query_ratio = channels // heads, query_ratio
        self.dropout, self.patch_chunk_size = dropout, patch_chunk_size
        self.max_distance, self.checkpoint_chunks = max_distance, checkpoint_chunks
        self.norm = nn.LayerNorm(channels)
        # Separate projections avoid computing queries for unselected points.
        self.query = nn.Linear(channels, channels)
        self.key_value = nn.Linear(channels, channels * 2)
        self.output = nn.Linear(channels, channels)
        self.rope = QuatRoPE(heads, self.head_dim) if position_encoding == "quaternion" else None
        self.scale = nn.Parameter(torch.full((channels,), .1))

    def _chunk(self, normalized, coord, members, key_valid, query, query_valid):
        b, p = members.shape
        m = query.shape[1]
        q = self.query(normalized[query.clamp_min(0)]).reshape(b, m, self.heads, self.head_dim).transpose(1, 2)
        kv = self.key_value(normalized[members.clamp_min(0)]).reshape(b, p, 2, self.heads, self.head_dim)
        k, v = kv.unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        qcoord, kcoord = coord[query.clamp_min(0)], coord[members.clamp_min(0)]
        # Shared patch origin is valid because QuatRoPE has linear phases and
        # a fixed axis for each head/group. It reduces large-phase roundoff.
        origin = kcoord[:, :1]
        if self.rope is not None:
            q, k = self.rope(q, qcoord - origin), self.rope(k, kcoord - origin)
        allowed = key_valid[:, None, :].expand(-1, m, -1)
        if self.max_distance is not None:
            # coord is already normalized by stage scale; the caller scales
            # the physical distance guard by the same amount.
            distance2 = (qcoord[:, :, None] - kcoord[:, None]).square().sum(-1)
            allowed = allowed & (distance2 <= self.max_distance**2)
        # Valid queries include themselves. Padded queries get a safe key and
        # are discarded before scattering, so no row is entirely masked.
        fallback = torch.zeros_like(allowed)
        fallback[:, :, 0] = True
        allowed = torch.where(query_valid[:, :, None], allowed, fallback)
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None],
            dropout_p=self.dropout if self.training else 0.,
        )
        result = result.transpose(1, 2).reshape(b, m, self.channels)
        return self.output(result)

    def forward(self, feature, coord, patches):
        query, query_valid = select_queries(patches, self.query_ratio, self.training)
        normalized = self.norm(feature)
        outputs, targets = [], []
        for start in range(0, len(query), self.patch_chunk_size):
            end = start + self.patch_chunk_size
            args = (normalized, coord, patches.members[start:end], patches.valid[start:end],
                    query[start:end], query_valid[start:end])
            if self.checkpoint_chunks and self.training:
                update = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                update = self._chunk(*args)
            outputs.append(update[query_valid[start:end]])
            targets.append(query[start:end][query_valid[start:end]])
        selected = torch.cat(targets)
        updated = feature[selected] + self.scale * torch.cat(outputs)
        return feature.index_copy(0, selected, updated.to(feature.dtype)), selected
