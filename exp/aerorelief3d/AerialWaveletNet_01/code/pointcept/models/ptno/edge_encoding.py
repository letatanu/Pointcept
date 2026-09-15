"""Explicit edge quaternion encoding, used INSIDE vector attention.
Derived from the earlier neighbor revision. No neighbor search or Pointcept imports.
"""
import torch
from torch import nn

def apply_quat_rotation_to_features(feat, quat):
    """q*f*q* on scalar/vector groups. q must be unit length."""
    if feat.ndim != 2 or feat.shape[1] % 4 or quat.shape != (feat.shape[0], 4):
        raise ValueError("Expected feat=(N,4K), quat=(N,4)")
    f = feat.reshape(feat.shape[0], -1, 4)
    q = quat.to(feat.dtype)
    v = f[..., 1:]
    axis = q[:, None, 1:].expand_as(v)
    t = 2 * torch.cross(axis, v, dim=-1)
    rotated = v + q[:, None, :1] * t + torch.cross(axis, t, dim=-1)
    return torch.cat((f[..., :1], rotated), dim=-1).reshape_as(feat)


class QuaternionEdgeEncoding(nn.Module):
    """Pairwise x_i-x_j encoding with local RMS scale and true sandwich algebra.

    The learned projection is not an SO(3) intertwiner. No full equivariance claim.
    """
    def __init__(self, out_channels, num_freqs=8, distance_scale=1.0):
        super().__init__()
        if out_channels < 4 or out_channels % 4 or distance_scale <= 0:
            raise ValueError("Positive channels divisible by 4 and distance_scale required")
        self.distance_scale = distance_scale
        self.proj = nn.Sequential(nn.Linear(5+num_freqs,out_channels),
                                  nn.LayerNorm(out_channels), nn.GELU(),
                                  nn.Linear(out_channels,out_channels))
        self.register_buffer("freqs",2.0**torch.arange(num_freqs,dtype=torch.float32))

    @staticmethod
    def _build_quaternion(delta):
        raw = torch.cat((torch.ones_like(delta[..., :1]), delta), dim=-1)
        return raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)

    def encode_edges(self, coord, neighbor_index, query_index=None, batch=None):
        """Return (edge_features[M,K,C], valid[M,K]) for explicit neighbors.

        Indices refer to coord rows. -1 denotes padding; self edges are masked.
        Pass batch when supplying an external graph to reject cross-scene edges.
        Duplicate neighbor entries are caller error (internal kNN never adds them).
        """
        if neighbor_index.ndim != 2 or neighbor_index.dtype != torch.long:
            raise ValueError("neighbor_index must be an M,K int64 tensor")
        if query_index is None:
            query_index = torch.arange(len(coord), device=coord.device)
        if len(query_index) != len(neighbor_index):
            raise ValueError("query_index rows must match neighbor_index")
        if neighbor_index.numel() and bool(((neighbor_index < -1) | (neighbor_index >= len(coord))).any()):
            raise ValueError("Invalid neighbor index")
        valid = (neighbor_index >= 0) & (neighbor_index != query_index[:, None])
        safe = neighbor_index.clamp_min(0)
        if batch is not None and bool((valid & (batch[safe] != batch[query_index, None])).any()):
            raise ValueError("Neighbor edges must stay within their scene")
        dtype = self.proj[0].weight.dtype
        coords = coord.to(dtype)
        delta = coords[query_index, None, :] - coords[safe]  # x_i - x_j, not reverse
        delta = delta * valid[..., None]
        radius = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        degree = valid.sum(dim=1, keepdim=True).clamp_min(1)
        scale = (radius.square().sum(dim=1) / degree).clamp_min(1e-12).sqrt()
        quat = self._build_quaternion(delta / scale[:, None, :])
        radial = radius / scale[:, None, :]
        features = torch.cat((quat, torch.sin(radial * self.freqs),
                              torch.log1p(radius / self.distance_scale)), dim=-1)
        encoded = self.proj(features)
        c = encoded.shape[-1]
        rotated = apply_quat_rotation_to_features(encoded.reshape(-1,c), quat.reshape(-1,4))
        return rotated.reshape_as(encoded) * valid[..., None], valid

