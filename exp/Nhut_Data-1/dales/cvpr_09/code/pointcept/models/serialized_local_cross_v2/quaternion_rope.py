"""Fixed-axis quaternion rotary embedding with linear Cartesian phases.

Scalar-first Hamilton convention. Every head/group has an axis shared across
points, so L(r_i)^T L(r_j) = L(r_i^-1 r_j) depends on displacement.
This is a latent feature rotation, not physical SO(3) equivariance.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def hamilton(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw - ax*bx - ay*by - az*bz,
                        aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw), dim=-1)


class QuaternionRotary(nn.Module):
    def __init__(self, heads, head_dim, min_frequency=.25, max_frequency=2.):
        super().__init__()
        if heads < 1 or head_dim < 4 or head_dim % 4:
            raise ValueError("Quaternion rotary needs positive heads and head_dim divisible by 4")
        if not (math.isfinite(min_frequency) and math.isfinite(max_frequency)
                and 0 < min_frequency <= max_frequency):
            raise ValueError("Frequencies must be finite, positive and ordered")
        self.heads, self.head_dim = heads, head_dim
        groups = head_dim // 4
        # Deterministic spatial directions; no random-number consumption, so
        # enabling rotary does not change the initialization of other modules.
        index = torch.arange(heads * groups, dtype=torch.float32)
        z = 1. - 2. * (index + .5) / len(index)
        angle = index * (math.pi * (3. - math.sqrt(5.)))
        radius = (1. - z.square()).clamp_min(0).sqrt()
        direction = torch.stack((radius * angle.cos(), radius * angle.sin(), z), -1)
        frequency = torch.logspace(math.log10(min_frequency), math.log10(max_frequency), groups)
        omega = direction.reshape(heads, groups, 3) * frequency[None, :, None]
        basis = F.normalize(torch.tensor([[1., 0, 0], [0., 1, 0], [0., 0, 1], [1., 1, 1]]), dim=-1)
        axis = basis[torch.arange(heads * groups) % len(basis)].reshape(heads, groups, 3)
        self.register_buffer("omega", omega)
        self.register_buffer("axis", axis)

    def rotors(self, position):
        """position [N,3] in stage-normalized, shared-origin coordinates."""
        with torch.autocast(device_type=position.device.type, enabled=False):
            phase = torch.einsum("nc,hgc->nhg", position.float(), self.omega.float()) * .5
            return torch.cat((phase.cos()[..., None],
                              phase.sin()[..., None] * self.axis.float()[None]), -1)

    def apply_rotor(self, feature, rotor):
        """feature [N,H,D]; rotor [N,H,D/4,4]. Left multiplication."""
        dtype = feature.dtype
        with torch.autocast(device_type=feature.device.type, enabled=False):
            return hamilton(rotor.float(), feature.float().unflatten(-1, (-1, 4))).flatten(-2).to(dtype)

    def forward(self, feature, position):
        return self.apply_rotor(feature, self.rotors(position))
