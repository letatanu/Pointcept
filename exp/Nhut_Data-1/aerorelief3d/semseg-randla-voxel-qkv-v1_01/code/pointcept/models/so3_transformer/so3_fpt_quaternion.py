"""Centroid-aware scalar/vector voxel transformer with grouped quaternion Q/K.

SO(3) guarantees apply to continuous operations at fixed voxel memberships and
neighborhoods. Axis-aligned voxelization itself is rotation dependent.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .so3_quaternion_dual_attention import (
    MODELS,
    VLinear,
    VNRMSNorm,
    segment_mean,
    counts,
)
from .so3_voxel_quaternion import voxel_partition, voxel_graph
from .so3_voxel_quaternion_lite import (
    VoxelQuaternionLite,
    VoxelQuaternionLiteSegmentor,
    recompute,
)


class QuaternionVoxelBlock(nn.Module):
    """Each consecutive four Q/K components are [real, x, y, z].

    Query chunks own complete neighborhoods: softmax and message intermediates
    are recomputed in backward, rather than retained for all scene edges.
    """

    def __init__(self, channels, vectors, qk_channels, query_chunk, checkpointing):
        super().__init__()
        if qk_channels < 4 or qk_channels % 4:
            raise ValueError("qk_channels must be a positive multiple of four")
        self.nq = qk_channels // 4
        self.query_chunk, self.checkpointing = query_chunk, checkpointing
        self.norm = nn.LayerNorm(channels)
        self.vnorm = VNRMSNorm(vectors)
        self.real_qk = nn.Linear(channels, 2 * self.nq)
        self.vector_q, self.vector_k = VLinear(vectors, self.nq), VLinear(
            vectors, self.nq
        )
        self.value = nn.Linear(channels, channels)
        self.vector_value = VLinear(vectors, vectors)
        self.score = nn.Sequential(
            nn.Linear(5 * self.nq + 1, 16), nn.SiLU(), nn.Linear(16, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(5 * self.nq + 1, 16), nn.SiLU(), nn.Linear(16, vectors), nn.Tanh()
        )
        self.out, self.vout = nn.Linear(channels, channels), VLinear(vectors, vectors)
        self.readout = nn.Linear(vectors, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def project_qk(self, x, v):
        a, b = self.real_qk(self.norm(x)).chunk(2, -1)
        vn = self.vnorm(v)
        return (
            torch.cat([a[..., None], self.vector_q(vn)], -1),
            torch.cat([b[..., None], self.vector_k(vn)], -1),
        )

    @staticmethod
    def invariants(q, k, r):
        a, u = q[..., 0].float(), q[..., 1:].float()
        b, w = k[..., 0].float(), k[..., 1:].float()
        real = a * b + (u * w).sum(-1)
        return torch.cat(
            [
                real,
                (u * r[:, None]).sum(-1),
                (w * r[:, None]).sum(-1),
                (torch.cross(u, w, dim=-1) * r[:, None]).sum(-1),
                (b[..., None] * u - a[..., None] * w - torch.cross(u, w, dim=-1))
                .square()
                .sum(-1),
                r.square().sum(-1, keepdim=True),
            ],
            -1,
        )

    def _chunk(self, q, k, sv, vv, coord, dst, src, a, b, radius):
        local = dst - a
        r = (coord[src] - coord[dst]) / radius
        inv = self.invariants(q[dst], k[src], r).to(sv.dtype)
        logits = self.score(inv).float().squeeze(-1)
        maximum = logits.new_full((b - a,), -torch.inf).scatter_reduce(
            0, local, logits, reduce="amax"
        )
        exp = (logits - maximum[local]).exp()
        denom = exp.new_zeros(b - a).index_add(0, local, exp)
        weight = exp / denom[local].clamp_min(1e-12)
        scalar = sv.new_zeros(b - a, sv.shape[-1], dtype=torch.float32).index_add(
            0, local, weight[:, None] * sv[src].float()
        )
        vector = vv.new_zeros(b - a, vv.shape[1], 3, dtype=torch.float32).index_add(
            0,
            local,
            weight[:, None, None]
            * (vv[src].float() + self.gate(inv).float()[..., None] * r[:, None]),
        )
        return scalar.to(sv.dtype), vector.to(vv.dtype)

    def forward(self, x, v, coord, graph):
        q, k = self.project_qk(x, v)
        sv, vv = self.value(self.norm(x)), self.vector_value(self.vnorm(v))
        ends = list(range(0, len(x), self.query_chunk)) + [len(x)]
        boundaries = torch.searchsorted(
            graph.dst, torch.tensor(ends, device=x.device)
        ).tolist()
        scalars, vectors = [], []
        for j, (a, b) in enumerate(zip(ends[:-1], ends[1:])):
            s, t = recompute(
                self._chunk,
                q,
                k,
                sv,
                vv,
                coord,
                graph.dst[boundaries[j] : boundaries[j + 1]],
                graph.src[boundaries[j] : boundaries[j + 1]],
                a,
                b,
                graph.radius,
                enabled=self.training and self.checkpointing,
            )
            scalars.append(s)
            vectors.append(t)
        v = v + 0.1 * self.vout(torch.cat(vectors))
        x = (
            x
            + self.out(torch.cat(scalars))
            + self.readout(v.float().square().sum(-1).to(x.dtype))
        )
        return x + self.ffn(self.ffn_norm(x)), v


@MODELS.register_module("FPT-SO3-Quaternion-v1")
class FPTQuaternion(VoxelQuaternionLite):
    def __init__(self, qk_channels=16, **kwargs):
        # Reuse validation, scalar selection and scalar skip projections only.
        kwargs["quaternion_stages"] = ()
        super().__init__(**kwargs)
        vectors = kwargs.get("vector_channels", 4)
        self.vector_channels = vectors
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        QuaternionVoxelBlock(
                            c,
                            vectors,
                            qk_channels,
                            self.query_chunk,
                            self.checkpoint_blocks,
                        )
                        for _ in range(d)
                    ]
                )
                for c, d in zip(self.channels, kwargs.get("depths", (1, 1, 2, 2)))
            ]
        )
        ns = max(1, len(self.indices))
        self.stem = nn.Sequential(
            nn.Linear(ns + 1, self.channels[0]),
            nn.SiLU(),
            nn.Linear(self.channels[0], self.channels[0]),
        )
        self.seed_gate = nn.Sequential(
            nn.Linear(ns + 1, 16), nn.SiLU(), nn.Linear(16, vectors), nn.Tanh()
        )
        self.vector_down = nn.ModuleList([VLinear(vectors, vectors) for _ in self.down])
        self.vector_up = nn.ModuleList([VLinear(2 * vectors, vectors) for _ in self.up])

    def _encode(self, feat, coord, centroid, parent, a, b):
        r = (coord[a:b] - centroid[parent[a:b]]) / self.voxel_sizes[0]
        z = torch.cat(
            [self.scalars(feat[a:b]), r.square().sum(-1, keepdim=True).to(feat.dtype)],
            -1,
        )
        return self.stem(z), self.seed_gate(z)[..., None] * r[:, None]

    def forward_voxels(self, data):
        coord, feat, offset = (
            data["coord"].detach().float(),
            data["feat"],
            data["offset"],
        )
        if (
            coord.ndim != 2
            or coord.shape[1] != 3
            or not len(coord)
            or not torch.isfinite(coord).all()
        ):
            raise ValueError("Expected finite nonempty coordinates [N,3]")
        if (
            feat.shape != (len(coord), self.in_channels)
            or not feat.is_floating_point()
            or not torch.isfinite(feat).all()
        ):
            raise ValueError("Invalid features")
        if (
            offset.ndim != 1
            or offset.dtype not in (torch.int32, torch.int64)
            or not len(offset)
            or int(offset[-1]) != len(coord)
            or (counts(offset) <= 0).any()
        ):
            raise ValueError("Invalid offsets")
        if feat.device != coord.device or offset.device != coord.device:
            raise ValueError("Input devices must match")
        inverse, _, c, off = voxel_partition(coord, offset, self.voxel_sizes[0])
        fine_centroid = c

        # Pool inside each checkpoint: no persistent N x hidden-width encoding.
        def encode_pool(feat, coord, c, parent, a, b):
            s, v = self._encode(feat, coord, c, parent, a, b)
            ids, local = parent[a:b].unique(sorted=True, return_inverse=True)
            sc = s.new_zeros(len(ids), s.shape[1], dtype=torch.float32).index_add(
                0, local, s.float()
            )
            vc = v.new_zeros(len(ids), v.shape[1], 3, dtype=torch.float32).index_add(
                0, local, v.float()
            )
            return ids, sc, vc

        x = feat.new_zeros(len(c), self.channels[0], dtype=torch.float32)
        v = feat.new_zeros(len(c), self.vector_channels, 3, dtype=torch.float32)
        for a in range(0, len(coord), self.query_chunk):
            ids, s, t = recompute(
                encode_pool,
                feat,
                coord,
                c,
                inverse,
                a,
                min(a + self.query_chunk, len(coord)),
                enabled=self.training and self.checkpoint_blocks,
            )
            x.index_add_(0, ids, s)
            v.index_add_(0, ids, t)
        occupancy = torch.bincount(inverse, minlength=len(c)).clamp_min(1).float()
        x = x / occupancy[:, None]
        v = v / occupancy[:, None, None]
        skips, maps, stats = [], [], []
        for i, blocks in enumerate(self.stages):
            graph = voxel_graph(
                c, off, self.voxel_sizes[i], self.radii[i], chunk_size=self.query_chunk
            )
            stats.append(dict(voxels=len(c), edges=len(graph.dst), channels=x.shape[1]))
            for block in blocks:
                x, v = recompute(
                    block,
                    x,
                    v,
                    c,
                    graph,
                    enabled=self.training and self.checkpoint_blocks,
                )
            skips.append((x, v))
            if i < len(self.down):
                parent, _, c, off = voxel_partition(c, off, self.voxel_sizes[i + 1])
                maps.append(parent)
                x = self.down[i](segment_mean(x, parent, len(c)))
                v = self.vector_down[i](segment_mean(v, parent, len(c)))
        for i in reversed(range(len(maps))):

            def decode(x, v, s, t, parent, a, b, level):
                ids = parent[a:b]
                return (
                    F.gelu(self.up[level](torch.cat([s[a:b], x[ids]], -1))),
                    self.vector_up[level](torch.cat([t[a:b], v[ids]], 1)),
                )

            chunks = [
                recompute(
                    decode,
                    x,
                    v,
                    *skips[i],
                    maps[i],
                    a,
                    min(a + self.query_chunk, len(maps[i])),
                    i,
                    enabled=self.training and self.checkpoint_blocks
                )
                for a in range(0, len(maps[i]), self.query_chunk)
            ]
            x = torch.cat([s for s, t in chunks])
            v = torch.cat([t for s, t in chunks])
        self.last_stats = dict(points=len(coord), stages=stats)
        return self._linear_chunks(self.output, x), v, inverse, fine_centroid

    def forward(self, data, return_stages=False):
        raise RuntimeError("Use FPTQuaternionSegmentor for checkpointed point decoding")


@MODELS.register_module("FPTQuaternionSegmentor")
class FPTQuaternionSegmentor(VoxelQuaternionLiteSegmentor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        h = self.backbone.head_channels
        self.refine = nn.Sequential(
            nn.Linear(
                h
                + max(1, len(self.backbone.indices))
                + 1
                + 2 * self.backbone.vector_channels,
                h,
            ),
            nn.GELU(),
            nn.Linear(h, h),
        )

    def _point_logits(self, x, v, inverse, centroid, coord, feat, a, b):
        ids = inverse[a:b]
        decoded = x[ids]
        r = (coord[a:b].detach().float() - centroid[ids]) / self.backbone.voxel_sizes[0]
        vec = v[ids].float()
        z = torch.cat(
            [
                decoded,
                self.backbone.scalars(feat[a:b]).to(decoded.dtype),
                r.square().sum(-1, keepdim=True).to(decoded.dtype),
                (vec * r[:, None]).sum(-1).to(decoded.dtype),
                vec.square().sum(-1).to(decoded.dtype),
            ],
            -1,
        )
        return self.seg_head(decoded + self.refine(z))

    def forward(self, input_dict, return_point=False):
        from contextlib import nullcontext

        target = input_dict.get("segment")
        if self.training and target is None:
            raise ValueError("Training requires segment labels")
        if target is not None and (
            target.dtype != torch.long or target.shape != (len(input_dict["coord"]),)
        ):
            raise ValueError("segment must be long [N]")
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            x, v, inverse, centroid = self.backbone.forward_voxels(input_dict)
        logits = torch.cat(
            [
                recompute(
                    self._point_logits,
                    x,
                    v,
                    inverse,
                    centroid,
                    input_dict["coord"],
                    input_dict["feat"],
                    a,
                    min(a + self.point_chunk, len(inverse)),
                    enabled=self.training,
                )
                for a in range(0, len(inverse), self.point_chunk)
            ]
        )
        out = {} if self.training else dict(seg_logits=logits)
        if target is not None:
            out["loss"] = self._loss(logits, target)
        if return_point:
            from .so3_quaternion_dual_attention import Point

            point = Point(dict(input_dict))
            point.feat = logits
            out["point"] = point
        return out
