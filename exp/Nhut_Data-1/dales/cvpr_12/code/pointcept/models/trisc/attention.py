import torch
from torch import nn
from .axial_rope import AxialRotary
from .quaternion_rope import QuaternionRotary
from .layout import packed_table


ROTARY_MODES = ("none", "qk", "qkv", "axial_qk")


def varlen(q, k, v, cuq, cuk, maxq, maxk, backend):
    if backend == "flash_varlen":
        if not all(t.is_cuda for t in (q, k, v)):
            raise RuntimeError("flash_varlen requires CUDA; use reference for CPU correctness tests")
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError as exc:
            raise ImportError("Install a CUDA/PyTorch-compatible flash-attn 2.x package") from exc
        original = q.dtype
        dtype = original
        if dtype == torch.float32:
            with torch.cuda.device(q.device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("Attention requires FP16/BF16 or FP32 inputs")
        q, k, v = (t.to(dtype).contiguous() for t in (q, k, v))
        return flash_attn_varlen_func(q, k, v, cuq.contiguous(), cuk.contiguous(), maxq, maxk,
                                     dropout_p=0., causal=False).to(original)
    if backend != "reference":
        raise ValueError("Unknown attention backend")
    aq, ak = cuq.cpu().tolist(), cuk.cpu().tolist()
    result = []
    for i in range(len(aq) - 1):
        query, key, value = q[aq[i]:aq[i+1]], k[ak[i]:ak[i+1]], v[ak[i]:ak[i+1]]
        score = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * q.shape[-1] ** -.5
        result.append(torch.einsum("hqk,khd->qhd", score.softmax(-1), value.float()).to(q.dtype))
    return torch.cat(result)


class PackedAttention(nn.Module):
    def __init__(self, channels, heads, rotary, backend, chunk_size, min_frequency, max_frequency):
        super().__init__()
        if rotary not in ROTARY_MODES:
            raise ValueError(f"Rotary mode must be one of {ROTARY_MODES}")
        self.heads, self.dim = heads, channels // heads
        self.mode, self.backend, self.chunk_size = rotary, backend, chunk_size
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.output = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.full((channels,), .1))
        self.rotary = None
        if rotary == "axial_qk":
            self.rotary = AxialRotary(self.dim)
        elif rotary in ("qk", "qkv"):
            self.rotary = QuaternionRotary(heads, self.dim, min_frequency, max_frequency)

    def forward(self, feature, coord, query_table, key_table, maxq, maxk, spatial_scale):
        qids, update = self.query_updates(
            feature, coord, query_table, key_table, maxq, maxk, spatial_scale)
        return feature.index_copy(0, qids, (feature[qids] + update).to(feature.dtype))

    def _rotate(self, q, k, v, q_position, k_position):
        if self.mode == "axial_qk":
            return self.rotary(q, q_position), self.rotary(k, k_position), v, None
        rq = self.rotary.rotors(q_position)
        rk = self.rotary.rotors(k_position)
        q = self.rotary.apply_rotor(q, rq)
        k = self.rotary.apply_rotor(k, rk)
        if self.mode == "qkv":
            v = self.rotary.apply_rotor(v, rk)
        return q, k, v, rq

    def _restore_values(self, out, rq):
        inverse = torch.cat((rq[..., :1], -rq[..., 1:]), -1)
        return self.rotary.apply_rotor(out, inverse)

    def query_updates(self, feature, coord, query_table, key_table, maxq, maxk, spatial_scale):
        """Return updates only for the sampled queries."""
        qids, cuq = packed_table(query_table)
        kids, cuk = packed_table(key_table)
        # Normalize/project only tokens used by this attention operation.
        normalized_q = self.norm(feature[qids])
        normalized_k = normalized_q if query_table is key_table else self.norm(feature[kids])
        q = self.q(normalized_q).reshape(-1, self.heads, self.dim)
        kv = self.kv(normalized_k).reshape(-1, 2, self.heads, self.dim)
        k, v = kv.unbind(1)
        # Transfer prefix metadata once, not scalar maxima in the chunk loop.
        hq, hk = cuq.cpu().tolist(), cuk.cpu().tolist()
        outputs = []
        for begin in range(0, len(query_table), self.chunk_size):
            end = min(begin + self.chunk_size, len(query_table))
            a, b = hq[begin], hq[end]
            c, d = hk[begin], hk[end]
            qc, kc, vc = q[a:b], k[c:d], v[c:d]
            if self.rotary is not None:
                qc, kc, vc, rq = self._rotate(qc, kc, vc, coord[qids[a:b]] / spatial_scale,
                                               coord[kids[c:d]] / spatial_scale)
            out = varlen(qc, kc, vc, cuq[begin:end+1] - cuq[begin], cuk[begin:end+1] - cuk[begin],
                         maxq, maxk, self.backend)
            if self.mode == "qkv":
                out = self._restore_values(out, rq)
            outputs.append(out.flatten(1))
        # A neighbor-only batch can have no cross windows at a coarse stage.
        # Keep its projections in the graph for DDP, even with an empty update.
        joined = torch.cat(outputs) if outputs else q.flatten(1) + kv.sum()
        update = self.output(joined)
        return qids, self.scale.to(feature.dtype) * update

    def sampled_window_updates(self, feature, coord, query_ids, key_ids, windows, spatial_scale):
        """Project each sampled token once, then gather it into its windows."""
        q = self.q(self.norm(feature[query_ids])).reshape(-1, self.heads, self.dim)
        kv = self.kv(self.norm(feature[key_ids])).reshape(-1, 2, self.heads, self.dim)
        k, v = kv.unbind(1)
        if self.rotary is not None:
            q, k, v, rq = self._rotate(q, k, v, coord[query_ids] / spatial_scale,
                                      coord[key_ids] / spatial_scale)

        # Grid windows are already chunked. FlashAttention uses device-side
        # offsets directly, with no per-chunk copy of lengths back to the CPU.
        used_queries, outputs = [], []
        for query_table, key_table in windows:
            query, cuq = packed_table(query_table)
            key, cuk = packed_table(key_table)
            out = varlen(q[query], k[key], v[key], cuq, cuk, query_table.shape[1],
                         key_table.shape[1], self.backend)
            if self.mode == "qkv":
                out = self._restore_values(out, rq[query])
            outputs.append(out.flatten(1))
            used_queries.append(query)

        # Preserve zero gradients for an entirely isolated grid.
        if not outputs:
            empty = q[:0].flatten(1) + kv[:0].sum()
            return query_ids[:0], self.scale.to(feature.dtype) * self.output(empty)
        return (query_ids[torch.cat(used_queries)],
                self.scale.to(feature.dtype) * self.output(torch.cat(outputs)))
