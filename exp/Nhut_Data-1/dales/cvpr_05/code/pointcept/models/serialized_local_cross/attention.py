import torch
from torch import nn
from .quaternion_rope import QuaternionRotary
from .layout import packed_table


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
        if rotary not in ("none", "qk", "qkv"):
            raise ValueError("Rotary mode must be none, qk, or qkv")
        self.heads, self.dim = heads, channels // heads
        self.mode, self.backend, self.chunk_size = rotary, backend, chunk_size
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.output = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.full((channels,), .1))
        self.rotary = QuaternionRotary(heads, self.dim, min_frequency, max_frequency) if rotary != "none" else None

    def forward(self, feature, coord, query_table, key_table, maxq, maxk, spatial_scale):
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
                rq = self.rotary.rotors(coord[qids[a:b]] / spatial_scale)
                rk = self.rotary.rotors(coord[kids[c:d]] / spatial_scale)
                qc = self.rotary.apply_rotor(qc, rq)
                kc = self.rotary.apply_rotor(kc, rk)
                if self.mode == "qkv":
                    vc = self.rotary.apply_rotor(vc, rk)
            out = varlen(qc, kc, vc, cuq[begin:end+1] - cuq[begin], cuk[begin:end+1] - cuk[begin],
                         maxq, maxk, self.backend)
            if self.mode == "qkv":
                inverse = torch.cat((rq[..., :1], -rq[..., 1:]), -1)
                out = self.rotary.apply_rotor(out, inverse)
            outputs.append(out.flatten(1))
        update = self.output(torch.cat(outputs))
        return feature.index_copy(0, qids, (feature[qids] + self.scale.to(feature.dtype) * update).to(feature.dtype))
