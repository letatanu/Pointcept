"""Packed cross-attention: real queries/keys only, no empty-cell attention slots."""
import torch
from torch import nn
from .sampling import sample_members
from .quaternion_rope import QuaternionRotary


def _prepare_flash_qkv(q, k, v, bf16_supported):
    """Keep AMP precision; handle FP32 callers such as Pointcept evaluation."""
    if any(t.dtype not in (torch.float16, torch.bfloat16, torch.float32) for t in (q, k, v)):
        raise TypeError("Flash attention accepts FP16, BF16 or FP32 inputs")
    dtype = q.dtype
    if dtype == torch.float32:
        dtype = torch.bfloat16 if bf16_supported else torch.float16
    return tuple(t.to(dtype=dtype).contiguous() for t in (q, k, v))


def packed_attention(q, k, v, cu_q, cu_k, max_q, max_k, backend):
    if backend == "flash_varlen":
        if not all(t.is_cuda for t in (q, k, v)):
            raise RuntimeError("flash_varlen requires CUDA tensors. Use attention_backend='reference' for CPU tests only.")
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError as exc:
            raise ImportError("Install a CUDA/PyTorch-compatible flash-attn 2.x package for packed training.") from exc
        output_dtype = q.dtype
        with torch.cuda.device(q.device):
            bf16_supported = torch.cuda.is_bf16_supported() if q.dtype == torch.float32 else False
        q, k, v = _prepare_flash_qkv(q, k, v, bf16_supported)
        output = flash_attn_varlen_func(q, k, v,
            cu_q.contiguous(), cu_k.contiguous(), max_q, max_k, dropout_p=0., causal=False)
        return output.to(output_dtype)
    if backend != "reference":
        raise ValueError("attention_backend must be flash_varlen or reference")
    # Deliberately slow exact reference, never an automatic GPU fallback.
    q_bounds, k_bounds = cu_q.cpu().tolist(), cu_k.cpu().tolist()
    outputs = []
    for i in range(len(q_bounds) - 1):
        a, b = q[q_bounds[i]:q_bounds[i + 1]], k[k_bounds[i]:k_bounds[i + 1]]
        value = v[k_bounds[i]:k_bounds[i + 1]]
        score = torch.einsum("qhd,khd->hqk", a.float(), b.float()) * (q.shape[-1] ** -.5)
        out = torch.einsum("hqk,khd->qhd", score.softmax(-1), value.float())
        outputs.append(out.to(q.dtype))
    return torch.cat(outputs)


class PackedNeighborAttention(nn.Module):
    def __init__(self, channels, heads, max_queries, keys_per_cell, query_ratio,
                 chunk_size, attention_backend="flash_varlen", rotary_encoding="quaternion",
                 rotary_min_frequency=.25, rotary_max_frequency=2.):
        super().__init__()
        if attention_backend not in ("flash_varlen", "reference"):
            raise ValueError("Unknown attention backend")
        if attention_backend == "flash_varlen" and (channels // heads) % 8:
            raise ValueError("This FlashAttention configuration requires head_dim divisible by 8")
        self.heads, self.dim = heads, channels // heads
        self.max_queries, self.keys_per_cell = max_queries, keys_per_cell
        self.query_ratio, self.chunk_size = query_ratio, chunk_size
        self.attention_backend = attention_backend
        if rotary_encoding not in ("none", "quaternion"):
            raise ValueError("rotary_encoding must be none or quaternion")
        self.rotary = QuaternionRotary(heads, self.dim, rotary_min_frequency, rotary_max_frequency) if rotary_encoding == "quaternion" else None
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.position = nn.Sequential(nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels))
        # Offset order matches itertools.product((-1,0,1), repeat=3).
        self.cell_offset = nn.Embedding(27, channels)
        nn.init.normal_(self.cell_offset.weight, std=.02)
        self.output = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.full((channels,), .1))

    def forward(self, feature, coord, layout, order, starts, neighbors, cell_size, return_debug=False):
        query, qvalid = sample_members(layout, order, starts, self.max_queries, self.query_ratio, self.training)
        key, kvalid = sample_members(layout, order, starts, self.keys_per_cell, 1., self.training)
        origin = layout.indices[layout.inverse, 1:].to(coord.dtype) * cell_size
        # ONE batched position-MLP evaluation per real point per block.
        point_position = self.position(((coord - origin) / cell_size).to(feature.dtype))
        normalized = self.norm(feature)
        target = query[qvalid]
        # Q only for selected queries; K/V only for actual sampled source keys.
        q_bank = self.q(normalized[target]) + point_position[target]
        key_target = key[kvalid]
        k_bank, v_bank = self.kv(normalized[key_target]).chunk(2, -1)
        k_bank = k_bank + point_position[key_target]
        # Cache Q rotations and sampled-source K rotors before window gathers.
        # coord is centered once per cloud by the backbone, not per-cell offsets.
        if self.rotary is not None:
            q_bank = self.rotary(q_bank.reshape(-1, self.heads, self.dim), coord[target] / cell_size).flatten(1)
            key_rotors = self.rotary.rotors(coord[key_target] / cell_size)
        key_map = kvalid.flatten().long().cumsum(0).reshape_as(key) - 1
        query_count = qvalid.sum(-1)
        # One metadata transfer per block; no per-chunk CUDA scalar reads.
        q_offsets = torch.cat((query_count.new_zeros(1), query_count.cumsum(0)))
        host_offsets = q_offsets.cpu().tolist()
        outputs, packed_counts = [], []
        for begin in range(0, len(query), self.chunk_size):
            end = min(begin + self.chunk_size, len(query))
            adjacent = neighbors[begin:end]
            # Compact occupied cell edges first; do not gather missing cells.
            row, offset_id = (adjacent >= 0).nonzero(as_tuple=True)
            source = adjacent[row, offset_id]
            edge_valid = kvalid[source]
            lengths = edge_valid.sum(-1)
            bank_ids = key_map[source][edge_valid]
            relative_ids = offset_id[:, None].expand(-1, self.keys_per_cell)[edge_valid]
            k = k_bank[bank_ids] + self.cell_offset(relative_ids).to(k_bank.dtype)
            if self.rotary is not None:
                # Rotate the complete key, including its cell-offset embedding.
                # Trigonometric terms are cached per sampled source point.
                k = self.rotary.apply_rotor(k.reshape(-1, self.heads, self.dim), key_rotors[bank_ids]).flatten(1)
            v = v_bank[bank_ids]
            counts = query_count.new_zeros(end - begin).index_add_(0, row, lengths)
            # Row-major compaction keeps each query cell's keys contiguous.
            cu_k = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
            local_q = query_count[begin:end]
            cu_q = torch.cat((local_q.new_zeros(1), local_q.cumsum(0))).to(torch.int32)
            q = q_bank[host_offsets[begin]:host_offsets[end]]
            out = packed_attention(q.reshape(-1, self.heads, self.dim),
                k.reshape(-1, self.heads, self.dim), v.reshape(-1, self.heads, self.dim),
                cu_q, cu_k, self.max_queries, 27 * self.keys_per_cell, self.attention_backend)
            outputs.append(out.reshape(-1, self.heads * self.dim))
            if return_debug:
                packed_counts.append(counts.detach())
        update = self.output(torch.cat(outputs))
        result = feature.index_copy(0, target, (feature[target] + self.scale * update).to(feature.dtype))
        record = None
        if return_debug:
            record = dict(selected=target.detach(), query=query.detach(), query_valid=qvalid.detach(),
                keys=key.detach(), key_valid=kvalid.detach(), packed_key_counts=torch.cat(packed_counts))
        return result, record
