# Packed Hybrid Cell Query v1

Sparse CNN + random point queries per grid cell + packed cross-attention across
27 neighboring cells. No quaternion encoding or serialization in this version.

## Two changes

1. A shared point-position MLP is evaluated ONCE on each real point per block,
   relative to its OWN query-cell origin. Its output is reused for query/key
   gathers. A learned 27-entry embedding supplies source-cell offset to keys.
   This changes the previous nonlinear window-relative encoding, so accuracy
   equivalence with the old hybrid is NOT claimed. Weights are shared across
   windows, but distinct blocks retain their own positional MLP parameters.
2. Occupied neighboring-cell edges and valid key slots are compacted. FlashAttention
   receives packed Q/K/V plus int32 cumulative sequence lengths. No masked empty
   neighbors or padded query/key slots enter the attention kernel. Small padded
   integer sampling tables still exist; sparse CNN and metadata also cost time.

Q is projected only for selected queries, K/V only for sampled source keys.
Source projections and point-position outputs are reused across neighborhoods.
Real keys still occur in multiple windows; overlap is inherent to this design.
The cell-offset embedding is added AFTER packing, not computed as a point MLP.

## Install

Requires your existing CUDA-compatible spconv and a PyTorch/CUDA-compatible
`flash-attn` 2.x installation exposing `flash_attn_varlen_func`.
Official API: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py
The A100 is the intended target; CUDA training/inference requires FP16 or BF16
autocast. Do not disable AMP. Default head dimension is 16 at every stage.

Copy this folder to `pointcept/models/packed_hybrid_cell_query/` and add:

```python
from .packed_hybrid_cell_query import adapter
```

to `pointcept/models/__init__.py`. Copy config.py into your DALES configs and use
your existing Pointcept launcher. Default registry type is PackedHybridCellQuery-v1.
Transforms have PHCQ prefixes and can coexist with earlier versions.
Fresh run required: geometry encoding adds parameters and changes semantics.
Batch size is 4, matching your stated A100 setup; point cap remains 65,536.
Losses, channels, depths, sparse CNN, pooling and decoder match the prior hybrid.
Refresh Pointcept's experiment/code snapshot before launching.

The packed GPU backend is explicit and never silently falls back to Python
loops or cuDNN. Missing flash-attn or non-AMP CUDA input produces an actionable
error. To run CPU correctness checks, set BOTH sparse_backend='reference' and
attention_backend='reference'. Reference attention loops over cells and is not
for GPU training or timing. PyTorch 2.3+ is the tested API baseline.

## Checks

From the directory containing this folder:

```text
python -m unittest discover -s packed_hybrid_cell_query -t . -v
```

Tests compare packed versus padded attention and gradients, verify actual key
counts after compaction, single positional-MLP invocation, sampling coverage,
batch isolation, chunk equivalence, input/parameter gradients, and singleton
clouds. Optional CUDA tests cover FlashAttention agreement and spconv AMP.
Those CUDA tests were skipped locally because the required flash-attn/spconv
packages were unavailable; target GPU speed, memory and accuracy are unverified.

## Remaining costs

Per-block point-position MLP still processes every point once. Sparse convolution,
all-point FFNs/pooling, stage grid construction and decoding remain. Packing uses
dynamic compaction (nonzero/boolean indexing), which can synchronize CUDA. Query
prefix offsets transfer to CPU once per block to define chunk slices; there are
no per-chunk max().item() calls. Chunking caps transient gather allocations but
does not discard backward activations. Lower chunk_size for memory, benchmark
the runtime tradeoff. This implementation is not synchronization-free.

Attention keys remain capped at keys_per_cell per occupied neighbor, at most
108 per cell window by default. Empty slots are eliminated; valid overlap is
retained. Query ratios are capped by max_queries. Random training and deterministic
midpoint evaluation are input-index-stratified and input-order dependent.

Use equal batches/point counts to compare old and packed versions. Warm up before
timing, synchronize CUDA around measurements, and report peak allocated memory.
Run a short overfit/validation experiment because positional encoding changed.
