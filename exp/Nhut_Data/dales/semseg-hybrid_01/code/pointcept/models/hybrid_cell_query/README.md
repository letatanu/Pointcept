# Hybrid Cell Query v1

Fine-grid sparse CNN -> random queries per larger cell -> attention to points
sampled from 27 neighboring cells -> all-point FFN -> all-point pooling.
No intra-cell-only attention, serialized patches, or quaternion encoding.
The sparse CNN crosses query-cell boundaries; it is not reset inside each cell.

## Install in Pointcept

1. Copy this folder to `pointcept/models/hybrid_cell_query/`.
2. Add `from .hybrid_cell_query import adapter` to `pointcept/models/__init__.py`.
3. Copy config.py to `configs/dales/semseg-hybrid-cell-query-v1.py`.
4. Train using your existing Pointcept launcher with this config and a new output
   directory. Ensure any experiment/code snapshot contains these new files.

Requires PyTorch 2.3+ and a CUDA-compatible spconv installation for training.
Reference backend is slow CPU PyTorch sparse convolution for tests only.
The package can be imported standalone without importing Pointcept.

Config retains raw-point DALES transforms, CE+Lovasz, optimizer and scheduler
settings from the previous implementation, but sets batch_size=8 as a cautious
memory starting point instead of 42. Tune for your GPU. This changes steps per
epoch and optimization; do not compare batch seconds or convergence directly
with batch_size=42. Compare points/second, GPU memory, and matched update/data
budgets. No performance gain is established by CPU correctness tests.

## Architecture details

Default fine voxel sizes: (0.2, 0.4, 0.8, 1.6) metres.
Default query cell sizes: (0.4, 0.8, 1.6, 3.2) metres.
Each encoder block mean-aggregates features to fine voxels, applies a 3x3x3
submanifold sparse convolution, and gathers the residual context to all points.
Selected point queries attend to keys from their own and adjacent occupied cells.
One point can supply keys to multiple windows, but queries have unique owners.
Each cell independently samples at most 4 keys by default: at most 108 valid
keys per query-cell window, with padding for absent cells. This is capped spatial
sampling, not exact KNN and not all-point attention. A sampled query is not
guaranteed to be among the sampled keys; its residual retains its own features.

Query quota: min(ceil(ratio * cell occupancy), max_queries), at least one.
Training uses random samples from disjoint input-index strata; eval uses their
midpoints. Selection is unique per cell but input-order dependent. Quota caps
still apply when ratio=1. Query/key samples refresh each block during training.

Keys and values are projected ONCE per real input point, before padded gathers.
Cartesian position MLP is relative to the query-cell grid origin. Neighbor keys
include the center cell, so singleton clouds cannot produce all-masked rows.
cuDNN SDPA is excluded; Flash/efficient/math backends are allowed. Padding masks
can cause math fallback, which may be expensive on some PyTorch/GPU versions.

All-point mean/max pooling merges enhanced and remaining features. There are
three pools (after stages 0,1,2), not four. Final-stage attention acts before
decoding. Fine-point skips, gated coarse lifting, and pointwise decoder FFNs
preserve output count/order. Backbone output (N,64) goes to DefaultSegmentorV2's
8-class head. Decoder has no additional convolution/attention in this baseline.
One token per pooled cell is still a bottleneck; this version moves neighboring
context BEFORE pooling, rather than introducing multiple representatives.

Coordinates use mass-weighted centroids. Feature means are unweighted token
means. Raw features keep the supplied coordinate/strength preprocessing.
No exact translation/rotation invariance is claimed.

## Efficiency and checks

Spatial layouts, cell sort, neighbor lookup and spconv indice cache are reused
across blocks of each stage. Fixed query/key widths avoid occupancy buckets and
per-chunk GPU maxima. Chunking (default 512 cells) controls gather size. There
are still stage-level synchronizations in grid construction and Python chunk
loops; this is not a custom fused attention implementation.

Attention score work is O(sum_c m_c * k_c), k_c <= 27 * keys_per_cell.
Include sparse CNN, all-point projections/FFNs, grouping and decoder in runtime.
Large batches can still consume substantial memory. Reduce chunk_size or batch
size on OOM; reducing batch size also changes the training budget.

From the directory containing the package:

```text
python -m unittest hybrid_cell_query.test_model -v
```

Tests cover exact neighboring-cell lookup against brute force, batch separation,
cross-cell edges, sampling quotas and uniqueness, parameter/input gradients,
chunk equivalence, deterministic evaluation and singleton inputs. An optional
CUDA/spconv/FP16 test is skipped if unavailable. Validate on your GPU before a
full run. Run a fixed-crop overfit check and inspect per-class support in center
crop validation; full-scene test metrics are needed for comparison to PTv3.
