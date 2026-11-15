# Cell Query Transformer v1

Initial convolution-free experiment for raw-point outdoor semantic segmentation.
No quaternion operations, spconv, or FlashAttention package dependency; attention
uses PyTorch scaled_dot_product_attention (PyTorch 2.0+).

## Block and hierarchy

1. Group current points/tokens into occupied Cartesian cells.
2. Select min(ceil(ratio * occupancy), max_queries) queries per cell, at least one.
   Training samples one point uniformly from each disjoint index stratum;
   evaluation takes deterministic stratum midpoints. This is stratified random
   sampling, not uniform subset sampling. Evaluation depends on input ordering.
3. Selected queries attend to all members of their own cell. Gathered offsets
   relative to the cell centroid, divided by cell size, feed a Cartesian MLP.
4. Scatter enhanced features back; apply a cheap FFN to all points.
5. Project then mean/max pool ALL cell members and fuse into one cell token.
6. Morton serialize occupied cell tokens and apply full attention within bounded
   patches. Change axis order each block; no explicit patch overlap in v1.
7. Project tokens into the next stage. Save pre-pool point skips and inverses.
8. Decode every stage with gated contextual lifting and pointwise refinement.

Four default cell sizes are 0.4, 0.8, 1.6, 3.2 metres. The last stage ALSO pools
before inter-cell attention; there are four inverse mappings in the decoder.
Coordinates are mass-weighted centroids, features use unweighted token means
and maxima. Mass is only used to preserve geometric centroid meaning.
All original input points receive output features in original order.

## Pointcept integration

Copy this folder into `pointcept/models/cell_query_transformer/`.
Add this line to `pointcept/models/__init__.py`:

```python
from .cell_query_transformer import adapter
```

The adapter registers CellQueryTransformer-v1 and CQT-prefixed transforms.
The package itself does not import Pointcept, allowing standalone CPU checks.
Copy config.py to `configs/dales/semseg-cell-query-transformer-v1.py` and run
your existing Pointcept training command with that config.
Use a fresh run; this architecture is not checkpoint-compatible with SQQ/PTv3.

The config retains the supplied SQQ dataset, losses, optimizer, batch size,
raw-point crops, strength preparation, and single-scale test protocol. It is an
initial experiment, not a matched PTv3 benchmark. Adjust batch size for hardware.
Dataset loader must supply coord, segment, and strength. Backbone input is
coord (N,3), feat (N,4), plus offset or sorted batch; output is (N,64).
DefaultSegmentorV2 supplies the final 8-class classifier.

## Checks

From the directory containing this folder:

```text
python -m unittest cell_query_transformer.test_model -v
```

Tests cover backward gradients, exact query quotas and uniqueness, batch-safe
patches, evaluation repeatability, independent-cloud equivalence, singleton and
dense-cell inputs. CUDA/AMP and Pointcept training require validation in the
target environment; CPU checks do not establish segmentation performance.

## Cost and limitations

Attention score work is O(sum_c m_c*n_c + C*P), omitting feature dimensions.
max_queries caps query count even with ratio=1; for full-query ablations set
max_queries above the maximum cell occupancy. All-point projections, FFNs,
pooling, sorting and geometry MLPs remain. Occupancy-bucketed local padding
limits padding waste, but dense cells still raise memory and runtime. No key
subsampling is performed. Python bucket/chunk loops may limit throughput.
Cell occupancies, GPU peak memory and measured latency should be reported.

One token per cell can merge mixed classes. Serialization approximates spatial
neighborhoods and does not guarantee all geometric neighbors exchange messages.
Absolute input coordinate channels and axis-aligned grids mean no exact
translation/rotation invariance is claimed. Coordinates are spatial geometry;
raw feat channels keep the preprocessing chosen by the dataset config.
Test fragments include halo predictions in standard index-based voting, as in
the supplied SQQ transforms. Strength normalization is unchanged (scale=1).
