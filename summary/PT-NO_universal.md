# PT-NO-Universal

## Overview
PT-NO-Universal is the updated Point Transformer V3 + Neural Operator architecture designed to reduce the extreme parameter cost of earlier NO-enhanced variants while preserving global spectral modeling. It replaces stage-specific NO branches with a single shared universal NO branch and aligns the downsampling / upsampling path with grid-based operators.

## Main idea
The core idea is to use exactly one shared `NOGlobalBranch` for the whole network. Instead of building separate Fourier Neural Operator branches for different feature widths, each encoder or decoder layer projects its local feature channels into a fixed universal latent width, sends them through the shared NO branch, and projects them back to the layer-specific width.

## Why this was introduced
Earlier NO-enhanced designs duplicated heavy FNO weights across stages, causing parameter counts to explode far beyond the PTv3 baseline. PT-NO-Universal addresses this by sharing one global spectral operator across all stages and keeping only lightweight per-layer adapters around it.

## Key Files

| File | Role |
|------|------|
|`pointcept/models/point_transformer_v3/NO_PTNet_v2.py` | PT-NO_universal (active)|
| `configs/s3dis/semseg-pt-v3-no-v2-1.py` | S3DIS config ro PT-NO_universal|
| `pointcept/models/point_transformer_v3/ptv3_no_lightweighthead_enhanced.py` | PTv3-NOEncoder-Enhanced |
| `configs/s3dis/semseg-pt-v3-no-v1.py` | S3DIS config for PTv3-NOEncoder-Enhanced |
| `configs/s3dis/semseg-pt-v3-no-base.py` | S3DIS config for PTv3-NO (U-Net + NO, reference) |
| `configs/s3dis/semseg-pt-v3-no-v0.py` | S3DIS config for PTv3-NOEncoder (reference) |
| `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py` | PTv3 base (Block, Embedding, SerializedPooling, SerializedUnpooling) |
| `configs/s3dis/semseg-pt-v3m1-0-base.py` | S3DIS config for PTv3 |
| `pointcept/engines/train.py` | Trainer / dataloader |

## Architecture changes

### 1. Universal shared NO branch
- Only one `NOGlobalBranch` is instantiated in the entire model.
- The branch operates at a fixed latent width, for example `universal_dim=64`.
- Every NO-enabled stage uses:
  - `down_proj`: project local stage features into universal width,
  - shared `NOGlobalBranch`,
  - `up_proj`: project universal output back to stage output width.

This means the expensive 3D Fourier weights exist only once.

### 2. Relative geometry encoding
The universal NO branch now receives both feature information and geometric information. Before rasterization to the dense grid, each point gets a relative position encoding derived from:
- centered relative coordinates `(dx, dy, dz)`,
- normalized radial distance from the centroid.

These geometry features are processed by a small MLP and fused into the universal NO features before the FNO block.

### 3. Grid-based hierarchy
The previous serialized pooling and unpooling path is replaced with grid-based operators inspired by Sonata:
- `GridPooling` groups points into voxel cells using integer grid coordinates and aggregates their features with a reduce operation such as `max`.
- `GridUnpooling` restores higher-resolution features using the saved pooling inverse mapping and skip connections.

This makes the multi-scale hierarchy more consistent with the dense-grid representation already used inside the NO branch.

### 4. AMP-safe spectral block
The FNO block runs FFT operations in float32 even when AMP is enabled. This avoids PyTorch complex-half issues during FFT and `einsum`, while the rest of the network can still benefit from mixed precision.

## Data flow

### Encoder / decoder NO path
For an NO-enabled stage, the flow is:

1. Take point features from the current stage.
2. Project them to the universal latent width.
3. Add relative geometry encoding.
4. Scatter features to a dense regular 3D grid.
5. Apply the shared FNO block.
6. Gather grid features back to points.
7. Project back to the stage-specific feature width.
8. Fuse with pooled or unpooled PTv3 features.

### Pooling path
`GridPooling`:
1. Computes voxel indices from `grid_coord`.
2. Clusters points that fall into the same voxel.
3. Aggregates projected features inside each voxel.
4. Produces a reduced point set with coarser coordinates.
5. Stores `pooling_inverse` and `pooling_parent` for decoder recovery.

### Unpooling path
`GridUnpooling`:
1. Reads the coarse point set and its saved parent mapping.
2. Projects coarse and skip features into a common decoder width.
3. Broadcasts coarse features back to the finer points using `pooling_inverse`.
4. Adds the recovered coarse signal to the skip-projected parent features.

## Components

### Shared component
- `NOGlobalBranch`
- `FNO3dBlock`

### Per-stage lightweight adapters
- `down_proj`
- `up_proj`
- fusion gate / concat projection
- optional learnable skip-stage weights

### Grid hierarchy operators
- `GridPooling`
- `GridUnpooling`

## Advantages
- Much lower parameter count than stage-specific NO branches.
- A single universal spectral operator is easier to control and profile.
- Relative geometry gives the NO path stronger spatial awareness.
- Grid pooling/unpooling is a better fit for dense-grid spectral processing than serialized pooling.

## Tradeoffs
- One universal branch may be less specialized than stage-specific branches.
- Larger dense grids increase memory cost quickly.
- The quality of the universal branch depends on the projection adapters learning good stage-to-universal mappings.

## Recommended default settings
- `type="PT-v3m1-NO-SharedBranch"`
- `share_no_branch=True`
- `universal_dim=64`
- `fno_modes=8`
- `base_grid_size=(64, 64, 64)`
- `pool_reduce="max"`
- `fusion="concat"`
- `learnable_stage_weights=True`

## Current naming
The architecture is now referred to as **PT-NO-Universal**.

## Before vs after

| Component | Previous NO-enhanced design | PT-NO-Universal |
|---|---|---|
| Global NO branch | Multiple stage-specific branches | One shared universal branch |
| FNO weights | Repeated across stages | Single shared `FNO3dBlock` |
| Geometry input | None or implicit | Explicit relative position encoding |
| Pooling | Serialized pooling | Grid pooling |
| Unpooling | Serialized unpooling | Grid unpooling |
| Spectral width | Tied to stage channels | Fixed universal width |
| Parameter efficiency | Very poor | Strongly improved |

## Notes
PT-NO-Universal keeps the PTv3 backbone structure but changes the global modeling path and the hierarchy operators. The active S3DIS config based on `semseg-pt-v3-no-v2.py` should use the universal-branch arguments such as `universal_dim`, `share_no_branch`, `base_grid_size`, and `pool_reduce`.