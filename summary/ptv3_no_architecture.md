# PTv3 vs PTv3-NO: Design & Architecture Reference

> **Files:**
> - Original PTv3: `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`
> - PTv3-NO: `pointcept/models/point_transformer_v3/ptv3_no.py`
> - S3DIS config: `configs/s3dis/semseg-pt-v3m1-no-base.py`

---

## 1. Motivation

PTv3 uses **serialized patch-wise attention** (space-filling curves) for efficient local feature modeling in 3D point clouds. However, attention is computed only **within fixed-size patches** — it has no mechanism for long-range global context across the full scene.

**PTv3-NO** adds a parallel **Neural Operator (NO) global branch** that operates on the full scene via a 3D Fourier Neural Operator (FNO), providing global context while keeping PTv3's serialized local path completely intact.

---

## 2. PTv3 Architecture (Baseline)

### 2.1 Overall Structure

```
Input (point cloud)
    ↓
Embedding                          [in_channels → enc_channels[0]]
    ↓
Encoder (5 stages):
    Stage 0: Block × enc_depths[0]
    Stage 1: SerializedPooling → Block × enc_depths[1]
    Stage 2: SerializedPooling → Block × enc_depths[2]
    Stage 3: SerializedPooling → Block × enc_depths[3]
    Stage 4: SerializedPooling → Block × enc_depths[4]
    ↓
Decoder (4 stages):
    Stage 3: SerializedUnpooling → Block × dec_depths[3]
    Stage 2: SerializedUnpooling → Block × dec_depths[2]
    Stage 1: SerializedUnpooling → Block × dec_depths[1]
    Stage 0: SerializedUnpooling → Block × dec_depths[0]
    ↓
Output (per-point features)
```

### 2.2 Key Mechanisms

#### Serialization
- Points are reordered by a **space-filling curve** (`z-order`, `Hilbert`, and their transpositions).
- Gives spatially coherent patch groupings without explicit kNN or radius search.
- `serialized_code`, `serialized_order`, and `pooling_inverse` are computed once and reused throughout the encoder-decoder.

#### SerializedAttention (inside `Block`)
- Features are reordered by `point.serialized_order[...]` and reshaped into fixed-size patches.
- **Attention is local** — computed only within each patch, not globally.
- Supports Flash Attention for memory-efficient long-sequence attention within patches.

#### SerializedPooling
- Coarsens points by **bit-shifting the serialized code** (halving resolution at each stage).
- Stores `pooling_inverse` mapping (fine → coarse) for use in unpooling.
- Tightly coupled to the space-filling-curve design; cannot be naively replaced.

#### SerializedUnpooling
- Restores fine-level features using `pooling_inverse`.
- Fuses decoder features with encoder skip features by **addition**.
- The U-Net skip connections are the primary source of global multi-scale information in PTv3.

### 2.3 PTv3 Default Hyperparameters (S3DIS)

| Parameter | Value |
|---|---|
| `enc_depths` | (2, 2, 2, 6, 2) |
| `enc_channels` | (32, 64, 128, 256, 512) |
| `enc_num_head` | (2, 4, 8, 16, 32) |
| `enc_patch_size` | (1024, 1024, 1024, 1024, 1024) |
| `dec_depths` | (2, 2, 2, 2) |
| `dec_channels` | (64, 64, 128, 256) |
| `mlp_ratio` | 4 |
| `drop_path` | 0.3 |

### 2.4 Memory Behavior

All tensor shapes in PTv3 are **fixed at config time** — determined by `patch_size` and `enc_channels`. There is no dynamic allocation. GPU memory usage is **flat and predictable** during training.

---

## 3. PTv3-NO Architecture

### 3.1 Design Philosophy

Rather than replacing `SerializedPooling`/`SerializedUnpooling` (which would break the serialization pipeline), PTv3-NO **augments** the existing path with a parallel NO global branch. The serialized path handles local hierarchical features; the NO branch contributes global context via Fourier-space operations over a dense 3D grid.

```
Input (point cloud, with grid_coord from GridSample @ 2cm)
    ↓
Embedding
    ↓
Encoder (5 stages):
    Stage 0: Block × 2
    Stage 1: NOFusedPooling (NO disabled) → Block × 2
    Stage 2: NOFusedPooling (NO disabled) → Block × 2
    Stage 3: NOFusedPooling (NO enabled ✓) → Block × 6
    Stage 4: NOFusedPooling (NO enabled ✓) → Block × 2
    ↓
Decoder (4 stages):
    Stage 3: NOFusedUnpooling (NO enabled ✓) → Block × 2
    Stage 2: NOFusedUnpooling (NO enabled ✓) → Block × 2
    Stage 1: NOFusedUnpooling (NO disabled) → Block × 2
    Stage 0: NOFusedUnpooling (NO disabled) → Block × 2
    ↓
Output (per-point features)
```

> **`no_stages=(False, False, True, True)`** — NO is active only at the two deepest stages of the encoder and corresponding decoder stages.

### 3.2 New Modules

#### `FNO3dBlock`

A 3D Fourier Neural Operator block that operates on a dense volumetric grid `[B, C, X, Y, Z]`.

```python
class FNO3dBlock(nn.Module):
    def __init__(self, channels, modes=8):
        self.modes = modes
        # Learnable spectral weights (real + imaginary)
        self.weight_real = nn.Parameter(scale * torch.rand(C, C, modes, modes, modes))
        self.weight_imag = nn.Parameter(scale * torch.rand(C, C, modes, modes, modes))
        # Bypass: nn.Linear (contiguous 2D weight, DDP-safe)
        self.bypass = nn.Linear(channels, channels)

    def forward(self, x):  # x: [B, C, X, Y, Z]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        # Truncated spectral mixing: only first `modes` frequencies
        out_ft[:, :, :mx, :my, :mz] = einsum("bcxyz,cdxyz->bdxyz", x_ft[...], weight[...])
        x_no = torch.fft.irfftn(out_ft, s=(X, Y, Z), dim=[-3, -2, -1])
        del x_ft, out_ft  # free FFT buffers before bypass
        # Bypass: permute to [..., C] for Linear, then restore
        x_bypass = self.bypass(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        return x_no + x_bypass
```

**Key design points:**
- Spectral mixing is done in **truncated Fourier space** (first `modes` frequencies only) — global receptive field at low compute cost.
- `bypass` uses `nn.Linear` instead of `nn.Conv3d(kernel_size=1)` to avoid DDP grad stride mismatch (the `Conv3d` weight shape `[C,C,1,1,1]` is non-contiguous under DDP).
- `del x_ft, out_ft` before the bypass reduces peak memory by freeing large FFT buffers early.

---

#### `NOGlobalBranch`

Maps point features to a dense 3D grid, applies `FNO3dBlock`, and maps output back to points.

```python
class NOGlobalBranch(PointModule):
    def forward(self, point: Point):
        feat = point.feat          # [N, C]
        coord = point.grid_coord   # [N, 3] — reuses GridSample output, no re-voxelization needed

        # Shift to 0-based, cap grid to MAX^3
        shifted = (coord - coord.min(dim=0).values).clamp(max=[X-1, Y-1, Z-1])
        MAX = 128  # critical: caps grid volume to 128^3 max

        # Scatter points → dense grid (mean aggregation)
        flat_idx = shifted[:,0]*(Y*Z) + shifted[:,1]*Z + shifted[:,2]
        grid_flat = scatter_mean(feat, flat_idx)    # [X*Y*Z, C]
        grid = grid_flat.view(1, X, Y, Z, C).permute(0, 4, 1, 2, 3)  # [1, C, X, Y, Z]

        # Apply FNO
        grid_out = self.fno(grid)

        # Gather output back to points by exact index lookup
        feat_out = grid_out.squeeze(0).permute(1,2,3,0).reshape(-1, C)[flat_idx]
        del grid_flat, grid, grid_out  # explicit free to prevent fragmentation

        return self.norm(feat_out)  # [N, C]
```

**Key design points:**
- Reuses `grid_coord` from upstream `GridSample` (2 cm grid) — no separate voxelization needed.
- `MAX=128` is critical: grid volume scales cubically, so `MAX=128` is **8× smaller** in memory than `MAX=256`.
- Exact index lookup (not trilinear interpolation) since points are already aligned to the grid.
- `del` of all large intermediates prevents CUDA allocator fragmentation under DDP.

---

#### `NOFusedPooling`

Wraps `SerializedPooling` and injects NO global features before fusion.

```python
class NOFusedPooling(PointModule):
    def forward(self, point):
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))  # [N_fine, C_out]

        point = self.pool(point)  # SerializedPooling (unchanged)

        if self.enable_no:
            inv = point.pooling_inverse
            # Max-pool NO features from fine to coarse points
            feat_no_coarse = scatter_max(feat_global, inv, dim=0)[0]
            point.feat = point.feat + feat_no_coarse  # fusion by addition
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point
```

---

#### `NOFusedUnpooling`

Wraps `SerializedUnpooling` and injects NO global features after unpooling.

```python
class NOFusedUnpooling(PointModule):
    def forward(self, point):
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))  # [N_coarse, C_out]

        if not self.use_skip:
            point.pooling_parent.feat = zeros_like(...)  # ablation: zero out skip

        inverse = point.pooling_inverse
        point = self.unpool(point)  # SerializedUnpooling (unchanged)

        if self.enable_no:
            feat_no_fine = feat_global[inverse]  # broadcast coarse NO → fine
            point.feat = point.feat + feat_no_fine
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point
```

---

### 3.3 PTv3-NO Additional Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `no_stages` | `(False, False, True, True)` | Which encoder stages have NO active |
| `fno_modes` | `8` | Number of Fourier modes per dimension |
| `use_skip` | `True` | Whether to use U-Net skip connections (ablation) |
| `fusion` | `"add"` | Fusion rule: `"add"` (concat planned) |

---

## 4. PTv3 vs PTv3-NO: Key Differences

| Aspect | PTv3 | PTv3-NO |
|---|---|---|
| **Pooling module** | `SerializedPooling` | `NOFusedPooling` (wraps `SerializedPooling`) |
| **Unpooling module** | `SerializedUnpooling` | `NOFusedUnpooling` (wraps `SerializedUnpooling`) |
| **Global context** | U-Net skip connections only | U-Net skip + parallel FNO global branch |
| **Attention scope** | Local patch only (`patch_size` points) | Local patch (serialized) + global (FNO over full grid) |
| **Feature representation** | Point-based throughout | Point-based + temporary dense grid in NO branch |
| **Memory allocation** | Static (fixed by config) | Dynamic at NO-active stages (variable grid per scene) |
| **Memory behavior** | Flat / predictable | Sawtooth (spikes on alloc, drops on `del`) |
| **Grid voxelization** | None | Reuses `grid_coord` from `GridSample`, no extra step |
| **New parameters** | — | `weight_real`, `weight_imag` in `FNO3dBlock`; `proj_no` linears |
| **Skip ablation** | Not supported | `use_skip=False` zeros encoder skip, testing NO as sole global path |
| **DDP notes** | No issues | `bypass=nn.Linear` required to avoid grad stride mismatch |

---

## 5. Data Flow Comparison

### PTv3 (per encoder stage s > 0)

```
point_fine
    → SerializedPooling → point_coarse
    → Block × N          → point_coarse (enriched)
```

### PTv3-NO (per encoder stage s > 0, NO enabled)

```
point_fine
    ├─ [Serialized path] → SerializedPooling → point_coarse
    └─ [NO branch]       → NOGlobalBranch
                              scatter to grid [1,C,X,Y,Z]
                              → FNO3dBlock (spectral mixing)
                              gather back to points [N_fine, C_out]
                              → proj_no → [N_fine, C_out]
                              → scatter_max to coarse [N_coarse, C_out]
                          + (add fusion)
    → point_coarse (serialized + global NO features)
    → Block × N → point_coarse (enriched)
```

---

## 6. Why NO is Only at Deeper Stages

Applying a dense 3D Fourier operator at fine (early) stages would require grids on the order of `1000³` voxels — infeasible in memory. At deep stages, `SerializedPooling` has already coarsened the point cloud by `stride^depth`, reducing the grid to a manageable size. With `MAX=128`, the NO branch runs on at most a `128³` volume regardless of scene size.

| Stage | Approx. resolution | NO active? |
|---|---|---|
| 0 (finest) | ~full input | ❌ |
| 1 | ÷2 | ❌ |
| 2 | ÷4 | ❌ |
| 3 | ÷8 | ✅ |
| 4 (coarsest) | ÷16 | ✅ |

---

## 7. Known Issues & Fixes

### 7.1 CUDA OOM (iter 8/100)

**Cause:** `MAX=1028` in `NOGlobalBranch` (code bug) — grid can reach `1028³ × C` per scene.

**Fix:** Set `MAX=128` and add explicit `del` of intermediates:
```python
MAX = 128
del grid_flat, count, grid, grid_out
```

### 7.2 DDP Grad Stride Warning

**Cause:** `nn.Conv3d(C, C, kernel_size=1)` bypass — non-contiguous `[C,C,1,1,1]` weight violates DDP bucket-view contract.

**Fix:** Already applied — `self.bypass = nn.Linear(channels, channels)` with `.permute()` in forward.

### 7.3 GPU Memory Imbalance (~20 GiB difference between ranks)

**Cause:** `DistributedSampler` splits by index, not scene size. Large rooms → large grids → more memory on one rank.

**Fix:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Plus `MAX=128` and `empty_cache_per_epoch=True` in config.

### 7.4 Memory Management Summary

| Action | Cost | Recommendation |
|---|---|---|
| `del grid_flat, grid_out` etc. | ~0 (ref count decrement) | ✅ Keep |
| `torch.cuda.empty_cache()` per step | ~10–50 ms/step overhead | ❌ Disable |
| `torch.cuda.empty_cache()` per epoch | ~50 ms/epoch | ✅ Fine |
| `MAX=128` grid cap | Saves alloc time | ✅ Keep |
| `expandable_segments=True` | ~0 runtime cost | ✅ Keep |

> **Memory sawtooth after `del` is normal.** `del` returns tensors to PyTorch's internal pool (no GPU sync). `empty_cache` returns them to CUDA (slow, requires sync). Use `del` every step, `empty_cache` only per epoch.

---

## 8. Suggested Ablations

| Experiment | Config | Purpose |
|---|---|---|
| **Baseline PTv3** | `semseg-pt-v3m1-0-base.py` | Reference mIoU |
| **PTv3-NO + skip** | `use_skip=True`, `no_stages=(F,F,T,T)` | Does NO add on top of skip? |
| **PTv3-NO, no skip** | `use_skip=False`, `no_stages=(F,F,T,T)` | Can NO replace U-Net global path? |
| **NO at all stages** | `no_stages=(T,T,T,T)` | Upper bound (memory permitting) |
| **FNO modes sweep** | `fno_modes ∈ {4, 8, 16}` | Quality vs. memory tradeoff |
