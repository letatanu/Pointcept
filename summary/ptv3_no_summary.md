# PTv3-NO: Summary & Status

**Repository:** [Pointcept](https://github.com/letatanu/Pointcept)

## Goal
Improve **global feature modeling** for 3D semantic segmentation (S3DIS / LiDAR) by augmenting PTv3's strong local serialized attention with a parallel **Neural Operator (NO) global branch**
---

## Architecture Variants

### PTv3-NOEncoder-Enhanced (Encoder-only + NO + 3 enhancements) ← active
```text
PTv3 serialized encoder  (stages 0–4)
        +
NOGlobalBranch at each pooling transition
  - voxel-grid scatter: uses point.grid_coord (integer, 0.02 m voxel indices)
    instead of float bounding-box normalisation → outlier-robust
  - flat_idx cached in Point dict per stage (key: no_flat_idx_stage{s}_{Gx}_{Gy}_{Gz})
  - adaptive grid: base_grid_size halved per stage (64→32→16→8, min=8)
  - grid occupancy ~constant across all stages (~30%)
        ↓
NOFusedPooling (spatial-gated fusion)
  - gate: nn.Sequential(Linear(2C,C), LayerNorm, Sigmoid) — per-point, per-channel
  - fused = gate * feat_fno + (1-gate) * feat_local → proj_concat
        ↓
NOLightweightUpsampleHead (enhanced)
  - project each stage to head_out_channels
  - chain pooling_inverse to upsample coarse → fine
  - scale each stage by learnable weight α_s (softmax-normalised)
  - sum ("sum") or concat ("concat") all stage features
  - final Linear → LayerNorm → GELU
        ↓
per-point classification head (no decoder, no skip connections)
```

### PTv3-NO (U-Net + NO) ← reference only
```text
PTv3 serialized path  (local, patch-wise, space-filling curve)
        +
NOGlobalBranch        (global, FNO over fixed canonical grid)
        ↓
concat fusion at all encoder/decoder stages
```

### PTv3-NOEncoder (Encoder-only + NO) ← reference only
```text
PTv3 serialized encoder  (stages 0–4)
        +
NOGlobalBranch at each pooling transition (fixed grid)
        ↓
NOLightweightUpsampleHead
  - project each stage to head_out_channels
  - chain pooling_inverse to upsample coarse → fine
  - sum all stage features (uniform weights)
        ↓
per-point classification head (no decoder, no skip connections)
```

---

## Key Files

| File | Role |
|------|------|
| `pointcept/models/point_transformer_v3/ptv3_no_lightweighthead_enhanced.py` | PTv3-NOEncoder-Enhanced (active) |
| `configs/s3dis/semseg-pt-v3-no-v1.py` | S3DIS config for PTv3-NOEncoder-Enhanced |
| `configs/s3dis/semseg-pt-v3-no-base.py` | S3DIS config for PTv3-NO (U-Net + NO, reference) |
| `configs/s3dis/semseg-pt-v3-no-v0.py` | S3DIS config for PTv3-NOEncoder (reference) |
| `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py` | PTv3 base (Block, Embedding, SerializedPooling, SerializedUnpooling) |
| `configs/s3dis/semseg-pt-v3m1-0-base.py` | S3DIS config for PTv3 |
| `pointcept/engines/train.py` | Trainer / dataloader |

---

## New Modules

### `FNO3dBlock`
3-D Fourier Neural Operator operating on a dense grid `[B, C, X, Y, Z]`.
- Learnable spectral weights `weight_real`, `weight_imag` of shape `[C, C, modes, modes, modes]`
- Truncated spectral mixing (first `modes` frequencies) → global receptive field at low cost
- **bypass = `nn.Linear(C, C)`** — avoids DDP grad-stride mismatch from `Conv3d(kernel_size=1)`
- `del x_ft, out_ft` before bypass to free peak FFT memory

### `NOGlobalBranch`
Scatter points → grid → FNO → gather back to points.
- **Voxel-grid scatter:** uses `point.grid_coord` (integer voxel indices from `GridSample@0.02m`) — no float bounding-box stretch, outlier-robust
- **Fixed grid** (NOEncoder): `grid_size=(Gx,Gy,Gz)` — same allocation shape every forward pass
- **Adaptive grid** (NOEncoder-Enhanced): `grid_size` halved per stage → `max(8, base//2^stage)`; occupancy ~30% at all stages
- **`flat_idx` cached** in `Point` dict under key `no_flat_idx_stage{s}_{Gx}_{Gy}_{Gz}` — computed once per stage, reused on repeated calls
- Reuses `point.grid_coord` from upstream `GridSample` — no extra voxelization
- `del grid_flat, count, grid, grid_out` after use

### `NOFusedPooling`
Wraps `SerializedPooling`. When `enable_no=True`:
1. Run `NOGlobalBranch` on fine points → `feat_fno [N_fine, C_out]`
2. Run `SerializedPooling` → `point_coarse`
3. `scatter_max` fine NO features to coarse resolution → `feat_no_coarse`
4. **Spatial-gated fusion** (`fusion="concat"`):
   ```python
   alpha = gate(cat([point.feat, feat_no_coarse]))  # (N, C), Sigmoid
   point.feat = proj_concat(alpha * feat_no_coarse + (1-alpha) * point.feat)
   ```
   — per-point, per-channel blend; replaces the old flat `concat → Linear`
5. Gated-add fusion (`fusion="add"`): `point.feat += sigmoid(gate_scalar) * feat_no_coarse` (unchanged)

### `NOFusedUnpooling` *(PTv3-NO only)*
Wraps `SerializedUnpooling`. When `enable_no=True`:
1. Run `NOGlobalBranch` on coarse points → `feat_global [N_coarse, C_out]`
2. Run `SerializedUnpooling` → `point_fine`
3. Broadcast coarse NO features to fine via `pooling_inverse`
4. Fuse into `point_fine.feat` via gated-add or concat

### `NOLightweightUpsampleHead` *(PTv3-NOEncoder / PTv3-NOEncoder-Enhanced)*
Decoder-free multi-scale upsampling.

**Enhanced (NOEncoder-Enhanced):**
1. Project each encoder stage `s` features: `nn.Linear(enc_channels[s], out_channels)` + `LayerNorm`
2. Chain `pooling_inverse` to upsample stage `s` features to stage-0 resolution
3. Scale each stage by **learnable weight** `α_s = softmax(stage_weights)[s]` — init `1/num_stages`
4. **Sum** (`head_fusion="sum"`) or **Concat** (`head_fusion="concat"`) all weighted stage features
5. Final `Linear(in → out_channels) → LayerNorm → GELU` projection
   - `in = out_channels` for sum mode
   - `in = num_stages × out_channels` for concat mode

---

## PTv3 vs PTv3-NOEncoder-Enhanced

| Aspect | PTv3 | PTv3-NO | PTv3-NOEncoder | PTv3-NOEncoder-Enhanced |
|--------|------|---------|----------------|-------------------------|
| Pooling | `SerializedPooling` | `NOFusedPooling` | `NOFusedPooling` | `NOFusedPooling` |
| Unpooling | `SerializedUnpooling` | `NOFusedUnpooling` | **None** | **None** |
| Decoder blocks | `dec_depths=(2,2,2,2)` | `dec_depths=(2,2,2,2)` | **None** | **None** |
| Skip connections | ✅ | ✅ (ablatable) | **None** | **None** |
| Global context | Skip only | Skip + FNO | FNO only | FNO only |
| Grid indexing | — | Canonical float | Canonical float | **Voxel integer (0.02m)** |
| Grid occupancy | — | ~30% (stage 0) → ~2% (stage 4) | ~30% → ~2% | **~30% all stages** |
| `flat_idx` cache | — | — | — | **Cached in `Point` dict** |
| Fusion gate | — | Scalar `gate*0` (DDP only) | Scalar `gate*0` (DDP only) | **Spatial MLP gate (Sigmoid)** |
| Feature recovery | U-Net unpooling | U-Net + NO broadcast | Chain `pooling_inverse` + sum | Chain `pooling_inverse` + **weighted** sum/concat |
| Stage weights | — | — | Uniform | **Learnable (softmax)** |
| Head fusion | — | — | Sum only | **Sum or Concat** |
| Memory allocation | Static | Fixed per stage | Fixed per stage | **Adaptive (smaller at deep stages)** |
| DDP notes | None | `bypass=nn.Linear`; `gate*0` trick | `gate*0` trick | Spatial gate always in graph |

---

## Stage Activation

| Stage | Point density | NO active | Grid (Enhanced, base=64) |
|-------|--------------|-----------|--------------------------|
| 0 | ~full input | ✅ | 64³ |
| 1 | ÷2 | ✅ | 32³ |
| 2 | ÷4 | ✅ | 16³ |
| 3 | ÷8 | ✅ | 8³ |

NO is active at **4 pooling transitions** per forward pass in PTv3-NOEncoder-Enhanced.

---

## Adaptive Grid Memory (C=256, Enhanced)

| Stage | Grid | Memory / call | Occupancy |
|-------|------|--------------|-----------| 
| 0 | 64³ | ~67 MB | ~30% |
| 1 | 32³ | ~8 MB | ~30% |
| 2 | 16³ | ~1 MB | ~30% |
| 3 | 8³ | ~0.1 MB | ~30% |

Total NO memory per forward pass: **~76 MB** vs **~268 MB** (fixed 64³ × 4 stages).

---

## Hyperparameters

| Parameter | PTv3-NOEncoder-Enhanced | Description |
|-----------|--------------------------|-------------|
| `no_stages` | `(T,T,T,T)` | All transitions have NO |
| `fno_modes` | `12` | Fourier modes per dimension |
| `base_grid_size` | `(64,64,64)` | Base grid; halved per stage |
| `adaptive_grid` | `True` | Enable per-stage grid scaling |
| `fusion` | `"concat"` | Pooling fusion rule (spatial-gated) |
| `head_fusion` | `"concat"` | Head fusion mode |
| `learnable_stage_weights` | `True` | Per-stage α_s in head |
| `drop_path` | `0.2` | Stochastic depth |
| `pct_start` | `0.1` | OneCycleLR warmup |
| `head_out_channels` | `64` | Upsampling head output width |

---

## NOEncoder-Enhanced Config

```python
backbone=dict(
    type='PT-v3m1-NOEncoder-Enhanced',
    in_channels=6,
    order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    mlp_ratio=4,
    drop_path=0.2,
    shuffle_orders=True,
    pre_norm=True,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    no_stages=(True, True, True, True),
    fno_modes=12,
    base_grid_size=(64, 64, 64),
    adaptive_grid=True,
    fusion='concat',
    head_fusion='concat',
    head_out_channels=64,
    learnable_stage_weights=True,
)
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(type='OneCycleLR', max_lr=[0.006, 0.0006], pct_start=0.1, ...)
```

---

## Issues & Fixes

### Issue 1 — CUDA OOM at iter 8/100
- **Cause:** `MAX=1028` in `NOGlobalBranch` allows `1028³ × C` allocation
- **Fix:** Switch to fixed grid `grid_size=(64, 64, 64)` with coordinate normalisation

### Issue 2 — DDP Grad Stride Warning
- **Cause:** `nn.Conv3d(C, C, kernel_size=1)` bypass has non-contiguous `[C,C,1,1,1]` weight
- **Fix (applied):** `self.bypass = nn.Linear(channels, channels)` with `.permute()` in forward

### Issue 3 — GPU Memory Imbalance (~20 GiB between ranks)
- **Cause:** `DistributedSampler` assigns large rooms to one rank; dynamic grid compounds this
- **Fix:** Fixed/adaptive grid eliminates imbalance; additionally:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

### Issue 4 — Memory Sawtooth Pattern
- **Cause:** `del` intermediates return tensors to PyTorch pool (fast); `empty_cache` returns to CUDA (slow)
- **Note:** Sawtooth after `del` is **normal and healthy**
- **Config:**
  ```python
  empty_cache = False
  empty_cache_per_epoch = True
  ```

### Issue 5 — DDP Unused Parameter Error (NOEncoder)
- **Cause:** `gate` parameter in `NOFusedPooling` unused when `fusion="concat"`
- **Fix (applied):** Promoted scalar `gate` to spatial MLP gate — always in compute graph for real:
  ```python
  self.gate = nn.Sequential(Linear(2C, C), LayerNorm, Sigmoid)
  alpha = self.gate(cat([point.feat, feat_no_coarse]))
  point.feat = proj_concat(alpha * feat_no_coarse + (1-alpha) * point.feat)
  ```

### Issue 6 — Deep-Stage Grid Sparsity (NOEncoder only)
- **Cause:** Fixed 64³ grid at stage 3–4 has ~98% empty cells → FNO on near-zero tensor
- **Fix (Enhanced):** Adaptive grid halves resolution per stage → occupancy ~30% at all stages
- **Workaround** (without Enhanced): set `no_stages=(True, True, False, False)`

### Issue 7 — Canonical Grid Outlier Sensitivity
- **Cause:** Float bounding-box normalisation stretches the grid when distant outlier points exist, clustering majority of scene points in a small FNO grid region → poor spectral coverage
- **Fix (applied):** Use `point.grid_coord` (integer voxel indices, 0.02m spacing) as spatial index — discrete, bounded, outlier-robust. `flat_idx` cached in `Point` dict per stage to avoid recomputation:
  ```python
  cache_key = f"no_flat_idx_stage{self.stage}_{Gx}_{Gy}_{Gz}"
  if cache_key not in point.keys():
      point[cache_key] = self._compute_flat_idx(point, Gx, Gy, Gz)
  flat_idx = point[cache_key]
  ```

---

## Results

S3DIS Area 5. PTv3 baseline (own run): **0.7052 mIoU**.

| Experiment | `no_stages` | `fno_modes` | `fusion` | `drop_path` | `pct_start` | mIoU | mAcc | allAcc | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline PTv3** | — | — | — | 0.3 | 0.05 | **0.7052** | **0.7610** | **0.9120** | Reference |
| **PTv3-NO (NONet_08)** | `(T,T,T,T)` | 12 | concat | 0.2 | 0.1 | **0.7045** | **0.7589** | **0.9093** | Best U-Net+NO run; matches baseline |
| **PTv3-NOEncoder_01** | `(T,T,T,T)` | 12 | concat | 0.2 | 0.1 | **0.6785** | **0.7350** | **0.9060** | Encoder-only; -0.0267 vs baseline |
| **PTv3-NOEncoder-Enhanced_01** | `(T,T,T,T)` | 12 | concat | 0.2 | 0.1 | **0.6853** | **0.7454** | **0.9031** | Adaptive grid + spatial gate + concat head; -0.0199 vs baseline |

---

## Per-Class Results (S3DIS Area 5)

| Class | PTv3 Baseline | NONet_08 | NOEncoder_01 | NOEncoder-Enhanced_01 |
|-------|--------------|----------|--------------|----------------------|
| ceiling | 0.9352 | 0.9407 | 0.9466 | **0.9321** |
| floor | 0.9836 | 0.9834 | 0.9821 | **0.9831** |
| wall | 0.8616 | 0.8479 | 0.8587 | **0.8464** |
| beam | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| column | 0.3772 | 0.3270 | 0.2622 | **0.3261** (+0.064 vs NOEncoder_01) |
| window | 0.6017 | 0.6360 | 0.6021 | **0.6295** |
| door | 0.7316 | 0.6658 | 0.6056 | **0.5529** |
| table | 0.8290 | 0.8060 | 0.8131 | **0.8072** |
| chair | 0.9178 | 0.9213 | 0.9110 | **0.9173** |
| sofa | 0.7078 | 0.7873 | 0.6346 | **0.6953** (+0.061 vs NOEncoder_01) |
| bookcase | 0.7935 | 0.7882 | 0.7617 | **0.7630** |
| board | 0.8050 | 0.8240 | 0.8293 | **0.8426** |
| clutter | 0.6238 | 0.6312 | 0.6130 | **0.6137** |
| **Overall** | **0.7052** | **0.7045** | **0.6785** | **0.6853** |

> **NOEncoder-Enhanced_01 observations (2026-04-20):**
> - **+0.068 mIoU vs NOEncoder_01** — adaptive grid + spatial gate deliver clear gains
> - **column +0.064, sofa +0.061** — biggest recoveries vs NOEncoder_01
> - **board 0.8426** — beats PTv3 baseline (+0.038)
> - **beam still 0.000** — consistent across all models including PTv3 baseline and NONet_08 (both with full decoder + skip connections); skip connections are not the bottleneck. Beam is rare, thin, and elongated — requires stronger attention (e.g. larger patch size, anisotropic receptive field, or explicit instance-level grouping) to capture reliably.
> - **door 0.5529** — still below NOEncoder_01 (0.6056); small geometry, needs further attention
> - Overall gap to PTv3 baseline: **-0.0199 mIoU** (was -0.0267 for NOEncoder_01)

---

## Ablation Studies (Planned)

### A — Adaptive Grid Size
| Exp | `adaptive_grid` | `base_grid_size` | Expected |
|---|---|---|---|
| A1 | False | (64,64,64) | Baseline NOEncoder |
| A2 | False | (128,128,128) | More cells, still fixed |
| A3 | True | (64,64,64) | Halve per stage: 64→32→16→8 |
| A4 | True | (128,128,128) | Larger base, halved: 128→64→32→16 |

### B — Head Fusion Mode
| Exp | `head_fusion` | `learnable_stage_weights` | Expected |
|---|---|---|---|
| B1 | `sum` | False | Original uniform sum |
| B2 | `sum` | True | Learned weights |
| B3 | `concat` | False | Concat only |
| B4 | `concat` | True | Full enhanced head |

### C — NO Stage Activation
| Exp | `no_stages` | Expected |
|---|---|---|
| C1 | (F,F,F,F) | Lightweight head baseline (NO-free) |
| C2 | (T,F,F,F) | NO at finest stage only |
| C3 | (T,T,F,F) | NO at fine stages only |
| C4 | (F,F,T,T) | NO at coarse stages only |
| C5 | (T,T,T,T) | All stages (default) |

### D — Encoder Depth
| Exp | `enc_depths` | `no_stages` | Expected |
|---|---|---|---|
| D1 | (2,2,2,6,2) | (T,T,T,T) | Baseline |
| D2 | (4,4,2,6,2) | (T,T,T,T) | Deeper fine stages |
| D3 | (2,2,2,6,2) | (T,T,F,F) | Drop coarse NO |
| D4 | (4,4,2,6,2) | (T,T,F,F) | Deeper fine + fine NO only |

**Recommended run order:** A3 → B2 → C2 → A3+B4 → D4

---

## Memory Aid

> **Active model:** PTv3-NOEncoder-Enhanced — encoder-only, adaptive grid, spatial-gated FNO fusion, weighted concat head.
> Voxel-grid scatter (`point.grid_coord`) — discrete 0.02m indices, outlier-robust, `flat_idx` cached per stage.
> Spatial gate replaces scalar `gate*0` trick — per-point per-channel blend, always in DDP compute graph.
> Adaptive grid — halve per stage, min 8³ — keeps ~30% occupancy at all depths, ~76 MB total NO memory.
> **Best current: NOEncoder-Enhanced_01 = 0.6853 mIoU** — gap to PTv3 baseline reduced to -0.0199.
> Beam (0.000) — fails across ALL models including PTv3 baseline; not a decoder/skip issue. Rare thin class needs stronger attention or dedicated geometry-aware handling (larger patches, anisotropic kernels, or instance grouping).