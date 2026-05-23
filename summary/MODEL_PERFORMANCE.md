# 3D Point Cloud Segmentation: Model Performance Tracking

This document tracks the evaluation metrics, hyperparameters, code locations, architectural decisions, and design choices for 3D point cloud segmentation models evaluated on the S3DIS dataset (Area 5).

## 📁 Code & File References

* **Point Transformer v3 (PTv3):** `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`
* **OPTNetPCA:** `pointcept/models/optnet/OPTNetPCA.py`
* **NOPNet (No Superpoint):** `pointcept/models/optnet/NOPNet_Normal.py`
* **NOPNet Config:** `configs/s3dis/config_optnet_NO.py`

---

## 📊 Overall Performance

| Model | mIoU | mAcc | allAcc |
| :--- | :--- | :--- | :--- |
| **Point Transformer v3** | **0.7052** | **0.7610** | **0.9120** |
| **OPTNetPCA** | 0.6435 | 0.7011 | 0.8940 |
| **NOPNet** (No Superpoint) | 0.5532* | 0.6603 | 0.8329 |

*\* NOPNet achieved a Best mIoU of 0.5757 during training; final tracked evaluation epoch yielded 0.5532.*

---

## 🎯 Class-wise IoU

| Class ID | Class Name | PTv3 IoU | OPTNetPCA IoU | NOPNet IoU |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Ceiling | **0.9352** | 0.9210 | 0.8699 |
| 1 | Floor | **0.9836** | 0.9776 | 0.9436 |
| 2 | Wall | **0.8616** | 0.8352 | 0.7480 |
| 3 | Beam | 0.0000 | 0.0000 | 0.0000 |
| 4 | Column | **0.3772** | 0.1411 | 0.3061 |
| 5 | Window | 0.6017 | **0.6097** | 0.3899 |
| 6 | Door | **0.7316** | 0.6058 | 0.4060 |
| 7 | Table | **0.8290** | 0.7743 | 0.7346 |
| 8 | Chair | **0.9178** | 0.8945 | 0.8428 |
| 9 | Sofa | **0.7078** | 0.4353 | 0.4116 |
| 10 | Bookcase | **0.7935** | 0.7452 | 0.6695 |
| 11 | Board | 0.8050 | **0.8461** | 0.4291 |
| 12 | Clutter | **0.6238** | 0.5791 | 0.4407 |

---

## ⚙️ Training Configurations & Hyperparameters

All models trained on **S3DIS** (Areas 1, 2, 3, 4, 6), evaluated on **Area 5**.

| Setting | PTv3 | OPTNetPCA | NOPNet |
| :--- | :--- | :--- | :--- |
| **Save Path** | `exp/s3dis/ptv3_01` | `exp/s3dis/optnet_PCA_02` | `exp/s3dis/optnet_NO_06` |
| **Batch Size** | 18 | 32 | 4 |
| **Max LR** | 0.006 | 0.001 | 0.001 |
| **Input Features** | `color`, `normal` | `coord`, `color` | `color`, `normal` |
| **TTA** | High (10 variants) | Low (1 variant) | Low (1 variant) |
| **Voxel Grid Size** | 0.02 | 0.02 | 0.02 |
| **Epochs** | 3000 | 3000 | 3000 |

### 💡 Key Takeaways
* **Feature Advantage:** PTv3 benefits from surface `normal` vectors + aggressive TTA. OPTNet models should be tested with identical TTA for a fair comparison.
* **Batch Size Discrepancy:** NOPNet trained with batch size 4 vs OPTNetPCA's 32 — likely hurt gradient stability.
* **The "Beam" Problem:** `beam` class fails universally (0.0000 IoU) — target for future dataset re-balancing.

---

## 🧠 Architectural Evolution: Standalone NOPNet

NOPNet was decoupled from `PointTransformerV3`. Morton-code serialization and z-order sorting were removed. Architecture is now a **pure U-Net** (`OPTNet` class in `NOPNet_Normal.py`).

### Core Components
1. **FPTLightweightAttentionBlock:** Local self-attention from Fast Point Transformer (FPT). Uses KNN graph for neighbor context. Incorporates relative-coordinate + relative-normal positional encoding.
2. **NOSuperpointPooling:** Wraps `SuperpointNeuralOperator` (SPT energy function). Pools point features to superpoints based on geometric boundaries. Runs dynamically so boundaries are learned end-to-end.
3. **NOSuperpointUnpooling:** Soft unpooling using stored `pooling_local_A` weights and `pooling_anchor_nn` indices. Falls back to hard nearest-neighbor if weights unavailable.
4. **Explicit Skip Connections:** Encoder saves full `Point` objects per stage. Decoder pops in reverse order.

### Feature Integration: Normals
* `in_channels=6` (3 color + 3 normal) fed into the initial embedding layer.
* `FPTLightweightAttentionBlock` uses `rel_normal = n_i - n_j` concatenated with `rel_pos` for positional encoding.
* `SuperpointNeuralOperator` receives `[coord, normal, feat]` concatenated in its `lift` layer.
* Pooled normals are scatter-summed and L2-normalized so coarse levels retain valid geometry.
* Dataloader `Collect` must include `'normal'` in both `keys` and `feat_keys` to prevent Pointcept from discarding it.

### Strategic Decision: Dynamic vs. Static Pooling
Offline precomputation (`precompute_NAG.py`) was considered but rejected. Dynamic superpoint computation allows boundaries to adapt based on learned semantic features, yielding higher accuracy for complex shapes.

---

## ✅ Correct NOPNet Config (`config_optnet_NO.py`)

### Architecture Depth — SPT Alignment
The original config inherited PTv3's 5-stage depth. The Superpoint Transformer paper (arxiv:2306.08045) uses **3 encoder levels** (2 pooling operations). Correct config:

```python
enc_depths=(2, 4, 2),   # 3 stages → 2 NOSuperpointPooling ops (matches SPT hierarchy)
dec_depths=(2, 2),       # must be len(enc_depths) - 1
base_channels=32,        # channel progression: 
backbone_out_channels=32,  # must match channels = base_channels
```

**Why 5-stage was wrong:** 4 pooling ops instead of SPT's 2 compounds approximation error at each unpool. Pushes channels to 512 (PTv3 scale), unnecessary for KNN-attention backbone.

### Valid OPTNet backbone parameters (only these are accepted by `OPTNet.__init__`)
```python
backbone=dict(
    type="OPTNet",
    in_channels=6,
    base_channels=32,
    enc_depths=(2, 4, 2),
    dec_depths=(2, 2),
    fpt_k=16,
    ordering_loss_weight=1.0,
)
```

All PTv3 params (`order`, `stride`, `enc_channels`, `enc_num_head`, `enc_patch_size`, `dec_channels`, `dec_num_head`, `dec_patch_size`, `mlp_ratio`, `qkv_bias`, `qk_scale`, `attn_drop`, `proj_drop`, `drop_path`, `pre_norm`, `enable_rpe`, `enable_flash`, `upcast_attention`, `upcast_softmax`, `enc_mode`, `pdnorm_*`) and stale sorter params (`ordering_k`, `warmup_epoch`, `enable_score_concat`, `tau`, `loss_weights`, `sorter_hidden_channels`, `sorter_k`, `num_score_buckets`, `code_depth`) must be removed — they are not accepted by `OPTNet.__init__`.

---

## 🧩 Design Decisions

### Positional Encoding in `FPTLightweightAttentionBlock`
**Decision:** Use mean-pooled relative geometry `pos_feat.mean(dim=1)` as input to `pos_enc`.

**Rationale:**
- `pos_feat = [rel_pos, rel_normal]` is `[N, K, 6]` — applying `pos_enc` directly to this is shape-correct but causes OOM (processes N×K rows through two Linear layers).
- Using absolute `[coords, normals]` is memory-efficient but not translation-equivariant.
- `pos_feat.mean(dim=1)` → `[N, 6]` is the **mean relative displacement** of all K neighbors — fully relative (translation-invariant), costs no extra memory, and only projects N points through `pos_enc`.
- The subsequent `proj_pos.unsqueeze(1) - proj_pos_j` encodes how each specific neighbor deviates from the average neighborhood in learned space.

```python
pos_input = pos_feat.mean(dim=1)          # [N, 6] — mean relative geometry
proj_pos = self.pos_enc(pos_input)         # [N, C] — project N points only
proj_pos_j = proj_pos[idx]                 # [N, K, C]
pos_emb = (proj_pos.unsqueeze(1) - proj_pos_j).view(N, k, self.num_heads, self.head_dim)
```

### `no_pooling_loss` Accumulation
**Decision:** Accumulate pooling loss in a local tensor in `OPTNet.forward()`, not via `point` attributes. Return `(feat, no_pooling_loss)` tuple from backbone. Unpack in `OPTNetSegmentor`.

**Rationale:** `torch.utils.checkpoint` strips non-tensor attributes from `Point` objects. Storing `no_pooling_loss` on `point` and reading it back after a checkpointed block silently returns `None` or raises `AttributeError`. A plain local tensor is checkpoint-safe.

```python
# OPTNet.forward() returns:
return point.feat, total_no_pooling_loss * self.no_pooling_loss_weight

# OPTNetSegmentor.forward() unpacks:
feat, no_pooling_loss = self.backbone(data_dict)
```

### Gradient Checkpointing Strategy
**Decision:** Checkpoint `enc_blocks` and `dec_blocks` (pure attention, no side effects). Run `enc_pools` and `dec_unpools` normally.

**Rationale:** Pooling layers write `no_pooling_loss` as a side effect — checkpointing them would lose this. Attention blocks are pure functions (tensor in → tensor out) and safe to checkpoint. This saves ~30-40% activation memory at ~20% extra compute cost.

---

## ⚡ Memory Optimization

Hardware: 2× NVIDIA H100 80GB HBM3. With `batch_size=4`, memory usage was ~65-74GB per GPU.

### Applied Optimizations
| Change | Location | Impact |
| :--- | :--- | :--- |
| `point_max=102400` (from 204800) | Config `SphereCrop` | Halves N → halves O(N×K) attention + operator memory |
| `hidden_channels=32` (from 64) | `SuperpointNeuralOperator` | Cuts operator intermediate tensors ~4× |
| `T=2` (from 3) | `SuperpointNeuralOperator` | One fewer `[N, k, hidden]` iteration |
| `k_anchor=4` (from 8) | `SuperpointNeuralOperator` | Halves soft-assignment matrix size |
| `fpt_k=8` (from 16) | Config backbone | Halves `[N, K, num_heads, head_dim]` attention tensors |
| Gradient checkpointing on `enc_blocks` + `dec_blocks` | `OPTNet.forward()` | ~30-40% activation memory saving |

Target after optimizations: ~25-30GB per sample → `batch_size=12-16` on H100 80GB.

---

## ⚖️ Strategic Decisions

* **Dynamic vs. Static Pooling:** Kept `SuperpointNeuralOperator` dynamic (not precomputed offline) — allows end-to-end learning of superpoint boundaries based on semantic features.
* **AMP:** `bfloat16` recommended over `float16` for stability with the `SuperpointNeuralOperator` energy loss.
* **Gradient Clipping:** `clip_grad=35.0` stabilizes the combined `seg_loss + no_pooling_loss` objective.