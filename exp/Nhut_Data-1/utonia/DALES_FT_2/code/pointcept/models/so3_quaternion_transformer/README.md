# Scalar–Vector Neuron network with PTv2-style attention

## Install
Copy `model/` to `pointcept/models/so3_vector_neuron/`. Add this line to `pointcept/models/__init__.py`:

```python
from .so3_vector_neuron import SO3VectorNeuronNet, VNPTv2Segmentor
```

Copy the standalone configs to `configs/aerorelief3d/`. Use your existing Pointcept launch command, selecting one config. For standard Pointcept CLI:

```bash
python tools/train.py --config-file configs/aerorelief3d/A0_full.py
```

Adapt distributed launcher options to your installation. All experiments require new training runs; older model checkpoints are incompatible. Keep dataset preprocessing and numeric labels identical across experiments. Class names in the config do not remap annotations.

## What changed
The full model follows the agreed formula:

`xi = [raw distance/shape descriptor, Q_vector dot displacement, K_vector dot displacement]`

`relation = multiplier_MLP(xi) * (q_scalar - k_scalar) + bias_MLP(xi)`

`logits = weight_encoder(grouped_linear(relation))`

`weights = neighbor_softmax(logits)`

The multiplier is a direct learned MLP output, as discussed. There is no 1+tanh bounding and no separate additive vector score. Both multiplier and bias output scalar-channel width, not group width. Scalar-attention mode instead uses elementwise q*k and scaled group reduction, with the same positional ordering. Vector messages use group-specific weights. Their coefficient MLP sees normalized query/source scalar features and xi. Optional geometric scalar values use another xi MLP.

All geometric/vector/scalar normalization and interpolation in this backbone use explicit nonzero/zero cases, with no additive epsilon. External optimizer, losses and transforms retain their implementations. Small nonzero denominators can cause large gradients.

RGB initializes invariant scalar features. A geometry stem initializes equivariant vectors from learned weighted relative displacements. Vector linear mixing, RMS normalization and directional activation preserve equivariance. Downsampling uses FPS neighborhoods, scalar max pooling, whole-vector max selection with exact-tie averaging (or mean in A6), and coordinate mean pooling. Decoder interpolation is inverse-distance weighted except that coincident coarse neighbors receive uniform zero-distance weights. Final invariant readout concatenates norms and projected Gram products.

## Ablation plan
Every config is complete and runnable without config inheritance. Each has a distinct save_path.

| Config | Change from A0 | Question |
|---|---|---|
| A0_full | None | Reference model |
| A1_scalar_only | vector_channels=0; no directional conditioning | Does the vector subsystem help? |
| A2_no_directional | Remove vector–displacement conditioning; retain vector messages/readout | Does directional conditioning help? |
| A3_no_multiplier | Positional multiplier replaced with 1 | Does geometry-conditioned multiplication help? |
| A4_no_bias | Remove channel-wise positional bias before encoder | Does geometric bias help? |
| A5_scalar_attention | Product relation and scaled group sum | Does grouped learned difference attention help? |
| A6_vector_mean_pool | Mean rather than VN max vector pooling | Does whole-vector selection help? |
| A7_geometry_values | Add xi embedding to scalar values | Does positional value encoding help? |
| A8_vectors32 | 32 vector channels instead of 16 | Is vector capacity limiting? |

Priority: A0, A1, A2, A3, A4; then A5–A8. Run all initially on 3DAeroRelief using the same split, sampling, optimizer, schedule and seed. For shortlisted models repeat with at least three matched seeds. Change the top-level seed consistently across variants. Record mIoU, mAcc, allAcc, all per-class IoUs (especially Road), parameters, peak memory and iteration time. Do not modify label mapping, crop policy or class weights between variants. Vector removal/width changes alter parameter counts, so these are component/capacity ablations, not parameter-matched controls. A5 changes relation and weight encoder together.

The supplied batch_size=32, crop size and schedule are inherited from your original config; choose a feasible batch size once and use it consistently for comparisons. edge_chunk_size=256 bounds edge intermediates but does not remove full point-feature, graph and geometric-cache memory. No improvement is assumed before measurements.

## Symmetry evaluation
On the same sampled points, rotate coordinates, regenerate the XYZ portion of input features, and leave RGB unchanged. Compare invariant logits/features and rotated vector outputs. Evaluate unrotated and randomly SO(3)-rotated copies to distinguish numerical equivariance from actual class performance. Test CUDA pointops separately: exact distance ties and FPS backend choices can change sampling. Axis-aligned GridSample does not commute with arbitrary rotations, so full preprocessing pipelines are not exactly invariant.

## Validation
Python compilation and static config/registry checks passed. PyTorch is unavailable in the authoring environment, so runtime/gradient/CUDA/distributed checks have NOT been executed. Run `python smoke_test.py` from this extracted folder in a PyTorch environment. It stubs Pointcept's registration/container interfaces to isolate the backbone and uses real PyTorch CPU operations. Actual Pointcept integration and CUDA pointops require a separate training smoke run.
