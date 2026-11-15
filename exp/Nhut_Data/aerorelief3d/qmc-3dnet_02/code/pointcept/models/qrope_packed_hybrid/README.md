# Quaternion rotary positional ablation

Packed hybrid with fixed-axis quaternion rotary encoding added to its existing
point-position MLP and 27-entry cell-offset embeddings. Sparse CNN, sampling,
pooling and decoder are unchanged. No serialization or spherical coordinates.

## Install

Copy this folder to `pointcept/models/qrope_packed_hybrid/` and add:

```python
from .qrope_packed_hybrid import adapter
```

to `pointcept/models/__init__.py`. Registry type: QuatRoPEPackedHybrid-v1.
New QPHCQ-prefixed transforms coexist with earlier packages.
Requires the same compatible CUDA spconv and flash-attn 2.x packages as the
packed baseline. AMP FP16/BF16 is recommended; the FP32 validation attention
casting fix is included. Attention does not use cuDNN SDPA.

Use config.py for quaternion rotary. Copy it to your DALES config directory and
launch with your usual Pointcept training command. Use config_no_rope.py for a
matched baseline. Both use batch=4, max_queries=(8,8,16,16), keys_per_cell=(4,4,4,4),
depths=(2,2,3,2), original widths/optimizer/data, and distinct save directories.
These do not inherit any later local depth/key experiments. Start fresh for the
ablation. Ensure active experiment/code snapshots include this package.

Backbone options:

```python
rotary_encoding = "quaternion"  # or "none"
rotary_min_frequency = 0.25
rotary_max_frequency = 2.0
```

## Definition

Each attention head has 16 channels, split into four scalar-first quaternions.
For head/group h,g: theta_i = omega[h,g] dot (coord_i / stage_cell_size).
r_i = [cos(theta_i/2), axis[h,g] * sin(theta_i/2)].
Rotated Q_i = r_i * Q_i and K_j = r_j * K_j using Hamilton LEFT multiplication.
V is unchanged. This is not quaternion conjugation of features or vector normals.
Axes are fixed unit vectors and frequencies fixed deterministic buffers, shared
across points. They consume no RNG and add no learned parameters. Spatial phase
directions cover a deterministic spherical set; group frequency magnitudes are
log-spaced from 0.25 to 2 radians per normalized coordinate unit.

With a shared axis per group, r_i^-1*r_j depends on coord_j-coord_i. The rotary
part of Q/K inner products thus has relative-position structure. Existing raw
coordinate features, additive geometry and axis-aligned grouping mean the FULL
network is not claimed translation-invariant or physically rotation-equivariant.
Do not interpret latent quaternion rotations as recovered physical orientations.

Coordinates use the backbone's single per-cloud origin, NOT point-to-cell
offsets. Scale by query-cell size at each stage. Rotary is applied AFTER the
point-position additions and (for keys) AFTER the neighboring-cell embedding.
This matters because that embedding depends on the query window.

Q rotations and sampled-source K rotors are cached per block. Each gathered key
still needs Hamilton multiplication after its window-specific cell embedding;
its trig terms are reused. FP32 phase/trig/multiplication then cast back to the
attention dtype. This adds linear token/gather work, not a pairwise bias tensor,
and retains the packed FlashAttention API. No speed or accuracy improvement is
claimed until measured on the target GPU.

## Tests

From the directory containing this folder:

```text
python -m unittest discover -s qrope_packed_hybrid -t . -v
```

Reference model tests, packed-attention tests and rotary tests cover norm and
zero-position identity, displacement inner products, shared-translation scores,
gradients, chunk/batch equivalence, and disabled-rotary baseline agreement.
test_rotary's original-model comparison uses the prior packed_hybrid_cell_query
package beside this folder and skips that check if absent; other tests stand alone.
The GPU FlashAttention and spconv integration tests require installed packages.

Keep the same training budget, key/query settings and test voting for comparison.
Record latency and peak allocated memory along with IoU. These configs isolate
quaternion rotary as the only backbone behavior change; frequencies/axes should
be separate follow-up ablations, not changed simultaneously with network depth.
