# Serialized Local + Cross Attention

Convolution-free encoder: pointwise residual MLP -> full local patch attention
with residual -> optional sampled cross-patch attention with residual.
No spconv, neighboring-cell attention, additive point-position MLP or cell-offset
embedding in this version. Geometry enters attention through independently
configurable quaternion rotary encoding. Grid coordinates still define pooling.
Raw coordinate channels remain part of the input features as in prior configs.

## Pointcept setup

Copy this folder into `pointcept/models/serialized_local_cross/` and add:

```python
from .serialized_local_cross import adapter
```

to `pointcept/models/__init__.py`. Copy the desired config into configs/dales/ and
use your usual launcher with one GPU. Registry: SerializedLocalCross-v1.
Transforms have SLC prefixes. Refresh experiment/code snapshots before training.
Requires PyTorch 2.3+ and a compatible flash-attn 2.x CUDA installation; no spconv
dependency. AMP is recommended. FP32 evaluation is handled by internally casting
only FlashAttention Q/K/V to BF16 if supported, otherwise FP16, then restoring
the output dtype. The backend never silently falls back to a per-cell Python loop.

Default config uses batch=4, raw point cap=65,536, original DALES pipeline/losses
and optimizer. Do not immediately assume batch=12 with 204,800 raw points fits.
Use a fresh run: architecture, pooling and positional treatment changed together
from the previous hybrid. This is a new architecture baseline, not a single-factor
comparison against that hybrid. No previous checkpoint compatibility is claimed.
Input coord (N,3), feat (N,4), offset or sorted batch. Output features (N,64) go
to DefaultSegmentorV2's 8-class head. All input point outputs preserve ordering.

## Local and cross attention

Each stage serializes using a selected named spatial order and forms patches of up to
128 tokens. Tail patches are unpadded, without duplicated tokens. Local attention
uses all tokens as Q/K/V. Named orders follow PTv3:

```python
order = ("z", "z-trans", "hilbert", "hilbert-trans")
shuffle_orders = True
serialization_backend = "pointcept"
```

The default backend calls `pointcept.models.utils.serialization.encode` directly.
Both transposed variants swap X/Y. In PTv3's Morton convention X occupies the
highest bit in each XYZ triplet, unlike the previous legacy axis-only helper.
Reference PyTorch encoders are included for standalone correctness checks;
set serialization_backend='reference' outside Pointcept. They are not optimized
GPU kernels. Named orders use PTv3's adaptive-depth convention and depth<=16 limit.
Increase serialization_resolution if the quantized cloud exceeds that limit.
Stable sorting makes duplicate-grid ties deterministic; upstream default tie
sorting need not choose the same order. Raw points sharing coordinates are kept.

Each stage shuffles the order list during training; evaluation uses a fixed list
for repeatability. Block order index is (stage+block) modulo list length, retaining
our previous stage offset. This is PTv3-compatible code generation, NOT an exact
copy of PTv3's whole grouping/shuffle/pooling schedule. Local and cross attention
use the same selected patch partition within a block. Old explicit axis_orders
configs are still supported as legacy Morton-only mode; omit that argument to
enable named orders. Patch layouts are cached per used order within each stage.

Upstream references:
https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/utils/serialization/default.py
https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/utils/structure.py

Every SECOND block within each stage also performs cross-patch attention
(`cross_every=2`; set 1 for every block). With depths=(2,2,3,2), this creates nine
local modules and four cross modules. Sparse CNN is entirely absent, including
the stem. The stem is a pointwise MLP; each block's FFN is at the beginning.

Cross groups contain two patches by default. Dilation alternates (1,2) over
enabled cross blocks across stages. G=2,D=1 groups [0,1],[2,3],... within each
cloud. G=2,D=2 groups [0,2],[1,3],[4,6],[5,7],... . Each patch belongs to exactly
one group per cross block. Incomplete tail groups may have only one patch;
their operation is then sampled within-patch attention, not actual cross-patch
interaction. Dilation is in patch index, not physical distance in metres.

Defaults M=16 total queries per group, K=32 keys per member patch (up to 64 keys).
Query budget is split across member patch slots; with two full patches it is
8+8. At least one query per occupied member patch requires M>=G, enforced.
Short patches/tail groups can use fewer queries; unused quotas are not reallocated.
Train samples one random token in each disjoint serialized-index stratum; eval
uses midpoint tokens. This is stratified random sampling, not uniform subset
sampling or guaranteed uniform physical coverage. Queries/keys are unique, but
a selected query need not also be a selected key. Members of its own patch ARE
allowed as cross keys. Unselected points keep their local attention features.

Sampling and attention groups stay inside each cloud. Packed Q/K/V contain only
valid tokens. No key is repeated across groups in the same cross block. Local
and cross are distinct operations with distinct learned projections.

## Rotary ablations

Modes `none`, `qk`, `qkv` independently configure local/cross attention:

| Config | Local | Cross |
|---|---|---|
| config.py | none | qk |
| config_no_rotary.py | none | none |
| config_local_qk.py | qk | none |
| config_both_qk.py | qk | qk |
| config_cross_qkv.py | qk | qkv |

Other settings are matched; output directories are distinct. QKV mode rotates
values with their key coordinates, then inverse-rotates each aggregated output
with the QUERY rotor before output projection/residual. It therefore has a
relative value pathway. It is not just rotating V and leaving the output rotated.
Fixed axes/frequencies and FP32 rotation math are reused from the previous
quaternion implementation. Physical coordinates divided by per-stage
position_scales are used, with a shared per-cloud origin. No physical rotation
equivariance or whole-network translation invariance is claimed.

## Pooling and decoder

Grid sizes (0.4,0.8,1.6)m define three reductions between four stages. A small
score network reads LayerNorm(Cin) features concatenated with cell-relative
coordinates, producing one scalar per point. Cell softmax weights form a weighted
sum of ORIGINAL Cin-dimensional features. Only then does an MLP project Cin to
Cout. There is no per-point Cout value projection, explicit mean/max feature
fusion, or score mass correction. All points participate, including unselected
cross queries. Coordinate centroids remain mass-weighted; mass tracks original
point representation. One token per cell is retained after pooling.
Fine-stage features/inverses are saved. Decoder projects coarse features, lifts
by parent cell, performs gated skip fusion, then a residual pointwise MLP.
No decoder attention/convolution. The backbone head produces 64 features.

## Memory, complexity and checks

For N tokens, patch width P, G patches/group, M queries/GROUP and K keys/PATCH,
score counts are approximately NP locally and NMK/P for each enabled cross block.
Channel projection/FFN costs, sorting, sampling, pooling and decoder remain.
No explicit pairwise positional tensor is allocated. Packed FlashAttention avoids
materializing full score matrices, but token activations still require memory.
MLP and attention residual scales use activation dtype to avoid accidental FP32
promotion under AMP. Weighted pooling accumulates FP32 in chunks. Chunking limits
temporary buffers; it does not discard saved backward graphs.

`checkpoint_blocks=True` recomputes encoder blocks during backward while preserving
sampling RNG state; it trades runtime for memory. Saved decoder skips remain.
Debug mode returns layouts and intentionally bypasses checkpointing. Some grouping
and dynamic compaction operations synchronize CUDA; this is not a fused custom
serialization/sampling implementation. Performance on your H100 is unverified.

From the directory containing this package:

```text
python -m unittest discover -s serialized_local_cross -t . -v
python -m serialized_local_cross.benchmark --points 65536 --batch 4
python -m serialized_local_cross.benchmark --points 204800 --batch 1 --checkpoint
```

The benchmark uses synthetic backbone-only loss, no real DALES loading or final
segmentation classifier; it is a memory smoke test, not an accuracy benchmark.
It warms up optimizer state, synchronizes timing, reports peak allocated/reserved
memory, and also checks evaluation without autocast. Start small and increase batch
size gradually. It will not establish full training memory requirements.

Tests verify exact dilation partitions, cloud isolation, unique bounded samples,
all-point local coverage, backward gradients for all five rotary settings,
checkpoint equality including random sampling, pooled weight normalization and
post-pooling projection shape, singleton cases, chunk equivalence, relative QKV
value transport, and optional CUDA AMP/FP32 evaluation. Reference tests can run
without flash-attn by setting attention_backend='reference'; that slow backend
is for correctness only. Also set serialization_backend='reference' for tests
outside Pointcept. Tests exercise all four order names, known Morton/Hilbert
codes, Hilbert continuity and coverage/cloud separation. Optional exact code
parity runs if Pointcept is available; otherwise that test skips. GPU test skips
if flash-attn/CUDA is unavailable.
