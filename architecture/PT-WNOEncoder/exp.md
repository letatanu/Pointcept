# 3D Feature Extraction Directions for PT-WNO

## Context

Current architecture: **PT-WNO encoder-only** with WNO as a global branch, QuatRPE for
positional embeddings, and a lightweight upsample head (no decoder).

Key observation: Removing color features only drops mIoU from ~71.3% → 70.04%, confirming
the "geometric shortcut" problem — the model over-relies on geometry and underutilizes
appearance cues (color, normals).

Goal: Deepen 3D feature extraction without resorting to heavy 2D lifting (e.g., DINOv2).

---

## Direction 1 — Separate RGB & Normal Encoding Streams

**Problem:** Flat concatenation of `(X,Y,Z,R,G,B,Nx,Ny,Nz)` into `Embedding` gives each
modality no dedicated gradient path.

**Idea:** Use a `MultiModalEmbedding` module with separate lightweight encoders per modality
(geometry, color, normals), then fuse via linear projection.

**Expected benefit:** Each modality learns its own representation before being mixed,
preventing geometry from dominating the early layers.

**Complexity:** Low — replaces the first `Embedding` layer only.

---

## Direction 2 — Normal-Aware QuatRPE ⭐ (Recommended First)

**Problem:** Current `QuatRPE` encodes *where* a point is (position), but not *how its
surface faces* (orientation). Surface normals are the strongest cue for structural boundaries
(wall vs. floor vs. ceiling).

**Idea:** Extend `QuatRPE` to build a **second quaternion from the surface normal vector**
(as rotation axis), then fuse both quaternions (position + orientation) into the RPE.

**Change:** Minimal — extend `QuatRPE.__init__` with a larger input dim and update
`forward()` to consume `point.normal`. Drop-in replacement for the current `QuatRPE` in
both `NOGlobalBranch` and `QuatRPEBlockWrapper`.

**Expected benefit:** The attention bias in every transformer block now encodes surface
orientation, helping the model distinguish co-located but differently-oriented surfaces.

---

## Direction 3 — Color Grid in NOGlobalBranch

**Problem:** The WNO/FNO operates on a geometry-only voxel grid; spatially coherent color
patterns (e.g., brown floor, white walls) are never seen globally.

**Idea:** Build a parallel **color voxel grid** alongside the feature grid inside
`NOGlobalBranch.forward()`, project it with a lightweight `Conv3d(3→C)`, and add it to the
feature grid before the WNO operator.

**Expected benefit:** The wavelet operator can now capture room-scale color layout.

**Complexity:** Low — ~5 lines added inside `NOGlobalBranch.forward()`.

---

## Direction 4 — HSV Color Space + Local Color Histogram

**Problem:** RGB conflates illumination (brightness) with true surface color. A dark brown
floor and a shadow on a light floor can have similar RGB values.

**Idea:** Convert colors to HSV, then build a per-point **soft color histogram** over
K-nearest neighbors to capture local color texture.

**Expected benefit:** Decouples illumination from appearance; local histograms capture
texture (smooth walls vs. cluttered tables).

**Complexity:** Medium — requires KNN lookup per point; feasible at input preprocessing step.

---

## Direction 5 — Normal Consistency Auxiliary Loss

**Problem:** The cross-entropy loss has no signal to encourage consistent predictions on
co-planar surfaces.

**Idea:** Add an auxiliary loss that penalizes adjacent points with similar normals
(= co-planar) but different predicted labels:
`L_aux = Σ_{(i,j)} sim(n_i, n_j) · 1[pred_i ≠ pred_j]`

**Expected benefit:** Forces the model to respect surface orientation during training
without changing architecture.

**Complexity:** Low — loss-only change, no new modules.

---

## Priority Order

| Priority | Direction | Effort | Expected Gain |
|----------|-----------|--------|---------------|
| 1 | Normal-Aware QuatRPE | Low | High (structural boundaries) |
| 2 | Color Grid in NOGlobalBranch | Low | Medium (room-scale appearance) |
| 3 | Separate Modality Streams | Low | Medium (early fusion quality) |
| 4 | Normal Consistency Loss | Low | Medium (training signal) |
| 5 | HSV + Color Histogram | Medium | Medium (illumination robustness) |