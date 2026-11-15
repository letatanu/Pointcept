# SO(3) loss ablations

Replace your existing `so3_quaternion_transformer.py` with the supplied file.
Keep its existing package imports: the registrations `SO3-QuatTransport-v1`
and `DefaultSegmentorV3SO3` are unchanged. No trainer modification is required;
Pointcept should backpropagate `output["loss"]` as usual.

`ablation_configs.py` supplies a model section to copy into your dataset/training
config. Set `ablation` to `final`, `deep_supervision`, `equivariance`, or `full`.
The full example enables both vector equivariance and invariant-feature
consistency. Each is independently switchable. All switches default to false.

| Mode | Changes to an existing model config |
|---|---|
| Final only | No additional flags |
| Deep supervision | `deep_supervision=True` |
| Equivariance | `equivariance_regularization=True` |
| Full | All three flags true, including `invariance_regularization=True` |

## Loss and correspondence

```
loss = final_criteria
     + aux_loss_weight * sum_s(aux_stage_weights[s] * aux_criteria_s)
     + equivariance_weight * sum_s(consistency_stage_weights[s] * vector_MSE_s)
     + invariance_weight * sum_s(consistency_stage_weights[s] * invariant_MSE_s)
```

Only enabled terms are included. Stage weights run from the finest encoder stage
(stage 0, before downsampling) to the coarsest; their lengths must match
`enc_depths`. Zero weights skip those losses. Auxiliary criteria default to the
final criteria and may be overridden with `aux_criteria`. Use the same
`ignore_index` in the wrapper and every configured criterion. All-ignored
segmentation targets produce differentiable zero losses.

Every state carries `sample_index`, the row indices into the original packed
input batch. Downsampling composes these indices through FPS; auxiliary labels
are exactly `segment[state.sample_index]`, including across batch boundaries.
Auxiliary heads consume `[scalar, vector_norms]` at each encoder stage.

Consistency adds one rotated **encoder** pass per training batch, with one
uniformly sampled proper rotation shared across its point clouds. Coordinates
and all configured `vector_feature_groups` rotate; scalar input channels stay
unchanged. The second pass reuses the original FPS indices and dropout random
draws. kNN is recomputed, so floating-point changes near distance ties can still
contribute to the loss. Replayed FPS guarantees correspondence but does not test
the rotation stability of an independently resampled FPS hierarchy.

Vector MSE compares `V_rotated` to `V_original @ R.T`. Invariant MSE compares
`[S_rotated, ||V_rotated||]` to `[S_original, ||V_original||]`. Each MSE averages
over all point, channel, and component entries, including unlabeled points;
stage sums are not normalized. Both branches receive gradients unless
`consistency_detach_target=True`. A frozen backbone supports auxiliary head
training but rejects active consistency regularization, which could not update
it. Exact equivariant features should already yield nearly zero regularization;
these terms do not guarantee an accuracy improvement.

## Compatibility and checks

- Evaluation/inference uses only the final head, with the original output keys
  (`seg_logits`, optional `loss` and `point`); no auxiliary or rotated pass runs.
- With all additions disabled there are no new checkpoint parameters. To load an
  old checkpoint with auxiliary heads enabled, allow the missing `aux_heads.*`
  keys. The normal backbone call still returns a `Point`; explicit
  `return_stages=True` returns `(point, encoder_states)`.
- Inputs are not mutated. CPU execution uses the fallback even if CUDA pointops
  is installed. Quaternion rotation now promotes mixed input dtypes to prevent
  the existing autocast cross-product error.
- `loss_final`, `loss_aux_stage_*`, `loss_aux`, `loss_equivariance`, and
  `loss_invariance` are detached diagnostics when their terms are active.
  Aggregate diagnostics include stage weights but exclude their global weights.
  Backpropagate only `loss`; do not sum every diagnostic key.
- `python test_so3_ablation.py` passes CPU forward/backward checks for all four
  modes with both output-head variants, label mapping, ignored targets, disabled
  weights, rotation/dropout consistency, a deliberately broken rotation,
  frozen-backbone behavior, evaluation, and CPU bfloat16 autocast. These tests
  use minimal Pointcept interface doubles; CUDA pointops and a full Pointcept
  training run remain unverified in this CPU-only environment.
