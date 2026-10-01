"""Copy this model section into your existing Pointcept dataset/train config.

Select one ablation below. Example input feat = [RGB, normal_xyz]; adjust the
feature indices to your actual data pipeline. Raw XYZ is never a scalar input.
"""

ablation = "final"  # "final", "deep_supervision", "equivariance", or "full"

ablation_options = dict(
    final=dict(),
    deep_supervision=dict(deep_supervision=True),
    equivariance=dict(equivariance_regularization=True),
    full=dict(
        deep_supervision=True,
        equivariance_regularization=True,
        invariance_regularization=True,
    ),
)

model = dict(
    type="DefaultSegmentorV3SO3",
    num_classes=5,
    backbone_out_channels=64,
    ignore_index=-1,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    # Weights are illustrative starting values, ordered finest -> coarsest.
    # Omitted stage weights default to 1 at every stage; sums are not normalized.
    aux_loss_weight=1.0,
    aux_stage_weights=(0.1, 0.1, 0.1, 0.1, 0.1),
    aux_criteria=None,  # None reuses the final CE + Lovasz criteria.
    equivariance_weight=0.01,
    invariance_weight=0.01,
    consistency_stage_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
    consistency_detach_target=False,
    backbone=dict(
        type="SO3-QuatTransport-v1",
        in_channels=6,
        scalar_feature_indices=(0, 1, 2),
        vector_feature_groups=((3, 4, 5),),
        enc_depths=(2, 2, 2, 4, 2),
        scalar_channels=(32, 64, 128, 256, 512),
        vector_channels=(16, 32, 64, 128, 256),
        num_heads=(2, 4, 8, 16, 16),
        neighbors=(16, 16, 24, 24, 32),
        sample_ratios=(0.25, 0.25, 0.25, 0.25),
        use_decoder=False,
        head_channels=64,
    ),
    **ablation_options[ablation],
)

# Independent invariance-only ablation:
# model.update(invariance_regularization=True)  # with ablation = "final"
# Full without invariance:
# model.update(invariance_regularization=False)  # with ablation = "full"
# CE-only auxiliary heads while retaining final CE + Lovasz:
# model.update(aux_criteria=[dict(
#     type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1,
# )])
