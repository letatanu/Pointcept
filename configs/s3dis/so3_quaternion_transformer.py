# configs/s3dis/semseg-pt-v3m1-no-encoder-enhanced.py

_base_ = ["../_base_/default_runtime.py"]

"""Copy this model section into your existing Pointcept dataset/train config.

Select one ablation below. Example input feat = [RGB, normal_xyz]; adjust the
feature indices to your actual data pipeline. Raw XYZ is never a scalar input.
"""

ablation = "equivariance"  # "final", "deep_supervision", "equivariance", or "full"

batch_size = 4
num_worker = 24
mix_prob = 0.8
empty_cache = True
empty_cache_per_epoch = True
enable_amp = True

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
    num_classes=13,
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
        neighbors=(8, 8, 12, 12, 16),
        sample_ratios=(0.25, 0.25, 0.25, 0.25),
        use_decoder=False,
        head_channels=64,
    ),
    **ablation_options[ablation],
)

epoch = 3000
eval_epoch = 100

optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

param_dicts = [dict(keyword="block", lr=0.0006)]

dataset_type = "S3DISDataset"
data_root = "data/s3dis"

data = dict(
    num_classes=13, 
    ignore_index=-1,
    names=["ceiling", "floor", "wall", "beam", "column", "window", "door",
           "table", "chair", "sofa", "bookcase", "board", "clutter"],
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=40960, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment"),
                 feat_keys=("color", "normal")),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type, split="Area_5", data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                 feat_keys=("color", "normal")),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type, split="Area_5", data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", 
                          grid_size=0.02, hash_type="fnv", mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect",
                     keys=("coord", "grid_coord", "index"),
                     feat_keys=("color", "normal")),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[1.0, 1.0])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
            ],
        ),
    ),
)