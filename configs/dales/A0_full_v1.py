# PT-QWNO redesigned configuration for AeroRelief3D.
# Reference dataset pipeline retained from the supplied configuration.

_base_ = ["../_base_/default_runtime.py"]

batch_size = 12
num_worker = 24
mix_prob = 0.8
enable_amp = True
amp_dtype = "bfloat16"
clip_grad = 3.0

ignore_index = -1
names = [
    "Ground",
    "Vegetation",
    "Cars",
    "Trucks",
    "Power lines",
    "Fences",
    "Poles",
    "Buildings",
]

# Airborne LiDAR tiles are large — 0.5m voxel is a reasonable starting point
# Adjust down (e.g. 0.3) if GPU memory allows and detail is needed
grid_size = 0.3

model = dict(
    type="VNPTv2IndependentSegmentor",
    num_classes=len(names),
    backbone_out_channels=64,
    ignore_index=-1,
    deep_supervision=False,
    aux_loss_weight=0.2,
    freeze_backbone=False,
    backbone=dict(
        type="SO3-VN-PTv2-Independent-v1",
        in_channels=4,
        scalar_feature_indices=(3,),
        channels=(32, 64, 128, 256, 384),
        depths=(2, 2, 2, 4, 2),
        heads=(2, 4, 8, 16, 16),
        neighbors=(8, 8, 12, 12, 16),
        sample_ratios=(0.25, 0.25, 0.25, 0.25),
        head_channels=64,
        attention_type="grouped_vector",
        pos_channels=16,
        vector_channels=16,
        use_decoder=True,
        edge_chunk_size=16384,
        checkpoint_edges=True,
        share_geometry=True,
        vector_pooling="max",
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 3000
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="backbone", lr=0.0001)]

dataset_type = "DALESDataset"
data_root = "data/dales/pointcept"

data = dict(
    num_classes=len(names),
    ignore_index=-1,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # Airborne LiDAR: small tilt augmentation is fine but keep it minimal
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # No chromatic augmentations — DALES has no RGB
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),  # normalizes whatever is in feat_keys
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),  # strength replaces color
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                # Identity baseline
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                ],
                # One arbitrary 3D rotation: angles are multiples of pi
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0.23],
                        axis="x",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0.37],
                        axis="y",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0.61],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                ],
            ],
        ),
    ),
)
