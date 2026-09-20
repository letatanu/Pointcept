# PT-QWNO redesigned configuration for AeroRelief3D.
# Reference dataset pipeline retained from the supplied configuration.

_base_ = ["../_base_/default_runtime.py"]

batch_size = 168
num_worker = 20
mix_prob = 0.8
empty_cache = False
empty_cache_per_epoch = True
enable_amp = True
amp_dtype = "bfloat16"
clip_grad = 3.0

names = [
        "Road",
        "Walkway",
        "Dirt",
        "Gravel",
        "Rock",
        "Grass",
        "Vegetation",
        "Tree",
        "Ground Obstacle",
        "Person",
        "Bicycle",
        "Vehicle",
        "Water",
        "Building",
        "Roof",
        "Cable",
        "Cable Tower",
        "Parking Lot",
        "Construction",
        "Crane",
        "Truck",
    ]
grid_size = 0.5

model = dict(
    type="DefaultSegmentorV3Redesigned",
    num_classes=len(names),
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1-QWNO-Redesigned",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        # enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(4, 8, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        drop_path=0.2,
        # Pairwise quaternion attention uses explicit attention bias.
        enable_flash=True,
        use_quaternion_rpe=True,
        # quaternion_pair_chunk_size=65536,
        # Select: haar_wno, haar_wno_cnn, fno, or none.
        context_operator="haar_wno_cnn",
        context_stages=(True, True, True, True),
        context_channels=32,
        context_grid_size=(32, 32, 32),
        haar_levels=3,
        fno_modes=4,
        # Decoder-enabled configuration.
        use_decoder=True,
        dec_depths=(2, 2, 2, 2),
        dec_channels=(48, 96, 192, 384),
        dec_num_head=(4, 8, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        head_channels=64,
        quaternion_base=10000.0,
        quaternion_learnable_frequencies=True,
        quaternion_learnable_axes=True
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 1000
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

dataset_type = "OccuFlyDataset"
data_root = "data/occufly/voxel_semantic"

common_train = [
    dict(type="RandomScale", scale=[0.5, 0.5]),
    dict(type="CenterShift", apply_z=True),
    dict(
        type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=1.0
    ),
    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.0025, clip=0.01),
    dict(type="ElasticDistortion", distortion_params=[[0.1, 0.2], [0.4, 0.8]]),
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
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
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "color")),
]

data = dict(
    num_classes=len(names),
    ignore_index=-1,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=common_train,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"), feat_keys=("coord", "color")),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="test", return_grid_coord=True),
            crop=None,
           post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)