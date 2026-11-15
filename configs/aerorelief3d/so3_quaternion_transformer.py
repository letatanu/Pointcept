# configs/s3dis/semseg-pt-v3m1-no-encoder-enhanced.py

_base_ = ["../_base_/default_runtime.py"]

"""Copy this model section into your existing Pointcept dataset/train config.

Select one ablation below. Example input feat = [RGB, normal_xyz]; adjust the
feature indices to your actual data pipeline. Raw XYZ is never a scalar input.
"""

batch_size = 32
num_worker = 24
mix_prob = 0.8
# empty_cache = True
# empty_cache_per_epoch = True
enable_amp = True
names = ["Background", "Building-Damage", "Building-No-Damage", "Road", "Tree"]
grid_size = 0.22
amp_dtype = "bfloat16"
clip_grad = 3.0
ignore_index = -1


model = dict(
    type="QuatFrameSegmentor",
    num_classes=len(names),
    backbone_out_channels=64,
    ignore_index=ignore_index,
    # Start without auxiliary supervision.
    deep_supervision=False,
    aux_loss_weight=0.2,
    freeze_backbone=False,
    backbone=dict(
        type="SO3-QuatFrame-v1",
        # Your Collect uses feat_keys=("coord", "color"): XYZRGB.
        # Only RGB enters the scalar feature stem.
        in_channels=6,
        scalar_feature_indices=(3, 4, 5),
        # Keep the existing hierarchy for a controlled comparison.
        channels=(32, 64, 128, 256, 384),
        depths=(2, 2, 2, 4, 2),
        heads=(2, 4, 8, 16, 16),
        neighbors=(8, 8, 12, 12, 16),
        sample_ratios=(0.25, 0.25, 0.25, 0.25),
        head_channels=64,
        # Scalar attention conditioned by quaternion-frame geometry.
        attention_type="grouped_vector",
        pos_channels=16,
        position_multiplier=True,
        position_bias=True,
        geometry_in_values=True,
        # Separate world-coordinate equivariant vector stream.
        vector_channels=16,
        # Transport selected K/V vector groups into the query frame.
        # Q is already expressed in the query frame.
        transport_qkv=True,
        transport_ratio=0.75,
        # Baseline frame descriptor: [delta_ij, t_ij], four channels.
        # These flags control descriptor concatenation, NOT transport.
        use_relative_quat=False,
        use_shape=False,
        # Dimensionless PCA-frame validity thresholds.
        frame_gap_eps=0.05,
        frame_skew_eps=0.05,
        # Final vector readout: norms + projected Gram entries.
        readout_triple=False,
        # Decoder.
        use_decoder=True,
        decoder_depths=(1, 1, 1, 1),
        interp_k=3,
        # Transformer blocks.
        mlp_ratio=2.0,
        attn_drop=0.0,
        proj_drop=0.0,
        # Smaller chunks limit temporary edge-tensor allocations.
        edge_chunk_size=16384,
        checkpoint_edges=True,
        share_geometry=True,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ],
)

epoch = 1000
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

dataset_type = "AeroRelief3DDataset"
data_root = "data/aerorelief3d/pointcept"

data = dict(
    num_classes=len(names),
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_3", "Area_4", "Area_5", "Area_6", "Area_7", "Area_8"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
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
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_2",
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
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_2",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
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
                    feat_keys=("coord", "color"),
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
                ]
            ],
        ),
    ),
)
