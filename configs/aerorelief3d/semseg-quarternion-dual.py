# PT-QWNO redesigned configuration for AeroRelief3D.
# Reference dataset pipeline retained from the supplied configuration.

_base_ = ["../_base_/default_runtime.py"]

batch_size = 36
num_worker = 24
mix_prob = 0
empty_cache = False
enable_amp = False
amp_dtype = "bfloat16"
clip_grad = 3.0

ignore_index = -1
names = ["Background", "Building-Damage", "Building-No-Damage",  "Road", "Tree"]
grid_size = 0.2

model = dict(
    type="SO3QuaternionDualSegmentor",
    num_classes=len(names),
    backbone_out_channels=64,
    ignore_index=-1,
    backbone=dict(
        type="SO3-Quaternion-Dual-v1",
        in_channels=3,
        scalar_feature_indices=(0,1,2),  # Intensity
        # vector_feature_indices=((3, 4, 5),),
        vector_feature_indices=None,
        channels=(48, 96, 192, 384),
        vector_channels=(8, 8, 16, 16),
        depths=(2, 2, 4, 4),
        heads=(3, 6, 8, 12),
        radii=(0.4, 0.8, 1.6, 3.2),  # tune against physical density and object sizes
        patch_sizes=(32, 64, 64, 128),
        max_neighbors=(24, 24, 32, 32),  # messages only, never coverage graph
        quaternion_channels=4,
        geometry_groups=4,
        geometry_width=32,
        decoder_depths=(1, 1, 1),
        decoder_scalar_attention=False,
        use_scalar_attention=True,
        use_quaternion_attention=True,
        global_context_tokens=8,
        head_channels=64,
        mlp_ratio=2.,
        attn_drop=0.,
        proj_drop=0.,
        edge_chunk_size=32768,
        patch_chunk_size=64,
        checkpoint_edges=True,
        radius_backend='hash',
        radius_query_chunk=128,
        radius_candidate_limit=2_000_000,
        max_radius_edges=8_000_000,
        readout_triple=True,
    ),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1., ignore_index=ignore_index),
        dict(type='LovaszLoss', mode='multiclass', loss_weight=1., ignore_index=ignore_index),
    ],
)

epoch = 1000
eval_epoch = 100

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

param_dicts = [dict(keyword="block", lr=0.0001)]

dataset_type = "AeroRelief3DDataset"
data_root = "data/aerorelief3d/pointcept"

data = dict(
    num_classes=len(names),
    ignore_index=-1,
    names=names,
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_3", "Area_4", "Area_5", "Area_6", "Area_7", "Area_8"),
        data_root=data_root,
        transform=[
            # Creates point_id before coordinate-independent subsampling.
            # Mean centering; no world-Z alignment.
            dict(
                type="SO3Prepare",
                max_points=8192,
                center=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "point_id"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="Area_2",
        data_root=data_root,
        transform=[
            # Keeps every point: no inverse mapping is required.
            dict(type="SO3Prepare", max_points=8192, center=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "point_id"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="Area_2",
        data_root=data_root,
        transform=[
            dict(type="SO3Prepare", center=True),
        ],
        test_mode=True,
        test_cfg=dict(
            # One identity fragment per input block.
            # Raises an error for oversized blocks rather than dropping points.
            voxelize=dict(
                type="SO3IdentityPartition",
                max_points=16384,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index", "point_id"),
                    feat_keys=("color",),
                ),
            ],
            aug_transform=[[]],  # Single identity pass; no rotation voting.
        ),
    ),
)