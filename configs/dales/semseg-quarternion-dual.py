# Full-coverage spatial-block training for the SO3 DALES network.
# Install as configs/dales/semseg-so3-dual-blocks-dales.py in Pointcept.
# Requires so3_block_dataset.py, so3_transforms.py and the matching model.
# Experimental training recipe; no PTv3-equivalent accuracy is claimed.

_base_ = ["../_base_/default_runtime.py"]

# Total across GPUs; four full-coverage spatial blocks per step.
batch_size = 28
batch_size_val = 1
batch_size_test = 1
seed = 42
weight = None
resume = False
evaluate = True
enable_wandb = False
num_worker = 8
mix_prob = 0
empty_cache = False
enable_amp = False
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

# Coordinate units must match these radii (normally meters).
# No GridSample transform is used; local crops preserve the source density.
train_points = 16384
val_points = 16384
test_points = 16384

model = dict(
    type="SO3QuaternionDualSegmentor",
    debug_numerics=False,  # Re-enable if non-finite losses recur.
    num_classes=len(names),
    backbone_out_channels=64,
    freeze_backbone=False,
    deep_supervision=True,
    aux_loss_weight=0.2,
    aux_stage_weights=(1.0, 0.5, 0.25, 0.125),
    ignore_index=-1,
    backbone=dict(
        type="SO3-Quaternion-Dual-v1",
        in_channels=1,
        scalar_feature_indices=(0,),  # Intensity
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
        decoder_scalar_attention=True,
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

# One data pass per displayed epoch with the standard Pointcept runtime.
epoch = 100
eval_epoch = 100
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.001,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
# Train the full network at one LR; no frozen or reduced-LR backbone group.
param_dicts = None

dataset_type = "DALESDataset"
data_root = "data/dales/pointcept"

data = dict(
    num_classes=len(names),
    ignore_index=-1,
    names=names,

    train=dict(
        type="SO3BlockDataset",
        source=dict(type=dataset_type, split="train", data_root=data_root,
                    transform=[], test_mode=False, loop=1),
        cache_dir="data/dales/so3_block_cache/train",
        max_points=train_points,
        # Match total batch_size so drop_last=True cannot discard unique blocks.
        pad_to_multiple=batch_size,
        transform=[
            dict(type="SO3Prepare", max_points=None, center=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment", "point_id"),
                 feat_keys=("strength",)),
        ],
        test_mode=False,
    ),
    val=dict(
        type="SO3BlockDataset",
        source=dict(type=dataset_type, split="test", data_root=data_root,
                    transform=[], test_mode=False, loop=1),
        cache_dir="data/dales/so3_block_cache/val",
        max_points=val_points,
        pad_to_multiple=1,
        transform=[
            dict(type="SO3Prepare", max_points=None, center=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment", "point_id"),
                 feat_keys=("strength",)),
        ],
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="SO3Prepare", center=True),
        ],
        test_mode=True,
        test_cfg=dict(
            # Spatial fragments cover the full scene and preserve original indices.
            voxelize=dict(
                type="SO3SpatialPartition",
                max_points=test_points,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index", "point_id"),
                    feat_keys=("strength",),
                ),
            ],
            aug_transform=[[]],  # Single identity pass; no rotation voting.
        ),
    ),
)