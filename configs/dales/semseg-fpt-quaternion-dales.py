# Import so3_fpt_quaternion after the supporting model modules.
# Experimental voxel/quaternion model with full-scene DALES inputs.
# Install as configs/dales/semseg-voxel-quaternion-dales.py in Pointcept.
# Requires so3_transforms.py and both model modules.
# Experimental training recipe; no PTv3-equivalent accuracy is claimed.

_base_ = ["../_base_/default_runtime.py"]

# Start with one full scene per step on one GPU.
batch_size = 34
seed = 42
enable_wandb = False
num_worker = 20
mix_prob = 0  # Keep full scenes separate; avoid mixing their voxel coordinates.
empty_cache = False
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


model = dict(
    type="FPTQuaternionSegmentor",
    point_chunk=32768,
    debug_numerics=False,  # Re-enable if non-finite losses recur.
    num_classes=len(names),
    backbone_out_channels=32,
    freeze_backbone=False,
    deep_supervision=False,
    aux_loss_weight=0.2,
    aux_stage_weights=(1.0, 0.5, 0.25, 0.125),
    ignore_index=-1,
    backbone=dict(
        type="FPT-SO3-Quaternion-v1",
        in_channels=1,
        scalar_feature_indices=(0,),
        channels=(32, 64, 128, 192),
        depths=(1, 1, 2, 2),
        voxel_sizes=(0.4, 0.8, 1.6, 3.2),
        radii=(0.8, 1.6, 3.2, 6.4),
        qk_channels=16,  # Four quaternions per Q/K, each [real,x,y,z].
        vector_channels=4,
        head_channels=32,
        query_chunk=4096,
        checkpoint_blocks=True,
        diagnostics=False,  # Logs input points, voxel counts and edges per stage.
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ],
)

# One data pass per displayed epoch with the standard Pointcept runtime.
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.001,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
# Train the full network at one LR; no frozen or reduced-LR backbone group.
param_dicts = None

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
            dict(type="SO3Prepare", max_points=None, center=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "point_id"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
        loop=1,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="SO3Prepare", max_points=None, center=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "point_id"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
        loop=1,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="SO3Prepare", max_points=None, center=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=None,  # Full scene; model voxelizes internally
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index", "point_id"),
                    feat_keys=("strength",),
                ),
            ],
            aug_transform=[[]],
        ),
    ),
)
