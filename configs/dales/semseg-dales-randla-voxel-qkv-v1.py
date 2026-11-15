# Memory-oriented starting point; GPU peak and DALES accuracy are unmeasured.
# Optimizer rates/schedule retained; retune for the changed effective batch.
# DALES config in the supplied PT-v3m1 base format, using RandLA-Voxel-QKV-v3.
# Put this config in configs/dales/ and randla_voxel_qkv.py in pointcept/models/.
# Add to pointcept/models/__init__.py:
# from .randla_voxel_qkv import RandLAVoxelQKVNetV3
# Requires your existing DALESDataset registration and Pointcept runtime base.
# The centroid-aware decoder is FPT-inspired; this is not an exact FPT model.
# No GridSample, grid_coord, inverse map, or dataset grid_size is required.
# XYZ+intensity input; within-voxel point attention plus selectable voxel context.
# Options below are supported alternatives, not measured DALES improvements.

_base_ = ["../_base_/default_runtime.py"]

# Standalone DALES config for the supplied Pointcept environment.
# Requires its DALESDataset, DefaultSegmentorV2, and losses/hooks.
# epoch=3000 / eval_epoch=100 / train.loop=30 follows the supplied
# Pointcept repeated-dataset schedule; these are NOT 3000 full validation cycles.
# Batch size is global under standard Pointcept distributed launch.
# Neighbor context is enabled to implement the requested decoder; disable it
# for the smallest baseline. DALES class mapping must match your dataset.


num_worker = 20
batch_size = 34

epoch = 3000
eval_epoch = 100

clip_grad = 1.0

enable_amp = True
amp_dtype = "float16"
empty_cache = False


enable_wandb = False

mix_prob = 0.8

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

model = {
    "type": "DefaultSegmentorV2",
    "num_classes": 8,
    "backbone_out_channels": 64,
    "backbone": {
        "type": "SerializedVoxelLite",
        "in_channels": 4,
        "channels": (32, 64, 96, 128),
        "depths": (1, 1, 2, 2),
        "num_heads": (2, 4, 4, 4),
        "voxel_size": 0.3,
        "sample_ratio": 0.5,
        "patch_size": 64,
        "point_width": 16,
        "out_channels": 64,
        "ffn_ratio": 2.0,
        "neighbor_context": True,
        "checkpoint_blocks": True,
    },
    "criteria": [
        {"type": "CrossEntropyLoss", "loss_weight": 1.0, "ignore_index": -1},
        {
            "type": "LovaszLoss",
            "mode": "multiclass",
            "loss_weight": 1.0,
            "ignore_index": -1,
        },
    ],
}

epoch = 3000
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
param_dicts = [dict(keyword='backbone', lr=0.0001)]

dataset_type = "DALESDataset"

data_root = "data/dales/pointcept"

data = {
    "num_classes": 8,
    "ignore_index": -1,
    "names": [
        "Ground",
        "Vegetation",
        "Cars",
        "Trucks",
        "Power lines",
        "Fences",
        "Poles",
        "Buildings",
    ],
    "train": {
        "type": "DALESDataset",
        "split": "train",
        "data_root": "data/dales/pointcept",
        "transform": [
            {"type": "CenterShift", "apply_z": True},
            {
                "type": "RandomDropout",
                "dropout_ratio": 0.2,
                "dropout_application_ratio": 0.2,
            },
            {
                "type": "RandomRotate",
                "angle": [-1, 1],
                "axis": "z",
                "center": [0, 0, 0],
                "p": 0.5,
            },
            {
                "type": "RandomRotate",
                "angle": [-0.015625, 0.015625],
                "axis": "x",
                "p": 0.5,
            },
            {
                "type": "RandomRotate",
                "angle": [-0.015625, 0.015625],
                "axis": "y",
                "p": 0.5,
            },
            {"type": "RandomScale", "scale": [0.9, 1.1]},
            {"type": "RandomFlip", "p": 0.5},
            {"type": "RandomJitter", "sigma": 0.005, "clip": 0.02},
            {"type": "SphereCrop", "sample_rate": 0.6, "mode": "random"},
            {"type": "SphereCrop", "point_max": 102400, "mode": "random"},
            {"type": "CenterShift", "apply_z": False},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "segment"),
                "feat_keys": ("coord", "strength"),
            },
        ],
        "test_mode": False,
        "loop": 30,
    },
    "val": {
        "type": "DALESDataset",
        "split": "test",
        "data_root": "data/dales/pointcept",
        "transform": [
            {"type": "CenterShift", "apply_z": True},
            {"type": "SphereCrop", "point_max": 102400, "mode": "center"},
            {"type": "CenterShift", "apply_z": False},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "segment"),
                "feat_keys": ("coord", "strength"),
            },
        ],
        "test_mode": False,
    },
    "test": {
        "type": "DALESDataset",
        "split": "test",
        "data_root": "data/dales/pointcept",
        "transform": [{"type": "CenterShift", "apply_z": True}],
        "test_mode": True,
        "test_cfg": {
            "voxelize": None,
            "crop": {"type": "SphereCrop", "point_max": 102400, "mode": "all"},
            "post_transform": [
                {"type": "CenterShift", "apply_z": False},
                {"type": "ToTensor"},
                {
                    "type": "Collect",
                    "keys": ("coord", "index"),
                    "feat_keys": ("coord", "strength"),
                },
            ],
            "aug_transform": [[{"type": "RandomScale", "scale": [1.0, 1.0]}]],
        },
    },
}
