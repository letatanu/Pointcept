# RandLA-Voxel-Context-v1 on S3DIS Area 5, WITHOUT dataset GridSample.
# Copy this file to configs/s3dis/ to resolve the base config.
# Required imports after copying the supplied companion modules:
#   pointcept/models/__init__.py:
#       from .randla_voxel_context import RandLAVoxelContextNet
#   pointcept/datasets/__init__.py:
#       from .point_fragments import PointFragments
# PointFragments keeps every test point, supplies original indices, and splits
# large rooms into spatial fragments. It does NOT voxelize or downsample.
# Training keeps one fixed-size spatial crop; validation uses complete rooms.
# No dataset RandomDropout, rate-based SphereCrop, grid_coord or inverse mapping.
# Model input/stage sampling remains enabled, with outputs restored to input points.
# Starting settings only; benchmark memory and accuracy on your target GPU.

_base_ = ["../_base_/default_runtime.py"]

batch_size = 16  # Total across GPUs; raise only after profiling raw-point memory.
num_worker = 8
mix_prob = 0.0  # Preserve room boundaries for voxel grouping.
empty_cache = False
empty_cache_per_epoch = True
clip_grad = 3.0
enable_amp = True
# spconv uses float16 CUDA autocast; do not inherit the wavelet BF16 setting.
amp_dtype = "float16"

model = dict(
    type="DefaultSegmentorV2",
    num_classes=13,
    backbone_out_channels=64,
    backbone=dict(
        type="RandLA-Voxel-Context-v1",
        # Collect concatenates coord (3), color (3), and normal (3).
        in_channels=9,
        channels=(32, 64, 128),
        depths=(1, 1, 1),
        # Sizes are measured AFTER the data transforms (including scale=0.5).
        voxel_sizes=(0.1, 0.2, 0.4),
        # Random selection preserves at least one point per occupied voxel.
        # Set to 1.0 to retain every pipeline input point at stage zero.
        input_sample_ratio=0.5,
        sample_ratios=(0.5, 0.5),
        # All retained queries remain; K limits keys INSIDE the same voxel only.
        # Options: 16, "all", or (16, 16, "all") for per-stage settings.
        # All mode has quadratic cost within a voxel; it has no silent cap.
        neighbors=(16, 16, "all"),
        geometry_width=16,
        attention_width=32,
        # None gives channel-wise RandLA-style scores; 4 shares four groups.
        attention_groups=None,
        sparse_backend="spconv",
        edge_chunk_size=32768,
        checkpoint_edges=True,
        attn_drop=0.0,
        out_channels=64,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 3000
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)

# Preserve the supplied backbone/head learning-rate groups and scheduler order.
param_dicts = [
    dict(keyword="backbone", lr=0.0001),
]

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

dataset_type = "S3DISDataset"
data_root = "data/s3dis"

# test_cfg.voxelize is a loader API slot: PointFragments performs no voxelization.
# Set its point_max=None for one whole-room test fragment (higher peak memory).
# Finite test fragments limit cross-fragment context; every point is still evaluated.
data = {
    "num_classes": 13,
    "ignore_index": -1,
    "names": [
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    "train": {
        "type": "S3DISDataset",
        "split": ("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        "data_root": "data/s3dis",
        "transform": [
            {"type": "RandomScale", "scale": [0.5, 0.5]},
            {"type": "CenterShift", "apply_z": True},
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
            {"type": "RandomJitter", "sigma": 0.0025, "clip": 0.01},
            {
                "type": "ElasticDistortion",
                "distortion_params": [[0.1, 0.2], [0.4, 0.8]],
            },
            {"type": "ChromaticAutoContrast", "p": 0.2, "blend_factor": None},
            {"type": "ChromaticTranslation", "p": 0.95, "ratio": 0.05},
            {"type": "ChromaticJitter", "p": 0.95, "std": 0.05},
            {"type": "SphereCrop", "point_max": 65536, "mode": "random"},
            {"type": "CenterShift", "apply_z": False},
            {"type": "NormalizeColor"},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "segment"),
                "feat_keys": ("coord", "color", "normal"),
            },
        ],
        "test_mode": False,
    },
    "val": {
        "type": "S3DISDataset",
        "split": "Area_5",
        "data_root": "data/s3dis",
        "transform": [
            {"type": "RandomScale", "scale": [0.5, 0.5]},
            {"type": "CenterShift", "apply_z": True},
            {"type": "CenterShift", "apply_z": False},
            {"type": "NormalizeColor"},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "segment"),
                "feat_keys": ("coord", "color", "normal"),
            },
        ],
        "test_mode": False,
    },
    "test": {
        "type": "S3DISDataset",
        "split": "Area_5",
        "data_root": "data/s3dis",
        "transform": [
            {"type": "RandomScale", "scale": [0.5, 0.5]},
            {"type": "CenterShift", "apply_z": True},
            {"type": "NormalizeColor"},
        ],
        "test_mode": True,
        "test_cfg": {
            "voxelize": None,
            "crop": None,
            "post_transform": [
                {"type": "CenterShift", "apply_z": False},
                {"type": "ToTensor"},
                {
                    "type": "Collect",
                    "keys": ("coord", "index"),
                    "feat_keys": ("coord", "color", "normal"),
                },
            ],
            "aug_transform": [
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [0],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    }
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [0.5],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    }
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [1],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    }
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [1.5],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    }
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [0],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    },
                    {"type": "RandomScale", "scale": [0.95, 0.95]},
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [0.5],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    },
                    {"type": "RandomScale", "scale": [0.95, 0.95]},
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [1],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    },
                    {"type": "RandomScale", "scale": [0.95, 0.95]},
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [1.5],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    },
                    {"type": "RandomScale", "scale": [0.95, 0.95]},
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [0],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    },
                    {"type": "RandomScale", "scale": [1.05, 1.05]},
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [0.5],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    },
                    {"type": "RandomScale", "scale": [1.05, 1.05]},
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [1],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    },
                    {"type": "RandomScale", "scale": [1.05, 1.05]},
                ],
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [1.5],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    },
                    {"type": "RandomScale", "scale": [1.05, 1.05]},
                ],
                [{"type": "RandomFlip", "p": 1}],
            ],
        },
    },
}
