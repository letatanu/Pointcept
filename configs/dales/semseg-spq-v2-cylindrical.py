# Independent serialized sampled-point model. No PTv3 backbone or GridSample.
_base_ = ["../_base_/default_runtime.py"]

embedding_mode = "cylindrical"  # alternatively spherical; or use the inherited config
# patch: fixed-count serial groups. grid: metric cells AFTER serialization.
grouping = "patch"
data_root = "data/dales/pointcept"
strength_scale = 1.0
# Compatibility with your supplied train/test folder layout. For paper tuning,
# create a held-out train-tile val split and edit this field to "val".
validation_split = "test"
point_max = 204800
batch_size = 52  # global; must be divisible by GPU count
num_worker = 20
mix_prob = 0.0
empty_cache = False
empty_cache_per_epoch = False
enable_amp = True
amp_dtype = "float16"  # switch to bfloat16 on a supporting GPU if desired
clip_grad = 1.0
seed = 3407
find_unused_parameters = False
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
    type="DefaultSegmentorV2",
    num_classes=8,
    backbone_out_channels=64,
    backbone=dict(
        type="SerializedPatchPQ-v2",
        in_channels=4,
        out_channels=64,
        channels=(48, 96, 192, 256),
        depths=(2, 2, 3, 2),
        edge_channels=(32, 48, 64, 64),
        patch_sizes=(64, 64, 64, 32),
        pool_strides=(4, 4, 4),
        samples_per_group=(2, 2, 2, 2),
        k_each_side=(8, 8, 8, 8),  # eight preceding + self + eight following
        stage_scales=(0.5, 1.0, 2.0, 4.0),
        max_neighbor_distances=(2.0, 4.0, 8.0, 16.0),  # meters; None disables a guard
        serialization_resolution=0.05,  # keys ONLY; does not remove raw points
        axis_orders=((0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0)),
        grouping=grouping,
        grid_pool_sizes=(1.0, 2.0, 4.0, 8.0),
        embedding_mode=embedding_mode,
        use_quaternion=True,
        learned_skip=True,
        ffn_expansion=2.0,
        residual_scale=0.1,
        checkpoint_blocks=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Keep your original repeated-data schedule, one consistent optimizer group.
epoch = 3000
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
param_dicts = None
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
dataset_type = (
    "DALESDataset"  # use your existing dataset registration and label mapping
)

# val is a bounded center-crop diagnostic, not a full-tile benchmark metric.
# Final test uses full-coverage raw-point fragments with context halos.
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
            {"type": "SPQPrepareStrength", "scale": 1.0},
            {"type": "CenterShift", "apply_z": True},
            {
                "type": "RandomRotate",
                "angle": [-1, 1],
                "axis": "z",
                "center": [0, 0, 0],
                "p": 0.5,
            },
            {"type": "RandomScale", "scale": [0.9, 1.1]},
            {"type": "RandomFlip", "p": 0.5},
            {"type": "SphereCrop", "point_max": point_max, "mode": "random"},
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
    "val": {
        "type": "DALESDataset",
        "split": "test",
        "data_root": "data/dales/pointcept",
        "transform": [
            {"type": "SPQPrepareStrength", "scale": 1.0},
            {"type": "CenterShift", "apply_z": True},
            {"type": "SphereCrop", "point_max": point_max, "mode": "center"},
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
        "transform": [
            {"type": "SPQPrepareStrength", "scale": 1.0},
            {"type": "CenterShift", "apply_z": True},
        ],
        "test_mode": True,
        "test_cfg": {
            "voxelize": {"type": "SPQIdentityFragments"},
            "crop": {
                "type": "SPQSpatialTestCrop",
                "point_max": point_max,
                "halo_fraction": 0.125,
                "halo_distance": 2.0,
            },
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
for split in ("train", "val", "test"):
    data[split]["type"] = dataset_type
    data[split]["data_root"] = data_root
    for transform in data[split]["transform"]:
        if transform["type"] == "SPQPrepareStrength":
            transform["scale"] = strength_scale
        elif transform["type"] == "SphereCrop":
            transform["point_max"] = point_max
data["val"]["split"] = validation_split
data["test"]["test_cfg"]["crop"]["point_max"] = point_max
