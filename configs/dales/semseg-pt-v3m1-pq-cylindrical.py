# Experimental DALES config, based on the supplied semseg-pt-v3m1-0-base.py.
# Install the model package and its registry import before loading this config.
_base_ = ["../_base_/default_runtime.py"]

# Change this to "spherical", or use model.backbone.embedding_mode via --options.
embedding_mode = "cylindrical"
# Both GridSample and the backbone must use the same base grid size.
grid_size = 0.5
# Set to the actual divisor for your preprocessed intensity; 1 means already scaled.
strength_scale = 1.0
# Preserves the supplied train/test layout. This uses test labels for diagnostics.
# For research: create a held-out train-tile "val" split, then set this to "val";
# tune/checkpoint on val and run final evaluation on test only once.
validation_split = "test"

batch_size = 24  # global batch size; must be divisible by GPU count
num_worker = 20
mix_prob = 0.0  # preserve each cloud's radius neighborhood and spatial extent
empty_cache = False
empty_cache_per_epoch = True
enable_amp = True
amp_dtype = "bfloat16"  # use float16 on GPUs without BF16 support
clip_grad = 3.0
seed = 3407
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
        "type": "PT-v3m1-PQ",
        "in_channels": 4,
        "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        "stride": (2, 2, 2, 2),
        "enc_depths": (2, 2, 2, 6, 2),
        "enc_channels": (32, 64, 128, 256, 512),
        "enc_num_head": (2, 4, 8, 16, 32),
        "enc_patch_size": (512, 512, 512, 512, 512),
        "dec_depths": (2, 2, 2, 2),
        "dec_channels": (64, 64, 128, 256),
        "dec_num_head": (4, 4, 8, 16),
        "dec_patch_size": (512, 512, 512, 512),
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "drop_path": 0.3,
        "shuffle_orders": True,
        "pre_norm": True,
        "enable_rpe": False,
        "enable_flash": True,
        "upcast_attention": False,
        "upcast_softmax": False,
        "enc_mode": False,
        "pdnorm_bn": False,
        "pdnorm_ln": False,
        "pdnorm_decouple": True,
        "pdnorm_adaptive": False,
        "pdnorm_affine": True,
        "embedding_mode": "cylindrical",
        "use_quaternion": True,
        "attention_position": "rope",
        "base_grid_size": 0.5,
        "local_stages": (0, 1),
        "query_ratio": 0.125,
        "radius_multiplier": 2.0,
        "max_neighbors": 16,
        "geometry_channels": 32,
        "radius_query_chunk_size": 128,
        "radius_max_candidate_edges": 262144,
        "learned_pool": True,
        "pool_max_members": 8,
        "enhanced_skip": True,
        "deterministic_eval": True,
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
model["backbone"]["embedding_mode"] = embedding_mode
model["backbone"]["base_grid_size"] = grid_size

# One optimizer group; avoids the original lr/max_lr parameter-group mismatch.
# Retains Pointcept's supplied 3000/100 repeated-data schedule. Not 3000
# independent complete validation cycles; see your trainer's eval_epoch semantics.
epoch = 3000
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
param_dicts = None
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

dataset_type = "DALESDataset"  # requires your existing registered DALES loader
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
            {"type": "PQPrepareStrength", "scale": 1.0},
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
            {
                "type": "GridSample",
                "grid_size": 0.5,
                "hash_type": "fnv",
                "mode": "train",
                "return_grid_coord": True,
            },
            {"type": "SphereCrop", "sample_rate": 0.6, "mode": "random"},
            {"type": "SphereCrop", "point_max": 102400, "mode": "random"},
            {"type": "CenterShift", "apply_z": False},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "grid_coord", "segment"),
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
            {"type": "PQPrepareStrength", "scale": 1.0},
            {"type": "CenterShift", "apply_z": True},
            {"type": "Copy", "keys_dict": {"segment": "origin_segment"}},
            {
                "type": "GridSample",
                "grid_size": 0.5,
                "hash_type": "fnv",
                "mode": "train",
                "return_grid_coord": True,
                "return_inverse": True,
            },
            {"type": "CenterShift", "apply_z": False},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "grid_coord", "segment", "origin_segment", "inverse"),
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
            {"type": "PQPrepareStrength", "scale": 1.0},
            {"type": "CenterShift", "apply_z": True},
        ],
        "test_mode": True,
        "test_cfg": {
            "voxelize": {
                "type": "GridSample",
                "grid_size": 0.5,
                "hash_type": "fnv",
                "mode": "test",
                "return_grid_coord": True,
            },
            "crop": {"type": "PQSpatialTestCrop", "point_max": 102400},
            "post_transform": [
                {"type": "CenterShift", "apply_z": False},
                {"type": "ToTensor"},
                {
                    "type": "Collect",
                    "keys": ("coord", "grid_coord", "index"),
                    "feat_keys": ("coord", "strength"),
                },
            ],
            "aug_transform": [[{"type": "RandomScale", "scale": [1, 1]}]],
        },
    },
}

# Apply the top-level switches consistently to every split and test voxelizer.
for split in ("train", "val", "test"):
    data[split]["type"] = dataset_type
    data[split]["data_root"] = data_root
    for transform in data[split]["transform"]:
        if transform["type"] == "GridSample":
            transform["grid_size"] = grid_size
        elif transform["type"] == "PQPrepareStrength":
            transform["scale"] = strength_scale
data["val"]["split"] = validation_split
data["test"]["test_cfg"]["voxelize"]["grid_size"] = grid_size
