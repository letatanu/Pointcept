# Complete native-Pointcept FPT S3DIS Area-5 configuration.
# No _base_ is needed: runtime settings are included below.
# Install helper fpt_recipe.py into pointcept/models/ before loading this file.
# Install/import native FPT as described in the package README.
# import pointcept.models.FPT.fpt_recipe

# RGB input; start a NEW run rather than resume a six-channel checkpoint.
# All data_root values below point to processed Pointcept S3DIS data.
# epoch=1600, eval_epoch=100 gives ~40800 steps for 204 rooms / one GPU / bs8.
# Actual steps depend on loader length, world size, and dataset size.
# No external GridSample: native model voxelizes internally at 4 cm.
# Full-resolution room validation/test can be expensive or run out of memory.
# Augmentation and schedule are approximations of the original recipe.

weight = None
resume = False
evaluate = True
test_only = False
seed = None
num_worker = 20
batch_size = 24
gradient_accumulation_steps = 1

batch_size_val = 1
batch_size_test = 1

epoch = 1600

eval_epoch = 100

clip_grad = None
sync_bn = False
enable_amp = False
amp_dtype = "float16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = "pointcept"
wandb_key = None
mix_prob = 0.0

param_dicts = None

hooks = [
    {"type": "CheckpointLoader"},
    {"type": "ModelHook"},
    {"type": "IterationTimer", "warmup_iter": 2},
    {"type": "InformationWriter"},
    {"type": "SemSegEvaluator"},
    {"type": "CheckpointSaver", "save_freq": None},
    {"type": "PreciseEvaluator", "test_last": False},
]

train = {"type": "DefaultTrainer"}

test = {"type": "SemSegTester", "verbose": True}

model = {
    "type": "FastPointTransformerSegmentor",
    "num_classes": 13,
    "ignore_index": -1,
    "backbone": {
        "type": "FastPointTransformer-v1",
        "in_channels": 3,
        "voxel_size": 0.04,
        "channels": (64, 128, 384, 640, 384, 384, 256, 128),
        "depths": (2, 3, 4, 6, 2, 2, 2, 2),
        "heads": 8,
    },
    "criteria": [{"type": "CrossEntropyLoss", "loss_weight": 1.0, "ignore_index": -1}],
}

optimizer = {"type": "SGD", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0001}

scheduler = {"type": "FPTWarmupCosineLR", "warmup_ratio": 0.01}

dataset_type = "S3DISDataset"

data_root = "data/s3dis"

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
            {
                "type": "RandomRotate",
                "angle": [-1, 1],
                "axis": "z",
                "center": [0, 0, 0],
                "p": 0.9,
            },
            {
                "type": "FPTRectangularCrop",
                "size": 4.0,
                "min_points": 100,
                "retries": 40,
            },
            {
                "type": "RandomDropout",
                "dropout_ratio": 0.2,
                "dropout_application_ratio": 0.2,
            },
            {"type": "ChromaticTranslation", "p": 0.95, "ratio": 0.1},
            {"type": "ChromaticJitter", "p": 0.7, "std": 0.01},
            {"type": "RandomFlip", "p": 0.5},
            {
                "type": "RandomShift",
                "shift": ((-0.12, 0.12), (-0.12, 0.12), (-0.12, 0.12)),
            },
            {"type": "FPTNormalizeColor"},
            {"type": "ToTensor"},
            {"type": "Collect", "keys": ("coord", "segment"), "feat_keys": ("color",)},
        ],
        "test_mode": False,
    },
    "val": {
        "type": "S3DISDataset",
        "split": "Area_5",
        "data_root": "data/s3dis",
        "transform": [
            {"type": "FPTNormalizeColor"},
            {"type": "ToTensor"},
            {"type": "Collect", "keys": ("coord", "segment"), "feat_keys": ("color",)},
        ],
        "test_mode": False,
    },
    "test": {
        "type": "S3DISDataset",
        "split": "Area_5",
        "data_root": "data/s3dis",
        "transform": [{"type": "FPTNormalizeColor"}],
        "test_mode": True,
        "test_cfg": {
            "voxelize": {"type": "FPTFullResolutionTest"},
            "crop": None,
            "post_transform": [
                {"type": "ToTensor"},
                {
                    "type": "Collect",
                    "keys": ("coord", "index"),
                    "feat_keys": ("color",),
                },
            ],
            "aug_transform": [[]],
        },
    },
}
