"""Pointcept configuration for AerialWaveletNet and AeroRelief3DDataset.

The supplied aerorelief3d.py defines a DATASET, not training hyperparameters.
Training settings below are explicit starting defaults, not reproduced results.
Install at configs/aerorelief3d/semseg-aerial-wavelet-aerorelief.py.
Prepare the JSON scene manifests with the companion script before training.
"""
_base_ = ["../_base_/default_runtime.py"]

# Set this to YOUR preprocessed directory. CLI overrides must set all three
# data.train.data_root, data.val.data_root and data.test.data_root fields.
dataset_type = "AeroRelief3DDataset"
data_root = "data/aerorelief3d/pointcept"
# This repeats the held-out validation set for precise inference; not a third split.
# Preserve the IDs declared by the supplied dataset. Background (0) IS evaluated.
class_names =  ["Building-Damage", "Building-No-Damage",  "Road", "Tree", "Background"]
num_classes = 5
ignore_index = -1

# Conservative initial single-GPU settings. Batch size is TOTAL across all GPUs.
batch_size = 24
num_worker = 20
mix_prob = 0.8  # Do not merge unrelated scenes into a common context domain.
empty_cache = False
empty_cache_per_epoch = True
enable_amp = True
amp_dtype = "bfloat16"
sync_bn = False  # This model uses LayerNorm.
clip_grad = 1.0

# Pointcept uses epoch//eval_epoch as the dataset repetition factor.
# Equal values give one dataset pass per evaluation epoch.
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

grid_sample_size = 0.22  # Assumes metric coordinates; verify your preprocessing.
train_point_max = 204800
# Explicit rotation protocol. Choose 'yaw' for a matched augmentation experiment.
# 'none' does not assert structural rotation invariance of this model.
rotation_augmentation = "none"

model = dict(
    type="AerialWaveletSegmentor",  # Optional trainer adapter in the model package.
    network=dict(
        in_channels=6,  # Collect below builds [centered XYZ, normalized RGB].
        num_classes=num_classes,
        channels=(32, 64, 128, 256),
        depths=(2, 2, 4, 2),
        strides=(2, 2, 2),
        base_voxel_size=grid_sample_size,
        k=16,
        neighbor_mode="curve_knn",
        exact_backend="torch",  # Used only if neighbor_mode='exact_knn'.
        orders=("hilbert:xyz", "morton:yzx", "serpentine:zxy"),
        curve_bits=10,
        window_radius=16,
        chunk_size=512,
        position="quaternion",
        distance_scale=1.0,
        mlp_ratio=2.0,
        dropout=0.0,
        wavelet_stages=(True, True, True),
        # Smaller starting grid than the model's 64^3 default. Retune after profiling.
        wavelet_dim=32,
        grid_size=(32, 32, 32),
        wavelet_levels=3,
        wavelet_rank=16,
        wavelet_global_mixing=True,
        context_position=True,
        head_channels=64,
        pool_reduce="max",
        checkpoint_blocks=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
)

rotation_transforms = []
if rotation_augmentation == "yaw":
    rotation_transforms = [dict(type="RandomRotate", angle=[-1, 1], axis="z",
                                center=[0, 0, 0], p=1.0)]
elif rotation_augmentation != "none":
    raise ValueError("rotation_augmentation must be 'none' or 'yaw'")

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split=("Area_1", "Area_3", "Area_4", "Area_5", "Area_6", "Area_7", "Area_8"),
        ignore_index=ignore_index,
        cache=False,
        test_mode=False,
        transform=[
            dict(type="CenterShift", apply_z=True),
            *rotation_transforms,
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="GridSample", grid_size=grid_sample_size, hash_type="fnv",
                 mode="train", return_grid_coord=False),
            dict(type="SphereCrop", point_max=train_point_max, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),  # Assumes stored RGB is in [0,255].
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color")),
        ],
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split="Area_2",
        ignore_index=ignore_index,
        cache=False,
        test_mode=False,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=grid_sample_size, hash_type="fnv",
                 mode="train", return_grid_coord=False, return_inverse=True),
            # No crop: preserve a valid inverse map to ALL original labels.
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment", "origin_segment", "inverse"),
                 feat_keys=("coord", "color")),
        ],
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split="Area_2",
        ignore_index=ignore_index,
        cache=False,
        test_mode=True,
        transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=grid_sample_size, hash_type="fnv",
                          mode="test", return_grid_coord=False),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "index"), feat_keys=("coord", "color")),
            ],
            aug_transform=[[]],  # Identity only: no hidden voting or rotations.
        ),
    ),
)
