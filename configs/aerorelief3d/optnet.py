_base_ = ["../_base_/default_runtime.py"]

# Data Configuration
# ----------------------------------------------------------------------------
data_root = "data/AeroRelief3D/pcd_1"
grid_size = 0.22
data = dict(
    num_classes=5,
    ignore_index=-1,
    names=[
        "Background",
        "Building-Damage",
        "Building-No-Damage",
        "Road",
        "Tree"
    ],
    train=dict(
        type="AeroRelief3DDataset",
        split=["Area_1", "Area_3", "Area_4", "Area_5", "Area_6", "Area_7", "Area_8"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # Random dropout to simulate sensor noise/occlusion
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # Geometric Augmentations
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # Photometric Augmentations
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # Voxelization (Grid Sampling) - Key for OPTNet base_grid_size
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=560000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "color"))
        ],
        test_mode=False,
        loop=1 # Increase if dataset is small
    ),
    val=dict(
        type="AeroRelief3DDataset",
        split=["Area_2"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "color"))
        ],
        test_mode=False,
    ),
    test=dict(
        type="AeroRelief3DDataset",
        split=["Area_2"], # or 'test' if labels hidden
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="test", keys=("coord", "color", "segment"), return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=("coord", "color"))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)]
            ]
        )
    ),
)

# Model Configuration
# ----------------------------------------------------------------------------
model = dict(
    type="OPTNetSegmentor", # Uses the custom segmentor to handle aux_loss
    num_classes=5,
    backbone_out_channels=128,
    backbone=dict(
        type="OPTNet",
        in_channels=6,          # 3 (Coord) + 3 (Color)
        embed_dim=128,
        enc_depths=(2, 2, 6, 2),
        dec_depths=(1, 1, 1, 1),
        num_heads=4,
        win_sizes=(64, 64, 64, 64),
        win_chunk=256,
        base_grid_size=grid_size,    # Matches GridSample grid_size
        pool_factors=(2, 2, 2, 2),
        dropout=0.0,
        ffn_ratio=3.0,
        ordering_loss_weight=0.1,
        warmup_epoch=10,        # 10 Epochs of Teacher Forcing
        ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)
    ],
    # Freeze backbone if doing fine-tuning, otherwise False
    freeze_backbone=False
)

# Scheduler & Optimizer
# ----------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(type="OneCycleLR", max_lr=0.002, 
                 pct_start=0.04, anneal_strategy="cos", 
                 div_factor=10.0, final_div_factor=100.0)

# Training Settings
# ----------------------------------------------------------------------------
batch_size = 6  # Adjust based on GPU memory
epoch = 600
eval_epoch = 5
save_freq = 5
enable_amp = True
empty_cache = False