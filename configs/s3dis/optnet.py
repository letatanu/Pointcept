# config_optnet_NO.py
_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4         # NO: reduced from 12 — GNO neighbor search uses more memory
mix_prob = 0.8
empty_cache = False
enable_amp = True
amp_dtype = 'bfloat16'
clip_grad = 35.0

# model settings
model = dict(
    type="OPTNetSegmentor",
    backbone=dict(
        type="OPTNet",          # NO: same class name — OPTNet now uses PointSorterNO internally
        in_channels=6,

        # ============================================
        # NO: Ordering / Neural Operator Parameters
        # ============================================
        ordering_loss_weight=0.5,   # NO: reduced from 1.0 — GNO loss is simpler (dist only)
        ordering_k=16,              # kept — used by OrderingLoss fallback if needed
        warmup_epoch=3,             # NO: reduced from 5 — GNO converges faster (no locality warmup needed)
        enable_score_concat=True,
        tau=0.1,
        loss_weights=[1, 0.5, 0, 0],  # NO: changed from [0,0,0,1] → enable only distribution loss
                                    #     locality (w_loc=0): handled architecturally by GNO
                                    #     global feature (w_glob=0): remove contrastive, keep dist

        # NO: New parameters for PointSorterNO
        # radius must match your voxel/scene scale.
        # grid_size=0.02 → radius=0.04 (2x voxel) is a good start
        # This is passed to PointSorterNO(radius=...) inside OPTNet.__init__
        # Add this field to OPTNet.__init__ kwargs and forward to PointSorterNO:
        # sorter_radius=0.04,         # NO: NEW — radius for GNO NeighborSearch
        sorter_hidden_channels=64,  # NO: NEW — hidden dim of GNO layers (same as MLP default)
        sorter_k=16,

        # NO: SemanticGridPooling parameters
        num_score_buckets=8,        # kept — number of semantic score splits per voxel
        code_depth=10,              # kept

        # ============================================
        # PTv3 Backbone Parameters — UNCHANGED
        # ============================================
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    num_classes=13,
    backbone_out_channels=64,
)

# scheduler settings — UNCHANGED
epoch = 3000
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings — UNCHANGED
dataset_type = "S3DISDataset"
data_root = "data/S3DIS/pointcept"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        "ceiling", "floor", "wall", "beam", "column",
        "window", "door", "table", "chair", "sofa",
        "bookcase", "board", "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "normal"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "normal"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "normal"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
            ],
        ),
    ),
)

# hook — UNCHANGED
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# runtime — UNCHANGED
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)
