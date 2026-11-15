"""Complete Pointcept configuration for AeroRelief3D (five classes).

Copy to configs/aerorelief3d/semseg-so3-ptv2-lite.py AFTER installing/importing
the two companion modules (see README). Uses your existing AeroRelief3DDataset.
Old SO3-VN-PTv2 checkpoints are incompatible: start a new experiment directory.
The model is mathematically rotation-invariant; measured accuracy/speed are TBD.
"""

# ---------- Runtime: keep a bounded POINT budget, not a huge crop batch ----------
weight = None
resume = False
evaluate = True
test_only = False
seed = 3407
save_path = "exp/aerorelief3d/so3_ptv2_lite"
num_worker = 8                 # CPU spatial trees run here; tune on the H100 host.
batch_size = 4                 # Global batch; on one GPU: at most 4 * 32768 points.
batch_size_val = 1             # Required for the local inverse map in fast validation.
batch_size_test = 1
gradient_accumulation_steps = 1

# Pointcept uses eval_epoch outer cycles; the dataset loops inside each cycle.
# 300 nominal dataset passes = 100 evaluation cycles * 3 repetitions.
# With your 56 train files and batch 4, approximately 42 updates per cycle.
epoch = 300
eval_epoch = 100
clip_grad = 3.0
sync_bn = False                # This model uses scalar LayerNorm, not BatchNorm.
enable_amp = True
amp_dtype = "bfloat16"         # H100 training; geometry and loss reductions use FP32.
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = "pointcept"
wandb_key = None

# CRITICAL: cached local graph indices require original cloud boundaries.
# Pointcept's scene-mixing collator would invalidate them. The model checks this.
mix_prob = 0.0
param_dicts = None             # One LR for a model trained from scratch.

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # At training completion, full sphere-cover evaluation of the best checkpoint.
    dict(type="PreciseEvaluator", test_last=False),
]
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)

# ---------- Model: PTv2 multiplier/bias BEFORE grouped weight encoding ----------
stage_neighbors = (12, 16, 16, 24, 24)
stage_ratios = (0.25, 0.25, 0.25, 0.25)
model = dict(
    type="SO3PTv2LiteSegmentor",
    num_classes=5,
    ignore_index=-1,
    ce_weight=1.0,
    lovasz_weight=1.0,
    # Do not guess rare-class weights. Inspect the post-crop class histogram first.
    class_weights=None,
    backbone=dict(
        type="SO3-PTv2-Lite",
        in_channels=3,         # RGB ONLY. XYZ stays in coord, never scalar features.
        channels=(32, 64, 96, 160, 256),
        vector_channels=(4, 8, 8, 8, 8),
        depths=(1, 1, 2, 2, 2),
        groups=(2, 4, 4, 8, 8),  # Must divide BOTH scalar and vector channels.
        neighbors=stage_neighbors,
        pos_hidden=16,          # Separate small MLPs for delta_mul and delta_bias.
        direction_channels=4,  # e_ij has 4 geometry + 3*4 directional scalar terms.
        # 'scalar': gamma = q_i-k_j on invariant scalar features (PTv2 ordering).
        # 'vector': gamma = Linear(||Q_i^c-K_j^c||^2), an invariant VN ablation.
        # BOTH modes apply grouped attention to whole XYZ vector values.
        relation_mode="scalar",
        head_channels=64,
        mlp_ratio=2,
        # Start off. If memory is tight, reduce point_max/batch first or turn on
        # whole-block checkpointing. There are no tiny per-edge checkpoint loops.
        checkpoint_blocks=False,
    ),
)

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR", max_lr=0.001, pct_start=0.05,
    anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0,
)

# ---------- Data: no axis-aligned voxel sampling / pooling ----------
dataset_type = "AeroRelief3DDataset"
data_root = "data/aerorelief3d/pointcept"
ignore_index = -1
names = ["Background", "Building-Damage", "Building-No-Damage", "Road", "Tree"]

# This assumes coord is measured in meters. Tune the metric radius for your
# actual density/building sizes. Randomly retain points THROUGHOUT the sphere.
crop_radius = 20.0
train_point_max = 32768
val_point_max = 65536
test_point_max = 32768

data = dict(
    num_classes=5,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_3", "Area_4", "Area_5", "Area_6", "Area_7", "Area_8"),
        data_root=data_root,
        loop=epoch // eval_epoch,
        test_mode=False,
        transform=[
            dict(type="SO3TrainCrop", point_max=train_point_max, radius=crop_radius, min_points=512),
            dict(type="SO3Center"),
            dict(type="SO3RandomRotate", p=0.5),
            # Uniform scale preserves rotations, but scale invariance is NOT
            # required: metric size remains a useful signal for outdoor classes.
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="ChromaticTranslation", p=0.2, ratio=0.03),
            dict(type="ChromaticJitter", p=0.2, std=0.02),
            dict(type="NormalizeColor"),
            # SO3Pack replaces BOTH ToTensor and Collect. It MUST run last.
            dict(type="SO3Pack", neighbors=stage_neighbors, ratios=stage_ratios,
                 interp_k=3, mode="train", feature_keys=("color",)),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="Area_2",
        data_root=data_root,
        test_mode=False,
        transform=[
            dict(type="SO3Center"),
            # Fast whole-scene SAMPLE proxy, mapped back to original labels.
            # Its mIoU is not interchangeable with full-resolution test mIoU.
            dict(type="SO3ValidationSample", point_max=val_point_max, seed=12345),
            dict(type="NormalizeColor"),
            dict(type="SO3Pack", neighbors=stage_neighbors, ratios=stage_ratios,
                 interp_k=3, mode="eval", seed=12345, feature_keys=("color",)),
        ],
    ),
    test=dict(
        type=dataset_type,
        split="Area_2",        # Same held-out area as your supplied config.
        data_root=data_root,
        test_mode=True,
        transform=[dict(type="SO3Center"), dict(type="NormalizeColor")],
        test_cfg=dict(
            # Pointcept calls this field voxelize; our transform does NOT voxelize.
            voxelize=dict(type="SO3IdentityFragments"),
            crop=dict(type="SO3SphereCover", point_max=test_point_max,
                      radius=crop_radius, seed=12345),
            post_transform=[
                dict(type="SO3Pack", neighbors=stage_neighbors, ratios=stage_ratios,
                     interp_k=3, mode="eval", seed=12345, feature_keys=("color",)),
            ],
            # No rotation voting needed. One identity augmentation preserves every
            # raw point via sphere-cover index maps and the existing SemSegTester.
            aug_transform=[[]],
        ),
    ),
)
