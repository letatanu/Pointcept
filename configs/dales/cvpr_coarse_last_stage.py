# Generated experiment variant; inspect the setting values below.
"""DALES: local + sliding cross-patch + coarse global attention.

Copy this file into Pointcept/configs/dales/ so the relative _base_ resolves.
Copy serialized_local_cross_v2/ into pointcept/models/ and import its adapter.
Train fresh: v2 with the new branches is not checkpoint-compatible with v1.
This retains cvpr.py's four-stage widths, depths, optimizer and data pipeline.
The new central cross branch changes sampling semantics; this is a new baseline.
"""

_base_ = ["../_base_/default_runtime.py"]  # Pointcept hooks, trainer and tester defaults.

# Independent attention ablations. Edit ONLY these flags for the 2^3 factorial
# study; the name below automatically gives each combination its own directory.
# Disabling a branch removes its modules/parameters. FFNs, pooling and decoder stay.
enable_local_attention = True   # Full self-attention inside each serialized patch.
enable_cross_attention = True   # Central-patch queries read surrounding patches.
enable_coarse_attention = True  # Master switch for global context at selected encoder locations.
ablation_name = (
    f"L{int(enable_local_attention)}_X{int(enable_cross_attention)}_G{int(enable_coarse_attention)}"
)
save_path = "exp/dales/cvpr_coarse_last_stage"  # Keep checkpoints/logs separate.

seed = 40697102  # Same as source cvpr.py; repeat promising ablations with other seeds.
weight = None   # No pretrained checkpoint; fresh training for architecture ablations.
resume = False  # True only when resuming a compatible run, not another ablation.
num_worker = 20  # Data loader worker count; tune for your host CPU/storage.
batch_size = 14  # Total batch size, as in cvpr.py; GPU fit is not verified here.
epoch = 3000    # Pointcept total effective epochs, as in cvpr.py.
eval_epoch = 100  # Pointcept evaluation-cycle setting; see its trainer semantics.
clip_grad = 1.0  # Maximum gradient norm.
enable_amp = True  # Mixed precision, required for practical FlashAttention training.
amp_dtype = "float16"  # Change to bfloat16 if supported and desired.
empty_cache = False  # Avoid emptying the CUDA allocator cache every iteration.
enable_wandb = False  # Keep external experiment tracking disabled by default.
mix_prob = 0.0  # Disable Pointcept cloud mixing; context remains per input cloud.
param_dicts = None  # Use one optimizer parameter group.
ignore_index = -1  # Label excluded from the loss and metrics.
names = ["Ground", "Vegetation", "Cars", "Trucks", "Power lines", "Fences", "Poles", "Buildings"]

model = dict(
    type="DefaultSegmentorV2",  # Standard Pointcept segmentation wrapper.
    num_classes=8,  # DALES semantic classes, in the order listed above.
    backbone_out_channels=64,  # Must match backbone.out_channels.
    backbone=dict(
        type="SerializedLocalCross-v2",  # Separate registry entry; v1 can coexist.
        in_channels=4,  # XYZ + strength, supplied by Collect below.
        out_channels=64,  # Final per-point backbone feature width.
        channels=(48, 96, 192, 256),  # Encoder width at each of four stages.
        heads=(3, 6, 12, 16),  # Attention heads per stage; head width is 16 here.
        depths=(2, 2, 3, 2),  # Number of FFN/local/cross blocks per stage.
        patch_sizes=(128, 128, 128, 128),  # Maximum points per serialized patch.
        pool_sizes=(0.4, 0.8, 1.6),  # Absolute cell widths in coordinate units for metric partition ONLY.
        pooling_mode="learned",  # learned = score-weighted mean -> MLP; mean = equal mean -> same MLP.
        # ptv3 = pointwise Linear -> segment reduction -> normalization -> GELU.
        pooling_reduce="max",  # PTv3 mode only: max (upstream default), mean, sum, min.
        pooling_norm="batch",  # PTv3 mode only: batch (upstream default), layer, none.
        # Batch uses eps=1e-3, momentum=.01; training requires >1 pooled token across batch.
        pooling_partition="metric",  # metric = rebin current physical centroids at pool_sizes.
        # grid_stride = retain integer base grid and coarsen it by each stride, PTv3-style.
        pooling_grid_size=0.2,  # Base grid cell width for grid_stride, in coordinate units.
        pooling_strides=(2, 2, 2),  # grid_stride ONLY; power-of-two factors, one per transition.
        # Base .2 with strides 2/2/2 produces .4/.8/1.6 parent-cell widths.
        # In grid_stride mode pool_sizes is ignored for grouping (still retain its tuple).
        # Only ptv3 mode uses equal CURRENT-token coordinate means like upstream;
        # learned/mean modes preserve original-point mass weighted centroids.
        position_scales=(0.4, 0.8, 1.6, 3.2),  # Divide physical XYZ by these for stage rotary encoding.

        # "none" below disables POSITION ENCODING, not the attention branch.
        enable_local_attention=enable_local_attention,
        local_rotary="qk",  # Quaternion rotary on queries/keys; also supports none or qkv.
        enable_cross_attention=enable_cross_attention,
        cross_rotary="qk",  # Independent rotary setting for cross-patch attention.
        cross_mode="central",  # central = sliding window; grouped = original disjoint grouping.
        cross_window_size=5,  # Odd size >=3; source offsets [-2,-1,0,1,2] times dilation.
        cross_group_size=2,  # Used ONLY in grouped mode; ignored by central layout.
        cross_queries=(16, 16, 16, 16),  # CENTRAL: queries per central patch; capped by patch count.
        # GROUPED: cross_queries is instead TOTAL per group, split among patches.
        # To reproduce source cvpr.py layout, use grouped + queries=(32,)*4 + keys=(128,)*4,
        # enable local/cross, disable coarse, and keep the remaining source settings.
        cross_keys=(32, 32, 32, 32),  # Keys per SOURCE patch; <=160 keys/window with window=5.
        cross_every=1,  # Apply cross attention every nth block, counted within each stage.
        cross_dilations=(1, 2),  # Cycle over enabled cross blocks, across stages.
        # D=1 reads i-2,i-1,i,i+1,i+2; D=2 reads i-4,i-2,i,i+2,i+4.
        # This is serialized PATCH-INDEX distance, not Euclidean distance in metres.
        # Queries are unique across windows. K/V reuse across windows is intentional.
        # Training samples one random token per serialized stratum; eval uses midpoints.

        enable_coarse_attention=enable_coarse_attention,  # Master switch; overrides placement selection.
        coarse_stages=None,  # None = last encoder stage only, preserving the original v2 behavior.
        # Zero-based stage selection: (2,3) = last two; (0,1,2,3) = all four stages.
        coarse_placement="stage_end",  # stage_end = once after selected stages, before skips/pooling.
        # every_block = after EVERY encoder block in selected stages; stage weights are shared.
        # Earlier selected stages get independent context modules with the correct channel widths.
        coarse_grid_size=(8, 8, 8),  # <=512 OCCUPIED tokens/cloud, using each crop's XYZ bounds.
        # This is bounding-box-normalized spatial pooling, not fixed-metre voxels or Haar-WNO.
        # Empty cells are omitted. Grid shape changes both token count and spatial resolution.
        coarse_channels=64,  # Project pooled final-stage features to this context width.
        coarse_heads=8,  # 64/8 = 8 channels/head, compatible with FlashAttention and quaternion rotary.
        coarse_depth=1,  # Number of global attention+FFN blocks PER context invocation.
        coarse_rotary="qk",  # Position encoding for coarse-token global attention.
        coarse_position_scale=32.0,  # Physical token centroid XYZ / 32m for coarse rotary.
        # Original-point mass weights coarse feature/centroid pooling. Global updates are
        # gathered to every point of the SELECTED stage with a learned gate and residual scale.
        # Existing gated decoder then carries context to the original point resolution.
        # "Global" means the current INPUT CROP; unseen parts of a tile are unavailable.

        ffn_ratio=2,  # Hidden width multiplier of encoder/decoder pointwise MLPs.
        score_hidden=16,  # Learned pooling score width; unused when pooling_mode is mean or ptv3.
        chunk_size=128,  # Attention rows/windows processed per chunk; does not alter neighborhoods.
        attention_backend="flash_varlen",  # GPU training; use reference only for small CPU checks.
        serialization_resolution=0.05,  # Metre grid used to generate space-filling-curve codes.
        order=("z", "z-trans", "hilbert", "hilbert-trans"),  # Available serialization orders.
        shuffle_orders=True,  # Shuffle stage order choices during training; deterministic evaluation.
        serialization_backend="pointcept",  # Uses Pointcept encode; reference is for standalone tests.
        rotary_min_frequency=0.25,  # Minimum frequency in the fixed quaternion rotary bank.
        rotary_max_frequency=2.0,  # Maximum frequency; shared bounds across all three branches.
        checkpoint_blocks=False,  # Recompute encoder blocks on backward to save activation memory.
        # Checkpointing does not cover decoder skips or any coarse context invocation.
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),  # Per-point classification.
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),  # IoU surrogate.
    ],
)

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)  # Match cvpr.py optimizer.
scheduler = dict(
    type="OneCycleLR",  # Warm up, then decay learning rate over the run.
    max_lr=0.001,  # Peak learning rate; initial = max_lr/div_factor.
    pct_start=0.05,  # Fraction of scheduler steps spent warming up.
    anneal_strategy="cos",  # Cosine learning-rate decay.
    div_factor=10.0,  # Initial learning rate = 0.0001.
    final_div_factor=1000.0,  # Final rate = initial rate/1000.
)

dataset_type = "DALESDataset"  # Pointcept DALES loader; must return coord, strength, segment.
data_root = "data/dales/pointcept"  # Prepared DALES dataset location relative to Pointcept root.
data = dict(
    num_classes=8,
    ignore_index=-1,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",  # Training data split.
        data_root=data_root,
        transform=[
            dict(type="SLC3PrepareStrength", scale=1.0),  # Keep finite Nx1 strength; divide by scale.
            dict(type="CenterShift", apply_z=True),  # Center input coordinates, including height.
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),  # Z rotation augmentation.
            dict(type="RandomScale", scale=[0.9, 1.1]),  # Isotropic scale augmentation.
            dict(type="RandomFlip", p=0.5),  # Horizontal flip augmentation.
            dict(type="SphereCrop", point_max=204800, mode="random"),  # Maximum raw points per training crop.
            dict(type="CenterShift", apply_z=False),  # Recenter XY after cropping.
            dict(type="ToTensor"),  # Convert NumPy data to tensors.
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "strength")),  # Build Nx4 input features.
        ],
        test_mode=False,
        loop=30,  # Dataset repetition setting; preserve Pointcept/cvpr.py training semantics.
    ),
    val=dict(
        type=dataset_type,
        split="test",  # Matches cvpr.py. Use held-out TRAIN data for architecture selection if test is final reporting.
        data_root=data_root,
        transform=[
            dict(type="SLC3PrepareStrength", scale=1.0),
            dict(type="CenterShift", apply_z=True),
            dict(type="SphereCrop", point_max=204800, mode="center"),  # Deterministic validation crop.
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "strength")),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[dict(type="SLC3PrepareStrength", scale=1.0), dict(type="CenterShift", apply_z=True)],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="SLC3IdentityFragments"),  # Preserve all raw points; tester expects a fragment list.
            crop=dict(
                type="SLC3SpatialTestCrop",  # Spatial cores plus overlapping context halos.
                point_max=204800,  # Fragment cap, including halo points.
                halo_fraction=0.125,  # Reserve up to 12.5% of capacity for halo context.
                halo_distance=2.0,  # Halo extension in metres along each coordinate axis.
            ),
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "index"), feat_keys=("coord", "strength")),  # Index merges fragment predictions.
            ],
            aug_transform=[[dict(type="RandomScale", scale=[1.0, 1.0])]],  # One identity test augmentation.
        ),
    ),
)
