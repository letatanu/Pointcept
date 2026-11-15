# Memory-oriented AeroRelief3D variant; original config is unchanged.
# Requires memory_revision/randla_voxel_qkv.py installed at your existing model path.
# GPU peak, forward/backward behavior and accuracy remain unmeasured.
# Optimizer schedule retained; retune for the changed effective batch size.
# AeroRelief3D configuration for RandLA-Voxel-QKV-v3.
# Install in configs/aerorelief3d/ so ../_base_/default_runtime.py resolves.
# Put randla_voxel_qkv.py in pointcept/models/ and add to __init__.py:
# from .randla_voxel_qkv import RandLAVoxelQKVNetV3
# Requires your existing AeroRelief3DDataset registration.
# No dataset grid sampling: internal backbone voxel_sizes define grouping.
# If your dataset unconditionally builds a test voxelizer, adapt its constructor
# to accept voxelize=None (the local dataset implementation is not supplied here).
# New design: direct voxel context + centroid-aware decoder + point refinement.

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
# Conservative starting batch; increase only after measuring GPU peak memory.
batch_size = 12
# num_worker = 24
mix_prob = 0.8
empty_cache = False
# Mixed precision for the selected spconv backend.
# Option: enable_amp=False for FP32 debugging.
enable_amp = True
# "bfloat16" requires GPU support. If selecting spconv, use "float16" unless
# your installed spconv build explicitly supports bfloat16.
amp_dtype = "bfloat16"
ignore_index = -1
enable_wandb = False
names = ["Background", "Building-Damage", "Building-No-Damage", "Road", "Tree"]

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=len(names),
    backbone_out_channels=64,  # Must match backbone.out_channels.
    backbone=dict(
        type="RandLA-Voxel-QKV-v3",
        in_channels=6,  # Collect concatenates coord (XYZ) + normalized color (RGB).
        # Feature width per encoder stage; length defines the number of stages.
        # Larger option: (32, 64, 128, 256), with matching stage tuples below.
        channels=(32, 64, 96, 128, 192),
        # Point-attention blocks per encoder stage. Option: (2, 2, 2).
        # More blocks increase compute and activation memory.
        enc_depths=(3,3,3,12,3),
        # Each head count must divide its corresponding channels entry.
        num_heads=(2, 4, 4, 4, 6),
        # Strictly increasing voxel sizes, in coord units (meters if input is meters).
        # Smaller voxels: finer detail, more voxel sites, fewer point pairs per voxel.
        # Larger voxels: broader support, potentially much more all-pairs memory.
        # Four-stage example: (0.3, 0.6, 1.2, 2.4).
        voxel_sizes=(0.3, 0.6, 1.2, 2.4, 4.8),
        # Width of learned pair geometry and decoder centroid-offset embeddings.
        # Option: 32 for greater embedding capacity at additional cost.
        geometry_width=32,
        # Stratified random training sampling retains >=1 point per voxel.
        # All original pipeline input points receive decoded output features.
        # 1.0: every input point enters the encoder; best preserves fine skips.
        # Option: 0.5 reduces dense-voxel points before the first learned stage.
        # Discarded inputs then use a separate restoration MLP at the output.
        input_sample_ratio=1.0,
        # One ratio per transition (number of stages minus one).
        # Keep ceil(ratio * occupancy), at least one point per sampling voxel.
        # Thus 0.5 does NOT necessarily halve the total point count.
        # Train: random representatives. Eval: first representatives in input order.
        # Option: (1.0, 1.0) disables inter-stage point removal at higher cost.
        sample_ratios=(0.5, 0.5, 0.5, 0.5),
        # Exact within-voxel KNN, including self: at most N*K retained edges.
        # Search is tiled but still quadratic in voxel occupancy.
        point_attention_mode="knn",
        point_knn_k=16,
        # Context backend used in BOTH encoder and decoder refinement:
        # "spconv": two 3x3x3 sparse convolutions over occupied voxel sites;
        #           requires a compatible installed spconv package.
        # "neighbor_attention": QKV-softmax over occupied 3x3x3 voxel neighbors
        #           (at most 27 tokens including self); pure PyTorch operations.
        # "torch": slow reference implementation of the sparse CNN, for checks.
        # None of these adds direct point-to-point edges across voxel boundaries.
        # Point attention gathers at most K retained points in its own voxel.
        sparse_backend="neighbor_attention",
        # Point-attention blocks after each coarse/fine decoder fusion.
        # Options: 0 disables refinement; 1 is the starting setting; 2 adds depth.
        # Refinement always enables its own direct voxel-context residual,
        # independently of context_residual below.
        dec_depths=(0,0,0,0),
        # True: concatenate fine skip, coarse voxel mean, and learned [u, ||u||],
        # where u = (fine point - ALL-fine-point voxel centroid) / coarse voxel size.
        # A fusion MLP creates point-specific features before refinement.
        # False: v2-style fusion using fine skip and coarse voxel mean only.
        # Neither option performs KNN, distance interpolation, or neighbor-voxel
        # cross-attention during upsampling itself.
        decoder_geometry=True,
        # Chunk size for point-pair edges and point-wise FFN rows.
        # Option: 8192 for smaller temporary buffers, usually with more overhead.
        # Chunking does NOT bound total graph indices or attention score memory.
        edge_chunk_size=16384,
        # Recompute edge operations during backward to reduce saved activations.
        # Option: False for faster execution when memory is sufficient.
        checkpoint_edges=False,
        # Requires the accompanying revised randla_voxel_qkv.py.
        # Recompute voxel context, QKV, softmax and FFN during backward.
        checkpoint_blocks=False,
        # Per-stage edge budget across the whole batch; raises, never truncates.
        # None: no safety cap. Option: 8_000_000 to raise before graph allocation
        # when a point-attention stage exceeds this number of ordered edges.
        # KNN edges = sum(n_v * min(n_v, K)), including self, across the batch.
        # This cap does not truncate neighborhoods or cover voxel-token edges.
        max_edges=None,
        # FFN hidden width = channels * ratio after each point-attention block.
        # Options: 0 disables FFN; 4.0 increases capacity and memory.
        ffn_ratio=1.0,
        # Dropout on point-attention weights during training, in [0, 1).
        # Option: 0.1 for regularization. Does not affect voxel-token attention.
        attn_drop=0.0,
        # True transfers voxel context directly into encoder point outputs.
        # False conditions Q/K only; singleton softmax weights are then always 1,
        # so neighboring context cannot enter that block through its scores.
        # Decoder refinement uses a direct context residual regardless of this flag.
        context_residual=True,
        # Final backbone feature width; must match backbone_out_channels above.
        out_channels=64,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ],
)


# scheduler settings
epoch = 3000
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
# Retain the supplied two-group schedule for comparison. "blocks" now matches
# both encoder attention blocks and decoder-refinement attention blocks.
# Uniform-rate option: optimizer.lr=0.001, scheduler.max_lr=0.001, param_dicts=[].
# Edit these entries together; max_lr must match the optimizer group count.
param_dicts = [dict(keyword="blocks", lr=0.0006)]

# dataset settings
dataset_type = "AeroRelief3DDataset"
data_root = "data/aerorelief3d/pointcept"

data = dict(
    num_classes=len(names),
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_3", "Area_4", "Area_5", "Area_6", "Area_7", "Area_8"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=16384, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_2",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # Center-cropped diagnostic only; excludes points outside this crop.
            # Test mode is full-scene only when the evaluator covers all indices.
            dict(type="SphereCrop", point_max=16384, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_2",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            # No external sampling or fragmentation is configured. Full scenes may
            # exceed memory even with at most N*K point-attention edges.
            # For large scenes, use your registered fragmenter and index aggregation.
            # Requires the optional-voxelizer constructor patch if your
            # Pointcept dataset unconditionally builds this None value.
            voxelize=None,
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
    ),
)
