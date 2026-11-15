# TRiSC on AeroRelief3D.
_base_ = ["../_base_/default_runtime.py"]


batch_size = 4
num_worker = 24
mix_prob = 0.0
# empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
clip_grad = 3.0

names = ["Background", "Building-Damage", "Building-No-Damage", "Road", "Tree"]
grid_size = 0.22

enable_local_attention = True
enable_cross_attention = True
# The residual MLP runs first. Both attention branches add their own residual.
# Switch to 'cross_then_local' for the order ablation.
attention_order = "local_then_cross"
# For the patch-size control, turn cross and coarse attention off, then change
# patch_size from 128 to 1024. Keep widths, depths, and the training recipe fixed.
patch_size = 128
# 'grid': center-cell queries attend to neighboring grid cells.
# 'central': use the serialized patch neighbors from run 10 instead.
cross_mode = "grid"
cross_window_size = 5  # 5 x 5 x 5 encoder grid cells.
cross_query_cap = 4 if cross_mode == "grid" else 128  # N per center cell/patch.
cross_key_cap = 2 if cross_mode == "grid" else 128  # M per source cell/patch.
cross_dilations = (1,) if cross_mode == "grid" else (1, 2)
# Larger grid batches reduce FlashAttention launch overhead. Lower this if a
# single attention chunk needs too much GPU memory. Sampling budgets stay fixed.
cross_grid_chunk_size = 1024
# Grid mode always excludes the center. In central mode, True reproduces
# the center-inclusive patch layout from runs 7 and 9.
cross_include_center = False
# Grid cross-attention shares these encoder grids. Multiplying by the next
# transition factor gives 0.5 m -> 1 m -> 2 m -> 4 m in this example.
pooling_grid_size = 0.5
pooling_strides = (2, 2, 2)  # One power-of-two factor per encoder transition.

# The pointwise MLP runs in every mode. 'pointwise' disables the residual.
# 'relative_xyz' uses individual offsets, 'bezier' uses pair geometry, and
# 'bezier_features' adds the selected neighbors' pointwise features.
stem_mode = "bezier_features"
stem_radius = 0.25  # Meters. Also the lookup cell width, separate from encoder pooling.
stem_samples_per_cell = 4  # Deterministic representatives per occupied lookup cell.
stem_neighbors = 8  # Closest in-radius candidates among those representatives.
stem_chunk_size = 2048  # Point rows per lookup chunk. Lower to reduce temporary memory.

# Match the attached Utonia runs: 18 channels/head, base 10, raw coordinate
# scale, and shared training jitter/rescale factors for each attention layer.
# 'axial_qk' keeps the partial-rotation setup used by run 10.
# 'qk' keeps legacy quaternion RoPE. 'none' uses the legacy attention widths.
rotary_mode = "utonia_qk"

# Run shared_fused with all three cell-pooling modes for the reducer ablation.
# 'mean_ln_project': mass-weighted mean -> cell LN -> linear (legacy baseline).
# 'ln_mean_project': token LN -> mass-weighted mean -> linear.
# 'ln_max_project':  token LN -> channel-wise max -> linear (no mass weighting).
coarse_cell_pool = "ln_max_project"  # Match run 10 for the cross-branch comparison.
coarse_ablation = "shared_fused"
coarse_grid_size = (8, 8, 8)  # Try (16, 16, 2) for the same maximum cell count.
coarse_options = {
    "none": dict(enable_coarse_attention=False),
    # Stage IDs below are zero-based. Stage 3 in the figure is index 2 here.
    "final_only": dict(coarse_stages=(3,)),
    "independent_both": dict(coarse_stages=(2, 3)),
    # Match the shared branch's injection timing without fusing its cell tokens.
    "independent_delayed": dict(coarse_stages=(2, 3), coarse_placement="decoder_entry"),
    # Same union grid and both decoder updates, but mask finer input features
    # and their presence indicator. Compare directly against shared_fused.
    "shared_final": dict(
        coarse_mode="shared_final",
        coarse_stages=(2, 3),
        coarse_placement="decoder_entry",
    ),
    "shared_fused": dict(
        coarse_mode="shared_fused",
        coarse_stages=(2, 3),
        coarse_placement="decoder_entry",
    ),
}
coarse_config = dict(
    enable_coarse_attention=True,
    coarse_mode="independent",
    coarse_placement="stage_end",
    coarse_cell_pool=coarse_cell_pool,
)
coarse_config.update(coarse_options[coarse_ablation])
model = dict(
    type="DefaultSegmentorV2",
    num_classes=len(names),
    backbone_out_channels=64,
    backbone=dict(
        type="SerializedLocalCross-v2",
        in_channels=6,
        stem_mode=stem_mode,
        stem_radius=stem_radius,
        stem_samples_per_cell=stem_samples_per_cell,
        stem_neighbors=stem_neighbors,
        stem_chunk_size=stem_chunk_size,
        out_channels=64,
        channels=(48, 96, 192, 256),
        heads=(3, 6, 12, 16),
        depths=(2, 2, 3, 2),
        patch_sizes=(patch_size,) * 4,
        pool_sizes=(0.4, 0.8, 1.6),
        pooling_mode="ptv3",
        pooling_reduce="max",
        pooling_norm="batch",
        pooling_partition="grid_stride",
        pooling_grid_size=pooling_grid_size,
        pooling_strides=pooling_strides,
        position_scales=(0.4, 0.8, 1.6, 3.2),
        enable_local_attention=enable_local_attention,
        local_rotary=rotary_mode,
        enable_cross_attention=enable_cross_attention,
        attention_order=attention_order,
        cross_rotary=rotary_mode,
        cross_mode=cross_mode,
        cross_window_size=cross_window_size,
        cross_include_center=cross_include_center,
        cross_group_size=2,
        cross_queries=(cross_query_cap,) * 4,
        cross_keys=(cross_key_cap,) * 4,
        cross_every=1,
        cross_dilations=cross_dilations,
        cross_grid_chunk_size=cross_grid_chunk_size,
        coarse_grid_size=coarse_grid_size,
        coarse_channels=64,
        coarse_heads=8,
        coarse_depth=1,
        coarse_rotary=rotary_mode,
        coarse_position_scale=32.0,
        ffn_ratio=2,
        score_hidden=16,
        chunk_size=128,
        attention_backend="flash_varlen",
        serialization_resolution=0.05,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders=True,
        serialization_backend="pointcept",
        rotary_min_frequency=0.25,
        rotary_max_frequency=2.0,
        checkpoint_blocks=False,
        **coarse_config,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 3000
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="backbone", lr=0.0001)]

dataset_type = "AeroRelief3DDataset"
data_root = "data/aerorelief3d/pointcept"

common_train = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(
        type="GridSample",
        grid_size=grid_size,
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
        keys=("coord", "grid_coord", "segment"),
        feat_keys=("coord", "color"),
    ),
]

data = dict(
    num_classes=len(names),
    ignore_index=-1,
    names=names,
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_3", "Area_4", "Area_5", "Area_6", "Area_7", "Area_8"),
        data_root=data_root,
        transform=common_train,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_2",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
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
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_2",
        data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
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
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[[]],
        ),
    ),
)
