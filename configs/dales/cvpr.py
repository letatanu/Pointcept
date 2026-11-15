# Enable only for the frozen recipe after choosing it on the validation tiles.
_base_ = [
    "../_base_/default_runtime.py"
]  # Pointcept hooks, trainer and tester defaults.

# Enable only for the frozen recipe after choosing it on the validation tiles.

seed = 10492522
num_worker = 20
batch_size = 4

epoch = 3000
eval_epoch = 100
clip_grad = 1.0
enable_amp = True
amp_dtype = "float16"
empty_cache = False

mix_prob = 0.0
param_dicts = None

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
cross_window_size = 3  # 5 x 5 x 5 grid cells.
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
# Set aside whole tiles from the official training split in a separate 'val'
# split in the DALES data root. Keep those tiles out of 'train' and leave the
# official 'test' split untouched until final reporting.
train_split = "train"
val_split = "test"

# Fine input sampling, separate from the 0.5 m encoder pooling grid.
# Keep one random representative per 0.5 m voxel before the training crop.
# None keeps the original points.
train_input_grid_size = 0.5
# Multi-pass GridSample visits the original points across test fragments.
test_input_grid_size = 0.5
for split, size in (("train", train_input_grid_size), ("test", test_input_grid_size)):
    if size is not None and (
        type(size) not in (int, float) or not 0 < size < float("inf")
    ):
        raise ValueError(
            f"{split}_input_grid_size must be None or a positive finite number"
        )

train_input_sampling = (
    []
    if train_input_grid_size is None
    else [
        dict(
            type="GridSample",
            grid_size=train_input_grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=False,
        ),
    ]
)
test_input_sampling = (
    dict(type="SLC3IdentityFragments")
    if test_input_grid_size is None
    else dict(
        type="GridSample",
        grid_size=test_input_grid_size,
        hash_type="fnv",
        mode="test",
        return_grid_coord=False,
    )
)
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
grid_tag = "x".join(str(size) for size in coarse_grid_size)
save_path = (
    f"exp/dales/trisc_{coarse_ablation}_{coarse_cell_pool}_{grid_tag}"
    f"_p{patch_size}_l{int(enable_local_attention)}_x{int(enable_cross_attention)}"
    f"_{attention_order}_{cross_mode}{cross_window_size}_base{pooling_grid_size}"
    f'_strides{"x".join(map(str, pooling_strides))}_d{"x".join(map(str, cross_dilations))}'
    f"_q{cross_query_cap}_k{cross_key_cap}_center{int(cross_include_center)}"
    f"_{rotary_mode}_s{seed}"
)
if train_input_grid_size is not None:
    save_path += f"_trainvox{train_input_grid_size}"
if test_input_grid_size is not None:
    save_path += f"_testvox{test_input_grid_size}"
if stem_mode != "pointwise":
    save_path += (
        f"_stem{stem_mode}_r{stem_radius}_c{stem_samples_per_cell}_n{stem_neighbors}"
    )
model = dict(
    type="DefaultSegmentorV2",
    num_classes=8,
    backbone_out_channels=64,
    backbone=dict(
        type="SerializedLocalCross-v2",
        in_channels=4,
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
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
dataset_type = "DALESDataset"
data_root = "data/dales/pointcept"
data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        "Ground",
        "Vegetation",
        "Cars",
        "Trucks",
        "Power lines",
        "Fences",
        "Poles",
        "Buildings",
    ],
    train=dict(
        type="DALESDataset",
        split=train_split,
        data_root="data/dales/pointcept",
        transform=[
            dict(type="SLC3PrepareStrength", scale=1.0),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            *train_input_sampling,
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        loop=30,
    ),
    val=dict(
        type="DALESDataset",
        split=val_split,
        data_root="data/dales/pointcept",
        transform=[
            dict(type="SLC3PrepareStrength", scale=1.0),
            dict(type="CenterShift", apply_z=True),
            dict(type="SphereCrop", point_max=204800, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type="DALESDataset",
        split="test",
        data_root="data/dales/pointcept",
        transform=[
            dict(type="SLC3PrepareStrength", scale=1.0),
            dict(type="CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=test_input_sampling,
            crop=dict(
                type="SLC3SpatialTestCrop",
                point_max=204800,
                halo_fraction=0.125,
                halo_distance=0.0,
            ),
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[[]],
        ),
    ),
)
