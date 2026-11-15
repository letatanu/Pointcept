weight = None
resume = False
evaluate = True
test_only = False
seed = 37225782
save_path = 'exp/aerorelief3d/trisc_full'
num_worker = 24
batch_size = 4
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 3000
eval_epoch = 100
clip_grad = 3.0
sync_bn = False
enable_amp = True
amp_dtype = 'bfloat16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.0
param_dicts = [dict(keyword='backbone', lr=0.0001)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
names = ['Background', 'Building-Damage', 'Building-No-Damage', 'Road', 'Tree']
grid_size = 0.22
enable_local_attention = True
enable_cross_attention = True
attention_order = 'local_then_cross'
patch_size = 128
cross_mode = 'grid'
cross_window_size = 5
cross_query_cap = 4
cross_key_cap = 2
cross_dilations = (1, )
cross_grid_chunk_size = 1024
cross_include_center = False
pooling_grid_size = 0.5
pooling_strides = (2, 2, 2)
stem_mode = 'bezier_features'
stem_radius = 0.25
stem_samples_per_cell = 4
stem_neighbors = 8
stem_chunk_size = 2048
rotary_mode = 'utonia_qk'
coarse_cell_pool = 'ln_max_project'
coarse_ablation = 'shared_fused'
coarse_grid_size = (8, 8, 8)
coarse_options = dict(
    none=dict(enable_coarse_attention=False),
    final_only=dict(coarse_stages=(3, )),
    independent_both=dict(coarse_stages=(2, 3)),
    independent_delayed=dict(
        coarse_stages=(2, 3), coarse_placement='decoder_entry'),
    shared_final=dict(
        coarse_mode='shared_final',
        coarse_stages=(2, 3),
        coarse_placement='decoder_entry'),
    shared_fused=dict(
        coarse_mode='shared_fused',
        coarse_stages=(2, 3),
        coarse_placement='decoder_entry'))
coarse_config = dict(
    enable_coarse_attention=True,
    coarse_mode='shared_fused',
    coarse_placement='decoder_entry',
    coarse_cell_pool='ln_max_project',
    coarse_stages=(2, 3))
model = dict(
    type='DefaultSegmentorV2',
    num_classes=5,
    backbone_out_channels=64,
    backbone=dict(
        type='SerializedLocalCross-v2',
        in_channels=6,
        stem_mode='bezier_features',
        stem_radius=0.25,
        stem_samples_per_cell=4,
        stem_neighbors=8,
        stem_chunk_size=2048,
        out_channels=64,
        channels=(48, 96, 192, 256),
        heads=(3, 6, 12, 16),
        depths=(2, 2, 3, 2),
        patch_sizes=(128, 128, 128, 128),
        pool_sizes=(0.4, 0.8, 1.6),
        pooling_mode='ptv3',
        pooling_reduce='max',
        pooling_norm='batch',
        pooling_partition='grid_stride',
        pooling_grid_size=0.5,
        pooling_strides=(2, 2, 2),
        position_scales=(0.4, 0.8, 1.6, 3.2),
        enable_local_attention=True,
        local_rotary='utonia_qk',
        enable_cross_attention=True,
        attention_order='local_then_cross',
        cross_rotary='utonia_qk',
        cross_mode='grid',
        cross_window_size=5,
        cross_include_center=False,
        cross_group_size=2,
        cross_queries=(4, 4, 4, 4),
        cross_keys=(2, 2, 2, 2),
        cross_every=1,
        cross_dilations=(1, ),
        cross_grid_chunk_size=1024,
        coarse_grid_size=(8, 8, 8),
        coarse_channels=64,
        coarse_heads=8,
        coarse_depth=1,
        coarse_rotary='utonia_qk',
        coarse_position_scale=32.0,
        ffn_ratio=2,
        score_hidden=16,
        chunk_size=128,
        attention_backend='flash_varlen',
        serialization_resolution=0.05,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        shuffle_orders=True,
        serialization_backend='pointcept',
        rotary_min_frequency=0.25,
        rotary_max_frequency=2.0,
        checkpoint_blocks=False,
        enable_coarse_attention=True,
        coarse_mode='shared_fused',
        coarse_placement='decoder_entry',
        coarse_cell_pool='ln_max_project',
        coarse_stages=(2, 3)),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'AeroRelief3DDataset'
data_root = 'data/aerorelief3d/pointcept'
common_train = [
    dict(type='CenterShift', apply_z=True),
    dict(
        type='RandomDropout', dropout_ratio=0.2,
        dropout_application_ratio=0.2),
    dict(
        type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
    dict(type='RandomRotate', angle=[-0.015625, 0.015625], axis='x', p=0.5),
    dict(type='RandomRotate', angle=[-0.015625, 0.015625], axis='y', p=0.5),
    dict(type='RandomScale', scale=[0.9, 1.1]),
    dict(type='RandomFlip', p=0.5),
    dict(type='RandomJitter', sigma=0.005, clip=0.02),
    dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
    dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
    dict(type='ChromaticJitter', p=0.95, std=0.05),
    dict(
        type='GridSample',
        grid_size=0.22,
        hash_type='fnv',
        mode='train',
        return_grid_coord=True),
    dict(type='SphereCrop', sample_rate=0.6, mode='random'),
    dict(type='SphereCrop', point_max=204800, mode='random'),
    dict(type='CenterShift', apply_z=False),
    dict(type='NormalizeColor'),
    dict(type='ToTensor'),
    dict(
        type='Collect',
        keys=('coord', 'grid_coord', 'segment'),
        feat_keys=('coord', 'color'))
]
data = dict(
    num_classes=5,
    ignore_index=-1,
    names=[
        'Background', 'Building-Damage', 'Building-No-Damage', 'Road', 'Tree'
    ],
    train=dict(
        type='AeroRelief3DDataset',
        split=('Area_1', 'Area_3', 'Area_4', 'Area_5', 'Area_6', 'Area_7',
               'Area_8'),
        data_root='data/aerorelief3d/pointcept',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(
                type='GridSample',
                grid_size=0.22,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', sample_rate=0.6, mode='random'),
            dict(type='SphereCrop', point_max=204800, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'color'))
        ],
        test_mode=False,
        loop=30),
    val=dict(
        type='AeroRelief3DDataset',
        split='Area_2',
        data_root='data/aerorelief3d/pointcept',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.22,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'origin_segment',
                      'inverse'),
                feat_keys=('coord', 'color'))
        ],
        test_mode=False),
    test=dict(
        type='AeroRelief3DDataset',
        split='Area_2',
        data_root='data/aerorelief3d/pointcept',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.22,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'color'))
            ],
            aug_transform=[[]])))
