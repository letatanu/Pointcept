weight = None
resume = False
evaluate = True
test_only = False
seed = 3407
save_path = 'exp/dales/semseg-sparse-query-quatrope-full-query_01'
num_worker = 20
batch_size = 42
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 3000
eval_epoch = 100
clip_grad = 1.0
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.0
param_dicts = None
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
data_root = 'data/dales/pointcept'
strength_scale = 1.0
validation_split = 'test'
point_max = 65536
ignore_index = -1
names = [
    'Ground', 'Vegetation', 'Cars', 'Trucks', 'Power lines', 'Fences', 'Poles',
    'Buildings'
]
model = dict(
    type='DefaultSegmentorV2',
    num_classes=8,
    backbone_out_channels=64,
    backbone=dict(
        type='SparseQueryQuatRoPE-v1',
        in_channels=4,
        out_channels=64,
        channels=(48, 96, 192, 256),
        depths=(2, 2, 3, 2),
        num_heads=(3, 6, 12, 16),
        patch_sizes=(64, 64, 64, 32),
        query_ratios=(1.0, 1.0, 1.0, 1.0),
        voxel_sizes=(0.2, 0.4, 0.8, 1.6),
        pool_sizes=(0.4, 0.8, 1.6),
        stage_scales=(0.5, 1.0, 2.0, 4.0),
        serialization_resolution=0.05,
        axis_orders=((0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0)),
        sparse_backend='spconv',
        use_sparse_cnn=True,
        position_encoding='quaternion',
        ffn_ratio=2.0,
        attention_dropout=0.0,
        max_neighbor_distances=(None, None, None, None),
        patch_chunk_size=128,
        checkpoint_attention=True),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'DALESDataset'
data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        'Ground', 'Vegetation', 'Cars', 'Trucks', 'Power lines', 'Fences',
        'Poles', 'Buildings'
    ],
    train=dict(
        type='DALESDataset',
        split='train',
        data_root='data/dales/pointcept',
        transform=[
            dict(type='SQQPrepareStrength', scale=1.0),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='SphereCrop', point_max=65536, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        loop=30),
    val=dict(
        type='DALESDataset',
        split='test',
        data_root='data/dales/pointcept',
        transform=[
            dict(type='SQQPrepareStrength', scale=1.0),
            dict(type='CenterShift', apply_z=True),
            dict(type='SphereCrop', point_max=65536, mode='center'),
            dict(type='CenterShift', apply_z=False),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False),
    test=dict(
        type='DALESDataset',
        split='test',
        data_root='data/dales/pointcept',
        transform=[
            dict(type='SQQPrepareStrength', scale=1.0),
            dict(type='CenterShift', apply_z=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type='SQQIdentityFragments'),
            crop=dict(
                type='SQQSpatialTestCrop',
                point_max=65536,
                halo_fraction=0.125,
                halo_distance=2.0),
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'index'),
                    feat_keys=('coord', 'strength'))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [1.0, 1.0]
            }]])))
split = 'test'
transform = dict(type='CenterShift', apply_z=True)
