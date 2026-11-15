weight = 'exp/dales/cvpr_01/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 15844540
save_path = 'exp/dales/cvpr_01'
num_worker = 20
batch_size = 8
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
ignore_index = -1
data_root = 'data/dales/pointcept'
strength_scale = 1.0
validation_split = 'test'
names = [
    'Ground', 'Vegetation', 'Cars', 'Trucks', 'Power lines', 'Fences', 'Poles',
    'Buildings'
]
model = dict(
    type='DefaultSegmentorV2',
    num_classes=8,
    backbone_out_channels=64,
    backbone=dict(
        type='SerializedLocalCross-v1',
        in_channels=4,
        out_channels=64,
        channels=(48, 96, 192, 256),
        heads=(3, 6, 12, 16),
        depths=(2, 2, 3, 2),
        patch_sizes=(128, 128, 128, 128),
        pool_sizes=(0.4, 0.8, 1.6),
        position_scales=(0.4, 0.8, 1.6, 3.2),
        cross_group_size=2,
        cross_queries=(16, 16, 16, 16),
        cross_keys=(32, 32, 32, 32),
        cross_every=2,
        cross_dilations=(1, 2),
        local_rotary='qk',
        cross_rotary='qkv',
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
        checkpoint_blocks=False),
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
            dict(type='SLCPrepareStrength', scale=1.0),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='SphereCrop', point_max=204800, mode='random'),
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
            dict(type='SLCPrepareStrength', scale=1.0),
            dict(type='CenterShift', apply_z=True),
            dict(type='SphereCrop', point_max=204800, mode='center'),
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
            dict(type='SLCPrepareStrength', scale=1.0),
            dict(type='CenterShift', apply_z=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type='SLCIdentityFragments'),
            crop=dict(
                type='SLCSpatialTestCrop',
                point_max=204800,
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
