weight = None
resume = False
evaluate = True
test_only = False
seed = 30517094
save_path = 'exp/s3dis/optnet_08'
num_worker = 16
batch_size = 4
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 800
eval_epoch = 100
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'bfloat16'
empty_cache = True
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0005)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
ignore_index = -1
names = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table',
    'chair', 'sofa', 'bookcase', 'board', 'clutter'
]
grid_size = 0.02
model = dict(
    type='OPTNetSegmentor',
    backbone=dict(
        type='OPTNet',
        in_channels=6,
        embed_dim=256,
        enc_depths=(2, 2, 6, 2),
        dec_depths=(2, 2, 2, 2),
        num_heads=8,
        pool_factors=(2, 2, 2, 2),
        dropout=0.1,
        ordering_loss_weight=1,
        pool_mode='max'),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ],
    num_classes=13,
    backbone_out_channels=256)
optimizer = dict(type='AdamW', lr=0.005, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.005,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'S3DISDataset'
data_root = 'data/S3DIS/pointcept'
data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ],
    train=dict(
        type='S3DISDataset',
        split=('Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'),
        data_root='data/S3DIS/pointcept',
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
                grid_size=0.02,
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
        loop=8),
    val=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root='data/S3DIS/pointcept',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.02,
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
        type='S3DISDataset',
        split='Area_5',
        data_root='data/S3DIS/pointcept',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
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
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }]])))
