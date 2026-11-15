weight = None
resume = False
evaluate = True
test_only = False
seed = 42
save_path = 'exp/dales/semseg-fpt-quaternion-dales_01'
num_worker = 20
batch_size = 16
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 100
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
mix_prob = 0
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
names = [
    'Ground', 'Vegetation', 'Cars', 'Trucks', 'Power lines', 'Fences', 'Poles',
    'Buildings'
]
model = dict(
    type='FPTQuaternionSegmentor',
    point_chunk=32768,
    debug_numerics=False,
    num_classes=8,
    backbone_out_channels=32,
    freeze_backbone=False,
    deep_supervision=False,
    aux_loss_weight=0.2,
    aux_stage_weights=(1.0, 0.5, 0.25, 0.125),
    ignore_index=-1,
    backbone=dict(
        type='FPT-SO3-Quaternion-v1',
        in_channels=1,
        scalar_feature_indices=(0, ),
        channels=(32, 64, 128, 192),
        depths=(1, 1, 2, 2),
        voxel_sizes=(0.4, 0.8, 1.6, 3.2),
        radii=(0.8, 1.6, 3.2, 6.4),
        qk_channels=16,
        vector_channels=4,
        head_channels=32,
        query_chunk=4096,
        checkpoint_blocks=True,
        diagnostics=False),
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
    max_lr=0.001,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'DALESDataset'
data_root = 'data/dales/pointcept'
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
            dict(type='SO3Prepare', max_points=None, center=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment', 'point_id'),
                feat_keys=('strength', ))
        ],
        test_mode=False,
        loop=1),
    val=dict(
        type='DALESDataset',
        split='test',
        data_root='data/dales/pointcept',
        transform=[
            dict(type='SO3Prepare', max_points=None, center=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment', 'point_id'),
                feat_keys=('strength', ))
        ],
        test_mode=False,
        loop=1),
    test=dict(
        type='DALESDataset',
        split='test',
        data_root='data/dales/pointcept',
        transform=[dict(type='SO3Prepare', max_points=None, center=True)],
        test_mode=True,
        test_cfg=dict(
            voxelize=None,
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'index', 'point_id'),
                    feat_keys=('strength', ))
            ],
            aug_transform=[[]])))
