weight = None
resume = False
evaluate = True
test_only = False
seed = 3407
save_path = 'exp/aerorelief3d/so3-ptv2_lite_01'
num_worker = 8
batch_size = 4
batch_size_val = 1
batch_size_test = 1
gradient_accumulation_steps = 1
epoch = 300
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
stage_neighbors = (12, 16, 16, 24, 24)
stage_ratios = (0.25, 0.25, 0.25, 0.25)
model = dict(
    type='SO3PTv2LiteSegmentor',
    num_classes=5,
    ignore_index=-1,
    ce_weight=1.0,
    lovasz_weight=1.0,
    class_weights=None,
    backbone=dict(
        type='SO3-PTv2-Lite',
        in_channels=3,
        channels=(32, 64, 96, 160, 256),
        vector_channels=(4, 8, 8, 8, 8),
        depths=(1, 1, 2, 2, 2),
        groups=(2, 4, 4, 8, 8),
        neighbors=(12, 16, 16, 24, 24),
        pos_hidden=16,
        direction_channels=4,
        relation_mode='scalar',
        head_channels=64,
        mlp_ratio=2,
        checkpoint_blocks=False))
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'AeroRelief3DDataset'
data_root = 'data/aerorelief3d/pointcept'
ignore_index = -1
names = ['Background', 'Building-Damage', 'Building-No-Damage', 'Road', 'Tree']
crop_radius = 20.0
train_point_max = 32768
val_point_max = 65536
test_point_max = 32768
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
        loop=3,
        test_mode=False,
        transform=[
            dict(
                type='SO3TrainCrop',
                point_max=32768,
                radius=20.0,
                min_points=512),
            dict(type='SO3Center'),
            dict(type='SO3RandomRotate', p=0.5),
            dict(type='RandomScale', scale=[0.95, 1.05]),
            dict(type='ChromaticTranslation', p=0.2, ratio=0.03),
            dict(type='ChromaticJitter', p=0.2, std=0.02),
            dict(type='NormalizeColor'),
            dict(
                type='SO3Pack',
                neighbors=(12, 16, 16, 24, 24),
                ratios=(0.25, 0.25, 0.25, 0.25),
                interp_k=3,
                mode='train',
                feature_keys=('color', ))
        ]),
    val=dict(
        type='AeroRelief3DDataset',
        split='Area_2',
        data_root='data/aerorelief3d/pointcept',
        test_mode=False,
        transform=[
            dict(type='SO3Center'),
            dict(type='SO3ValidationSample', point_max=65536, seed=12345),
            dict(type='NormalizeColor'),
            dict(
                type='SO3Pack',
                neighbors=(12, 16, 16, 24, 24),
                ratios=(0.25, 0.25, 0.25, 0.25),
                interp_k=3,
                mode='eval',
                seed=12345,
                feature_keys=('color', ))
        ]),
    test=dict(
        type='AeroRelief3DDataset',
        split='Area_2',
        data_root='data/aerorelief3d/pointcept',
        test_mode=True,
        transform=[dict(type='SO3Center'),
                   dict(type='NormalizeColor')],
        test_cfg=dict(
            voxelize=dict(type='SO3IdentityFragments'),
            crop=dict(
                type='SO3SphereCover',
                point_max=32768,
                radius=20.0,
                seed=12345),
            post_transform=[
                dict(
                    type='SO3Pack',
                    neighbors=(12, 16, 16, 24, 24),
                    ratios=(0.25, 0.25, 0.25, 0.25),
                    interp_k=3,
                    mode='eval',
                    seed=12345,
                    feature_keys=('color', ))
            ],
            aug_transform=[[]])))
