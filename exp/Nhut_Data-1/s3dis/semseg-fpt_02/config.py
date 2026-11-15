weight = None
resume = False
evaluate = True
test_only = False
seed = 14553705
num_worker = 20
batch_size = 16
gradient_accumulation_steps = 1
batch_size_val = 1
batch_size_test = 1
epoch = 1600
eval_epoch = 100
clip_grad = None
sync_bn = False
enable_amp = False
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
model = dict(
    type='FastPointTransformerSegmentor',
    num_classes=13,
    ignore_index=-1,
    backbone=dict(
        type='FastPointTransformer-v1',
        in_channels=3,
        voxel_size=0.04,
        channels=(64, 128, 384, 640, 384, 384, 256, 128),
        depths=(2, 3, 4, 6, 2, 2, 2, 2),
        heads=8),
    criteria=[dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1)])
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = dict(type='FPTWarmupCosineLR', warmup_ratio=0.01)
dataset_type = 'S3DISDataset'
data_root = 'data/s3dis'
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
        data_root='data/s3dis',
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.9),
            dict(
                type='FPTRectangularCrop',
                size=4.0,
                min_points=100,
                retries=40),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.1),
            dict(type='ChromaticJitter', p=0.7, std=0.01),
            dict(type='RandomFlip', p=0.5),
            dict(
                type='RandomShift',
                shift=((-0.12, 0.12), (-0.12, 0.12), (-0.12, 0.12))),
            dict(type='FPTNormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment'),
                feat_keys=('color', ))
        ],
        test_mode=False,
        loop=16),
    val=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root='data/s3dis',
        transform=[
            dict(type='FPTNormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment'),
                feat_keys=('color', ))
        ],
        test_mode=False),
    test=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root='data/s3dis',
        transform=[dict(type='FPTNormalizeColor')],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type='FPTFullResolutionTest'),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'index'),
                    feat_keys=('color', ))
            ],
            aug_transform=[[]])))
save_path = 'exp/s3dis/semseg-fpt_02'
