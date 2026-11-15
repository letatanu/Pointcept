weight = None
resume = False
evaluate = True
test_only = False
seed = 59171282
save_path = 'exp/aerorelief3d/semseg-randla-voxel-qkv-v1_01'
num_worker = 16
batch_size = 12
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 3000
eval_epoch = 100
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'bfloat16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
param_dicts = [dict(keyword='blocks', lr=0.0006)]
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
names = ['Background', 'Building-Damage', 'Building-No-Damage', 'Road', 'Tree']
model = dict(
    type='DefaultSegmentorV2',
    num_classes=5,
    backbone_out_channels=64,
    backbone=dict(
        type='RandLA-Voxel-QKV-v3',
        in_channels=6,
        channels=(32, 64, 96, 128, 192),
        enc_depths=(3, 3, 3, 12, 3),
        num_heads=(2, 4, 4, 4, 6),
        voxel_sizes=(0.3, 0.6, 1.2, 2.4, 4.8),
        geometry_width=32,
        input_sample_ratio=1.0,
        sample_ratios=(0.5, 0.5, 0.5, 0.5),
        point_attention_mode='knn',
        point_knn_k=16,
        sparse_backend='neighbor_attention',
        dec_depths=(0, 0, 0, 0),
        decoder_geometry=True,
        edge_chunk_size=16384,
        checkpoint_edges=False,
        checkpoint_blocks=False,
        max_edges=None,
        ffn_ratio=1.0,
        attn_drop=0.0,
        context_residual=True,
        out_channels=64),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'AeroRelief3DDataset'
data_root = 'data/aerorelief3d/pointcept'
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
            dict(type='SphereCrop', sample_rate=0.6, mode='random'),
            dict(type='SphereCrop', point_max=16384, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment'),
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
            dict(type='SphereCrop', point_max=16384, mode='center'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'segment'),
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
            voxelize=None,
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'index'),
                    feat_keys=('coord', 'color'))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }]])))
