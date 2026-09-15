weight = None
resume = False
evaluate = True
test_only = False
seed = 12523293
save_path = 'exp/aerorelief3d/AerialWaveletNet_02'
num_worker = 20
batch_size = 30
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 1000
eval_epoch = 100
clip_grad = 1.0
sync_bn = False
enable_amp = True
amp_dtype = 'bfloat16'
empty_cache = False
empty_cache_per_epoch = True
find_unused_parameters = False
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0006)]
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
dataset_type = 'AeroRelief3DDataset'
data_root = 'data/aerorelief3d/pointcept'
class_names = [
    'Building-Damage', 'Building-No-Damage', 'Road', 'Tree', 'Background'
]
num_classes = 5
ignore_index = -1
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.006, 0.0006],
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
grid_sample_size = 0.22
train_point_max = 204800
rotation_augmentation = 'none'
model = dict(
    type='AerialWaveletSegmentor',
    network=dict(
        in_channels=6,
        num_classes=5,
        channels=(32, 64, 128, 256),
        depths=(2, 2, 4, 2),
        strides=(2, 2, 2),
        base_voxel_size=0.22,
        k=16,
        neighbor_mode='serialized',
        pooling_mode='serialized',
        patch_size=16,
        exact_backend='torch',
        orders=('hilbert:xyz', 'morton:yzx', 'serpentine:zxy'),
        curve_bits=10,
        window_radius=16,
        chunk_size=512,
        position='quaternion',
        distance_scale=1.0,
        mlp_ratio=2.0,
        dropout=0.0,
        wavelet_stages=(True, True, True, True),
        wavelet_dim=32,
        grid_size=(32, 32, 32),
        wavelet_levels=3,
        wavelet_rank=16,
        wavelet_global_mixing=True,
        context_position=True,
        head_channels=64,
        pool_reduce='max',
        checkpoint_blocks=True),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
rotation_transforms = []
data = dict(
    num_classes=5,
    ignore_index=-1,
    names=[
        'Building-Damage', 'Building-No-Damage', 'Road', 'Tree', 'Background'
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
        loop=10),
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
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [1.0, 1.0]
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }]])))
