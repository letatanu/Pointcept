weight = None
resume = False
evaluate = True
test_only = False
seed = 42
save_path = 'exp/dales/semseg-voxel-quaternion-dales_01'
num_worker = 20
batch_size = 8
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 100
eval_epoch = 100
clip_grad = 3.0
sync_bn = False
enable_amp = False
amp_dtype = 'bfloat16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
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
    type='SO3QuaternionDualSegmentor',
    debug_numerics=False,
    num_classes=8,
    backbone_out_channels=64,
    freeze_backbone=False,
    deep_supervision=False,
    aux_loss_weight=0.2,
    aux_stage_weights=(1.0, 0.5, 0.25, 0.125),
    ignore_index=-1,
    backbone=dict(
        type='Voxel-Quaternion-Dual-v1',
        voxel_sizes=(1.6, 3.2, 6.4, 12.8),
        quaternion_stages=(2, 3),
        in_channels=1,
        scalar_feature_indices=(0, ),
        vector_feature_indices=None,
        channels=(48, 96, 192, 384),
        vector_channels=(8, 8, 16, 16),
        depths=(1, 1, 2, 2),
        heads=(3, 6, 8, 12),
        radii=(1.6, 3.2, 6.4, 12.8),
        patch_sizes=(32, 64, 64, 128),
        max_neighbors=(27, 27, 27, 27),
        quaternion_channels=4,
        geometry_groups=4,
        geometry_width=32,
        decoder_depths=(1, 1, 1),
        decoder_scalar_attention=True,
        use_scalar_attention=True,
        use_quaternion_attention=True,
        global_context_tokens=8,
        head_channels=64,
        mlp_ratio=2.0,
        attn_drop=0.0,
        proj_drop=0.0,
        edge_chunk_size=32768,
        patch_chunk_size=64,
        checkpoint_edges=True,
        radius_backend='hash',
        radius_query_chunk=128,
        radius_candidate_limit=2000000,
        max_radius_edges=8000000,
        readout_triple=True),
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
