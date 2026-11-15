# Convolution-free serialized local + sampled cross attention. Fresh run.
weight = None
resume = False
evaluate = True
test_only = False
seed = 3407
save_path = 'exp/dales/semseg-serialized-local-cross-ptv3-orders-no-rotary'
num_worker = 20
batch_size = 4
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
hooks = [{'type': 'CheckpointLoader'},
 {'type': 'ModelHook'},
 {'type': 'IterationTimer', 'warmup_iter': 2},
 {'type': 'InformationWriter'},
 {'type': 'SemSegEvaluator'},
 {'type': 'CheckpointSaver', 'save_freq': None},
 {'type': 'PreciseEvaluator', 'test_last': False}]
train = {'type': 'DefaultTrainer'}
test = {'type': 'SemSegTester', 'verbose': True}
data_root = 'data/dales/pointcept'
strength_scale = 1.0
validation_split = 'test'
point_max = 65536
ignore_index = -1
names = ['Ground',
 'Vegetation',
 'Cars',
 'Trucks',
 'Power lines',
 'Fences',
 'Poles',
 'Buildings']
model = {'type': 'DefaultSegmentorV2',
 'num_classes': 8,
 'backbone_out_channels': 64,
 'backbone': {'type': 'SerializedLocalCross-v1',
              'in_channels': 4,
              'out_channels': 64,
              'channels': (48, 96, 192, 256),
              'heads': (3, 6, 12, 16),
              'depths': (2, 2, 3, 2),
              'patch_sizes': (128, 128, 128, 128),
              'pool_sizes': (0.4, 0.8, 1.6),
              'position_scales': (0.4, 0.8, 1.6, 3.2),
              'cross_group_size': 2,
              'cross_queries': (16, 16, 16, 16),
              'cross_keys': (32, 32, 32, 32),
              'cross_every': 2,
              'cross_dilations': (1, 2),
              'local_rotary': 'none',
              'cross_rotary': 'none',
              'ffn_ratio': 2,
              'score_hidden': 16,
              'chunk_size': 128,
              'attention_backend': 'flash_varlen',
              'serialization_resolution': 0.05,
              'order': ('z', 'z-trans', 'hilbert', 'hilbert-trans'),
              'shuffle_orders': True,
              'serialization_backend': 'pointcept',
              'rotary_min_frequency': 0.25,
              'rotary_max_frequency': 2.0,
              'checkpoint_blocks': False},
 'criteria': [{'type': 'CrossEntropyLoss',
               'loss_weight': 1.0,
               'ignore_index': -1},
              {'type': 'LovaszLoss',
               'mode': 'multiclass',
               'loss_weight': 1.0,
               'ignore_index': -1}]}
optimizer = {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01}
scheduler = {'type': 'OneCycleLR',
 'max_lr': 0.001,
 'pct_start': 0.05,
 'anneal_strategy': 'cos',
 'div_factor': 10.0,
 'final_div_factor': 1000.0}
dataset_type = 'DALESDataset'
data = {'num_classes': 8,
 'ignore_index': -1,
 'names': ['Ground',
           'Vegetation',
           'Cars',
           'Trucks',
           'Power lines',
           'Fences',
           'Poles',
           'Buildings'],
 'train': {'type': 'DALESDataset',
           'split': 'train',
           'data_root': 'data/dales/pointcept',
           'transform': [{'type': 'SLCPrepareStrength', 'scale': 1.0},
                         {'type': 'CenterShift', 'apply_z': True},
                         {'type': 'RandomRotate',
                          'angle': [-1, 1],
                          'axis': 'z',
                          'center': [0, 0, 0],
                          'p': 0.5},
                         {'type': 'RandomScale', 'scale': [0.9, 1.1]},
                         {'type': 'RandomFlip', 'p': 0.5},
                         {'type': 'SphereCrop',
                          'point_max': 65536,
                          'mode': 'random'},
                         {'type': 'CenterShift', 'apply_z': False},
                         {'type': 'ToTensor'},
                         {'type': 'Collect',
                          'keys': ('coord', 'segment'),
                          'feat_keys': ('coord', 'strength')}],
           'test_mode': False,
           'loop': 30},
 'val': {'type': 'DALESDataset',
         'split': 'test',
         'data_root': 'data/dales/pointcept',
         'transform': [{'type': 'SLCPrepareStrength', 'scale': 1.0},
                       {'type': 'CenterShift', 'apply_z': True},
                       {'type': 'SphereCrop',
                        'point_max': 65536,
                        'mode': 'center'},
                       {'type': 'CenterShift', 'apply_z': False},
                       {'type': 'ToTensor'},
                       {'type': 'Collect',
                        'keys': ('coord', 'segment'),
                        'feat_keys': ('coord', 'strength')}],
         'test_mode': False},
 'test': {'type': 'DALESDataset',
          'split': 'test',
          'data_root': 'data/dales/pointcept',
          'transform': [{'type': 'SLCPrepareStrength', 'scale': 1.0},
                        {'type': 'CenterShift', 'apply_z': True}],
          'test_mode': True,
          'test_cfg': {'voxelize': {'type': 'SLCIdentityFragments'},
                       'crop': {'type': 'SLCSpatialTestCrop',
                                'point_max': 65536,
                                'halo_fraction': 0.125,
                                'halo_distance': 2.0},
                       'post_transform': [{'type': 'CenterShift',
                                           'apply_z': False},
                                          {'type': 'ToTensor'},
                                          {'type': 'Collect',
                                           'keys': ('coord', 'index'),
                                           'feat_keys': ('coord', 'strength')}],
                       'aug_transform': [[{'type': 'RandomScale',
                                           'scale': [1.0, 1.0]}]]}}}
split = 'test'
transform = {'type': 'CenterShift', 'apply_z': True}
