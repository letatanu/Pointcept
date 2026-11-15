"""Optional Pointcept transforms for the supplied gravity-free starter config.

Copy to pointcept/datasets/so3_transforms.py and import it in datasets/__init__.py.
Input scenes should be spatial blocks at a manageable density. These transforms
do not voxel-subsample, rotate, apply world-Z height, or use axis-aligned crops.
"""
import numpy as np
from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class SO3Prepare:
    def __init__(self, max_points=None, center=True):
        self.max_points = max_points
        self.center = center
        if max_points is not None and max_points < 1:
            raise ValueError('max_points must be positive or None')

    def __call__(self, data):
        n = len(data['coord'])
        if n == 0:
            raise ValueError('Empty scene block')
        if 'point_id' not in data:
            data['point_id'] = np.arange(n, dtype=np.int64)
        if np.unique(data['point_id']).size != n:
            raise ValueError('point_id must be unique within a scene block')
        keys = list(data.get('index_valid_keys',
                    ['coord', 'color', 'normal', 'strength', 'segment', 'instance']))
        if 'point_id' not in keys:
            keys.append('point_id')
        data['index_valid_keys'] = keys
        if self.max_points is not None and n > self.max_points:
            # Coordinate-independent subset. Sort back to original identity order.
            selected = np.sort(np.random.choice(n, self.max_points, replace=False))
            for key in keys:
                if key in data:
                    data[key] = data[key][selected]
        if self.center:
            coord = data['coord'].astype(np.float64)
            data['coord'] = (coord - coord.mean(0)).astype(np.float32)
        return data


@TRANSFORMS.register_module()
class SO3IdentityPartition:
    """One fragment per prepared block, with evaluator-compatible original index.

    Explicitly no subsampling at test time. Fail rather than dropping points from
    oversized blocks; prepare manageable blocks before benchmarking.
    """
    def __init__(self, max_points=16384):
        self.max_points = max_points

    def __call__(self, data):
        n = len(data['coord'])
        if n > self.max_points:
            raise ValueError(f'Test block has {n} points, exceeding {self.max_points}. '
                             'Prepare smaller blocks; test points were not dropped.')
        data['index'] = np.arange(n, dtype=np.int64)
        return [data]
