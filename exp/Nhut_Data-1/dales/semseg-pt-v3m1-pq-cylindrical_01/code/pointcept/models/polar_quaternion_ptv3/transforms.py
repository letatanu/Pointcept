"""Validate the strength channel without assuming its acquisition scale."""

import numpy as np
from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class PQPrepareStrength:
    def __init__(self, scale=1.0):
        if scale <= 0:
            raise ValueError("strength scale must be positive")
        self.scale = scale

    def __call__(self, data):
        if "strength" not in data:
            raise KeyError("DALESDataset must return strength; map your intensity field to strength")
        value = np.asarray(data["strength"], dtype=np.float32).reshape(-1, 1)
        if value.shape[0] != len(data["coord"]) or not np.isfinite(value).all():
            raise ValueError("strength must contain one finite scalar per point")
        data["strength"] = value / self.scale
        return data


@TRANSFORMS.register_module()
class PQSpatialTestCrop:
    """Full-coverage test fragments via recursive longest-axis spatial splits.

    Returns a list as required by Pointcept's test_cfg.crop interface. No
    training/validation labels are needed. Fragments are disjoint (no halo),
    so boundary effects should be evaluated against uncropped inference.
    """
    def __init__(self, point_max=102400):
        if point_max < 1:
            raise ValueError("point_max must be positive")
        self.point_max = point_max

    def __call__(self, data):
        n = len(data["coord"])
        if n == 0:
            raise ValueError("Cannot crop an empty point cloud")
        work = [np.arange(n)]
        fragments = []
        while work:
            ids = work.pop()
            if len(ids) > self.point_max:
                xyz = data["coord"][ids]
                axis = np.argmax(np.ptp(xyz, axis=0))
                order = np.argsort(xyz[:, axis], kind="stable")
                mid = len(ids) // 2
                work.extend((ids[order[:mid]], ids[order[mid:]]))
                continue
            fragment = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.ndim and len(value) == n:
                    fragment[key] = value[ids].copy()
                else:
                    fragment[key] = value
            # index maps fragment rows to original test points, supplied by GridSample.
            if "index" not in fragment:
                fragment["index"] = ids.copy()
            fragments.append(fragment)
        return fragments
