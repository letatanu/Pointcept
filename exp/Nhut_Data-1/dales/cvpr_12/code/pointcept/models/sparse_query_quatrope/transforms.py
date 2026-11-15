"""Pointcept data adapters; no point-reducing voxelization."""
import numpy as np
from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class SQQPrepareStrength:
    def __init__(self, scale=1.):
        if scale <= 0:
            raise ValueError("strength scale must be positive")
        self.scale = scale

    def __call__(self, data):
        if "strength" not in data:
            raise KeyError("DALESDataset must return strength (map intensity to strength)")
        strength = np.asarray(data["strength"],dtype=np.float32).reshape(-1,1)
        if len(strength) != len(data["coord"]) or not np.isfinite(strength).all():
            raise ValueError("Expected one finite strength value per point")
        data["strength"] = strength / self.scale
        return data


@TRANSFORMS.register_module()
class SQQIdentityFragments:
    """Adapt test_cfg.voxelize's list interface without gridding or sampling."""
    def __call__(self, data):
        data = dict(data)
        if "index" not in data:
            data["index"] = np.arange(len(data["coord"]))
        return [data]


@TRANSFORMS.register_module()
class SQQSpatialTestCrop:
    """Full-coverage spatial cores plus bounded context halos.

    Every core point is retained. Extra halo points overlap fragments; the
    standard Pointcept tester merges predictions by their original index.
    max_points limits actual fragments including halos. Crop creation is CPU
    work and may be noticeable for large raw tiles.
    """
    def __init__(self, point_max=65536, halo_fraction=.125, halo_distance=2.):
        if point_max < 1 or not 0 <= halo_fraction < 1 or halo_distance < 0:
            raise ValueError("Invalid crop/halo settings")
        self.point_max = point_max
        self.core_max = max(1,int(point_max*(1-halo_fraction)))
        self.halo_distance = halo_distance

    def __call__(self, data):
        xyz = data["coord"]
        n = len(xyz)
        if n == 0:
            raise ValueError("Cannot crop empty clouds")
        work, fragments = [np.arange(n)], []
        while work:
            core = work.pop()
            if len(core) > self.core_max:
                local = xyz[core]
                axis = np.argmax(np.ptp(local,axis=0))
                permutation = np.argsort(local[:,axis],kind="stable")
                mid = len(core)//2
                work.extend((core[permutation[:mid]],core[permutation[mid:]]))
                continue
            ids = core
            if self.halo_distance > 0 and len(core) < n and len(core) < self.point_max:
                local = xyz[core]
                low, high = local.min(0)-self.halo_distance, local.max(0)+self.halo_distance
                mask = np.all((xyz>=low)&(xyz<=high),axis=1)
                mask[core] = False
                halo = np.flatnonzero(mask)
                room = self.point_max-len(core)
                if len(halo)>room:
                    distance = np.square(xyz[halo]-local.mean(0)).sum(1)
                    halo = halo[np.argpartition(distance,room-1)[:room]]
                ids = np.concatenate((core,halo))
            part = {}
            for key,value in data.items():
                part[key] = (value[ids].copy() if isinstance(value,np.ndarray) and value.ndim
                             and len(value)==n else value)
            if "index" not in part:
                part["index"] = ids.copy()
            fragments.append(part)
        return fragments
