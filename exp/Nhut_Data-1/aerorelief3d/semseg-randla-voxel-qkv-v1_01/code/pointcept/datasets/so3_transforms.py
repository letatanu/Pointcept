"""Optional Pointcept transforms for the supplied gravity-free starter config.

Copy to pointcept/datasets/so3_transforms.py and import it in datasets/__init__.py.
Use local sampling for large training tiles to preserve neighborhood density. These transforms
do not voxel-subsample, rotate, apply world-Z height, or use axis-aligned crops.
"""

import numpy as np
from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class SO3Prepare:
    def __init__(self, max_points=None, center=True, sampling="random", seed=None):
        self.max_points = max_points
        self.center = center
        if sampling not in ("random", "local"):
            raise ValueError("sampling must be random or local")
        self.sampling = sampling
        self.seed = seed
        if max_points is not None and max_points < 1:
            raise ValueError("max_points must be positive or None")

    def __call__(self, data):
        n = len(data["coord"])
        if n == 0:
            raise ValueError("Empty scene block")
        if "point_id" not in data:
            data["point_id"] = np.arange(n, dtype=np.int64)
        if np.unique(data["point_id"]).size != n:
            raise ValueError("point_id must be unique within a scene block")
        keys = list(
            data.get(
                "index_valid_keys",
                ["coord", "color", "normal", "strength", "segment", "instance"],
            )
        )
        if "point_id" not in keys:
            keys.append("point_id")
        data["index_valid_keys"] = keys
        if self.max_points is not None and n > self.max_points:
            # A local RNG gives repeatable validation without resetting global RNG.
            rng = np.random if self.seed is None else np.random.RandomState(self.seed)
            if self.sampling == "local":
                # Distance-based crop around a uniformly sampled point. Keeps
                # local density; invariant to rigid motions away from ties.
                anchor = rng.randint(n)
                relative = data["coord"].astype(np.float64) - data["coord"][
                    anchor
                ].astype(np.float64)
                distance2 = np.einsum("ij,ij->i", relative, relative)
                selected = np.sort(
                    np.argpartition(distance2, self.max_points - 1)[: self.max_points]
                )
            else:
                # Coordinate-independent subset for already-small blocks.
                selected = np.sort(rng.choice(n, self.max_points, replace=False))
            for key in keys:
                if key in data:
                    data[key] = data[key][selected]
        if self.center:
            coord = data["coord"].astype(np.float64)
            data["coord"] = (coord - coord.mean(0)).astype(np.float32)
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
        n = len(data["coord"])
        if n > self.max_points:
            raise ValueError(
                f"Test block has {n} points, exceeding {self.max_points}. "
                "Prepare smaller blocks; test points were not dropped."
            )
        data["index"] = np.arange(n, dtype=np.int64)
        return [data]


@TRANSFORMS.register_module()
class SO3SpatialPartition:
    """Full-coverage test fragments using recursive spatial bisection.

    Split planes follow point-derived directions rather than world axes.
    Each original point appears exactly once; index maps it back to the scene.
    Fragments do not overlap, so boundary context differs from whole-scene inference.
    """

    def __init__(self, max_points=16384, center=True):
        if (
            isinstance(max_points, bool)
            or not isinstance(max_points, (int, np.integer))
            or max_points < 1
        ):
            raise ValueError("max_points must be a positive integer")
        self.max_points = int(max_points)
        self.center = center

    def __call__(self, data):
        coord = np.asarray(data["coord"])
        if (
            coord.ndim != 2
            or coord.shape[1] != 3
            or len(coord) == 0
            or not np.isfinite(coord).all()
        ):
            raise ValueError("coord must be nonempty finite [N, 3]")
        n = len(coord)
        keys = set(data.get("index_valid_keys", [])) | {
            "coord",
            "color",
            "normal",
            "strength",
            "segment",
            "instance",
            "displacement",
            "point_id",
            "index",
        }
        for key in keys & data.keys():
            if (
                not isinstance(data[key], np.ndarray)
                or data[key].ndim == 0
                or len(data[key]) != n
            ):
                raise ValueError(
                    f"{key} must be a point-aligned numpy array with {n} rows"
                )
        scene_index = data.get("index", np.arange(n, dtype=np.int64))
        point_id = data.get("point_id", np.arange(n, dtype=np.int64))
        pending = [np.arange(n, dtype=np.int64)]
        fragments = []
        while pending:
            selected = pending.pop()
            if len(selected) > self.max_points:
                x = coord[selected].astype(np.float64)
                relative = x - x.mean(0)
                anchor = np.argmax(np.einsum("ij,ij->i", relative, relative))
                relative = x - x[anchor]
                opposite = np.argmax(np.einsum("ij,ij->i", relative, relative))
                direction = relative[opposite]
                projection = relative @ direction
                # Balanced split guarantees termination, even for duplicate coordinates.
                order = np.argsort(projection, kind="stable")
                middle = len(order) // 2
                pending.append(selected[order[middle:]])
                pending.append(selected[order[:middle]])
                continue
            selected = np.sort(selected)
            fragment = dict(data)
            for key in keys & data.keys():
                fragment[key] = data[key][selected].copy()
            fragment["index"] = scene_index[selected].copy()
            fragment["point_id"] = point_id[selected].copy()
            fragment["index_valid_keys"] = sorted(keys)
            if self.center:
                x = coord[selected].astype(np.float64)
                fragment["coord"] = (x - x.mean(0)).astype(np.float32)
            fragments.append(fragment)
        return fragments
