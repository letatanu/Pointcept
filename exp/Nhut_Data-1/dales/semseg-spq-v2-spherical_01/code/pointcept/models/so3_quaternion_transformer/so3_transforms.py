"""Rotation-consistent crops and CPU spatial-tree graphs for SO3-PTv2-Lite.

Install in pointcept/datasets/so3_transforms.py and import from datasets/__init__.py.
SO3Pack is the LAST transform: it replaces ToTensor + Collect and preserves all
hierarchy tensors. Standard Pointcept collate_fn concatenates them; the model
converts local indices to packed indices. mix_prob MUST be 0.

Geometric decisions use Euclidean distance, not axis-aligned voxel membership.
Fixed-seed evaluation preserves sampled point identities under rotation. Exact
distance ties are resolved by point index; floating-point near-ties can still
change under rotation. Point permutation requires permuting persistent IDs / RNG
choices consistently; this is not an unconditional permutation-invariance claim.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.spatial import cKDTree

try:
    from pointcept.datasets.transform import TRANSFORMS
except ModuleNotFoundError as exc:
    if exc.name != "pointcept":
        raise
    TRANSFORMS = None


def register(cls):
    return TRANSFORMS.register_module()(cls) if TRANSFORMS is not None else cls


def coordinates(data):
    xyz = np.asarray(data["coord"], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("coord must be finite, nonempty [N,3]")
    return xyz


def select_points(data, index):
    """Select only per-point arrays, preserving scalar/string metadata."""
    n = len(data["coord"])
    return {key: (value[index].copy() if isinstance(value, np.ndarray)
                  and value.ndim > 0 and len(value) == n else value)
            for key, value in data.items()}


def random_generator(mode, seed):
    # NumPy's worker-seeded global RNG for training; repeatable evaluation RNG.
    return np.random if mode == "train" else np.random.RandomState(seed)


def exact_knn(source, query, k, tree=None):
    """Exact cKDTree search, deterministic exact ties, explicit padding mask.

    This is not the exhaustive O(N*M) GPU pointops query. SciPy's tree prunes
    candidates with exact Euclidean bounds (eps=0). workers=1 avoids multiplying
    thread counts inside multiple data-loader processes. No coordinate gradients.
    """
    if k < 1 or len(source) == 0:
        raise ValueError("k and source size must be positive")
    tree = cKDTree(source) if tree is None else tree
    valid_k = min(k, len(source))
    request = min(valid_k + 1, len(source))
    distances, indices = tree.query(query, k=request, eps=0.0, workers=1)
    if request == 1:
        distances, indices = distances[:, None], indices[:, None]
    # Sort equal-distance neighbors by their local point identity.
    order = np.lexsort((indices, distances), axis=1)
    distances = np.take_along_axis(distances, order, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    if request > valid_k:
        tied = np.flatnonzero(distances[:, valid_k-1] == distances[:, valid_k])
        for row in tied:
            radius = np.nextafter(distances[row, valid_k-1], np.inf)
            candidates = np.asarray(tree.query_ball_point(query[row], radius), dtype=np.int64)
            dd = np.linalg.norm(source[candidates] - query[row], axis=1)
            chosen = np.lexsort((candidates, dd))[:valid_k]
            indices[row, :valid_k] = candidates[chosen]
            distances[row, :valid_k] = dd[chosen]
    indices, distances = indices[:, :valid_k], distances[:, :valid_k]
    mask = np.zeros((len(query), k), dtype=bool)
    mask[:, :valid_k] = True
    if valid_k < k:
        pad = k - valid_k
        indices = np.concatenate((indices, np.repeat(indices[:, :1], pad, axis=1)), 1)
        distances = np.concatenate((distances, np.repeat(distances[:, :1], pad, axis=1)), 1)
    return indices.astype(np.int64), distances, mask


def interpolation_weights(distance, mask):
    """Inverse-distance interpolation with an explicit exact-match branch."""
    zero = (distance == 0) & mask
    has_zero = zero.any(1, keepdims=True)
    inverse = np.zeros_like(distance)
    np.divide(1.0, distance, out=inverse, where=mask & (distance > 0))
    ordinary = inverse / np.maximum(inverse.sum(1, keepdims=True), np.finfo(np.float64).tiny)
    exact = zero / np.maximum(zero.sum(1, keepdims=True), 1)
    return np.where(has_zero, exact, ordinary).astype(np.float32)


@register
class SO3Center:
    """Centroid subtraction, done before float32 conversion when possible."""
    def __call__(self, data):
        xyz = coordinates(data)
        data = dict(data)
        data["coord"] = (xyz - xyz.mean(0)).astype(np.float32)
        return data


@register
class SO3RandomRotate:
    """Uniform SO(3) rotation from a normalized Gaussian quaternion.

    This augmentation is optional: mathematical invariance does not require it.
    It is included as a useful training-time numerical/sampling robustness check.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if np.random.random() >= self.p:
            return data
        q = np.random.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        data = dict(data)
        data["coord"] = (coordinates(data) @ rotation.T).astype(np.float32)
        if "normal" in data:
            data["normal"] = (data["normal"] @ rotation.T).astype(np.float32)
        return data


@register
class SO3TrainCrop:
    """Point-centered metric sphere, then random cap. radius uses dataset units.

    Unlike taking only the nearest 32k points in a dense aerial cloud, this keeps
    a fixed spatial context before thinning. Set radius according to object size
    and actual coordinate units. Rare-class crop sampling is a separate ablation.
    """
    def __init__(self, point_max=32768, radius=20.0, min_points=512):
        if point_max < 1 or radius <= 0 or not 1 <= min_points <= point_max:
            raise ValueError("Invalid crop sizes/radius")
        self.point_max, self.radius, self.min_points = point_max, radius, min_points

    def __call__(self, data):
        xyz = coordinates(data)
        center = np.random.randint(len(xyz))
        d2 = ((xyz - xyz[center]) ** 2).sum(-1)
        index = np.flatnonzero(d2 <= self.radius ** 2)
        # Sparse areas may need a larger support to obtain min_points.
        if len(index) < min(self.min_points, len(xyz)):
            index = np.lexsort((np.arange(len(xyz)), d2))[:min(self.min_points, len(xyz))]
        if len(index) > self.point_max:
            others = index[index != center]
            index = np.r_[center, np.random.choice(others, self.point_max-1, replace=False)]
        return select_points(data, np.sort(index))


@register
class SO3ValidationSample:
    """FAST validation proxy: a fixed random whole-scene sample + nearest map.

    All raw labels contribute to reported validation IoU through inverse mapping,
    but predictions are based on the sampled scene. This is NOT the same protocol
    as full-resolution sphere-cover inference. Use the test pipeline for final
    metrics and compare baselines under identical sampling/evaluation protocols.
    Validation batch_size must be 1 because Pointcept's inverse map is local.
    """
    def __init__(self, point_max=65536, seed=12345):
        if point_max < 1:
            raise ValueError("point_max must be positive")
        self.point_max, self.seed = point_max, seed

    def __call__(self, data):
        xyz = coordinates(data)
        labels = np.asarray(data["segment"]).reshape(-1).copy()
        if len(xyz) <= self.point_max:
            index = np.arange(len(xyz))
            inverse = index.copy()
        else:
            rng = np.random.RandomState(self.seed)
            index = np.sort(rng.choice(len(xyz), self.point_max, replace=False))
            inverse = exact_knn(xyz[index], xyz, 1)[0][:, 0]
        result = select_points(data, index)
        result["origin_segment"], result["inverse"] = labels, inverse
        return result


@register
class SO3IdentityFragments:
    """Satisfy Pointcept's test_cfg.voxelize API without voxelizing the cloud."""
    def __call__(self, data):
        data = dict(data)
        data["index"] = np.arange(len(data["coord"]), dtype=np.int64)
        return [data]


@register
class SO3SphereCover:
    """Deterministic overlapping spherical fragments covering EVERY input point.

    Use as test_cfg.crop. Coverage is guaranteed by reserving at least half of
    each capped fragment for previously uncovered points, always including its
    center. Very sparse data can create small fragments; this is intentional.
    Pointcept averages/votes fragment predictions by the preserved index array.
    """
    def __init__(self, point_max=32768, radius=20.0, seed=12345):
        if point_max < 2 or radius <= 0:
            raise ValueError("Invalid sphere-cover settings")
        self.point_max, self.radius, self.seed = point_max, radius, seed

    def __call__(self, data):
        xyz = coordinates(data)
        if "index" not in data:
            data = dict(data, index=np.arange(len(xyz), dtype=np.int64))
        tree = cKDTree(xyz)
        rng = np.random.RandomState(self.seed)
        order = rng.permutation(len(xyz))
        covered = np.zeros(len(xyz), dtype=bool)
        fragments = []
        for center in order:
            if covered[center]:
                continue
            candidates = np.asarray(sorted(tree.query_ball_point(xyz[center], self.radius)), dtype=np.int64)
            if len(candidates) > self.point_max:
                fresh = candidates[(~covered[candidates]) & (candidates != center)]
                n_fresh = min(len(fresh), max(self.point_max // 2 - 1, 0))
                chosen = np.r_[center, rng.choice(fresh, n_fresh, replace=False)]
                remaining = np.setdiff1d(candidates, chosen, assume_unique=True)
                extra = rng.choice(remaining, self.point_max-len(chosen), replace=False)
                candidates = np.sort(np.r_[chosen, extra])
            covered[candidates] = True
            fragments.append(select_points(data, candidates))
        if not covered.all():
            raise RuntimeError("Internal error: incomplete test coverage")
        return fragments


@register
class SO3Pack:
    """Build multiresolution graphs in a worker, then return tensor-only inputs.

    Run AFTER every crop, rotation, scaling, jitter, dropout and color transform.
    Altering point coordinates/identity after this transform invalidates graphs.
    Fine->coarse pooling reuses fine kNN rows. Coarse->fine interpolation is
    searched once here. Model blocks and the decoder reuse this hierarchy.
    """
    def __init__(self, neighbors=(12, 16, 16, 24, 24), ratios=(0.25,)*4,
                 interp_k=3, mode="train", seed=12345, feature_keys=("color",)):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be train or eval")
        if len(ratios) != len(neighbors)-1 or any(not 0 < r <= 1 for r in ratios):
            raise ValueError("Need one valid sampling ratio per transition")
        if min(neighbors) < 1 or interp_k < 1:
            raise ValueError("Neighbor counts must be positive")
        if "coord" in feature_keys or "normal" in feature_keys:
            raise ValueError("feature_keys must contain scalar invariants, not XYZ/normals")
        self.neighbors, self.ratios = tuple(neighbors), tuple(ratios)
        self.interp_k, self.mode, self.seed = interp_k, mode, seed
        self.feature_keys = tuple(feature_keys)

    def __call__(self, data):
        xyz = coordinates(data)
        # Center before float32 storage; absolute position is deliberately unused.
        xyz = (xyz - xyz.mean(0)).astype(np.float32).astype(np.float64)
        rng = random_generator(self.mode, self.seed)
        feature_parts = []
        for key in self.feature_keys:
            part = np.asarray(data[key], dtype=np.float32)
            if part.ndim == 1:
                part = part[:, None]
            if len(part) != len(xyz) or not np.isfinite(part).all():
                raise ValueError(f"Invalid scalar feature: {key}")
            feature_parts.append(part)
        result = {"coord": torch.from_numpy(xyz.astype(np.float32)),
                  "feat": torch.from_numpy(np.concatenate(feature_parts, -1)),
                  "offset": torch.tensor([len(xyz)], dtype=torch.int64)}
        for key in ("segment", "origin_segment", "inverse", "index"):
            if key in data:
                result[key] = torch.from_numpy(np.asarray(data[key], dtype=np.int64).reshape(-1).copy())
        sizes = []
        for level, k in enumerate(self.neighbors):
            tree = cKDTree(xyz)
            index, _, mask = exact_knn(xyz, xyz, k, tree)
            result[f"so3_knn_{level}"] = torch.from_numpy(index)
            result[f"so3_mask_{level}"] = torch.from_numpy(mask)
            sizes.append(len(xyz))
            if level < len(self.ratios):
                count = max(1, math.ceil(len(xyz) * self.ratios[level]))
                take = np.sort(rng.choice(len(xyz), count, replace=False)).astype(np.int64)
                coarse = xyz[take]
                up, distance, valid = exact_knn(coarse, xyz, self.interp_k)
                result[f"so3_take_{level+1}"] = torch.from_numpy(take)
                result[f"so3_up_{level+1}"] = torch.from_numpy(up)
                result[f"so3_up_weight_{level+1}"] = torch.from_numpy(interpolation_weights(distance, valid))
                xyz = coarse
        result["so3_sizes"] = torch.tensor([sizes], dtype=torch.int64)
        return result


def collate_so3(samples):
    """Standalone equivalent of Pointcept collation for these tensor-only inputs.

    Pointcept users do not need to install a custom collator. This helper is used
    by the numerical tests and optional standalone inference/benchmark scripts.
    """
    result = {}
    if not samples or any(set(item) != set(samples[0]) for item in samples):
        raise ValueError("Need nonempty samples with matching keys")
    for key in samples[0]:
        tensors = [sample[key] for sample in samples]
        if key == "offset":
            counts = [torch.diff(t, prepend=t.new_zeros(1)) for t in tensors]
            result[key] = torch.cat(counts).cumsum(0)
        else:
            result[key] = torch.cat(tensors, 0)
    return result
