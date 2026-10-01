"""Create RGB semantic point clouds from OccuFly RGB images and depth maps.

Each valid depth pixel becomes one 3D point. Its semantic label is looked up
from the corresponding OccuFly full-resolution voxel grid. This is a visible
camera point cloud; it does not contain occluded or depth-missing voxels.
"""

import argparse
import json
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
from PIL import Image

GRID = (192, 128, 128)
VOXEL = 0.5
RAW_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 21, 22, 33, 34, 35, 36)
LABEL_MAP = np.full(256, -1, dtype=np.int32)
LABEL_MAP[list(RAW_IDS)] = np.arange(len(RAW_IDS))


def unpack_invalid(path):
    bits = np.unpackbits(np.fromfile(path, np.uint8), bitorder="big", count=np.prod(GRID))
    return bits.reshape(GRID).astype(bool)


def read_labels(frame_dir):
    frame = frame_dir.name
    labels = np.fromfile(frame_dir / f"{frame}.label", np.uint8)
    if labels.size != np.prod(GRID):
        raise ValueError(f"{frame_dir}: label size is {labels.size}, expected {np.prod(GRID)}")
    labels = labels.reshape(GRID)
    invalid_path = frame_dir / f"{frame}.invalid"
    if invalid_path.exists():
        labels[unpack_invalid(invalid_path)] = 255
    return labels


def calibration(path, image_shape, depth_shape, override):
    """Read fx,fy,cx,cy; override is already in depth-pixel coordinates."""
    if override is not None:
        return np.asarray(override, dtype=np.float64)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    nums = [[float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", line)] for line in lines]
    nums = [x for x in nums if x]
    if not nums:
        raise ValueError(f"No numeric calibration values found in {path}")
    # Common formats: one line fx fy cx cy, or a line beginning with Pinhole.
    k = next((x[:4] for x in nums if len(x) >= 4), None)
    if k is None:
        raise ValueError(f"Could not find fx fy cx cy in {path}; pass --fx --fy --cx --cy")
    fx, fy, cx, cy = k
    ih, iw = image_shape
    dh, dw = depth_shape
    # Calibration may be normalized (0..1) or already in image pixels.
    if max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 2.0:
        fx, fy, cx, cy = fx * iw, fy * ih, cx * iw, cy * ih
    fx, cx = fx * dw / iw, cx * dw / iw
    fy, cy = fy * dh / ih, cy * dh / ih
    return np.array([fx, fy, cx, cy], dtype=np.float64)


def load_depth(path, image_shape):
    depth = np.asarray(np.load(path, allow_pickle=False))
    if depth.ndim == 1:
        ih, iw = image_shape
        # Official depth maps are one quarter RGB resolution.
        candidates = [(ih // 4, iw // 4), (ih, iw)]
        shape = next((s for s in candidates if np.prod(s) == depth.size), None)
        if shape is None:
            ratio = iw / ih
            h = int(round(np.sqrt(depth.size / ratio)))
            shape = (h, depth.size // h)
        depth = depth.reshape(shape)
    if depth.ndim != 2:
        raise ValueError(f"{path}: expected 2D or flattened depth map, got {depth.shape}")
    return depth.astype(np.float32)


def convert(task):
    scene, altitude, frame, output, depth_kind, intrinsics = task
    try:
        frame_dir = scene / str(altitude) / "ground_truth" / frame
        image_path = scene / str(altitude) / "images" / "visual" / f"{frame}.png"
        depth_path = scene / str(altitude) / "depth_maps" / f"{frame}.npy"
        if depth_kind == "predicted":
            depth_path = scene.parent.parent / "OccuFly_Predicted_DepthMaps" / scene.name / str(altitude) / "depth_maps" / f"{frame}.npy"
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        depth = load_depth(depth_path, rgb.shape[:2])
        rgb = np.asarray(image.resize((depth.shape[1], depth.shape[0]), Image.Resampling.BILINEAR))
        fx, fy, cx, cy = calibration(scene / "calibration.txt", image.size[::-1], depth.shape, intrinsics)
        v, u = np.indices(depth.shape)
        # OccuFly depth documentation notes zero pixels and occasional very
        # large outliers. The released semantic frustum ends at 64 m, so an
        # outlier beyond that boundary cannot produce a valid voxel label.
        valid = np.isfinite(depth) & (depth > 0) & (depth < GRID[2] * VOXEL)
        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        ix = np.floor(x / VOXEL + GRID[0] / 2).astype(np.int32)
        iy = np.floor(y / VOXEL + GRID[1] / 2).astype(np.int32)
        iz = np.floor(z / VOXEL).astype(np.int32)
        inside = (ix >= 0) & (ix < GRID[0]) & (iy >= 0) & (iy < GRID[1]) & (iz >= 0) & (iz < GRID[2])
        if not np.any(inside):
            return {"status": "empty", "source": f"{scene.name}/{altitude}/{frame}", "points": 0,
                    "reason": "no valid depth points inside 96x64x64 m frustum"}
        labels = read_labels(frame_dir)[ix[inside], iy[inside], iz[inside]]
        keep = (labels != 0) & (labels != 255) & (LABEL_MAP[labels] >= 0)
        coord = np.column_stack((x[inside], y[inside], z[inside]))[keep].astype(np.float32)
        color = rgb[v[valid][inside], u[valid][inside]][keep].astype(np.uint8)
        segment = LABEL_MAP[labels[keep]].astype(np.int32)
        if len(coord) == 0:
            return {"status": "empty", "source": f"{scene.name}/{altitude}/{frame}", "points": 0,
                    "reason": "depth points did not hit occupied semantic voxels"}
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "coord.npy", coord, allow_pickle=False)
        np.save(output / "color.npy", color, allow_pickle=False)
        np.save(output / "segment.npy", segment, allow_pickle=False)
        np.save(output / "instance.npy", np.full(len(coord), -1, np.int32), allow_pickle=False)
        return {"status": "ok", "output": str(output), "points": len(coord)}
    except Exception as exc:
        return {"status": "error", "source": f"{scene.name}/{altitude}/{frame}", "error": str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    p.add_argument("--scenes", nargs="+", type=int, choices=range(1, 10))
    p.add_argument("--altitudes", nargs="+", type=int, choices=(30, 40, 50), default=(30, 40, 50))
    p.add_argument("--depth", choices=("ground_truth", "predicted"), default="ground_truth")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--fx", type=float); p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float); p.add_argument("--cy", type=float)
    args = p.parse_args()
    if args.num_workers < 1 or (args.limit is not None and args.limit < 1): p.error("invalid worker count or limit")
    root = args.dataset_root / "OccuFly_Dataset" if (args.dataset_root / "OccuFly_Dataset").is_dir() else args.dataset_root
    scenes = {f"scene_{n:02d}" for n in args.scenes} if args.scenes else {f"scene_{n:02d}" for n in range(1, 10)}
    split_for = lambda n: "train" if n <= 5 else "val" if n <= 7 else "test"
    override = None if all(v is None for v in (args.fx, args.fy, args.cx, args.cy)) else (args.fx, args.fy, args.cx, args.cy)
    if override is not None and any(v is None for v in override): p.error("provide all of --fx --fy --cx --cy")
    tasks = []
    for scene in sorted(root.glob("scene_*")):
        if scene.name not in scenes: continue
        split = split_for(int(scene.name[-2:]))
        if split not in args.splits: continue
        for altitude in sorted(set(args.altitudes)):
            for image_path in sorted((scene / str(altitude) / "images" / "visual").glob("*.png")):
                frame = image_path.stem
                if (scene / str(altitude) / "ground_truth" / frame / f"{frame}.label").exists():
                    tasks.append((scene, altitude, frame, args.output_root / split / f"{scene.name}_{altitude}_{frame}", args.depth, override))
    if args.limit: tasks = tasks[:args.limit]
    if not tasks: p.error("no matching image/label frames found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    mapper = map if args.num_workers == 1 else None
    pool = None
    if args.num_workers == 1:
        iterator = map(convert, tasks)
    else:
        pool = mp.Pool(args.num_workers)
        iterator = pool.imap(convert, tasks, chunksize=1)
    try:
        for i, result in enumerate(iterator, 1):
            results.append(result); print(f"[{i}/{len(tasks)}] {result}", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    (args.output_root / "conversion_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = sum(x["status"] == "error" for x in results)
    return 1 if failed or not any(x["status"] == "ok" for x in results) else 0


if __name__ == "__main__":
    mp.freeze_support(); raise SystemExit(main())
