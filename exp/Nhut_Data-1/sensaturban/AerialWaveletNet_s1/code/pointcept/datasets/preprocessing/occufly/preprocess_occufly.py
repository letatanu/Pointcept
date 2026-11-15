"""Convert OccuFly voxel labels to Pointcept point segmentation arrays.

Requires Python >= 3.9 and numpy. See README_occufly.md for usage and scope.
"""

import argparse
import json
import multiprocessing as mp
import pickle
import tempfile
from pathlib import Path

import numpy as np

GRID_SHAPE = (192, 128, 128)
VOXEL_SIZE = 0.5
CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 21, 22, 33, 34, 35, 36)
CLASS_NAMES = (
    "road", "walkway", "dirt", "gravel", "rock", "grass", "vegetation",
    "tree", "ground_obstacle", "person", "bicycle", "vehicle", "water",
    "building", "roof", "cable", "cable_tower", "parking_lot",
    "construction", "crane", "truck",
)
LABEL_MAP = np.full(256, -1, dtype=np.int32)
LABEL_MAP[list(CLASS_IDS)] = np.arange(len(CLASS_IDS), dtype=np.int32)


def read_mask(path):
    count = int(np.prod(GRID_SHAPE))
    packed = np.fromfile(path, dtype=np.uint8)
    if packed.size != (count + 7) // 8:
        raise ValueError(f"{path}: incorrect packed mask size: {packed.size}")
    # Matches np.unpackbits used in the official OccuFly visualizer.
    return np.unpackbits(packed, bitorder="big", count=count).reshape(GRID_SHAPE).astype(bool)


def load_grid(path, source):
    if source == "raw":
        grid = np.fromfile(path, dtype=np.uint8)
        if grid.size != int(np.prod(GRID_SHAPE)):
            raise ValueError(f"{path}: expected {np.prod(GRID_SHAPE)} label bytes, got {grid.size}")
        grid = grid.reshape(GRID_SHAPE)
        invalid = read_mask(path.with_suffix(".invalid"))
    else:
        # Only use pickle files from a trusted dataset source.
        with path.open("rb") as handle:
            grid = np.asarray(pickle.load(handle)["1_1"])
        invalid = None  # OccuFly preprocessing has already set invalid labels to 255.
    if grid.shape != GRID_SHAPE or grid.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8 {GRID_SHAPE}, got {grid.dtype} {grid.shape}")
    return grid, invalid


def to_points(grid, invalid=None, selection_mask=None):
    valid = (grid != 0) & (grid != 255)
    if invalid is not None:
        valid &= ~invalid
    if selection_mask is not None:
        valid &= selection_mask
    raw_labels = grid[valid]
    segment = LABEL_MAP[raw_labels]
    if np.any(segment < 0):
        raise ValueError(f"Unexpected semantic IDs: {np.unique(raw_labels[segment < 0]).tolist()}")
    # Voxel centers in camera coordinates, meters; C-order keeps labels aligned.
    indices = np.argwhere(valid).astype(np.float32)
    coord = (indices + np.float32(0.5) - np.array([96, 64, 0], dtype=np.float32)) * VOXEL_SIZE
    return {
        "coord": coord.astype(np.float32),
        "color": np.full((len(coord), 3), 255, dtype=np.uint8),
        "segment": segment.astype(np.int32),
        "instance": np.full(len(coord), -1, dtype=np.int32),
    }


def convert_frame(task):
    path, destination, source, selection = task
    try:
        grid, invalid = load_grid(path, source)
        mask = None
        if selection != "occupied":
            # Selection masks are raw ground-truth assets, also when reading pickle labels.
            gt = path.parent if source == "raw" else path.parent.parent / "ground_truth" / path.stem
            suffix = ".surface" if selection == "surface" else ".occluded"
            mask = read_mask(gt / (path.stem + suffix))
            if selection == "visible":
                mask = ~mask
        arrays = to_points(grid, invalid, mask)
        count = len(arrays["coord"])
        if count == 0:
            return {"source": str(path), "status": "empty", "points": 0}
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Publish only fully written samples into the split directory.
        staging = Path(tempfile.mkdtemp(prefix=".frame-", dir=destination.parent.parent))
        try:
            for name, array in arrays.items():
                np.save(staging / (name + ".npy"), array, allow_pickle=False)
            staging.rename(destination)
        finally:
            if staging.exists():
                for file in staging.iterdir():
                    file.unlink()
                staging.rmdir()
        return {"source": str(path), "output": str(destination), "status": "ok", "points": count}
    except Exception as exc:
        return {"source": str(path), "status": "error", "error": str(exc)}


def discover(root, output, source, selection, splits, scenes, altitudes):
    if (root / "OccuFly_Dataset").is_dir():
        root = root / "OccuFly_Dataset"
    tasks = []
    for scene in sorted(root.glob("scene_*")):
        if not scene.is_dir() or scene.name not in {f"scene_{n:02d}" for n in range(1, 10)}:
            continue
        scene_id = int(scene.name.split("_")[1])
        split = "train" if scene_id <= 5 else "val" if scene_id <= 7 else "test"
        if split not in splits or (scenes and scene_id not in scenes):
            continue
        for altitude in altitudes:
            base = scene / str(altitude)
            pattern = "ground_truth/*/*.label" if source == "raw" else "preprocess/*.pkl"
            for path in sorted(base.glob(pattern)):
                if not path.stem.isdigit():
                    continue
                if source == "raw" and path.parent.name != path.stem:
                    raise ValueError(f"Frame directory and filename disagree: {path}")
                name = f"{scene.name}_{altitude}_{path.stem}"
                tasks.append((path, output / split / name, source, selection))
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--source", choices=("raw", "pickle"), default="raw",
                        help="raw .label + .invalid (default), or trusted preprocess/*.pkl")
    parser.add_argument("--selection", choices=("occupied", "surface", "visible"), default="occupied")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--scenes", nargs="+", type=int, choices=range(1, 10))
    parser.add_argument("--altitudes", nargs="+", type=int, choices=(30, 40, 50), default=[30, 40, 50])
    parser.add_argument("--limit", type=int, help="Convert the first N discovered frames for a smoke test")
    args = parser.parse_args()
    if args.num_workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("num_workers and limit must be positive")
    root, output = args.dataset_root.resolve(), args.output_root.resolve()
    if not root.is_dir():
        parser.error(f"Dataset directory not found: {root}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output_root must be new or empty; choose another directory")
    tasks = discover(root, output, args.source, args.selection, set(args.splits), args.scenes, sorted(set(args.altitudes)))
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        parser.error("No matching frames. Check dataset_root, source, scenes and splits.")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "num_classes": 21, "ignore_index": -1, "class_names": CLASS_NAMES,
        "raw_to_train_id": {str(raw): train for train, raw in enumerate(CLASS_IDS)},
        "grid_shape": GRID_SHAPE, "voxel_size_m": VOXEL_SIZE,
        "coordinates": "camera x-right y-down z-forward; voxel centers in meters",
        "source": args.source, "selection": args.selection, "color": "constant white; no measured RGB",
        "split_scenes": {"train": [1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Found {len(tasks)} frames. Converting {args.selection} voxels...", flush=True)
    results = []

    def collect(iterator):
        for result in iterator:
            results.append(result)
            print(f"[{len(results)}/{len(tasks)}] {result['status']}: {result['source']}"
                  + (f" -- {result['error']}" if result['status'] == "error" else f" ({result['points']} points)"), flush=True)

    if args.num_workers == 1:
        collect(map(convert_frame, tasks))
    else:
        with mp.Pool(args.num_workers) as pool:
            collect(pool.imap(convert_frame, tasks, chunksize=1))
    (output / "conversion_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    failures = sum(r["status"] == "error" for r in results)
    converted = sum(r["status"] == "ok" for r in results)
    print(f"Finished: {converted} converted, {failures} failed, {len(results)-converted-failures} empty.")
    if failures or not converted:
        print("Output is incomplete. Check conversion_report.json before training.")
        return 1
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
