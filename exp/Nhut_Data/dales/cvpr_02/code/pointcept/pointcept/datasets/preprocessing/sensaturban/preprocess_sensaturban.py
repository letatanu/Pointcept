"""
Preprocess SensatUrban PLY files into Pointcept's per-scene NumPy format.

Input (official SensatUrban):
    <dataset_root>/data_release/train/*.ply
    <dataset_root>/data_release/test/*.ply

or:
    <dataset_root>/train/*.ply
    <dataset_root>/test/*.ply

or the flattened layout used by the official RandLA-Net code:
    <dataset_root>/original_block_ply/*.ply

Each labeled training PLY is expected to contain:
    x, y, z, red, green, blue, class

Official SensatUrban test PLYs do not contain semantic labels. For those files,
segment.npy is filled with -1 so that test points are not accidentally treated
as class 0.

Output:
    <output_root>/train/<scene>/coord.npy
                                color.npy
                                segment.npy
                                instance.npy
    <output_root>/val/<scene>/...
    <output_root>/test/<scene>/...

SensatUrban semantic IDs are already contiguous and are preserved unchanged:
    0  Ground
    1  High Vegetation
    2  Buildings
    3  Walls
    4  Bridge
    5  Parking
    6  Rail
    7  Traffic Roads
    8  Street Furniture
    9  Cars
    10 Footpath
    11 Bikes
    12 Water

The validation split follows the official SensatUrban RandLA-Net code:
    birmingham_block_1
    birmingham_block_5
    cambridge_block_7
    cambridge_block_10

Example:
    python pointcept/datasets/preprocessing/senseturban/preprocess_sensaturban.py \
        --dataset_root data/senseturban/ply \
        --output_root data/senseturban/pointcept \
        --num_workers 2

Dependencies:
    pip install numpy plyfile
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from plyfile import PlyData


# Official validation and test block names used in the SensatUrban reference code.
OFFICIAL_VAL_SCENES = {
    "birmingham_block_1",
    "birmingham_block_5",
    "cambridge_block_7",
    "cambridge_block_10",
}

OFFICIAL_TEST_SCENES = {
    "birmingham_block_2",
    "birmingham_block_8",
    "cambridge_block_15",
    "cambridge_block_16",
    "cambridge_block_22",
    "cambridge_block_27",
}

CLASS_NAMES = {
    0: "Ground",
    1: "High Vegetation",
    2: "Buildings",
    3: "Walls",
    4: "Bridge",
    5: "Parking",
    6: "Rail",
    7: "Traffic Roads",
    8: "Street Furniture",
    9: "Cars",
    10: "Footpath",
    11: "Bikes",
    12: "Water",
}


@dataclass(frozen=True)
class SceneTask:
    ply_path: str
    split: str
    output_root: str
    overwrite: bool


def _property_names(vertex_data) -> set[str]:
    return {prop.name for prop in vertex_data.properties}


def _extract_xyz(vertex_data, properties: set[str], scene_name: str) -> np.ndarray:
    required = {"x", "y", "z"}
    missing = required - properties
    if missing:
        raise ValueError(
            f"{scene_name}: missing coordinate properties {sorted(missing)}. "
            f"Found: {sorted(properties)}"
        )

    coord = np.column_stack(
        (vertex_data["x"], vertex_data["y"], vertex_data["z"])
    )
    return coord.astype(np.float32, copy=False)


def _extract_rgb(vertex_data, properties: set[str], scene_name: str) -> np.ndarray:
    if {"red", "green", "blue"}.issubset(properties):
        color = np.column_stack(
            (vertex_data["red"], vertex_data["green"], vertex_data["blue"])
        )
    elif {"r", "g", "b"}.issubset(properties):
        color = np.column_stack(
            (vertex_data["r"], vertex_data["g"], vertex_data["b"])
        )
    else:
        raise ValueError(
            f"{scene_name}: RGB properties were not found. Expected "
            f"red/green/blue (official SensatUrban) or r/g/b. "
            f"Found: {sorted(properties)}"
        )

    # Official SensatUrban colors are 0..255. The extra branch makes the
    # converter safe for PLYs that were rewritten with colors normalized to 0..1.
    if np.issubdtype(color.dtype, np.floating):
        finite = color[np.isfinite(color)]
        if finite.size and float(np.max(finite)) <= 1.0 + 1e-6:
            color = color * 255.0

    color = np.nan_to_num(color, nan=0.0, posinf=255.0, neginf=0.0)
    color = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    return color


def _extract_segment(
    vertex_data,
    properties: set[str],
    num_points: int,
    scene_name: str,
) -> Tuple[np.ndarray, bool]:
    """Return semantic labels and whether real labels were present."""
    # 'class' is the official SensatUrban field. The fallbacks are useful if a
    # user has re-exported the cloud through another point-cloud tool.
    label_candidates = ("class", "label", "semantic", "segment", "segment_id")
    label_field = next((name for name in label_candidates if name in properties), None)

    if label_field is None:
        return np.full(num_points, -1, dtype=np.int16), False

    segment = np.asarray(vertex_data[label_field]).reshape(-1)
    if segment.shape[0] != num_points:
        raise ValueError(
            f"{scene_name}: label count ({segment.shape[0]}) does not match "
            f"point count ({num_points})."
        )

    # int16 is more than enough for IDs 0..12 and also supports ignore label -1.
    segment = segment.astype(np.int16, copy=False)

    valid = segment >= 0
    unexpected = np.unique(segment[valid & (segment > 12)])
    if unexpected.size:
        print(
            f"Warning: {scene_name} contains label IDs outside SensatUrban 0..12: "
            f"{unexpected.tolist()}"
        )

    return segment, True


def parse_scene(task: SceneTask) -> Tuple[str, str, int, bool]:
    ply_path = task.ply_path
    scene_name = os.path.splitext(os.path.basename(ply_path))[0]
    save_path = os.path.join(task.output_root, task.split, scene_name)

    coord_path = os.path.join(save_path, "coord.npy")
    color_path = os.path.join(save_path, "color.npy")
    segment_path = os.path.join(save_path, "segment.npy")
    instance_path = os.path.join(save_path, "instance.npy")

    expected_outputs = (coord_path, color_path, segment_path, instance_path)
    if not task.overwrite and all(os.path.exists(p) for p in expected_outputs):
        print(f"Skipping existing scene: {task.split}/{scene_name}")
        n = np.load(coord_path, mmap_mode="r").shape[0]
        has_labels = bool(np.any(np.load(segment_path, mmap_mode="r") >= 0))
        return task.split, scene_name, int(n), has_labels

    print(f"Parsing: {task.split}/{scene_name} <- {ply_path}")

    plydata = PlyData.read(ply_path)
    if "vertex" not in plydata:
        raise ValueError(f"{scene_name}: PLY does not contain a 'vertex' element.")

    vertex_data = plydata["vertex"]
    properties = _property_names(vertex_data)

    coord = _extract_xyz(vertex_data, properties, scene_name)
    color = _extract_rgb(vertex_data, properties, scene_name)
    segment, has_labels = _extract_segment(
        vertex_data, properties, coord.shape[0], scene_name
    )

    if color.shape[0] != coord.shape[0]:
        raise ValueError(
            f"{scene_name}: RGB count ({color.shape[0]}) does not match "
            f"point count ({coord.shape[0]})."
        )

    # SensatUrban does not provide instance annotations. -1 is a safe ignore
    # value for loaders/configs that expect instance.npy to exist.
    instance = np.full(coord.shape[0], -1, dtype=np.int16)

    os.makedirs(save_path, exist_ok=True)
    np.save(coord_path, coord)
    np.save(color_path, color)
    np.save(segment_path, segment)
    np.save(instance_path, instance)

    if has_labels:
        labels, counts = np.unique(segment[segment >= 0], return_counts=True)
        label_summary = ", ".join(
            f"{int(label)}:{int(count)}" for label, count in zip(labels, counts)
        )
        print(
            f"Saved {task.split}/{scene_name}: {coord.shape[0]:,} points; "
            f"labels {{{label_summary}}}"
        )
    else:
        print(
            f"Saved {task.split}/{scene_name}: {coord.shape[0]:,} points; "
            "unlabeled -> segment=-1"
        )

    return task.split, scene_name, int(coord.shape[0]), has_labels


def _ply_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.ply")))


def _split_flat_file(ply_path: str) -> str:
    scene_name = os.path.splitext(os.path.basename(ply_path))[0]
    if scene_name in OFFICIAL_TEST_SCENES:
        return "test"
    if scene_name in OFFICIAL_VAL_SCENES:
        return "val"
    return "train"


def discover_tasks(
    dataset_root: str,
    output_root: str,
    overwrite: bool,
) -> List[SceneTask]:
    """Discover common SensatUrban directory layouts."""
    dataset_root = os.path.abspath(dataset_root)
    output_root = os.path.abspath(output_root)

    # 1) Official downloaded structure: <root>/data_release/{train,test}
    # 2) dataset_root may itself be data_release: <root>/{train,test}
    split_roots: Sequence[Tuple[str, str]] = (
        (os.path.join(dataset_root, "data_release", "train"),
         os.path.join(dataset_root, "data_release", "test")),
        (os.path.join(dataset_root, "train"),
         os.path.join(dataset_root, "test")),
    )

    for train_dir, test_dir in split_roots:
        train_files = _ply_files(train_dir) if os.path.isdir(train_dir) else []
        test_files = _ply_files(test_dir) if os.path.isdir(test_dir) else []

        if train_files or test_files:
            tasks: List[SceneTask] = []

            for ply_path in train_files:
                scene_name = os.path.splitext(os.path.basename(ply_path))[0]
                split = "val" if scene_name in OFFICIAL_VAL_SCENES else "train"
                tasks.append(SceneTask(ply_path, split, output_root, overwrite))

            for ply_path in test_files:
                tasks.append(SceneTask(ply_path, "test", output_root, overwrite))

            return tasks

    # 3) Flattened official-repository layout.
    flat_candidates = [
        os.path.join(dataset_root, "original_block_ply"),
        dataset_root,
    ]
    for folder in flat_candidates:
        files = _ply_files(folder) if os.path.isdir(folder) else []
        if files:
            return [
                SceneTask(p, _split_flat_file(p), output_root, overwrite)
                for p in files
            ]

    return []


def print_split_summary(tasks: Iterable[SceneTask]) -> None:
    counts = {"train": 0, "val": 0, "test": 0}
    for task in tasks:
        counts[task.split] += 1
    print(
        "Discovered scenes: "
        f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SensatUrban PLY point clouds to Pointcept .npy format."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help=(
            "SensatUrban root. Supports data_release/{train,test}, "
            "{train,test}, original_block_ply, or a flat directory of PLY files."
        ),
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Destination root for train/val/test scene directories.",
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help=(
            "Number of PLY files processed concurrently. SensatUrban blocks are large, "
            "so 1-2 workers are safer if RAM is limited. Default: 2."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess scenes even when all four output .npy files already exist.",
    )
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num_workers must be >= 1")

    tasks = discover_tasks(args.dataset_root, args.output_root, args.overwrite)
    if not tasks:
        raise FileNotFoundError(
            "No SensatUrban .ply files were found. Expected one of:\n"
            "  <dataset_root>/data_release/train/*.ply\n"
            "  <dataset_root>/data_release/test/*.ply\n"
            "  <dataset_root>/train/*.ply\n"
            "  <dataset_root>/test/*.ply\n"
            "  <dataset_root>/original_block_ply/*.ply\n"
            "  <dataset_root>/*.ply"
        )

    os.makedirs(args.output_root, exist_ok=True)
    print_split_summary(tasks)
    print(f"Output root: {os.path.abspath(args.output_root)}")

    # starmap is not needed because the complete job is stored in SceneTask.
    if args.num_workers == 1:
        results = [parse_scene(task) for task in tasks]
    else:
        # spawn is the most portable choice and avoids inheriting a large PLY
        # object if this script is run from a long-lived Python process.
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.num_workers) as pool:
            results = pool.map(parse_scene, tasks)

    point_counts = {"train": 0, "val": 0, "test": 0}
    for split, _, num_points, _ in results:
        point_counts[split] += num_points

    print("\nProcessing complete.")
    for split in ("train", "val", "test"):
        n_scenes = sum(task.split == split for task in tasks)
        print(
            f"  {split:5s}: {n_scenes:2d} scenes, "
            f"{point_counts[split]:,} points"
        )


if __name__ == "__main__":
    main()
