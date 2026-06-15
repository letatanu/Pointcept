"""
Preprocessing Script for DALES (Airborne LiDAR)
Reads train/<scene>.ply and test/<scene>.ply containing x, y, z, reflectance, class.
Converts to Pointcept .npy format: coord.npy, strength.npy, segment.npy
"""

import os
import argparse
import glob
import numpy as np
import multiprocessing as mp
from plyfile import PlyData


def parse_scene(split, scene_name, dataset_root, output_root):
    try:
        scene_file_path = os.path.join(dataset_root, split, f"{scene_name}.ply")
        save_path = os.path.join(output_root, split, scene_name)
        os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(scene_file_path):
            print(f"Warning: File not found {scene_file_path}, skipping.")
            return

        print(f"Parsing: {split}/{scene_name}")

        # 1. Load PLY
        plydata = PlyData.read(scene_file_path)
        vertex_data = plydata["vertex"]
        properties = [p.name for p in vertex_data.properties]
        print(f"  Properties found: {properties}")

        # 2. Coordinates (x, y, z) — normalize to local origin
        x = np.asarray(vertex_data["x"], dtype=np.float64)
        y = np.asarray(vertex_data["y"], dtype=np.float64)
        z = np.asarray(vertex_data["z"], dtype=np.float64)
        coords = np.stack([x, y, z], axis=1)
        # Shift to local origin to avoid floating point precision loss
        coords -= coords.min(axis=0)
        coords = coords.astype(np.float32)

        # 3. Strength / Reflectance
        strength = None
        for name in ["reflectance", "intensity", "Reflectance", "Intensity"]:
            if name in properties:
                strength = np.asarray(vertex_data[name], dtype=np.float32)
                # Normalize to [0, 1] if stored as uint16 or large range
                max_val = strength.max()
                if max_val > 1.0:
                    strength = strength / max_val
                strength = strength.reshape(-1, 1)
                print(f"  Using '{name}' as strength.")
                break

        if strength is None:
            print(f"  Warning: No reflectance/intensity field found. Using zeros.")
            strength = np.zeros((coords.shape[0], 1), dtype=np.float32)

        # 4. Semantic Labels
        segment = None
        for name in ["class", "label", "Class", "Label", "scalar_class", "scalar_label"]:
            if name in properties:
                segment = np.asarray(vertex_data[name], dtype=np.int16)
                print(f"  Using '{name}' as segment label.")
                break

        if segment is None:
            print(f"  Warning: No label field found. Assigning -1 (ignore).")
            segment = np.full(coords.shape[0], -1, dtype=np.int16)

        # 5. Save
        np.save(os.path.join(save_path, "coord.npy"), coords)
        np.save(os.path.join(save_path, "strength.npy"), strength)
        np.save(os.path.join(save_path, "segment.npy"), segment)

        print(f"  Saved {coords.shape[0]} points -> {save_path}")

    except Exception as e:
        print(f"Error processing {split}/{scene_name}: {e}")
        raise


def main_process():
    parser = argparse.ArgumentParser(description="Preprocess DALES PLY dataset for Pointcept")
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to raw DALES PLY root (contains train/ and test/ folders)"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path for processed .npy files"
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of parallel workers"
    )
    args = parser.parse_args()

    tasks = []
    for split in ["train", "test"]:
        split_path = os.path.join(args.dataset_root, split)
        if not os.path.isdir(split_path):
            print(f"Warning: Split folder not found: {split_path}, skipping.")
            continue

        ply_files = sorted(glob.glob(os.path.join(split_path, "*.ply")))
        for ply_path in ply_files:
            scene_name = os.path.splitext(os.path.basename(ply_path))[0]
            tasks.append((split, scene_name, args.dataset_root, args.output_root))

    if not tasks:
        print("No .ply files found. Check your dataset_root structure.")
        return

    print(f"Found {len(tasks)} scenes to process.")

    with mp.Pool(args.num_workers) as pool:
        pool.starmap(parse_scene, tasks)

    print("Processing complete.")


if __name__ == "__main__":
    main_process()