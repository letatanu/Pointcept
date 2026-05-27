"""
Preprocessing Script for AeroRelief3D (Single PLY format)
Reads Area_<n>/pp<n>.ply containing xyz, rgb, and label.
Converts to Pointcept .npy format (coord, color, segment).
"""

import os
import argparse
import glob
import numpy as np
import multiprocessing as mp
from plyfile import PlyData

def parse_scene(area, scene_name, dataset_root, output_root):
    try:
        # Input: dataset_root/Area_1/pp1.ply
        scene_file_path = os.path.join(dataset_root, area, f"{scene_name}.ply")
        
        # --- CHANGED HERE ---
        # Output: output_root/Area_1/pp1/
        save_path = os.path.join(output_root, area, scene_name)
        os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(scene_file_path):
            print(f"Warning: File not found {scene_file_path}, skipping.")
            return

        print(f"Parsing: {area}/{scene_name}")

        # 1. Load Data
        plydata = PlyData.read(scene_file_path)
        vertex_data = plydata['vertex']
        properties = [p.name for p in vertex_data.properties]

        # 2. Extract Coordinates (x, y, z)
        coords = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

        # 3. Extract Colors (red, green, blue or r, g, b)
        if 'red' in properties and 'green' in properties and 'blue' in properties:
            colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
        elif 'r' in properties and 'g' in properties and 'b' in properties:
            colors = np.vstack([vertex_data['r'], vertex_data['g'], vertex_data['b']]).T
        else:
            print(f"Warning: No color data found in {scene_name}, using white.")
            colors = np.ones_like(coords) * 255

        # 4. Extract Labels
        # Common names for label fields in PLY files
        possible_label_names = ['label', 'class', 'scalar_Label', 'scalar_class', 'segment_id', 'semantic']
        semantic_gt = None

        for name in possible_label_names:
            if name in properties:
                semantic_gt = np.asarray(vertex_data[name])
                break
        
        if semantic_gt is None:
            # Fallback: Check if it's stored in the 'alpha' channel or a generic scalar
            print(f"Warning: Could not identify label property in {scene_name}. Found: {properties}")
            # Assign ignore_index (-1) if no label is found
            semantic_gt = np.zeros(coords.shape[0]) - 1
        
        # 5. Save to Pointcept .npy format
        # Pointcept expects: coord.npy (float32), color.npy (uint8), segment.npy (int16/int32)
        np.save(os.path.join(save_path, "coord.npy"), coords.astype(np.float32))
        np.save(os.path.join(save_path, "color.npy"), colors.astype(np.uint8))
        np.save(os.path.join(save_path, "segment.npy"), semantic_gt.astype(np.int16))
        
        # Create dummy instance labels (required by some Pointcept loaders even if not used)
        instance_gt = np.ones_like(semantic_gt) * -1
        np.save(os.path.join(save_path, "instance.npy"), instance_gt.astype(np.int16))

    except Exception as e:
        print(f"Error processing {area}/{scene_name}: {e}")

def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Path to raw dataset (containing Area_*)")
    parser.add_argument("--output_root", required=True, help="Output path for processed .npy files")
    parser.add_argument("--num_workers", default=4, type=int, help="Num workers for multiprocessing")
    args = parser.parse_args()

    # Discover all scenes
    # Looks for structure: dataset_root/Area_<n>/pp<n>.ply
    area_folders = glob.glob(os.path.join(args.dataset_root, "Area_*"))
    
    tasks = []
    for area_path in area_folders:
        area_name = os.path.basename(area_path)
        
        # Find all pp*.ply files in this area
        ply_files = glob.glob(os.path.join(area_path, "pp*.ply"))
        
        for ply_path in ply_files:
            # Extract scene name (e.g., 'pp1' from 'pp1.ply')
            scene_filename = os.path.basename(ply_path)
            scene_name = os.path.splitext(scene_filename)[0]
            
            tasks.append((area_name, scene_name, args.dataset_root, args.output_root))

    if len(tasks) == 0:
        print("No 'pp*.ply' files found. Check your dataset_root structure.")
        return

    print(f"Found {len(tasks)} scenes to process.")
    
    # Run in parallel
    with mp.Pool(args.num_workers) as pool:
        pool.starmap(parse_scene, tasks)
    
    print("Processing complete.")

if __name__ == "__main__":
    main_process()