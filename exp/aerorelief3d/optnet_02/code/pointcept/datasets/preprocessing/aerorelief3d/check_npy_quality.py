import os
import argparse
import numpy as np
import open3d as o3d

# Same color mapping used in your original convert_segment_to_fpt.py
label_to_color = {
    0: (0, 0, 0),       # Background (Black)
    1: (230, 25, 75),   # Building-Damage (Red)
    2: (70, 240, 240),  # Building-No-Damage (Cyan)
    3: (255, 255, 25),  # Road (Yellow)
    4: (0, 128, 0),     # Tree (Green)
    -1: (128, 128, 128) # Unassigned/Ignore (Gray)
}

def labels_to_colors(labels):
    """Map integer labels back to RGB colors."""
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for label_id, color in label_to_color.items():
        # Mask points that match the current label
        mask = (labels == label_id)
        colors[mask] = color
    return colors

def reverse_npy_to_ply(scene_folder, output_ply, color_mode="label"):
    """
    Reads coord.npy, color.npy, and segment.npy from a folder and saves a PLY.
    color_mode: 
        - "label": saves the PLY with colors mapped from the segmentation labels.
        - "rgb": saves the PLY with the original RGB colors.
    """
    coord_path = os.path.join(scene_folder, "coord.npy")
    color_path = os.path.join(scene_folder, "color.npy")
    segment_path = os.path.join(scene_folder, "segment.npy")

    # 1. Load Data
    if not os.path.exists(coord_path) or not os.path.exists(segment_path):
        print(f"Error: Missing required .npy files in {scene_folder}")
        return

    coords = np.load(coord_path)
    segments = np.load(segment_path)
    
    print(f"Loaded {len(coords)} points.")
    print(f"Unique labels found in segment.npy: {np.unique(segments)}")

    # 2. Assign Colors
    if color_mode == "label":
        # Check label quality visually by applying the color dictionary
        ply_colors = labels_to_colors(segments)
    else:
        # Check if original RGB values were preserved
        if os.path.exists(color_path):
            ply_colors = np.load(color_path)
        else:
            print(f"Warning: color.npy not found. Defaulting to gray.")
            ply_colors = np.ones((len(coords), 3), dtype=np.uint8) * 128

    # 3. Create Open3D Point Cloud and Save
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    
    # Open3D expects colors in [0, 1] float format
    pcd.colors = o3d.utility.Vector3dVector(ply_colors.astype(np.float64) / 255.0)

    # Save to file
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Successfully saved validation PLY to: {output_ply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check .npy quality by converting back to .ply")
    parser.add_argument("--scene_folder", type=str, required=True, help="Path to folder containing coord.npy, segment.npy")
    parser.add_argument("--output_ply", type=str, default="check_quality.ply", help="Output .ply file path")
    parser.add_argument("--mode", type=str, choices=["label", "rgb"], default="label", 
                        help="Visualize the segmentation 'label' map, or original 'rgb' colors")
    args = parser.parse_args()

    reverse_npy_to_ply(args.scene_folder, args.output_ply, args.mode)