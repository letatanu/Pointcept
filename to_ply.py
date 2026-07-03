import numpy as np
# Assuming visualization.py is in the same directory
import os
import open3d as o3d
import torch

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x

def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(coord) if color is None else color
    )
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")

def save_ply_with_class(coord, color, preds, file_path="output.ply"):
    """
    Saves a PLY file with coordinates, colors, and a custom 'class' scalar field.
    
    coord: (N, 3) float array
    color: (N, 3) int array (0-255)
    preds: (N,) int array
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    num_points = coord.shape[0]

    # Convert everything to the exact types we want to write
    coord = np.ascontiguousarray(coord, dtype=np.float32)
    
    # If colors are [0, 1] floats, scale them back to [0, 255] integers
    if color.dtype == np.float32 or color.dtype == np.float64:
        if color.max() <= 1.0:
            color = (color * 255.0)
    color = np.ascontiguousarray(color, dtype=np.uint8)
    
    preds = np.ascontiguousarray(preds.flatten(), dtype=np.int32)

    # Open file in binary mode for fast writing
    with open(file_path, 'wb') as f:
        # 1. Write the PLY Header (must be ASCII)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property int class\n"  # <--- Custom field for your Semantic Seg predictions!
            "end_header\n"
        )
        f.write(header.encode('ascii'))

        # 2. Combine the data into a structured NumPy array
        # This allows us to interleave the data and dump it to binary in one fast step
        ply_dtype = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('class', 'i4')
        ])
        
        ply_data = np.empty(num_points, dtype=ply_dtype)
        ply_data['x'] = coord[:, 0]
        ply_data['y'] = coord[:, 1]
        ply_data['z'] = coord[:, 2]
        ply_data['red'] = color[:, 0]
        ply_data['green'] = color[:, 1]
        ply_data['blue'] = color[:, 2]
        ply_data['class'] = preds

        # 3. Write data to binary file
        f.write(ply_data.tobytes())

def convert_pointcept_npy_to_ply(pred_npy_path, coord_npy_path, ply_path, logger=None):
    # 1. Load predictions
    preds = np.load(pred_npy_path)
    
    # 2. Load coordinates
    orig_data = np.load(coord_npy_path)
    
    # Check what orig_data actually is to avoid segfaults!
    print(f"DEBUG - orig_data shape: {orig_data.shape}, dtype: {orig_data.dtype}")
    
    # Extract coordinates based on the shape of the dataset
    if len(orig_data.shape) == 1:
        # If it's a 1D structured array (e.g., has named fields like 'x', 'y', 'z')
        try:
            coord = np.vstack((orig_data['x'], orig_data['y'], orig_data['z'])).T
        except ValueError:
            raise ValueError(f"1D array detected but missing x, y, z fields. Fields are: {orig_data.dtype.names}")
    else:
        # If it's a standard 2D array (N, C)
        coord = orig_data[:, :3]
        
    print(f"DEBUG - Extracted coord shape: {coord.shape}, dtype: {coord.dtype}")

    # ENFORCE C-CONTIGUOUS FLOATS
    # Open3D strictly requires float64 (or float32) contiguous memory. 
    # Slicing orig_data[:, :3] creates a non-contiguous view, which causes segfaults in Open3D C++ bindings!
    coord = np.ascontiguousarray(coord, dtype=np.float64)

    if coord.shape[0] != preds.shape[0]:
        raise ValueError(f"Mismatch: Coords ({coord.shape[0]}) vs Preds ({preds.shape[0]})")

    # 3. Generate color map
    np.random.seed(42)
    max_class = preds.max() + 1
    cmap = np.random.rand(max_class, 3)
    color = cmap[preds]
    color = np.ascontiguousarray(color, dtype=np.float64)
        
    # 4. Save
    save_point_cloud(coord=coord, color=color, file_path=ply_path, logger=logger)
    if logger is None:
        print(f"Successfully saved {ply_path}")

# Example usage for your DALES dataset:
# You need the original data file that corresponds to test-5080_54400
# convert_pointcept_npy_to_ply(
#     pred_npy_path="exp/dales/semseg-pt-v3-no-v2-1_01/result/test-5080_54400_pred.npy", 
#     coord_npy_path="data/dales/pointcept/test/5080_54400/coord.npy", # Update this path!
#     ply_path="test-5080_54400_pred.ply"
# )

"""
batch_to_ply.py
---------------
Batch convert Pointcept experiment results to PLY files for multiple datasets.

Supported datasets: S3DIS, ScanNet, SemanticKITTI, DALES, 3DAeroRelief

For each scene, outputs three PLY files:
  - <scene>_pc.ply    : raw point cloud with real RGB colors (no class field)
  - <scene>_pred.ply  : prediction labels + dataset colormap colors + 'class' field
  - <scene>_gt.ply    : ground truth labels + dataset colormap colors + 'class' field

Requires: visualization.py in the same directory (from Pointcept).
"""

import os
import glob
import numpy as np

# =============================================================================
# Dataset Colormaps  (all in [0, 1] float RGB)
# =============================================================================

COLORMAPS = {
    "s3dis": np.array([
        [0.00, 0.75, 1.00],  # 0: ceiling
        [0.75, 0.38, 0.00],  # 1: floor
        [0.00, 0.50, 0.00],  # 2: wall
        [0.75, 0.75, 0.00],  # 3: beam
        [1.00, 0.50, 0.00],  # 4: column
        [1.00, 0.00, 0.00],  # 5: window
        [0.00, 0.00, 1.00],  # 6: door
        [0.50, 0.00, 0.50],  # 7: table
        [1.00, 0.00, 1.00],  # 8: chair
        [0.00, 1.00, 0.00],  # 9: sofa
        [1.00, 1.00, 0.00],  # 10: bookcase
        [0.25, 0.25, 0.25],  # 11: board
        [0.00, 1.00, 1.00],  # 12: clutter
    ], dtype=np.float64),

    "scannet": np.array([
        [0.00, 0.00, 0.00],  # 0: wall
        [0.17, 0.63, 0.17],  # 1: floor
        [1.00, 0.75, 0.80],  # 2: cabinet
        [0.55, 0.34, 0.29],  # 3: bed
        [0.89, 0.47, 0.76],  # 4: chair
        [0.50, 0.50, 0.50],  # 5: sofa
        [0.74, 0.74, 0.13],  # 6: table
        [0.09, 0.75, 0.81],  # 7: door
        [0.65, 0.38, 0.78],  # 8: window
        [0.13, 0.47, 0.71],  # 9: bookshelf
        [1.00, 0.50, 0.06],  # 10: picture
        [0.44, 0.19, 0.63],  # 11: counter
        [0.81, 0.73, 0.55],  # 12: desk
        [1.00, 0.60, 0.60],  # 13: curtain
        [0.77, 0.69, 0.83],  # 14: refrigerator
        [0.35, 0.70, 0.35],  # 15: shower curtain
        [0.20, 0.53, 0.74],  # 16: toilet
        [0.91, 0.59, 0.48],  # 17: sink
        [0.99, 0.99, 0.60],  # 18: bathtub
        [0.70, 0.87, 0.54],  # 19: otherfurniture
    ], dtype=np.float64),

    "semantickitti": np.array([
        [0.00, 0.00, 0.00],  # 0: unlabeled
        [0.87, 0.06, 0.94],  # 1: outlier
        [0.00, 0.00, 1.00],  # 2: car
        [0.00, 0.60, 0.89],  # 3: bicycle
        [0.47, 0.00, 0.20],  # 4: bus
        [0.00, 0.00, 0.55],  # 5: motorcycle
        [0.80, 0.31, 0.00],  # 6: on-rails
        [1.00, 0.00, 0.00],  # 7: truck
        [1.00, 0.60, 0.00],  # 8: other-vehicle
        [0.50, 0.25, 0.00],  # 9: person
        [0.96, 0.00, 0.32],  # 10: bicyclist
        [1.00, 0.45, 0.95],  # 11: motorcyclist
        [0.00, 0.00, 0.00],  # 12: road
        [0.50, 0.50, 0.50],  # 13: parking
        [0.60, 0.60, 0.60],  # 14: sidewalk
        [0.00, 0.50, 0.00],  # 15: other-ground
        [0.62, 0.31, 0.00],  # 16: building
        [0.00, 0.60, 0.00],  # 17: fence
        [0.00, 1.00, 0.00],  # 18: vegetation
        [0.63, 0.94, 0.00],  # 19: trunk
    ], dtype=np.float64),

    "dales": np.array([
        [0.50, 0.50, 0.50],  # 0: Unknown
        [0.00, 0.00, 1.00],  # 1: Ground
        [0.00, 1.00, 0.00],  # 2: Vegetation
        [1.00, 0.00, 0.00],  # 3: Cars
        [1.00, 1.00, 0.00],  # 4: Trucks
        [0.00, 1.00, 1.00],  # 5: Power Lines
        [1.00, 0.00, 1.00],  # 6: Fences
        [1.00, 0.50, 0.00],  # 7: Poles
        [0.50, 0.00, 1.00],  # 8: Buildings
    ], dtype=np.float64),

    "3daerorelief": np.array([
        [0.902, 0.098, 0.294],  # 0: Building-Damage
        [0.275, 0.941, 0.941],  # 1: Building-No-Damage
        [1.000, 1.000, 0.098],  # 2: Road
        [0.000, 0.502, 0.000],  # 3: Tree
        [0.000, 0.000, 0.000],  # 4: Background
    ], dtype=np.float64),
}

DATASET_ALIASES = {
    "s3dis": "s3dis", "scannet": "scannet",
    "semantickitti": "semantickitti", "semantic_kitti": "semantickitti", "kitti": "semantickitti",
    "dales": "dales",
    "3daerorelief": "3daerorelief", "aerorelief": "3daerorelief",
}



# =============================================================================
# Helpers
# =============================================================================

def load_coord(path):
    data = np.load(path)
    coord = np.vstack((data['x'], data['y'], data['z'])).T if data.ndim == 1 else data[:, :3]
    return np.ascontiguousarray(coord, dtype=np.float64)

def load_color(path):
    c = np.load(path)
    if c.ndim == 1:
        c = np.vstack((c['r'], c['g'], c['b'])).T
    elif c.shape[1] > 3:
        c = c[:, :3]
    c = np.ascontiguousarray(c, dtype=np.float64)
    return c / 255.0 if c.max() > 1.0 else c

def get_label_colors(labels, colormap):
    labels = labels.flatten()  +1
    return np.ascontiguousarray(colormap[np.clip(labels, 0, len(colormap) - 1)], dtype=np.float64)


def resolve_paths(base_name, data_dir):
    split, scene_name = base_name.split('-', 1) if '-' in base_name else ('test', base_name)
    scene_dir = os.path.join(data_dir, split, scene_name)
    if not os.path.isdir(scene_dir):
        scene_dir = os.path.join(data_dir, scene_name)
    return {k: os.path.join(scene_dir, f'{k}.npy') for k in ('coord', 'color', 'segment', 'segment20')}


# =============================================================================
# Main
# =============================================================================

def process_dataset(dataset_name, result_dir, data_dir, output_dir=None):
    key = DATASET_ALIASES.get(dataset_name.lower())
    if key is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")
    colormap = COLORMAPS[key]

    if output_dir is None:
        output_dir = result_dir
    os.makedirs(output_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(result_dir, '*_pred.npy')))
    print(f"\n[{dataset_name.upper()}] Found {len(pred_files)} scene(s)")

    for pred_path in pred_files:
        base_name = os.path.basename(pred_path).replace('_pred.npy', '')
        paths = resolve_paths(base_name, data_dir)

        if not os.path.exists(paths['coord']):
            print(f"  [SKIP] coord.npy missing: {paths['coord']}"); continue
        # if not os.path.exists(paths['color']):
        #     print(f"  [SKIP] color.npy missing: {paths['color']}"); continue

        coord = load_coord(paths['coord'])
        if os.path.exists(paths['color']):
            color = load_color(paths['color'])
                    # Raw point cloud — real RGB, no class field
            try:
                save_point_cloud(coord=coord, color=color,
                                file_path=os.path.join(output_dir, f"{base_name}_pc.ply"))
                print(f"    [OK] _pc.ply")
            except Exception as e:
                print(f"    [ERR] _pc.ply: {e}")
                
        print(f"  {base_name} ({coord.shape[0]:,} pts)")

   

        # Prediction
        try:
            preds = np.load(pred_path)
            save_ply_with_class(coord, get_label_colors(preds, colormap), preds,
                                os.path.join(output_dir, f"{base_name}_pred.ply"))
            print(f"    [OK] _pred.ply")
        except Exception as e:
            print(f"    [ERR] _pred.ply: {e}")

        # # Ground truth
        # gt_path = None
        
        # if os.path.exists(paths['segment']): gt_path = paths['segment']
        # elif os.path.exists(paths['segment20']):gt_path = paths['segment20']
        # else:
        #     print(f"    [WARN] segment.npy not found, skipping _gt.ply")
        # try:
        #     gt = np.load(gt_path)
        #     save_ply_with_class(coord, get_label_colors(gt, colormap), gt,
        #                         os.path.join(output_dir, f"{base_name}_gt.ply"))
        #     print(f"    [OK] _gt.ply")
        # except Exception as e:
        #     print(f"    [ERR] _gt.ply: {e}")
    


# =============================================================================
# Configure your experiments here
# =============================================================================

DATASETS = [
    # {"name": "s3dis",          "result_dir": "exp/s3dis/semseg-pt-v3m1-0-base/result",
    #                             "data_dir":   "data/s3dis"},
    # {"name": "scannet",        "result_dir": "exp/scannet/semseg-pt-v3m1-0-base/result",
    #                             "data_dir":   "data/scannet/val"},
    {"name": "dales",          "result_dir": "exp/dales/semseg-pt-v3m1-0-base/result",
                                "data_dir":   "data/dales/pointcept/test"},
    # {"name": "3daerorelief",   "result_dir": "exp/aerorelief3d/semseg-pt-v3m1-0-base/result",
    #                             "data_dir":   "data/aerorelief3d/pointcept/Area_2"},
]

if __name__ == "__main__":
    for ds in DATASETS:
        process_dataset(ds["name"], ds["result_dir"], ds["data_dir"],
                        ds.get("output_dir", None))