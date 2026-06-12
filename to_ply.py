import numpy as np
# Assuming visualization.py is in the same directory
import os
import open3d as o3d
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
    
    preds = np.ascontiguousarray(preds, dtype=np.int32)

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
convert_pointcept_npy_to_ply(
    pred_npy_path="exp/dales/semseg-pt-v3-no-v2-1_01/result/test-5080_54400_pred.npy", 
    coord_npy_path="data/dales/pointcept/test/5080_54400/coord.npy", # Update this path!
    ply_path="test-5080_54400_pred.ply"
)