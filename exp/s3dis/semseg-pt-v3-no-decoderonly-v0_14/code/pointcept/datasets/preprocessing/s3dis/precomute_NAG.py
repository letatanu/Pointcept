import os
import glob
import torch
import numpy as np
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.nn.pool import voxel_grid
from tqdm import tqdm

def build_hierarchical_nag(pos, features, num_levels=3, k_neighbors=16, voxel_sizes=[0.05, 0.15, 0.45]):
    """
    Builds a multi-level Neural Adjacency Graph (NAG) for a single point cloud.
    
    Args:
        pos (torch.Tensor): [N, 3] Point coordinates.
        features (torch.Tensor): [N, C] Point features (color, normals, etc.).
        num_levels (int): How many levels of superpoints to create.
        k_neighbors (int): Maximum degree for the adjacency graph to limit memory.
        voxel_sizes (list): The resolution for grouping points into superpoints at each level.
        
    Returns:
        dict: A dictionary containing the hierarchical structure.
    """
    nag_dict = {
        "level_0": {
            "pos": pos,
            "features": features,
            # Limit the initial graph connectivity using KNN
            "edge_index": knn_graph(pos, k=k_neighbors, loop=True)
        }
    }
    
    current_pos = pos
    current_batch = torch.zeros(pos.size(0), dtype=torch.long) # Dummy batch vector for 1 scene
    
    for lvl in range(1, num_levels + 1):
        # 1. Group points into superpoints using voxelization (or your custom partitioner)
        v_size = voxel_sizes[lvl - 1]
        cluster_indices = voxel_grid(current_pos, current_batch, size=v_size)
        
        # 2. Compute centroids of the new superpoints
        # scatter_mean is a fast way to average positions based on cluster indices
        from torch_scatter import scatter_mean
        superpoint_pos = scatter_mean(current_pos, cluster_indices, dim=0)
        
        # 3. Create the adjacency graph for the superpoints
        # We use radius_graph or knn_graph to connect neighboring superpoints
        sp_edge_index = knn_graph(superpoint_pos, k=k_neighbors, loop=True)
        
        # 4. Save the level data
        nag_dict[f"level_{lvl}"] = {
            "pos": superpoint_pos,
            "edge_index": sp_edge_index,
            "super_index": cluster_indices # Maps level_{lvl-1} points to level_{lvl} superpoints
        }
        
        # Move up the hierarchy
        current_pos = superpoint_pos
        current_batch = torch.zeros(superpoint_pos.size(0), dtype=torch.long)
        
    return nag_dict

def precompute_s3dis_nag(pointcept_data_root, output_dir):
    """
    Iterates through the Pointcept pre-processed S3DIS structure
    and computes the multi-level NAG for each room.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Grab all the pre-processed .npy files (e.g., Area_1_conferenceRoom_1.npy)
    file_paths = glob.glob(os.path.join(pointcept_data_root, "**", "*.npy"), recursive=True)
    print(f"Found {len(file_paths)} rooms. Starting NAG pre-computation...")
    
    for file_path in tqdm(file_paths):
        file_name = os.path.basename(file_path)
        print(file_path)
        # 1. Load the Pointcept .npy dictionary
        data = np.load(file_path, allow_pickle=True).item()
        
        # 2. Extract arrays and convert to PyTorch tensors
        # Pointcept standard keys are usually 'coord', 'color', and 'semantic_gt'
        pos = torch.tensor(data['coord'], dtype=torch.float32)
        features = torch.tensor(data['color'], dtype=torch.float32) 
        labels = torch.tensor(data['semantic_gt'], dtype=torch.long)
        
        # 3. Build the hierarchical graph
        nag = build_hierarchical_nag(
            pos=pos, 
            features=features,
            num_levels=3,
            k_neighbors=16,          # Capped to prevent memory issues!
            voxel_sizes=[0.04, 0.12, 0.36] # S3DIS appropriate scales
        )
        
        # 4. Attach ground truth labels to the base level so the loss function can use them
        nag["level_0"]["y"] = labels
        
        # 5. Save the pre-computed graph
        save_name = file_name.replace('.npy', '_nag.pth')
        save_path = os.path.join(output_dir, save_name)
        torch.save(nag, save_path)

if __name__ == "__main__":
    # Adjust these paths to where your Pointcept data is stored
    DATA_ROOT = "data/S3DIS/pointcept/" 
    OUTPUT_DIR = "data/S3DIS/s3dis_nag_precomputed"
    
    precompute_s3dis_nag(DATA_ROOT, OUTPUT_DIR)