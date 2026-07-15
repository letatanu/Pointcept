import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from pointcept.utils.config import Config, DictAction
from pointcept.datasets import build_dataset
from pointcept.utils.visualization import save_point_cloud

def get_scannet_color_map():
    # Standard ScanNet color map for 20 classes
    return np.array([
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),   # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),   # chair
        (140, 86, 75),    # sofa
        (255, 152, 150),  # table
        (214, 39, 40),    # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),   # counter
        (178, 76, 76),    # desk
        (247, 182, 210),  # curtain
        (66, 188, 102),   # refrigerator
        (219, 219, 141),  # shower curtain
        (140, 57, 197),   # toilet
        (202, 185, 52),   # sink
        (51, 176, 203),   # bathtub
        (200, 54, 131),   # otherfurniture
        (92, 193, 61),    # (extra)
        (78, 71, 183),    # (extra)
        (172, 114, 82),   # (extra)
    ]) / 255.0

def get_aerorelief3d_color_map():
    # Standard ScanNet color map for 20 classes
    return np.array([
        [0, 0, 0],          # Background
        [230, 25, 75],      # Building-Damage
        [70, 240, 240],     # Building-No-Damage
        [255, 255, 25],     # Road
        [0, 128, 0]         # Tree
    ]) / 255.0


def get_random_color_map(num_classes):
    return np.random.rand(num_classes, 3)

def main():
    parser = argparse.ArgumentParser(description="Convert .npy predictions to .ply for visualization")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--options", nargs="+", action=DictAction, help="custom options")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name (folder name inside exp/)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to visualize (val/test)")
    args = parser.parse_args()

    # Load Config
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # Setup paths
    # Assuming standard structure: exp/{exp_name}/result
    result_dir = os.path.join("exp", args.exp_name, "result")
    output_dir = os.path.join(result_dir, "ply_viz")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=> Loading dataset ({args.split})...")
    # Switch dataset split if necessary
    if args.split == "test":
        dataset_cfg = cfg.data.test
    else:
        dataset_cfg = cfg.data.val
        
    dataset = build_dataset(dataset_cfg)

    # Setup Color Map
    if "scannet" in cfg.data.test.type.lower():
        color_map = get_scannet_color_map()
    if "aerorelief3d" in cfg.data.test.type.lower():
        color_map = get_aerorelief3d_color_map()
    # else:
    #     num_classes = cfg.model.num_classes
    #     color_map = get_random_color_map(num_classes)
    print(f"=> Converting .npy to .ply in {output_dir} ...")
    
    # Iterate over dataset
    for idx in tqdm(range(len(dataset))):
        # Use get_data(idx) directly to avoid transforms (like normalization/shifting) 
        # so we get original coordinates
        data_dict = dataset.get_data(idx)
        name = data_dict["name"]
        
        # Path to the prediction .npy file
        # SemSegTester saves as "{name}_pred.npy"
        pred_path = os.path.join(result_dir, f"{name}_pred.npy")
        
        if not os.path.exists(pred_path):
            print(f"Prediction not found for {name}, skipping.")
            continue

        # Load prediction
        pred = np.load(pred_path) # Shape (N,)
        coord = data_dict["coord"] # Shape (N, 3)

        if pred.shape[0] != coord.shape[0]:
            print(f"Size mismatch for {name}: Pred {pred.shape[0]} vs Coord {coord.shape[0]}")
            continue

        # Map predictions to colors
        # Handle ignore index or invalid classes if necessary
        pred_colors = np.zeros_like(coord)
        valid_mask = (pred >= 0) & (pred < len(color_map))
        pred_colors[valid_mask] = color_map[pred[valid_mask]]

        # Save to PLY
        save_path = os.path.join(output_dir, f"{name}.ply")
        save_point_cloud(coord, pred_colors, save_path)

if __name__ == "__main__":
    main()