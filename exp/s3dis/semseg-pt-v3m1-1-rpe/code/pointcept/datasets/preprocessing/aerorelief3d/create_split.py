import os
import json
import argparse

def create_splits(data_root):
    # Find all processed scene folders (e.g., Area_1-pp1)
    # We filter for directories to ensure we don't pick up files
    scene_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    scene_folders.sort()

    train_list = []
    val_list = []

    print(f"Found {len(scene_folders)} scenes in {data_root}")

    for scene in scene_folders:
        # Standard AeroRelief3D logic: Area 2 is testing/val, others are training
        if "Area_2" in scene:
            val_list.append(scene)
        else:
            train_list.append(scene)

    # Save to .json files as expected by DefaultDataset
    train_json_path = os.path.join(data_root, "train.json")
    val_json_path = os.path.join(data_root, "val.json")

    with open(train_json_path, "w") as f:
        json.dump(train_list, f, indent=4)
    
    with open(val_json_path, "w") as f:
        json.dump(val_list, f, indent=4)

    print(f"Created {train_json_path} with {len(train_list)} scenes.")
    print(f"Created {val_json_path} with {len(val_list)} scenes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/aerorelief3d", help="Path to processed data containing Area folders")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root '{args.data_root}' does not exist.")
    else:
        create_splits(args.data_root)