import os
import numpy as np
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class DALESDataset(DefaultDataset):
    """
    DALES (Dayton Annotated LiDAR Earth Scan) Dataset for Pointcept.
    Airborne LiDAR with x, y, z, reflectance, and semantic labels.

    Labels in raw data: 0 (unlabeled/ignore), 1-8 (valid classes).
    We remap to 0-indexed: class i -> i-1, label 0 -> ignore_index (-1).

    Expected data structure after preprocessing:
    data_root/
        train/   (29 tiles)
            5080_54435/
                coord.npy       # (N, 3) float32
                strength.npy    # (N, 1) float32
                segment.npy     # (N,)   int32, already remapped to 0-7 / -1
            ...
        test/    (11 tiles)
            ...
    """

    class_names = [
        "Ground",       # 1 -> 0
        "Vegetation",   # 2 -> 1
        "Cars",         # 3 -> 2
        "Trucks",       # 4 -> 3
        "Power lines",  # 5 -> 4
        "Fences",       # 6 -> 5
        "Poles",        # 7 -> 6
        "Buildings",    # 8 -> 7
    ]

    colors = [
        [128, 128, 128],   # Ground        - gray
        [0, 192, 0],       # Vegetation    - green
        [255, 0, 0],       # Cars          - red
        [255, 128, 0],     # Trucks        - orange
        [255, 255, 0],     # Power lines   - yellow
        [0, 128, 255],     # Fences        - light blue
        [128, 0, 255],     # Poles         - purple
        [0, 0, 255],       # Buildings     - blue
    ]

    VALID_ASSETS = ["coord", "strength", "segment"]

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        # Remap: 1-8 -> 0-7, 0 (unlabeled) -> ignore_index
        if "segment" in data_dict:
            seg = data_dict["segment"].copy()
            valid_mask = seg > 0
            seg[valid_mask] = seg[valid_mask] - 1
            seg[~valid_mask] = self.ignore_index
            data_dict["segment"] = seg.astype(np.int32)
        return data_dict

    def get_data_name(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        scene_name = os.path.basename(data_path)
        split_name = os.path.basename(os.path.dirname(data_path))
        return f"{split_name}-{scene_name}"