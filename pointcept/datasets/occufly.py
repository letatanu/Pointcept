"""Pointcept dataset for the output of preprocess_occufly.py.

Install as pointcept/datasets/occufly.py and import OccuFlyDataset in
pointcept/datasets/__init__.py to register it.
"""

import os

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class OccuFlyDataset(DefaultDataset):
    """OccuFly point segmentation with 21 contiguous training classes.

    Expected layout:
        data_root/
            train/scene_01_30_000000/{coord,color,segment,instance}.npy
            val/scene_06_30_000000/{coord,color,segment,instance}.npy
            test/scene_08_30_000000/{coord,color,segment,instance}.npy

    segment.npy already contains training IDs 0..20. Do not remap it again.
    Training ID 0 is Road, not empty/background. Ignore index is -1.
    Colors below are for visualization, not replacements for color.npy.
    DefaultDataset handles array loading, split discovery, and transforms.
    """

    class_names = [
        "Road",
        "Walkway",
        "Dirt",
        "Gravel",
        "Rock",
        "Grass",
        "Vegetation",
        "Tree",
        "Ground Obstacle",
        "Person",
        "Bicycle",
        "Vehicle",
        "Water",
        "Building",
        "Roof",
        "Cable",
        "Cable Tower",
        "Parking Lot",
        "Construction",
        "Crane",
        "Truck",
    ]

    # Index = training ID; value = original OccuFly semantic ID.
    raw_class_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 21, 22, 33, 34, 35, 36
    ]

    # Official OccuFly RGB palette in the same order as class_names.
    colors = [
        [128, 0, 128],
        [204, 163, 72],
        [128, 0, 0],
        [192, 192, 192],
        [246, 120, 40],
        [0, 255, 0],
        [112, 148, 32],
        [64, 64, 0],
        [255, 255, 0],
        [255, 16, 255],
        [255, 204, 153],
        [0, 128, 128],
        [0, 0, 255],
        [255, 0, 0],
        [64, 160, 120],
        [255, 160, 0],
        [106, 0, 255],
        [128, 64, 128],
        [240, 120, 120],
        [255, 255, 128],
        [128, 128, 64],
    ]

    VALID_ASSETS = ["coord", "color", "segment", "instance"]

    def get_data_name(self, idx):
        # Scene + altitude + frame is already unique across the official splits.
        # Preserve this name to match <name>_pred.npy and the PLY exporter.
        data_path = self.data_list[idx % len(self.data_list)]
        return os.path.basename(os.path.normpath(data_path))
