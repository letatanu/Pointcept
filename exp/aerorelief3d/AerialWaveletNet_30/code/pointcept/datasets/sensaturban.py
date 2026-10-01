import os

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SensatUrbanDataset(DefaultDataset):
    """
    SensatUrban dataset for Pointcept.

    Expected data structure after preprocessing:

    data_root/
        ├── train/
        │   ├── birmingham_block_0/
        │   │   ├── coord.npy
        │   │   ├── color.npy
        │   │   ├── segment.npy
        │   │   └── instance.npy
        │   └── ...
        ├── val/
        │   └── ...
        └── test/
            ├── cambridge_block_22/
            │   ├── coord.npy
            │   ├── color.npy
            │   ├── segment.npy
            │   └── instance.npy
            └── ...

    SensatUrban semantic labels:
        0  Ground
        1  High Vegetation
        2  Buildings
        3  Walls
        4  Bridge
        5  Parking
        6  Rail
        7  Traffic Roads
        8  Street Furniture
        9  Cars
        10 Footpath
        11 Bikes
        12 Water

    Notes:
        - Training/validation scenes should contain labels in segment.npy.
        - Official SensatUrban test data is unlabeled. If preprocessing stores
          -1 in segment.npy for test points, Pointcept can still load the data
          for inference.
    """

    # Official SensatUrban visualization colors.
    colors = [
        [85, 107, 47],    # Ground
        [0, 255, 0],      # High Vegetation
        [255, 165, 0],    # Buildings
        [41, 49, 101],    # Walls
        [0, 0, 0],        # Bridge
        [0, 0, 255],      # Parking
        [255, 0, 255],    # Rail
        [200, 200, 200],  # Traffic Roads
        [89, 47, 95],     # Street Furniture
        [255, 0, 0],      # Cars
        [255, 255, 0],    # Footpath
        [0, 255, 255],    # Bikes
        [0, 191, 255],    # Water
    ]

    class_names = [
        "Ground",
        "High Vegetation",
        "Buildings",
        "Walls",
        "Bridge",
        "Parking",
        "Rail",
        "Traffic Roads",
        "Street Furniture",
        "Cars",
        "Footpath",
        "Bikes",
        "Water",
    ]

    VALID_ASSETS = ["coord", "color", "segment", "instance"]

    def get_data_name(self, idx):
        """
        Return the SensatUrban scene name.

        Example:
            /data/sensaturban/train/birmingham_block_0
            -> birmingham_block_0
        """
        data_path = self.data_list[idx % len(self.data_list)]
        return os.path.basename(os.path.normpath(data_path))
