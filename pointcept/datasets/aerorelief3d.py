import os
from .builder import DATASETS
from .defaults import DefaultDataset

@DATASETS.register_module()
class AeroRelief3DDataset(DefaultDataset):
    """
    AeroRelief3D Dataset for Pointcept.
    
    Expected Data Structure after preprocessing:
    data_root/
        ├── Area_1_pp_1
        │   ├── coord.npy
        │   └── ...
    """
    colors = [
        [0, 0, 0],          # Background
        [230, 25, 75],      # Building-Damage
        [70, 240, 240],     # Building-No-Damage
        [255, 255, 25],     # Road
        [0, 128, 0]         # Tree
    ]
    class_names = [
        "Background",
        "Building-Damage",
        "Building-No-Damage",
        "Road",
        "Tree"
    ]
    VALID_ASSETS = [ "coord", "color", "segment" ]
    def get_data_name(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        dir_path, scene_name = os.path.split(data_path)
        _, parent_name = os.path.split(dir_path)
        if "Area" in parent_name:
            return f"{parent_name}-{scene_name}"
        return scene_name