"""Preserve RGB and intensity for adapted Utonia fine-tuning."""
import numpy as np
from .transform import TRANSFORMS


@TRANSFORMS.register_module()
class UtoniaRGBStrength:
    def __init__(self, rgb_divisor=255.0):
        if rgb_divisor <= 0:
            raise ValueError("rgb_divisor must be positive")
        self.rgb_divisor = rgb_divisor

    def __call__(self, data_dict):
        n = data_dict["coord"].shape[0]
        color = np.asarray(data_dict["color"], dtype=np.float32)
        if color.shape != (n, 3):
            raise ValueError(f"Expected RGB shape {(n,3)}, got {color.shape}")
        data_dict["color"] = color / self.rgb_divisor
        strength = np.asarray(data_dict["strength"], dtype=np.float32)
        if strength.shape not in ((n,), (n, 1)):
            raise ValueError(f"Expected scalar strength per point, got {strength.shape}")
        data_dict["strength"] = strength.reshape(n, 1)
        if "normal" not in data_dict:
            data_dict["normal"] = np.zeros((n, 3), dtype=np.float32)
        else:
            data_dict["normal"] = np.asarray(data_dict["normal"], dtype=np.float32)
            if data_dict["normal"].shape != (n, 3):
                raise ValueError("Normals must have shape (N,3)")
        return data_dict