"""Pointcept registrations for the accompanying FPT config."""
import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from pointcept.datasets.transform import TRANSFORMS
from pointcept.utils.scheduler import SCHEDULERS


@TRANSFORMS.register_module()
class FPTNormalizeColor:
    def __call__(self, data):
        # Input must be raw RGB in [0,255], before NormalizeColor.
        data['color'] = (data['color'].astype(np.float32) - 128.) / 256.
        return data


@TRANSFORMS.register_module()
class FPTRectangularCrop:
    def __init__(self, size=4., min_points=100, retries=40):
        self.size, self.min_points, self.retries = size, min_points, retries

    def __call__(self, data):
        xyz = data['coord']
        shifted = xyz - xyz.min(0)
        span = np.maximum(shifted.max(0) - self.size, 0)
        if not np.any(span):
            return data
        for _ in range(self.retries):
            low = np.random.rand(3) * span
            keep = ((shifted > low) & (shifted < low + self.size)).all(1)
            if keep.sum() > self.min_points:
                for key in ('coord', 'color', 'normal', 'segment', 'instance', 'strength'):
                    if key in data:
                        data[key] = data[key][keep]
                break
        return data


@SCHEDULERS.register_module()
class FPTWarmupCosineLR(LambdaLR):
    """Linear warmup from zero, then cosine to zero; step-based."""
    def __init__(self, optimizer, total_steps, warmup_ratio=0.01, last_epoch=-1):
        warmup = max(2, int(total_steps * warmup_ratio))
        if total_steps <= warmup:
            raise ValueError('total_steps must exceed warmup')
        def factor(step):
            if step < warmup:
                return step / (warmup - 1)
            t = min(1., (step - warmup) / (total_steps - warmup))
            return .5 * (1. + math.cos(math.pi * t))
        super().__init__(optimizer, factor, last_epoch=last_epoch)


@TRANSFORMS.register_module()
class FPTFullResolutionTest:
    def __call__(self, data):
        data['index'] = np.arange(len(data['coord']))
        return [data]
