from pointcept.models.builder import MODELS
from .model import PackedHybridCellQuery
from . import transforms

MODELS.register_module("PackedHybridCellQuery-v1")(PackedHybridCellQuery)
