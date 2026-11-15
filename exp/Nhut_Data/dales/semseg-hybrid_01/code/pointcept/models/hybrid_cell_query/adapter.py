from pointcept.models.builder import MODELS
from .model import HybridCellQuery
from . import transforms

MODELS.register_module("HybridCellQuery-v1")(HybridCellQuery)
