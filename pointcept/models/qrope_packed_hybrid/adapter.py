from pointcept.models.builder import MODELS
from .model import QuatRoPEPackedHybrid
from . import transforms

MODELS.register_module("QuatRoPEPackedHybrid-v1")(QuatRoPEPackedHybrid)
