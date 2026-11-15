from pointcept.models.builder import MODELS
from .model import SerializedLocalCross
from . import transforms

MODELS.register_module("SerializedLocalCross-v1")(SerializedLocalCross)
