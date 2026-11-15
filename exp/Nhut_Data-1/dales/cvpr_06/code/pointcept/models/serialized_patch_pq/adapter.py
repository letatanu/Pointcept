"""Only registry/transform integration; no Pointcept backbone is imported."""
from pointcept.models.builder import MODELS
from .core import SerializedPatchPQ
from .transforms import SPQPrepareStrength, SPQIdentityFragments, SPQSpatialTestCrop

MODELS.register_module("SerializedPatchPQ-v2")(SerializedPatchPQ)
