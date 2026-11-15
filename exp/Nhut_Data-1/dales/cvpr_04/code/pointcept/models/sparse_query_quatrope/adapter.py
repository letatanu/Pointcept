"""Explicit Pointcept registry integration; no PTv3 backbone dependency."""
from pointcept.models.builder import MODELS

from .model import SparseQueryQuatRoPE
from .transforms import SQQPrepareStrength, SQQIdentityFragments, SQQSpatialTestCrop

MODELS.register_module("SparseQueryQuatRoPE-v1")(SparseQueryQuatRoPE)
