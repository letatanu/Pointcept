"""Import this module from Pointcept's models package to register the backbone."""
from pointcept.models.builder import MODELS
from .model import CellQueryTransformer
from . import transforms  # registers the raw-point DALES preprocessing

MODELS.register_module("CellQueryTransformer-v1")(CellQueryTransformer)
