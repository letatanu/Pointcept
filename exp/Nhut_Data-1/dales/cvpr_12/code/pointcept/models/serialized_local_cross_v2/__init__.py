from .model import SerializedLocalCross
# Register with Pointcept when installed; keep standalone CPU tests importable.
from .adapter import *
__all__ = ["SerializedLocalCross"]
