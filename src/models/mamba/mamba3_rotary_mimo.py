"""Compatibility alias for the renamed MIMO kernel backend."""
import sys
from src.models.mamba import mamba3_mimo_kernels
sys.modules[__name__] = mamba3_mimo_kernels
