"""
Liquid Metal Antenna Optimizer

Automated design and optimization of reconfigurable liquid-metal antennas 
using differentiable EM solvers and neural surrogate models.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core import AntennaSpec, LMAOptimizer
from .designs import ReconfigurablePatch, LiquidMetalArray, MetasurfaceAntenna
from .solvers import DifferentiableFDTD
from .liquid_metal import GalinStanModel, FlowSimulator

__all__ = [
    "__version__",
    "AntennaSpec",
    "LMAOptimizer",
    "ReconfigurablePatch",
    "LiquidMetalArray",
    "MetasurfaceAntenna",
    "DifferentiableFDTD",
    "GalinStanModel",
    "FlowSimulator",
]