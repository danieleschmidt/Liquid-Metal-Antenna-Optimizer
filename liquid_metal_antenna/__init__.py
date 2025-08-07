"""
Liquid Metal Antenna Optimizer

Automated design and optimization of reconfigurable liquid-metal antennas 
using differentiable EM solvers and neural surrogate models.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Core components that should always work
from .core import AntennaSpec, LMAOptimizer

# Optional components - only import if dependencies are available
try:
    from .designs import ReconfigurablePatch, LiquidMetalArray, MetasurfaceAntenna
    DESIGNS_AVAILABLE = True
except ImportError:
    DESIGNS_AVAILABLE = False
    ReconfigurablePatch = None
    LiquidMetalArray = None  
    MetasurfaceAntenna = None

try:
    from .solvers import DifferentiableFDTD
    SOLVERS_AVAILABLE = True
except ImportError:
    SOLVERS_AVAILABLE = False
    DifferentiableFDTD = None

try:
    from .liquid_metal import GalinStanModel, FlowSimulator
    LIQUID_METAL_AVAILABLE = True
except ImportError:
    LIQUID_METAL_AVAILABLE = False
    GalinStanModel = None
    FlowSimulator = None

# Export all available components
__all__ = ["__version__", "AntennaSpec", "LMAOptimizer"]

if DESIGNS_AVAILABLE:
    __all__.extend(["ReconfigurablePatch", "LiquidMetalArray", "MetasurfaceAntenna"])
if SOLVERS_AVAILABLE:
    __all__.append("DifferentiableFDTD")
if LIQUID_METAL_AVAILABLE:
    __all__.extend(["GalinStanModel", "FlowSimulator"])