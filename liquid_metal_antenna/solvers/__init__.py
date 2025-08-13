from .base import BaseSolver, SolverResult

# Optional imports with fallbacks
try:
    from .fdtd import DifferentiableFDTD
    FDTD_AVAILABLE = True
except ImportError:
    FDTD_AVAILABLE = False
    DifferentiableFDTD = None

try:
    from .simple_fdtd import SimpleFDTD
    SIMPLE_FDTD_AVAILABLE = True
except ImportError:
    SIMPLE_FDTD_AVAILABLE = False
    SimpleFDTD = None

# Export what's available
__all__ = ["BaseSolver", "SolverResult"]
if FDTD_AVAILABLE:
    __all__.append("DifferentiableFDTD")
if SIMPLE_FDTD_AVAILABLE:
    __all__.append("SimpleFDTD")