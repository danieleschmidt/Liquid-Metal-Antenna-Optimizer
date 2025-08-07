from .antenna_spec import AntennaSpec

# Use the fallback optimizer for now (since torch is not available)
try:
    from .optimizer_fallback import SimpleLMAOptimizer as LMAOptimizer
except ImportError:
    # Last resort: create a dummy optimizer
    class LMAOptimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("No optimizer implementation available.")

__all__ = ["AntennaSpec", "LMAOptimizer"]