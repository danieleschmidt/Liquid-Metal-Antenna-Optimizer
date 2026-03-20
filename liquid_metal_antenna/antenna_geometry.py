"""Patch antenna geometry parameterized for liquid metal channel optimization."""

import numpy as np


class AntennaGeometry:
    """
    Represents a patch antenna with liquid metal microfluidic channels.

    Parameters
    ----------
    length : float
        Patch length in meters.
    width : float
        Patch width in meters.
    channel_positions : list of (float, float)
        (x, y) positions of liquid metal channel centers in meters.
    """

    def __init__(self, length: float, width: float, channel_positions=None):
        if length <= 0:
            raise ValueError("length must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        self.length = float(length)
        self.width = float(width)
        self.channel_positions = list(channel_positions) if channel_positions else []

    def get_params(self) -> dict:
        """Return geometry parameters as a dictionary."""
        return {
            "length": self.length,
            "width": self.width,
            "channel_positions": list(self.channel_positions),
        }

    def perturb(self, delta: float) -> "AntennaGeometry":
        """
        Return a new AntennaGeometry with random perturbations of magnitude delta.

        Parameters
        ----------
        delta : float
            Maximum perturbation magnitude applied to length and width.
        """
        rng = np.random.default_rng()
        new_length = max(1e-4, self.length + rng.uniform(-delta, delta))
        new_width = max(1e-4, self.width + rng.uniform(-delta, delta))
        new_channels = [
            (x + rng.uniform(-delta, delta), y + rng.uniform(-delta, delta))
            for x, y in self.channel_positions
        ]
        return AntennaGeometry(new_length, new_width, new_channels)

    def area(self) -> float:
        """Return the patch area in m²."""
        return self.length * self.width

    def __repr__(self) -> str:
        return (
            f"AntennaGeometry(length={self.length:.4f}, width={self.width:.4f}, "
            f"channels={len(self.channel_positions)})"
        )
