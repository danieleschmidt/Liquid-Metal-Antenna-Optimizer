"""Analytical EM solver for patch antenna resonant frequency and S11."""

import numpy as np
from .antenna_geometry import AntennaGeometry


class EMSolver:
    """
    Analytical approximation of patch antenna EM characteristics.

    Uses the Hammerstad fringe-field correction to compute effective patch
    length and hence the dominant-mode resonant frequency.

    Parameters
    ----------
    substrate_height : float
        Dielectric substrate thickness in meters (default 1.6 mm, FR4).
    """

    C = 3e8  # speed of light (m/s)

    def __init__(self, substrate_height: float = 1.6e-3):
        self.h = float(substrate_height)

    def _effective_length(self, length: float, width: float) -> float:
        """
        Compute effective patch length using Hammerstad fringe correction.

        ΔL = 0.824 * h * (w + 0.264*h) / (w - 0.8*h)
        L_eff = length + ΔL
        """
        h = self.h
        w = width
        denom = w - 0.8 * h
        if denom <= 0:
            # width too small relative to substrate — fall back to no correction
            return length
        delta_L = 0.824 * h * (w + 0.264 * h) / denom
        return length + delta_L

    def resonant_frequency(self, geometry: AntennaGeometry) -> float:
        """
        Return the dominant-mode resonant frequency in Hz.

        f = c / (2 * L_eff)
        """
        L_eff = self._effective_length(geometry.length, geometry.width)
        return self.C / (2.0 * L_eff)

    def compute_s11(
        self,
        geometry: AntennaGeometry,
        freq_range: np.ndarray,
        q_factor: float = 50.0,
    ) -> np.ndarray:
        """
        Return S11 magnitude (dB) over freq_range using a Lorentzian resonance model.

        The dip occurs at the resonant frequency with depth controlled by q_factor.

        Parameters
        ----------
        geometry : AntennaGeometry
        freq_range : np.ndarray
            Array of frequencies in Hz.
        q_factor : float
            Quality factor determining resonance bandwidth.

        Returns
        -------
        np.ndarray
            S11 in dB, same length as freq_range.
        """
        freq_range = np.asarray(freq_range, dtype=float)
        f0 = self.resonant_frequency(geometry)
        # Lorentzian lineshape centred at f0
        gamma = f0 / q_factor  # half-width at half-maximum
        lorentzian = gamma**2 / ((freq_range - f0) ** 2 + gamma**2)
        # At resonance Γ → 0, away from resonance Γ → 1
        reflection = 1.0 - lorentzian
        reflection = np.clip(reflection, 1e-10, 1.0)
        return 20.0 * np.log10(reflection)
