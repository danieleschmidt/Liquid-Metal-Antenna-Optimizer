"""
Liquid metal material property models.
"""

import numpy as np
from typing import Union, Dict, Any


class GalinStanModel:
    """
    Galinstan (Ga-In-Sn) liquid metal material property model.
    
    This class provides temperature-dependent electrical and physical
    properties of Galinstan alloy commonly used in liquid metal antennas.
    """
    
    def __init__(self, composition: str = "68.5Ga-21.5In-10Sn"):
        """
        Initialize Galinstan model.
        
        Args:
            composition: Alloy composition (weight percentages)
        """
        self.composition = composition
        
        # Reference properties at 25°C
        self.ref_temperature = 25.0  # °C
        self.ref_conductivity = 3.46e6  # S/m
        self.ref_viscosity = 2.4e-3  # Pa·s
        self.ref_density = 6440.0  # kg/m³
        
        # Temperature coefficients
        self.conductivity_temp_coeff = -1.2e-3  # per °C
        self.viscosity_temp_coeff = -0.045  # per °C (exponential)
        self.density_temp_coeff = -0.6  # kg/m³ per °C
        
        # Physical constants
        self.melting_point = -19.0  # °C
        self.boiling_point = 1300.0  # °C
        self.surface_tension = 0.714  # N/m at 25°C
    
    def conductivity(self, temperature: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate electrical conductivity at given temperature.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Conductivity in S/m
        """
        temp_diff = np.asarray(temperature) - self.ref_temperature
        return self.ref_conductivity * (1 + self.conductivity_temp_coeff * temp_diff)
    
    def viscosity(self, temperature: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate dynamic viscosity at given temperature.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Viscosity in Pa·s
        """
        temp_diff = np.asarray(temperature) - self.ref_temperature
        return self.ref_viscosity * np.exp(self.viscosity_temp_coeff * temp_diff)
    
    def density(self, temperature: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate density at given temperature.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Density in kg/m³
        """
        temp_diff = np.asarray(temperature) - self.ref_temperature
        return self.ref_density + self.density_temp_coeff * temp_diff
    
    def surface_tension_coeff(self, temperature: float = 25.0) -> float:
        """
        Calculate surface tension coefficient.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Surface tension in N/m
        """
        # Simplified linear temperature dependence
        temp_coeff = -7e-4  # N/m per °C
        temp_diff = temperature - self.ref_temperature
        return self.surface_tension + temp_coeff * temp_diff
    
    def reynolds_number(
        self,
        velocity: float,
        characteristic_length: float,
        temperature: float = 25.0
    ) -> float:
        """
        Calculate Reynolds number for flow analysis.
        
        Args:
            velocity: Flow velocity in m/s
            characteristic_length: Characteristic length in m
            temperature: Temperature in Celsius
            
        Returns:
            Reynolds number (dimensionless)
        """
        rho = self.density(temperature)
        mu = self.viscosity(temperature)
        
        return rho * velocity * characteristic_length / mu
    
    def skin_depth(
        self,
        frequency: float,
        temperature: float = 25.0
    ) -> float:
        """
        Calculate electromagnetic skin depth.
        
        Args:
            frequency: Frequency in Hz
            temperature: Temperature in Celsius
            
        Returns:
            Skin depth in meters
        """
        sigma = self.conductivity(temperature)
        mu0 = 4 * np.pi * 1e-7  # Permeability of free space
        omega = 2 * np.pi * frequency
        
        return np.sqrt(2 / (omega * mu0 * sigma))
    
    def resistance_per_length(
        self,
        cross_sectional_area: float,
        temperature: float = 25.0
    ) -> float:
        """
        Calculate resistance per unit length for cylindrical conductor.
        
        Args:
            cross_sectional_area: Cross-sectional area in m²
            temperature: Temperature in Celsius
            
        Returns:
            Resistance per length in Ω/m
        """
        sigma = self.conductivity(temperature)
        return 1 / (sigma * cross_sectional_area)
    
    def thermal_time_constant(
        self,
        volume: float,
        surface_area: float,
        ambient_temperature: float = 25.0
    ) -> float:
        """
        Estimate thermal time constant for heating/cooling.
        
        Args:
            volume: Liquid metal volume in m³
            surface_area: Heat transfer surface area in m²
            ambient_temperature: Ambient temperature in Celsius
            
        Returns:
            Time constant in seconds
        """
        # Simplified thermal model
        specific_heat = 296.0  # J/kg·K (approximate for Galinstan)
        heat_transfer_coeff = 50.0  # W/m²·K (natural convection estimate)
        
        rho = self.density(ambient_temperature)
        thermal_mass = rho * volume * specific_heat
        thermal_conductance = heat_transfer_coeff * surface_area
        
        return thermal_mass / thermal_conductance
    
    def get_properties_dict(self, temperature: float = 25.0) -> Dict[str, Any]:
        """
        Get all material properties at specified temperature.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Dictionary of material properties
        """
        return {
            'temperature_celsius': temperature,
            'conductivity_s_per_m': self.conductivity(temperature),
            'viscosity_pa_s': self.viscosity(temperature),
            'density_kg_per_m3': self.density(temperature),
            'surface_tension_n_per_m': self.surface_tension_coeff(temperature),
            'melting_point_celsius': self.melting_point,
            'boiling_point_celsius': self.boiling_point,
            'composition': self.composition
        }
    
    def plot_temperature_dependence(
        self,
        temp_range: tuple = (0, 80),
        n_points: int = 100
    ) -> None:
        """
        Plot temperature dependence of key properties.
        
        Args:
            temp_range: Temperature range (min, max) in Celsius
            n_points: Number of temperature points
        """
        try:
            import matplotlib.pyplot as plt
            
            temperatures = np.linspace(temp_range[0], temp_range[1], n_points)
            conductivities = self.conductivity(temperatures)
            viscosities = self.viscosity(temperatures)
            densities = self.density(temperatures)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Conductivity
            ax1.plot(temperatures, conductivities / 1e6)
            ax1.set_xlabel('Temperature (°C)')
            ax1.set_ylabel('Conductivity (MS/m)')
            ax1.set_title('Electrical Conductivity')
            ax1.grid(True)
            
            # Viscosity
            ax2.plot(temperatures, viscosities * 1000)
            ax2.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('Viscosity (mPa·s)')
            ax2.set_title('Dynamic Viscosity')
            ax2.grid(True)
            
            # Density
            ax3.plot(temperatures, densities)
            ax3.set_xlabel('Temperature (°C)')
            ax3.set_ylabel('Density (kg/m³)')
            ax3.set_title('Density')
            ax3.grid(True)
            
            # Skin depth at 2.45 GHz
            skin_depths = [self.skin_depth(2.45e9, T) * 1e6 for T in temperatures]
            ax4.plot(temperatures, skin_depths)
            ax4.set_xlabel('Temperature (°C)')
            ax4.set_ylabel('Skin Depth (μm)')
            ax4.set_title('Skin Depth @ 2.45 GHz')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.suptitle('Galinstan Temperature Dependence', y=1.02)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def __repr__(self) -> str:
        props = self.get_properties_dict()
        return (
            f"GalinStanModel("
            f"σ={props['conductivity_s_per_m']/1e6:.2f}MS/m, "
            f"η={props['viscosity_pa_s']*1000:.2f}mPa·s, "
            f"ρ={props['density_kg_per_m3']:.0f}kg/m³ "
            f"@ {props['temperature_celsius']}°C"
            f")"
        )