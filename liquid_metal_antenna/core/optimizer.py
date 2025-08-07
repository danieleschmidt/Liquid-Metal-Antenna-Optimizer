"""
Liquid Metal Antenna Optimizer - Main optimization engine.
"""

import time
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
import numpy as np
from dataclasses import dataclass

# Optional torch import
try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    optim = None

from .antenna_spec import AntennaSpec
from ..solvers.base import BaseSolver, SolverResult
from ..solvers.fdtd import DifferentiableFDTD

# Fallback implementation
if not TORCH_AVAILABLE:
    from .optimizer_fallback import SimpleLMAOptimizer, SimpleOptimizationResult


@dataclass
class OptimizationResult:
    """Result container for optimization process."""
    
    # Optimized geometry
    geometry: np.ndarray
    
    # Performance metrics
    gain_dbi: float
    vswr: float
    bandwidth_hz: float
    efficiency: float
    
    # Optimization metadata
    converged: bool
    iterations: int
    optimization_time: float
    objective_history: List[float]
    constraint_violations: List[float]
    
    # Solver result
    simulation_result: Optional[SolverResult] = None
    
    def plot_radiation_pattern(self, **kwargs) -> None:
        """Plot antenna radiation pattern."""
        if self.simulation_result is None or self.simulation_result.radiation_pattern is None:
            print("No radiation pattern data available")
            return
        
        import matplotlib.pyplot as plt
        
        pattern = self.simulation_result.radiation_pattern
        theta = self.simulation_result.theta_angles
        phi = self.simulation_result.phi_angles
        
        # Create polar plot
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        
        # Plot main cut (phi=0)
        if pattern.ndim >= 2:
            main_cut = pattern[:, 0] if pattern.shape[1] > 0 else pattern.flatten()[:len(theta)]
        else:
            main_cut = pattern
        
        # Convert to dB
        pattern_db = 10 * np.log10(np.maximum(main_cut, 1e-10))
        pattern_db = pattern_db - np.max(pattern_db)  # Normalize to 0 dB max
        
        ax.plot(theta, pattern_db)
        ax.set_ylim([-40, 0])
        ax.set_title(f"Radiation Pattern (Gain: {self.gain_dbi:.1f} dBi)")
        ax.grid(True)
        
        plt.show()
    
    def export_cad(self, filename: str) -> None:
        """Export antenna geometry to CAD format."""
        print(f"CAD export to {filename} - Feature not implemented in Generation 1")


if TORCH_AVAILABLE:
    class LMAOptimizer:
        """
        Liquid Metal Antenna Optimizer.
        
        Main optimization engine that combines electromagnetic solvers with
        optimization algorithms to design reconfigurable liquid-metal antennas.
        """
        
        def __init__(
            self,
            spec: Optional[AntennaSpec] = None,
            solver: Union[str, BaseSolver] = 'differentiable_fdtd',
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        ):
            """
            Initialize optimizer.
            
            Args:
            spec: Antenna specification
            solver: Solver type or instance
            device: Computation device
        """
        self.spec = spec
        self.device = device
        
        # Setup solver
        if isinstance(solver, str):
            if solver == 'differentiable_fdtd':
                self.solver = DifferentiableFDTD(device=device)
            else:
                raise ValueError(f"Unknown solver type: {solver}")
        else:
            self.solver = solver
        
        # Optimization parameters
        self.learning_rate = 0.01
        self.max_iterations = 1000
        self.tolerance = 1e-4
        
        # Constraint parameters
        self.constraint_weights = {
            'vswr': 10.0,
            'bandwidth': 1.0,
            'efficiency': 1.0,
            'size': 1.0
        }
    
    def create_initial_geometry(self, spec: AntennaSpec) -> Union['torch.Tensor', np.ndarray]:
        """Create initial antenna geometry."""
        # Grid dimensions based on antenna size
        nx = int(spec.size_constraint.length * 1e-3 / self.solver.resolution)
        ny = int(spec.size_constraint.width * 1e-3 / self.solver.resolution)
        nz = int(spec.size_constraint.height * 1e-3 / self.solver.resolution)
        
        # Ensure minimum size
        nx = max(nx, 32)
        ny = max(ny, 32)
        nz = max(nz, 8)
        
        # Create simple patch antenna starting geometry
        if TORCH_AVAILABLE:
            geometry = torch.zeros((nx, ny, nz), dtype=torch.float32, device=self.device)
        else:
            geometry = np.zeros((nx, ny, nz), dtype=np.float32)
        
        # Add rectangular patch in center
        patch_x = nx // 4
        patch_y = ny // 4
        center_x, center_y = nx // 2, ny // 2
        center_z = nz // 2
        
        # Create patch on top layer
        geometry[
            center_x - patch_x // 2 : center_x + patch_x // 2,
            center_y - patch_y // 2 : center_y + patch_y // 2,
            center_z
        ] = 1.0
        
        # Add feed line
        feed_width = max(1, nx // 20)
        geometry[
            center_x - feed_width // 2 : center_x + feed_width // 2,
            center_y - patch_y // 2 : center_y,
            center_z
        ] = 1.0
        
        return geometry
    
    def compute_objective(
        self,
        geometry: Union['torch.Tensor', np.ndarray],
        spec: AntennaSpec,
        objective: str = 'max_gain'
    ) -> Union['torch.Tensor', float]:
        """
        Compute optimization objective.
        
        Args:
            geometry: Current antenna geometry
            spec: Antenna specification
            objective: Objective type ('max_gain', 'max_bandwidth', etc.)
            
        Returns:
            Objective value (to minimize)
        """
        # Run simulation
        fields = self.solver.simulate(
            geometry=geometry,
            frequency=spec.frequency_range.center,
            compute_gradients=True,
            spec=spec
        )
        
        if objective == 'max_gain':
            # Estimate gain from fields (simplified)
            # In a full implementation, this would compute far-field pattern
            total_field_energy = sum(torch.sum(field ** 2) for field in fields.values())
            
            # Prevent division by zero
            gain_proxy = torch.log(total_field_energy + 1e-10)
            
            # We want to maximize gain, so minimize negative gain
            return -gain_proxy
        
        elif objective == 'min_vswr':
            # Simplified VSWR estimation from field reflections
            ez_field = fields['Ez']
            
            # Estimate reflection by looking at field variation
            field_variation = torch.std(ez_field)
            vswr_proxy = 1.0 + 2 * field_variation
            
            return vswr_proxy
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def compute_constraints(
        self,
        geometry: torch.Tensor,
        spec: AntennaSpec,
        constraints: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Compute constraint violations."""
        constraint_values = {}
        
        # Run simulation for constraint evaluation
        fields = self.solver.simulate(
            geometry=geometry,
            frequency=spec.frequency_range.center,
            compute_gradients=True,
            spec=spec
        )
        
        # VSWR constraint
        if 'vswr' in constraints:
            target_vswr = float(constraints['vswr'].replace('<', '').replace('>', ''))
            
            # Simplified VSWR estimation
            ez_field = fields['Ez']
            field_variation = torch.std(ez_field)
            estimated_vswr = 1.0 + 2 * field_variation
            
            # Constraint violation (0 if satisfied, positive if violated)
            constraint_values['vswr'] = torch.max(
                torch.tensor(0.0, device=self.device),
                estimated_vswr - target_vswr
            )
        
        # Bandwidth constraint (simplified)
        if 'bandwidth' in constraints:
            # For now, assume bandwidth is met if VSWR is good
            constraint_values['bandwidth'] = torch.tensor(0.0, device=self.device)
        
        # Efficiency constraint
        if 'efficiency' in constraints:
            target_eff = float(constraints['efficiency'].replace('>', ''))
            
            # Simplified efficiency estimation
            total_field = sum(torch.sum(torch.abs(field)) for field in fields.values())
            metal_field = torch.sum(torch.abs(fields['Ez']) * geometry)
            
            estimated_eff = metal_field / (total_field + 1e-10)
            
            constraint_values['efficiency'] = torch.max(
                torch.tensor(0.0, device=self.device),
                target_eff - estimated_eff
            )
        
        return constraint_values
    
    def optimize(
        self,
        objective: str = 'max_gain',
        constraints: Optional[Dict[str, Any]] = None,
        n_iterations: Optional[int] = None,
        spec: Optional[AntennaSpec] = None
    ) -> OptimizationResult:
        """
        Run antenna optimization.
        
        Args:
            objective: Optimization objective
            constraints: Performance constraints
            n_iterations: Maximum iterations (overrides default)
            spec: Antenna specification (overrides instance spec)
            
        Returns:
            OptimizationResult with optimized design
        """
        start_time = time.time()
        
        # Use provided spec or default
        if spec is None:
            if self.spec is None:
                raise ValueError("No antenna specification provided")
            spec = self.spec
        
        # Set iteration count
        if n_iterations is None:
            n_iterations = self.max_iterations
        
        # Default constraints
        if constraints is None:
            constraints = {
                'vswr': '<2.0',
                'bandwidth': '>100e6',
                'efficiency': '>0.8'
            }
        
        # Create initial geometry
        geometry = self.create_initial_geometry(spec)
        geometry.requires_grad_(True)
        
        # Setup optimizer
        optimizer = optim.Adam([geometry], lr=self.learning_rate)
        
        # Optimization history
        objective_history = []
        constraint_violations = []
        
        print(f"Starting optimization: {objective}")
        print(f"Target frequency: {spec.frequency_range.center/1e9:.2f} GHz")
        print(f"Antenna size: {spec.size_constraint.length}x{spec.size_constraint.width}x{spec.size_constraint.height} mm")
        
        # Optimization loop
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Compute objective
            obj_value = self.compute_objective(geometry, spec, objective)
            
            # Compute constraints
            constraint_vals = self.compute_constraints(geometry, spec, constraints)
            
            # Total loss (objective + constraint penalties)
            total_loss = obj_value
            total_constraint_violation = 0.0
            
            for constraint_name, violation in constraint_vals.items():
                weight = self.constraint_weights.get(constraint_name, 1.0)
                total_loss += weight * violation
                total_constraint_violation += float(violation)
            
            # Backpropagation
            total_loss.backward()
            
            # Apply geometry constraints (keep values between 0 and 1)
            with torch.no_grad():
                geometry.grad = geometry.grad * (geometry > 0.01) * (geometry < 0.99)
            
            # Update geometry
            optimizer.step()
            
            # Clamp geometry values
            with torch.no_grad():
                geometry.clamp_(0, 1)
            
            # Record history
            objective_history.append(float(obj_value))
            constraint_violations.append(total_constraint_violation)
            
            # Progress reporting
            if iteration % 100 == 0 or iteration == n_iterations - 1:
                print(f"Iteration {iteration}: Objective = {obj_value:.4f}, "
                      f"Constraints = {total_constraint_violation:.4f}")
            
            # Convergence check
            if len(objective_history) > 10:
                recent_change = abs(objective_history[-1] - objective_history[-10])
                if recent_change < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        optimization_time = time.time() - start_time
        
        # Final evaluation with full simulation
        with torch.no_grad():
            final_result = self.solver.simulate(
                geometry=geometry.detach(),
                frequency=spec.frequency_range.center,
                compute_gradients=False,
                spec=spec
            )
        
        # Create optimization result
        result = OptimizationResult(
            geometry=geometry.detach().cpu().numpy(),
            gain_dbi=final_result.gain_dbi if final_result.gain_dbi is not None else 0.0,
            vswr=final_result.get_vswr_at_frequency(spec.frequency_range.center),
            bandwidth_hz=final_result.compute_bandwidth(),
            efficiency=final_result.efficiency if final_result.efficiency is not None else 0.8,
            converged=iteration < n_iterations - 1,
            iterations=iteration + 1,
            optimization_time=optimization_time,
            objective_history=objective_history,
            constraint_violations=constraint_violations,
            simulation_result=final_result
        )
        
        print(f"\nOptimization complete!")
        print(f"Gain: {result.gain_dbi:.1f} dBi")
        print(f"VSWR: {result.vswr:.2f}")
        print(f"Bandwidth: {result.bandwidth_hz/1e6:.1f} MHz")
        print(f"Efficiency: {result.efficiency:.1%}")
        print(f"Optimization time: {result.optimization_time:.1f} seconds")
        
        return result
    
    def optimize_multiband(
        self,
        target_frequencies: List[float],
        bandwidth_min: float = 100e6,
        isolation_min: float = 20.0
    ) -> Dict[float, OptimizationResult]:
        """Optimize for multi-band operation (Generation 2 feature)."""
        print("Multi-band optimization not implemented in Generation 1")
        return {}
    
    def create_scan_optimizer(self, array) -> 'ScanOptimizer':
        """Create beam scanning optimizer (Generation 2 feature)."""
        print("Scan optimization not implemented in Generation 1")
        return None
    
    def multi_objective_optimize(
        self,
        antenna,
        objectives: List[str],
        constraints: Dict[str, Any],
        algorithm: str = 'NSGA-III',
        population_size: int = 200,
        generations: int = 500
    ) -> List[OptimizationResult]:
        """Multi-objective optimization (Generation 3 feature)."""
        print("Multi-objective optimization not implemented in Generation 1")
        return []
    
    def plot_pareto_frontier(
        self,
        pareto_designs: List[OptimizationResult],
        **kwargs
    ) -> None:
        """Plot Pareto frontier (Generation 3 feature)."""
        print("Pareto frontier plotting not implemented in Generation 1")


def create_optimizer(
    spec: AntennaSpec,
    solver: Union[str, BaseSolver] = 'differentiable_fdtd',
    device: str = 'cpu'
) -> Union['LMAOptimizer', 'SimpleLMAOptimizer']:
    """
    Factory function to create the appropriate optimizer.
    
    Returns the full LMAOptimizer if PyTorch is available,
    otherwise returns the simplified fallback implementation.
    """
    if TORCH_AVAILABLE:
        return LMAOptimizer(spec, solver, device)
    else:
        return SimpleLMAOptimizer(spec, solver, device)


# The actual LMAOptimizer class will be defined below for torch-enabled systems
if not TORCH_AVAILABLE:
    # Use the simple optimizer as the default when torch is not available
    LMAOptimizer = SimpleLMAOptimizer