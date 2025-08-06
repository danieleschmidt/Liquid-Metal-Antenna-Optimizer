"""
Comprehensive tests for Liquid Metal Antenna (LMA) optimization functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from liquid_metal_antenna.core.antenna_spec import AntennaSpec, SubstrateMaterial, LiquidMetalType
from liquid_metal_antenna.optimization.lma_optimizer import LMAOptimizer, OptimizationResult
from liquid_metal_antenna.optimization.objectives import ObjectiveFunction, MultiObjective
from liquid_metal_antenna.solvers.base import SolverResult
from liquid_metal_antenna.utils.validation import ValidationError


class TestLMAOptimizer:
    """Test core LMA optimizer functionality."""
    
    @pytest.fixture
    def optimizer(self):
        """Create LMA optimizer for testing."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        return LMAOptimizer(spec)
    
    @pytest.fixture
    def mock_solver(self):
        """Create mock solver for testing."""
        solver = Mock()
        solver.simulate.return_value = SolverResult(
            s_parameters=np.array([[[0.1+0.05j]]]),
            frequencies=np.array([2.45e9]),
            radiation_pattern=np.ones((18, 36)),
            theta_angles=np.linspace(0, np.pi, 18),
            phi_angles=np.linspace(0, 2*np.pi, 36),
            gain_dbi=5.2,
            max_gain_dbi=5.2,
            vswr=1.8,
            converged=True,
            iterations=100,
            computation_time=2.5
        )
        return solver
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.spec is not None
        assert optimizer.n_iterations == 100
        assert optimizer.learning_rate == 0.01
        assert optimizer.tolerance == 1e-6
        assert optimizer.population_size == 50
    
    def test_population_initialization(self, optimizer):
        """Test population initialization."""
        geometry_shape = (20, 20, 4)
        population = optimizer._initialize_population(geometry_shape)
        
        assert population.shape[0] == optimizer.population_size
        assert population.shape[1:] == geometry_shape
        
        # Check that population values are in valid range
        assert np.all(population >= 0)
        assert np.all(population <= 1)
    
    def test_objective_function_evaluation(self, optimizer, mock_solver):
        """Test objective function evaluation."""
        optimizer.solver = mock_solver
        
        geometry = np.random.rand(20, 20, 4)
        objective_value = optimizer._evaluate_objective(geometry)
        
        assert isinstance(objective_value, float)
        assert not np.isnan(objective_value)
        assert not np.isinf(objective_value)
    
    def test_constraint_evaluation(self, optimizer):
        """Test constraint evaluation."""
        # Valid geometry (should satisfy constraints)
        valid_geometry = np.random.rand(20, 20, 4) * 0.3  # Sparse
        constraints = optimizer._evaluate_constraints(valid_geometry)
        assert constraints >= 0  # No constraint violation
        
        # Invalid geometry (dense - should violate manufacturing constraints)
        invalid_geometry = np.ones((20, 20, 4)) * 0.9  # Very dense
        constraints = optimizer._evaluate_constraints(invalid_geometry)
        assert constraints > 0  # Constraint violation
    
    def test_gradient_computation(self, optimizer, mock_solver):
        """Test gradient computation for geometry optimization."""
        optimizer.solver = mock_solver
        
        geometry = np.random.rand(20, 20, 4)
        gradients = optimizer._compute_gradients(geometry)
        
        assert gradients.shape == geometry.shape
        assert not np.any(np.isnan(gradients))
        assert not np.any(np.isinf(gradients))
    
    def test_geometry_update(self, optimizer):
        """Test geometry update using gradients."""
        geometry = np.random.rand(20, 20, 4) * 0.5
        gradients = np.random.randn(*geometry.shape)
        
        updated_geometry = optimizer._update_geometry(geometry, gradients)
        
        # Should maintain valid range
        assert np.all(updated_geometry >= 0)
        assert np.all(updated_geometry <= 1)
        assert updated_geometry.shape == geometry.shape
    
    def test_convergence_detection(self, optimizer):
        """Test optimization convergence detection."""
        # Convergence with small changes
        history = [1.5, 1.49, 1.491, 1.489, 1.490]  # Small variations
        assert optimizer._check_convergence(history, threshold=1e-2)
        
        # No convergence with large changes
        history = [1.5, 1.3, 1.7, 1.2, 1.8]  # Large variations
        assert not optimizer._check_convergence(history, threshold=1e-2)
    
    @patch('liquid_metal_antenna.optimization.lma_optimizer.torch')
    def test_optimization_loop(self, mock_torch, optimizer, mock_solver):
        """Test complete optimization loop."""
        # Mock torch for gradient computation
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.autograd.grad.return_value = [np.random.randn(20, 20, 4)]
        
        optimizer.solver = mock_solver
        optimizer.n_iterations = 10  # Short for testing
        
        initial_geometry = np.random.rand(20, 20, 4) * 0.5
        
        result = optimizer.optimize(initial_geometry)
        
        assert isinstance(result, OptimizationResult)
        assert result.optimized_geometry is not None
        assert result.objective_history is not None
        assert len(result.objective_history) <= optimizer.n_iterations


class TestMultiObjectiveOptimization:
    """Test multi-objective optimization functionality."""
    
    @pytest.fixture
    def multi_objective_optimizer(self):
        """Create multi-objective optimizer."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            performance_targets={
                'min_gain_dbi': 6.0,
                'max_vswr': 1.5,
                'min_efficiency': 0.85
            }
        )
        return LMAOptimizer(spec, optimization_mode='multi_objective')
    
    def test_pareto_front_computation(self, multi_objective_optimizer):
        """Test Pareto front computation."""
        # Mock population with different objectives
        population = np.random.rand(20, 10, 10, 4)
        objectives = np.random.rand(20, 3)  # 3 objectives
        
        pareto_indices = multi_objective_optimizer._compute_pareto_front(objectives)
        
        assert len(pareto_indices) > 0
        assert len(pareto_indices) <= len(population)
        assert all(0 <= idx < len(population) for idx in pareto_indices)
    
    def test_crowding_distance(self, multi_objective_optimizer):
        """Test crowding distance calculation for diversity."""
        objectives = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.5, 2.5],
            [1.5, 1.8, 2.8],
            [3.0, 1.0, 2.0]
        ])
        
        distances = multi_objective_optimizer._calculate_crowding_distance(objectives)
        
        assert len(distances) == len(objectives)
        assert all(d >= 0 for d in distances)
    
    def test_nsga2_selection(self, multi_objective_optimizer):
        """Test NSGA-II selection mechanism."""
        population = np.random.rand(50, 10, 10, 4)
        objectives = np.random.rand(50, 3)
        
        selected_indices = multi_objective_optimizer._nsga2_selection(
            population, objectives, n_select=20
        )
        
        assert len(selected_indices) == 20
        assert all(0 <= idx < len(population) for idx in selected_indices)


class TestOptimizationConstraints:
    """Test optimization constraints and penalties."""
    
    @pytest.fixture
    def constrained_optimizer(self):
        """Create optimizer with constraints."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        return LMAOptimizer(spec, constraints={
            'manufacturing': {
                'min_feature_size': 0.2e-3,
                'max_metal_fraction': 0.4,
                'connectivity_required': True
            },
            'performance': {
                'min_gain_dbi': 5.0,
                'max_vswr': 2.0
            }
        })
    
    def test_manufacturing_constraints(self, constrained_optimizer):
        """Test manufacturing constraint evaluation."""
        # Valid geometry - meets manufacturing constraints
        valid_geometry = np.zeros((20, 20, 4))
        valid_geometry[8:12, 8:12, 2] = 1.0  # Simple connected patch
        
        manufacturing_penalty = constrained_optimizer._evaluate_manufacturing_constraints(valid_geometry)
        assert manufacturing_penalty == 0  # No violation
        
        # Invalid geometry - violates feature size
        invalid_geometry = np.zeros((20, 20, 4))
        invalid_geometry[10, 10, 2] = 1.0  # Single isolated pixel
        
        manufacturing_penalty = constrained_optimizer._evaluate_manufacturing_constraints(invalid_geometry)
        assert manufacturing_penalty > 0  # Violation detected
    
    def test_connectivity_constraint(self, constrained_optimizer):
        """Test metal connectivity constraint."""
        # Connected geometry
        connected_geometry = np.zeros((20, 20, 4))
        connected_geometry[8:12, 8:12, 2] = 1.0  # Connected region
        
        connectivity_penalty = constrained_optimizer._check_connectivity(connected_geometry)
        assert connectivity_penalty == 0
        
        # Disconnected geometry
        disconnected_geometry = np.zeros((20, 20, 4))
        disconnected_geometry[5, 5, 2] = 1.0  # Isolated pixel
        disconnected_geometry[15, 15, 2] = 1.0  # Another isolated pixel
        
        connectivity_penalty = constrained_optimizer._check_connectivity(disconnected_geometry)
        assert connectivity_penalty > 0
    
    def test_performance_constraints(self, constrained_optimizer, mock_solver):
        """Test performance constraint evaluation."""
        constrained_optimizer.solver = mock_solver
        
        geometry = np.random.rand(20, 20, 4)
        
        # Mock solver to return low gain (violates constraint)
        mock_solver.simulate.return_value.gain_dbi = 3.0  # Below min_gain_dbi=5.0
        
        performance_penalty = constrained_optimizer._evaluate_performance_constraints(geometry)
        assert performance_penalty > 0  # Constraint violation


class TestOptimizationAlgorithms:
    """Test different optimization algorithms."""
    
    @pytest.fixture
    def spec(self):
        """Create antenna specification for algorithm testing."""
        return AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
    
    def test_genetic_algorithm(self, spec, mock_solver):
        """Test genetic algorithm optimization."""
        optimizer = LMAOptimizer(spec, algorithm='genetic')
        optimizer.solver = mock_solver
        
        initial_geometry = np.random.rand(16, 16, 4) * 0.5
        
        # Mock genetic operations
        optimizer._crossover = Mock(return_value=np.random.rand(16, 16, 4))
        optimizer._mutate = Mock(return_value=np.random.rand(16, 16, 4))
        
        result = optimizer.optimize(initial_geometry, max_iterations=5)
        
        assert isinstance(result, OptimizationResult)
        assert result.algorithm_used == 'genetic'
    
    def test_gradient_descent(self, spec, mock_solver):
        """Test gradient descent optimization."""
        optimizer = LMAOptimizer(spec, algorithm='gradient_descent')
        optimizer.solver = mock_solver
        
        initial_geometry = np.random.rand(16, 16, 4) * 0.5
        
        # Mock gradient computation
        optimizer._compute_gradients = Mock(return_value=np.random.randn(16, 16, 4))
        
        result = optimizer.optimize(initial_geometry, max_iterations=5)
        
        assert isinstance(result, OptimizationResult)
        assert result.algorithm_used == 'gradient_descent'
    
    def test_particle_swarm(self, spec, mock_solver):
        """Test particle swarm optimization."""
        optimizer = LMAOptimizer(spec, algorithm='particle_swarm')
        optimizer.solver = mock_solver
        
        initial_geometry = np.random.rand(16, 16, 4) * 0.5
        
        result = optimizer.optimize(initial_geometry, max_iterations=5)
        
        assert isinstance(result, OptimizationResult)
        assert result.algorithm_used == 'particle_swarm'


class TestOptimizationBenchmarks:
    """Test optimization performance benchmarks."""
    
    def test_convergence_speed_benchmark(self):
        """Test convergence speed for different algorithms."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Mock solver for consistent results
        mock_solver = Mock()
        mock_solver.simulate.return_value = SolverResult(
            s_parameters=np.array([[[0.1+0.05j]]]),
            frequencies=np.array([2.45e9]),
            gain_dbi=5.2,
            vswr=1.8,
            converged=True
        )
        
        algorithms = ['gradient_descent', 'genetic', 'particle_swarm']
        convergence_iterations = {}
        
        for algorithm in algorithms:
            optimizer = LMAOptimizer(spec, algorithm=algorithm, n_iterations=50)
            optimizer.solver = mock_solver
            
            initial_geometry = np.random.rand(12, 12, 4) * 0.5
            result = optimizer.optimize(initial_geometry)
            
            convergence_iterations[algorithm] = len(result.objective_history)
        
        # All algorithms should converge within reasonable iterations
        for algorithm, iterations in convergence_iterations.items():
            assert iterations <= 50
            assert iterations > 0
    
    def test_solution_quality_benchmark(self):
        """Test solution quality for different problem sizes."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        optimizer = LMAOptimizer(spec, n_iterations=20)
        
        # Mock solver with objective that favors certain patterns
        mock_solver = Mock()
        def mock_simulate(geometry, **kwargs):
            # Favor geometries with connected regions
            connected_score = np.sum(geometry > 0.5)
            gain = 3.0 + connected_score / 100  # Simple model
            return SolverResult(
                s_parameters=np.array([[[0.1+0.05j]]]),
                frequencies=np.array([2.45e9]),
                gain_dbi=gain,
                vswr=max(1.1, 3.0 - connected_score / 200),
                converged=True
            )
        
        mock_solver.simulate = mock_simulate
        optimizer.solver = mock_solver
        
        problem_sizes = [(8, 8, 4), (16, 16, 4), (24, 24, 4)]
        final_objectives = {}
        
        for size in problem_sizes:
            initial_geometry = np.random.rand(*size) * 0.3
            result = optimizer.optimize(initial_geometry)
            final_objectives[size] = result.final_objective_value
        
        # Should achieve reasonable objective values
        for size, objective in final_objectives.items():
            assert objective > 0  # Positive objective
            assert not np.isnan(objective)


class TestOptimizationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_geometry_optimization(self):
        """Test optimization with empty initial geometry."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        optimizer = LMAOptimizer(spec)
        empty_geometry = np.zeros((16, 16, 4))
        
        # Should handle empty geometry gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimizer.optimize(empty_geometry, max_iterations=5)
            assert result.optimized_geometry is not None
    
    def test_all_metal_geometry_optimization(self):
        """Test optimization with completely filled initial geometry."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        optimizer = LMAOptimizer(spec)
        full_geometry = np.ones((16, 16, 4))
        
        # Should handle over-filled geometry
        result = optimizer.optimize(full_geometry, max_iterations=5)
        assert result.optimized_geometry is not None
        assert np.any(result.optimized_geometry < 1.0)  # Should reduce metal fraction
    
    def test_optimization_with_solver_failures(self):
        """Test optimization robustness to solver failures."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        optimizer = LMAOptimizer(spec)
        
        # Mock solver that occasionally fails
        mock_solver = Mock()
        call_count = 0
        def failing_simulate(geometry, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise RuntimeError("Solver failure")
            return SolverResult(
                s_parameters=np.array([[[0.1+0.05j]]]),
                frequencies=np.array([2.45e9]),
                gain_dbi=5.0,
                vswr=1.5,
                converged=True
            )
        
        mock_solver.simulate = failing_simulate
        optimizer.solver = mock_solver
        
        initial_geometry = np.random.rand(12, 12, 4) * 0.5
        
        # Should handle solver failures gracefully
        result = optimizer.optimize(initial_geometry, max_iterations=10)
        assert result.optimized_geometry is not None
        assert len(result.objective_history) > 0
    
    def test_invalid_optimization_parameters(self):
        """Test handling of invalid optimization parameters."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Invalid learning rate
        with pytest.raises(ValidationError):
            LMAOptimizer(spec, learning_rate=-0.1)
        
        # Invalid population size
        with pytest.raises(ValidationError):
            LMAOptimizer(spec, population_size=0)
        
        # Invalid tolerance
        with pytest.raises(ValidationError):
            LMAOptimizer(spec, tolerance=-1e-6)


if __name__ == '__main__':
    pytest.main([__file__])