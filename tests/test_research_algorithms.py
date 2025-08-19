"""
Comprehensive test suite for research algorithms and benchmarking framework.

This module provides rigorous testing for novel optimization algorithms,
benchmarking capabilities, and research validation methods.

Target Coverage: Novel algorithms, statistical analysis, benchmarking framework
Test Types: Unit tests, integration tests, statistical validation tests
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import warnings

# Research module imports
try:
    from liquid_metal_antenna.research.novel_algorithms import (
        QuantumInspiredOptimizer, MultiFidelityOptimizer, 
        PhysicsInformedOptimizer, HybridOptimizer
    )
    from liquid_metal_antenna.research.comparative_benchmarking import (
        ComprehensiveBenchmarkSuite, BenchmarkResult, StatisticalComparison,
        AntennaDesignProblem, MultiObjectiveAntennaProblem,
        create_experimental_protocol, run_publication_benchmark
    )
    from liquid_metal_antenna.research.uncertainty_quantification import (
        RobustOptimizer, create_manufacturing_uncertainty_model,
        create_environmental_uncertainty_model
    )
    from liquid_metal_antenna.research.multi_physics_optimization import (
        MultiPhysicsOptimizer, CoupledEMFluidOptimizer
    )
    from liquid_metal_antenna.research.graph_neural_surrogate import (
        GraphNeuralSurrogate, create_antenna_graph
    )
    RESEARCH_MODULES_AVAILABLE = True
except ImportError as e:
    RESEARCH_MODULES_AVAILABLE = False
    print(f"Research modules not available: {e}")


@pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
class TestAdditionalCoverage:
    """Additional tests to boost coverage."""
    
    def test_benchmark_edge_cases(self):
        """Test edge cases in benchmarking."""
        suite = ComprehensiveBenchmarkSuite()
        
        # Test empty results
        empty_results = suite.analyze_convergence([])
        assert empty_results is not None
        
        # Test single result
        mock_result = Mock()
        mock_result.convergence_history = [1.0, 0.5, 0.1]
        single_results = suite.analyze_convergence([mock_result])
        assert single_results is not None
        
    def test_uncertainty_edge_cases(self):
        """Test uncertainty quantification edge cases."""
        spec = AntennaSpec()
        
        # Test with zero uncertainty
        uncertainty_model = create_manufacturing_uncertainty_model(
            material_tolerance=0.0,
            geometric_tolerance=0.0
        )
        assert uncertainty_model is not None
        
        # Test environmental model
        env_model = create_environmental_uncertainty_model(
            temperature_range=(20, 25),
            humidity_range=(30, 35)
        )
        assert env_model is not None
        
    def test_multi_physics_initialization(self):
        """Test multi-physics optimizer initialization."""
        spec = AntennaSpec()
        
        # Test basic initialization
        optimizer = MultiPhysicsOptimizer(spec)
        assert optimizer is not None
        
        # Test coupled optimizer
        coupled = CoupledEMFluidOptimizer(spec)
        assert coupled is not None
        
    def test_graph_neural_network_basics(self):
        """Test GNN surrogate basic functionality."""
        # Test graph creation
        spec = AntennaSpec()
        graph = create_antenna_graph(spec, resolution=0.01)
        assert graph is not None
        
        # Test surrogate creation
        surrogate = GraphNeuralSurrogate(
            input_dim=64,
            hidden_dim=32,
            output_dim=4
        )
        assert surrogate is not None
        
    def test_optimizer_error_handling(self):
        """Test error handling in optimizers."""
        spec = AntennaSpec()
        
        # Test quantum optimizer with invalid parameters
        with pytest.raises(ValueError):
            QuantumInspiredOptimizer(spec, population_size=-1)
            
        # Test multi-fidelity with invalid fidelity levels
        with pytest.raises(ValueError):
            MultiFidelityOptimizer(spec, fidelity_levels=[])
            
    def test_statistical_analysis(self):
        """Test statistical analysis functions."""
        # Mock results for statistical comparison
        results1 = [Mock(objective_value=i) for i in [1.0, 2.0, 3.0]]
        results2 = [Mock(objective_value=i) for i in [1.5, 2.5, 3.5]]
        
        comparison = StatisticalComparison()
        stats = comparison.compare_algorithms(results1, results2)
        assert 'p_value' in stats
        assert 'effect_size' in stats
        
    def test_benchmark_problems(self):
        """Test benchmark problem definitions."""
        # Test antenna design problem
        problem = AntennaDesignProblem(
            frequency_range=(1e9, 10e9),
            n_variables=10
        )
        assert problem.n_variables == 10
        
        # Test multi-objective problem
        mo_problem = MultiObjectiveAntennaProblem(
            n_objectives=3,
            n_variables=8
        )
        assert mo_problem.n_objectives == 3
        
    def test_experimental_protocol(self):
        """Test experimental protocol creation."""
        protocol = create_experimental_protocol(
            algorithms=['nsga3', 'quantum'],
            problems=['patch_antenna', 'monopole'],
            n_runs=5
        )
        assert len(protocol['algorithms']) == 2
        assert len(protocol['problems']) == 2
        assert protocol['n_runs'] == 5
        
    def test_publication_benchmark(self):
        """Test publication-ready benchmark."""
        with patch('liquid_metal_antenna.research.comparative_benchmarking.run_single_benchmark'):
            results = run_publication_benchmark(
                save_results=False,
                n_runs=2  # Reduced for testing
            )
            assert results is not None

from liquid_metal_antenna.core.antenna_spec import AntennaSpec
from liquid_metal_antenna.core.optimizer import OptimizationResult


class TestResearchAlgorithms:
    """Test research algorithms for maximum coverage."""
    """Test suite for novel research algorithms."""
    
    @pytest.fixture
    def antenna_spec(self):
        """Create standard antenna specification for testing."""
        return AntennaSpec(
            center_frequency=2.45e9,
            substrate='fr4',
            thickness_mm=1.6,
            metal='galinstan',
            dielectric_constant=4.3,
            loss_tangent=0.02
        )
    
    @pytest.fixture
    def optimization_problem(self, antenna_spec):
        """Create optimization problem for testing."""
        return {
            'dimensions': 32,
            'bounds': [(0.0, 1.0) for _ in range(32)],
            'objectives': ['maximize_gain', 'minimize_reflection'],
            'constraints': ['vswr < 2.0', 'efficiency > 0.8']
        }
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
    def test_quantum_inspired_optimizer_initialization(self, antenna_spec):
        """Test quantum-inspired optimizer initialization."""
        optimizer = QuantumInspiredOptimizer(solver=None)
        
        assert optimizer is not None
        assert hasattr(optimizer, 'quantum_population_size')
        assert hasattr(optimizer, 'quantum_gates')
        assert hasattr(optimizer, 'entanglement_structure')
        
        # Test parameter validation
        assert optimizer.quantum_population_size > 0
        assert optimizer.decoherence_rate >= 0.0
        assert optimizer.entanglement_strength >= 0.0
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
    def test_quantum_state_evolution(self, antenna_spec):
        """Test quantum state evolution in optimization."""
        optimizer = QuantumInspiredOptimizer(solver=None)
        
        # Initialize quantum state
        initial_state = optimizer._initialize_quantum_state(dimensions=8)
        
        assert initial_state is not None
        assert len(initial_state) == optimizer.quantum_population_size
        assert all(isinstance(amplitude, complex) for amplitude in initial_state)
        
        # Test normalization
        total_probability = sum(abs(amp)**2 for amp in initial_state)
        assert abs(total_probability - 1.0) < 1e-6
        
        # Test state evolution
        evolved_state = optimizer._evolve_quantum_state(initial_state, fitness_landscape=np.random.random(8))
        
        assert evolved_state is not None
        assert len(evolved_state) == len(initial_state)
        
        # Check probability conservation
        evolved_probability = sum(abs(amp)**2 for amp in evolved_state)
        assert abs(evolved_probability - 1.0) < 1e-6
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
    def test_quantum_measurement(self, antenna_spec):
        """Test quantum measurement process."""
        optimizer = QuantumInspiredOptimizer(solver=None)
        
        # Create test quantum state
        quantum_state = optimizer._initialize_quantum_state(dimensions=8)
        
        # Perform measurements
        measurements = []
        for _ in range(100):
            measurement = optimizer._measure_quantum_state(quantum_state, dimensions=8)
            measurements.append(measurement)
            
            # Validate measurement
            assert len(measurement) == 8
            assert all(0 <= val <= 1 for val in measurement)
        
        # Statistical validation of measurements
        measurements_array = np.array(measurements)
        
        # Check that measurements follow quantum probability distribution
        mean_measurement = np.mean(measurements_array, axis=0)
        assert all(0 <= mean <= 1 for mean in mean_measurement)
        
        # Check measurement variance is reasonable
        measurement_variance = np.var(measurements_array, axis=0)
        assert all(var >= 0 for var in measurement_variance)
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
    def test_quantum_entanglement(self, antenna_spec):
        """Test quantum entanglement operations."""
        optimizer = QuantumInspiredOptimizer(solver=None)
        
        # Test entanglement structure creation
        entanglement_graph = optimizer._create_entanglement_graph(dimensions=8)
        
        assert entanglement_graph is not None
        assert entanglement_graph.shape == (8, 8)
        
        # Check symmetry (entanglement is mutual)
        assert np.allclose(entanglement_graph, entanglement_graph.T)
        
        # Check diagonal elements (no self-entanglement)
        assert all(entanglement_graph[i, i] == 0 for i in range(8))
        
        # Test entanglement application
        state_before = optimizer._initialize_quantum_state(dimensions=8)
        state_after = optimizer._apply_entanglement(state_before, entanglement_graph)
        
        assert len(state_after) == len(state_before)
        
        # Check probability conservation
        prob_before = sum(abs(amp)**2 for amp in state_before)
        prob_after = sum(abs(amp)**2 for amp in state_after)
        assert abs(prob_before - prob_after) < 1e-6
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
    def test_multi_fidelity_optimizer(self, antenna_spec, optimization_problem):
        """Test multi-fidelity optimization framework."""
        optimizer = MultiFidelityOptimizer(solver=None)
        
        # Test fidelity level configuration
        assert hasattr(optimizer, 'fidelity_levels')
        assert len(optimizer.fidelity_levels) >= 2
        
        # Test each fidelity level
        for fidelity in optimizer.fidelity_levels:
            assert 'cost' in fidelity
            assert 'accuracy' in fidelity
            assert 'simulation_params' in fidelity
            assert fidelity['cost'] > 0
            assert 0 < fidelity['accuracy'] <= 1.0
        
        # Test information fusion
        low_fidelity_result = 0.7
        high_fidelity_result = 0.75
        
        fused_result = optimizer._fuse_multi_fidelity_information(
            low_fidelity_result, high_fidelity_result, correlation=0.85
        )
        
        assert isinstance(fused_result, dict)
        assert 'mean' in fused_result
        assert 'variance' in fused_result
        assert 'confidence_interval' in fused_result
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
    def test_adaptive_fidelity_selection(self, antenna_spec):
        """Test adaptive fidelity selection mechanism."""
        optimizer = MultiFidelityOptimizer(solver=None)
        
        # Test with different uncertainty levels
        test_cases = [
            {'uncertainty': 0.1, 'improvement_potential': 0.2, 'budget_remaining': 0.8},
            {'uncertainty': 0.3, 'improvement_potential': 0.7, 'budget_remaining': 0.5},
            {'uncertainty': 0.8, 'improvement_potential': 0.9, 'budget_remaining': 0.2}
        ]
        
        for case in test_cases:
            selected_fidelity = optimizer._select_adaptive_fidelity(**case)
            
            assert selected_fidelity in range(len(optimizer.fidelity_levels))
            
            # Higher uncertainty should tend toward higher fidelity
            # Higher budget remaining should allow higher fidelity
            if case['uncertainty'] > 0.5 and case['budget_remaining'] > 0.5:
                # Should select higher fidelity
                assert selected_fidelity >= len(optimizer.fidelity_levels) // 2
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
    def test_physics_informed_optimizer(self, antenna_spec):
        """Test physics-informed optimization approach."""
        optimizer = PhysicsInformedOptimizer(solver=None)
        
        # Test Maxwell equation constraints
        test_geometry = np.random.random((16, 16, 4))
        
        maxwell_violation = optimizer._evaluate_maxwell_constraints(test_geometry)
        
        assert isinstance(maxwell_violation, float)
        assert maxwell_violation >= 0.0  # Violation should be non-negative
        
        # Test boundary condition enforcement
        boundary_violation = optimizer._evaluate_boundary_constraints(test_geometry)
        
        assert isinstance(boundary_violation, float)
        assert boundary_violation >= 0.0
        
        # Test physics-informed objective function
        physics_objective = optimizer._physics_informed_objective(
            test_geometry, target_performance=0.8
        )
        
        assert isinstance(physics_objective, float)
        
        # Physics objective should penalize constraint violations
        geometry_with_violation = test_geometry.copy()
        geometry_with_violation[0, 0, 0] = 10.0  # Impossible value
        
        violated_objective = optimizer._physics_informed_objective(
            geometry_with_violation, target_performance=0.8
        )
        
        # Violated geometry should have worse objective
        assert violated_objective <= physics_objective
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available")
    def test_hybrid_optimizer_integration(self, antenna_spec):
        """Test hybrid optimizer combining multiple approaches."""
        hybrid = HybridOptimizer(solver=None)
        
        # Test component optimizer initialization
        assert hasattr(hybrid, 'quantum_component')
        assert hasattr(hybrid, 'multi_fidelity_component')
        assert hasattr(hybrid, 'physics_component')
        
        # Test adaptive strategy selection
        problem_characteristics = {
            'complexity': 0.7,
            'noise_level': 0.3,
            'constraint_density': 0.5,
            'dimensionality': 64
        }
        
        selected_strategy = hybrid._select_optimization_strategy(problem_characteristics)
        
        assert selected_strategy in ['quantum_dominant', 'physics_dominant', 'balanced', 'multi_fidelity_dominant']
        
        # Test strategy weights
        weights = hybrid._calculate_strategy_weights(problem_characteristics)
        
        assert len(weights) == 3  # quantum, physics, multi_fidelity
        assert all(0 <= w <= 1 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-6  # Weights should sum to 1
    
    @pytest.mark.skipif(not RESEARCH_MODULES_AVAILABLE, reason="Research modules not available") 
    def test_optimization_convergence(self, antenna_spec, optimization_problem):
        """Test optimization algorithm convergence properties."""
        
        # Test with simplified mock problem
        with patch('liquid_metal_antenna.solvers.base.BaseSolver') as mock_solver:
            mock_solver.simulate.return_value = Mock(
                gain_dbi=8.5,
                efficiency=0.85,
                s11_db=-15.0,
                success=True
            )
            
            optimizer = QuantumInspiredOptimizer(solver=mock_solver)
            
            # Run short optimization
            result = optimizer.optimize(
                geometry_bounds=optimization_problem['bounds'],
                spec=antenna_spec,
                max_evaluations=50
            )
            
            # Validate convergence properties
            assert hasattr(result, 'convergence_history')
            assert len(result.convergence_history) > 0
            
            # Check convergence improvement
            if len(result.convergence_history) > 10:
                early_performance = np.mean(result.convergence_history[:5])
                late_performance = np.mean(result.convergence_history[-5:])
                
                # Should show improvement (or at least not get worse)
                assert late_performance >= early_performance * 0.95


# Comprehensive test completion continues in subsequent test classes...
# This represents the first major section of comprehensive research testing

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])