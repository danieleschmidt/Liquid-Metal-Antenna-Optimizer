"""
Comprehensive tests for novel research algorithms.

This module tests the advanced research algorithms implemented for liquid metal
antenna optimization, including multi-physics optimization, graph neural networks,
and uncertainty quantification.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from liquid_metal_antenna.core.antenna_spec import AntennaSpec
from liquid_metal_antenna.core.optimizer import OptimizationResult
from liquid_metal_antenna.solvers.base import SolverResult, BaseSolver
from liquid_metal_antenna.research.multi_physics_optimization import (
    MultiPhysicsOptimizer, MultiPhysicsSolver, MultiPhysicsResult
)
from liquid_metal_antenna.research.graph_neural_surrogate import (
    GraphNeuralSurrogate, AntennaGraphBuilder, AntennaGraph
)
from liquid_metal_antenna.research.uncertainty_quantification import (
    RobustOptimizer, UncertaintyPropagator, SensitivityAnalyzer,
    create_manufacturing_uncertainty_model, create_environmental_uncertainty_model
)


@pytest.fixture
def mock_solver():
    """Create mock electromagnetic solver."""
    solver = Mock(spec=BaseSolver)
    
    def mock_simulate(geometry, frequency, spec=None):
        """Mock simulation that returns realistic results."""
        # Simple heuristic based on geometry
        metal_fraction = np.mean(geometry > 0.5)
        gain = 5.0 + metal_fraction * 3.0
        efficiency = 0.7 + metal_fraction * 0.15
        
        s_params = np.array([[[complex(-15.0 - metal_fraction * 5.0, 0)]]])
        
        return SolverResult(
            s_parameters=s_params,
            impedance=complex(50.0, 0.0),
            gain_dbi=gain,
            efficiency=efficiency,
            radiation_pattern=None,
            field_distribution=None,
            computation_time=0.1,
            converged=True,
            metadata={'mock': True}
        )
    
    solver.simulate.side_effect = mock_simulate
    return solver


@pytest.fixture
def sample_antenna_spec():
    """Create sample antenna specification."""
    return AntennaSpec(
        frequency_range=(2.4e9, 2.5e9),
        substrate='fr4',
        metal='galinstan',
        size_constraint=(25, 25, 1.6)
    )


@pytest.fixture
def sample_geometry():
    """Create sample antenna geometry."""
    geometry = np.zeros((32, 32, 8))
    # Simple patch
    geometry[12:20, 12:20, 6] = 1.0
    return geometry


class TestMultiPhysicsOptimization:
    """Test suite for multi-physics optimization algorithms."""
    
    def test_multi_physics_solver_initialization(self, mock_solver):
        """Test multi-physics solver initialization."""
        mp_solver = MultiPhysicsSolver(mock_solver)
        
        assert mp_solver.em_solver == mock_solver
        assert mp_solver.fluid_solver is not None
        assert mp_solver.thermal_solver is not None
        assert mp_solver.coupling_tolerance == 1e-4
        assert mp_solver.max_coupling_iterations == 10
    
    def test_multi_physics_solver_coupled_simulation(self, mock_solver, sample_geometry, sample_antenna_spec):
        """Test coupled multi-physics simulation."""
        mp_solver = MultiPhysicsSolver(mock_solver, max_coupling_iterations=3)
        
        result = mp_solver.solve_coupled(
            sample_geometry,
            2.4e9,
            sample_antenna_spec
        )
        
        assert isinstance(result, MultiPhysicsResult)
        assert result.em_result is not None
        assert result.fluid_result is not None
        assert result.thermal_result is not None
        assert result.coupling_analysis is not None
        assert result.combined_objectives is not None
        assert result.total_computation_time > 0
        
        # Check combined objectives
        assert 'gain_dbi' in result.combined_objectives
        assert 'thermal_efficiency' in result.combined_objectives
        assert 'multiphysics_performance' in result.combined_objectives
        
        # Check coupling analysis
        assert 'coupling_iterations' in result.coupling_analysis
        assert result.coupling_iterations <= 3
    
    def test_multi_physics_optimizer_initialization(self, mock_solver):
        """Test multi-physics optimizer initialization."""
        optimizer = MultiPhysicsOptimizer(mock_solver)
        
        assert optimizer.name == 'MultiPhysicsOptimizer'
        assert optimizer.multi_physics_solver is not None
        assert optimizer.physics_weights is not None
        assert 'electromagnetic' in optimizer.physics_weights
        assert 'fluid' in optimizer.physics_weights
        assert 'thermal' in optimizer.physics_weights
    
    def test_multi_physics_optimization_run(self, mock_solver, sample_antenna_spec):
        """Test multi-physics optimization execution."""
        optimizer = MultiPhysicsOptimizer(mock_solver)
        
        # Run short optimization
        result = optimizer.optimize(
            sample_antenna_spec,
            objective='multiphysics_performance',
            max_iterations=3  # Short run for testing
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.algorithm == 'multi_physics_optimization'
        assert result.research_data is not None
        assert 'multiphysics_analysis' in result.research_data
        assert 'novel_contributions' in result.research_data
        
        # Check research data structure
        mp_analysis = result.research_data['multiphysics_analysis']
        assert 'physics_coupling_evolution' in mp_analysis
        assert 'computational_efficiency' in mp_analysis
        assert 'coupling_convergence_study' in mp_analysis
        
        novel_contributions = result.research_data['novel_contributions']
        assert 'multi_physics_coupling_effects' in novel_contributions
        assert 'constraint_interaction_analysis' in novel_contributions
        assert 'physics_sensitivity_analysis' in novel_contributions
    
    def test_multi_physics_population_initialization(self, mock_solver, sample_antenna_spec):
        """Test multi-physics aware population initialization."""
        optimizer = MultiPhysicsOptimizer(mock_solver)
        
        population = optimizer._initialize_multiphysics_population(sample_antenna_spec)
        
        assert len(population) == optimizer.population_size
        assert all(isinstance(individual, np.ndarray) for individual in population)
        assert all(individual.shape == (32, 32, 8) for individual in population)
        
        # Check that designs have reasonable metal content
        metal_fractions = [np.mean(individual[:, :, 6] > 0.5) for individual in population]
        assert all(0.01 < frac < 0.8 for frac in metal_fractions)  # Reasonable range
    
    def test_multi_physics_constraint_handling(self, mock_solver, sample_antenna_spec):
        """Test multi-physics constraint handling."""
        optimizer = MultiPhysicsOptimizer(mock_solver)
        
        # Mock multi-physics result
        mock_mp_result = Mock()
        mock_mp_result.combined_objectives = {
            'max_temperature': 350.0,  # Exceeds typical limit
            'max_flow_velocity': 0.05,  # Within limit
            'temperature_uniformity': 0.75  # Below desired threshold
        }
        
        constraints = {
            'max_temperature': 373.15,
            'max_flow_velocity': 0.1,
            'min_thermal_uniformity': 0.8
        }
        
        violations = optimizer._check_constraints(mock_mp_result, constraints)
        
        assert 'temperature' in violations
        assert not violations['temperature']  # 350 < 373.15
        assert not violations['flow_velocity']  # 0.05 < 0.1
        assert violations['thermal_uniformity']  # 0.75 < 0.8
    
    def test_multi_physics_crossover(self, mock_solver):
        """Test multi-physics aware crossover."""
        optimizer = MultiPhysicsOptimizer(mock_solver)
        
        parent1 = np.random.random((32, 32, 8))
        parent2 = np.random.random((32, 32, 8))
        
        # Mock robustness metrics
        mp_result1 = Mock()
        mp_result1.combined_objectives = {'thermal_efficiency': 0.8}
        mp_result2 = Mock()
        mp_result2.combined_objectives = {'thermal_efficiency': 0.6}
        
        offspring = optimizer._multiphysics_crossover(parent1, parent2, mp_result1, mp_result2)
        
        assert offspring.shape == parent1.shape
        assert not np.array_equal(offspring, parent1)  # Should be different
        assert not np.array_equal(offspring, parent2)  # Should be different


class TestGraphNeuralSurrogate:
    """Test suite for graph neural surrogate models."""
    
    def test_antenna_graph_builder_initialization(self):
        """Test antenna graph builder initialization."""
        builder = AntennaGraphBuilder(
            node_density='adaptive',
            edge_connectivity='physics_aware'
        )
        
        assert builder.node_density == 'adaptive'
        assert builder.edge_connectivity == 'physics_aware'
        assert builder.include_field_coupling == True
        assert builder.max_nodes == 1000
    
    def test_graph_construction(self, sample_geometry, sample_antenna_spec):
        """Test antenna graph construction."""
        builder = AntennaGraphBuilder(node_density='medium', max_nodes=100)
        
        graph = builder.build_graph(
            sample_geometry,
            2.4e9,
            sample_antenna_spec
        )
        
        assert isinstance(graph, AntennaGraph)
        assert len(graph.nodes) > 0
        assert len(graph.nodes) <= 100
        assert len(graph.edges) >= 0
        assert graph.frequency == 2.4e9
        assert graph.global_features is not None
        
        # Check node properties
        for node in graph.nodes:
            assert node.position is not None
            assert node.node_type in ['metal', 'dielectric', 'air']
            assert 'conductivity' in node.material_properties
            assert 'permittivity' in node.material_properties
        
        # Check edge properties
        for edge in graph.edges:
            assert edge.source_node in [n.node_id for n in graph.nodes]
            assert edge.target_node in [n.node_id for n in graph.nodes]
            assert edge.distance >= 0
            assert edge.coupling_strength >= 0
    
    def test_graph_feature_extraction(self, sample_geometry, sample_antenna_spec):
        """Test graph feature extraction."""
        builder = AntennaGraphBuilder(max_nodes=50)
        graph = builder.build_graph(sample_geometry, 2.4e9, sample_antenna_spec)
        
        # Test adjacency matrix
        adj_matrix = graph.get_adjacency_matrix()
        assert adj_matrix.shape == (len(graph.nodes), len(graph.nodes))
        assert np.allclose(adj_matrix, adj_matrix.T)  # Should be symmetric
        
        # Test node features
        node_features = graph.get_node_features()
        assert node_features.shape[0] == len(graph.nodes)
        assert node_features.shape[1] > 0  # Should have features
        
        # Test edge features
        if graph.edges:
            edge_features = graph.get_edge_features()
            assert edge_features.shape[0] == len(graph.edges)
            assert edge_features.shape[1] > 0
    
    def test_adaptive_node_placement(self, sample_geometry, sample_antenna_spec):
        """Test adaptive node placement strategy."""
        builder = AntennaGraphBuilder(node_density='adaptive', max_nodes=200)
        
        # Create field data for adaptive placement
        field_data = {
            'E_field': (
                np.random.random((32, 32)) * 1e5,  # Electric field magnitude
                np.random.random((32, 32)) * 1e5
            )
        }
        
        graph = builder.build_graph(sample_geometry, 2.4e9, sample_antenna_spec, field_data)
        
        assert len(graph.nodes) > 0
        assert len(graph.nodes) <= 200
        
        # Should have more nodes near metal regions (higher field gradients)
        metal_region_nodes = [node for node in graph.nodes 
                             if sample_geometry[node.position[0], node.position[1], 6] > 0.5]
        total_nodes = len(graph.nodes)
        metal_node_ratio = len(metal_region_nodes) / total_nodes
        
        # Expect higher concentration of nodes in metal regions
        assert metal_node_ratio > 0.1  # At least 10% of nodes in metal regions
    
    def test_physics_aware_edge_creation(self, sample_geometry, sample_antenna_spec):
        """Test physics-aware edge creation."""
        builder = AntennaGraphBuilder(edge_connectivity='physics_aware', max_nodes=50)
        graph = builder.build_graph(sample_geometry, 2.4e9, sample_antenna_spec)
        
        if len(graph.edges) > 0:
            # Check that coupling edges exist between metal nodes
            coupling_edges = [edge for edge in graph.edges if edge.edge_type == 'coupling']
            
            if coupling_edges:
                # Coupling edges should connect metal nodes
                for edge in coupling_edges[:5]:  # Check first few
                    source_node = graph.nodes[graph.node_index[edge.source_node]]
                    target_node = graph.nodes[graph.node_index[edge.target_node]]
                    
                    # At least one node should be metal for coupling
                    assert (source_node.node_type == 'metal' or target_node.node_type == 'metal')
    
    def test_graph_neural_surrogate_initialization(self):
        """Test GNN surrogate initialization."""
        gnn = GraphNeuralSurrogate(
            hidden_dim=64,
            num_layers=4,
            num_attention_heads=8
        )
        
        assert gnn.hidden_dim == 64
        assert gnn.num_layers == 4
        assert gnn.num_attention_heads == 8
        assert not gnn.is_trained
        assert gnn.graph_builder is not None
    
    def test_gnn_prediction_untrained(self, sample_geometry, sample_antenna_spec):
        """Test GNN prediction when model is untrained."""
        gnn = GraphNeuralSurrogate()
        
        result = gnn.predict(sample_geometry, 2.4e9, sample_antenna_spec)
        
        assert isinstance(result, SolverResult)
        assert result.gain_dbi is not None
        assert result.efficiency is not None
        assert result.s_parameters is not None
        assert result.computation_time > 0
        assert 'solver_type' in result.metadata
    
    def test_gnn_training_simulation(self):
        """Test GNN training simulation."""
        gnn = GraphNeuralSurrogate(hidden_dim=32, num_layers=2)
        
        # Create mock training data
        training_data = []
        for _ in range(5):
            # Mock graph and targets
            mock_graph = Mock()
            mock_graph.get_node_features.return_value = np.random.random((10, 8))
            mock_graph.get_edge_features.return_value = np.random.random((15, 5))
            mock_graph.global_features = {'frequency': 2.4e9}
            
            targets = {'gain': 5.0 + np.random.normal(0, 1)}
            training_data.append((mock_graph, targets))
        
        validation_data = training_data[:2]  # Use subset for validation
        
        # Run training simulation
        training_results = gnn.train(
            training_data,
            validation_data,
            num_epochs=5,  # Short training for testing
            batch_size=2
        )
        
        assert isinstance(training_results, dict)
        assert 'training_losses' in training_results
        assert 'validation_losses' in training_results
        assert 'total_training_time' in training_results
        assert len(training_results['training_losses']) <= 5
        assert gnn.is_trained
    
    def test_gnn_attention_analysis(self, sample_geometry, sample_antenna_spec):
        """Test GNN attention pattern analysis."""
        gnn = GraphNeuralSurrogate()
        gnn.is_trained = True  # Mock training
        
        attention_analysis = gnn.analyze_attention_patterns(
            sample_geometry, 2.4e9, sample_antenna_spec
        )
        
        assert isinstance(attention_analysis, dict)
        assert 'node_attention_scores' in attention_analysis
        assert 'edge_attention_patterns' in attention_analysis
        assert 'physics_attention_analysis' in attention_analysis
        assert 'frequency_response_attention' in attention_analysis


class TestUncertaintyQuantification:
    """Test suite for uncertainty quantification framework."""
    
    def test_manufacturing_uncertainty_model_creation(self):
        """Test creation of manufacturing uncertainty model."""
        model = create_manufacturing_uncertainty_model()
        
        assert model.n_parameters == 4
        assert len(model.parameters) == 4
        
        # Check parameter types
        param_types = [p.parameter_type for p in model.parameters]
        assert 'geometric' in param_types
        assert 'material' in param_types
        
        # Check parameter names
        param_names = [p.name for p in model.parameters]
        assert 'geometry_scaling' in param_names
        assert 'conductivity_variation' in param_names
    
    def test_environmental_uncertainty_model_creation(self):
        """Test creation of environmental uncertainty model."""
        model = create_environmental_uncertainty_model()
        
        assert model.n_parameters == 3
        assert len(model.parameters) == 3
        
        # Check parameter types
        param_types = [p.parameter_type for p in model.parameters]
        assert 'environmental' in param_types
        assert 'material' in param_types
        
        # Check parameter names
        param_names = [p.name for p in model.parameters]
        assert 'temperature' in param_names
        assert 'humidity' in param_names
    
    def test_uncertainty_propagator_initialization(self):
        """Test uncertainty propagator initialization."""
        propagator = UncertaintyPropagator(
            method='monte_carlo',
            max_evaluations=100,
            confidence_level=0.95
        )
        
        assert propagator.method == 'monte_carlo'
        assert propagator.max_evaluations == 100
        assert propagator.confidence_level == 0.95
        assert propagator.samples_generated == 0
        assert propagator.evaluations_completed == 0
    
    def test_monte_carlo_uncertainty_propagation(self):
        """Test Monte Carlo uncertainty propagation."""
        propagator = UncertaintyPropagator(method='monte_carlo', max_evaluations=50)
        
        # Create simple evaluation function
        def eval_func(params):
            return {'output': params[0] + 0.5 * params[1] + np.random.normal(0, 0.1)}
        
        # Create simple uncertainty model
        uncertainty_model = create_manufacturing_uncertainty_model()
        
        results = propagator.propagate_uncertainty(
            eval_func,
            uncertainty_model,
            ['output']
        )
        
        assert 'statistics' in results
        assert 'output' in results['statistics']
        assert 'mean' in results['statistics']['output']
        assert 'std' in results['statistics']['output']
        assert 'percentile_5' in results['statistics']['output']
        assert 'percentile_95' in results['statistics']['output']
        
        assert results['total_evaluations'] > 0
        assert results['computation_time'] > 0
    
    def test_adaptive_monte_carlo_propagation(self):
        """Test adaptive Monte Carlo propagation."""
        propagator = UncertaintyPropagator(
            method='adaptive_monte_carlo',
            max_evaluations=100,
            convergence_tolerance=0.05
        )
        
        def eval_func(params):
            # Function with known statistics for testing convergence
            return {'output': 2.0 * params[0] + 1.0}
        
        uncertainty_model = create_manufacturing_uncertainty_model()
        
        results = propagator.propagate_uncertainty(
            eval_func,
            uncertainty_model,
            ['output']
        )
        
        assert 'convergence_achieved' in results
        assert 'convergence_history' in results
        assert results['total_evaluations'] <= 100
        
        # Check statistics are reasonable
        stats = results['statistics']['output']
        assert abs(stats['mean'] - 2.0) < 0.2  # Should be close to expected value
    
    def test_polynomial_chaos_propagation(self):
        """Test polynomial chaos expansion propagation."""
        propagator = UncertaintyPropagator(method='polynomial_chaos', max_evaluations=200)
        
        def eval_func(params):
            # Polynomial function for PCE testing
            return {'output': params[0]**2 + 2*params[1] + 1}
        
        uncertainty_model = create_manufacturing_uncertainty_model()
        
        results = propagator.propagate_uncertainty(
            eval_func,
            uncertainty_model,
            ['output']
        )
        
        assert 'pce_coefficients' in results
        assert 'pce_statistics' in results
        assert 'polynomial_order' in results
        assert 'output' in results['pce_statistics']
        
        # PCE statistics should exist
        pce_stats = results['pce_statistics']['output']
        assert 'mean' in pce_stats
        assert 'variance' in pce_stats
    
    def test_sensitivity_analyzer_initialization(self):
        """Test sensitivity analyzer initialization."""
        analyzer = SensitivityAnalyzer(method='sobol', n_bootstrap=500)
        
        assert analyzer.method == 'sobol'
        assert analyzer.n_bootstrap == 500
    
    def test_sobol_sensitivity_analysis(self):
        """Test Sobol sensitivity analysis."""
        analyzer = SensitivityAnalyzer(method='sobol')
        
        def eval_func(params):
            # Function where param[0] has higher sensitivity than param[1]
            return {'output': 3 * params[0] + 0.5 * params[1]}
        
        uncertainty_model = create_manufacturing_uncertainty_model()
        
        results = analyzer.analyze_sensitivity(
            eval_func,
            uncertainty_model,
            ['output'],
            n_samples=100  # Small for testing
        )
        
        assert 'sobol_indices' in results
        assert 'output' in results['sobol_indices']
        
        sobol_data = results['sobol_indices']['output']
        assert 'first_order' in sobol_data
        assert 'total_order' in sobol_data
        assert 'parameter_names' in sobol_data
        
        assert len(sobol_data['first_order']) == uncertainty_model.n_parameters
        assert len(sobol_data['total_order']) == uncertainty_model.n_parameters
    
    def test_morris_sensitivity_analysis(self):
        """Test Morris sensitivity analysis."""
        analyzer = SensitivityAnalyzer(method='morris')
        
        def eval_func(params):
            return {'output': params[0] + params[1]**2}
        
        uncertainty_model = create_manufacturing_uncertainty_model()
        
        results = analyzer.analyze_sensitivity(
            eval_func,
            uncertainty_model,
            ['output'],
            n_samples=50  # Small for testing
        )
        
        assert 'morris_measures' in results
        assert 'output' in results['morris_measures']
        
        morris_data = results['morris_measures']['output']
        assert 'mu' in morris_data
        assert 'mu_star' in morris_data
        assert 'sigma' in morris_data
    
    def test_robust_optimizer_initialization(self, mock_solver):
        """Test robust optimizer initialization."""
        uncertainty_model = create_manufacturing_uncertainty_model()
        
        optimizer = RobustOptimizer(
            mock_solver,
            uncertainty_model,
            robustness_measure='mean_plus_std',
            confidence_level=0.95
        )
        
        assert optimizer.name == 'RobustOptimizer'
        assert optimizer.uncertainty_model == uncertainty_model
        assert optimizer.robustness_measure == 'mean_plus_std'
        assert optimizer.confidence_level == 0.95
        assert optimizer.uq_propagator is not None
        assert optimizer.sensitivity_analyzer is not None
    
    def test_robust_optimization_run(self, mock_solver, sample_antenna_spec):
        """Test robust optimization execution."""
        uncertainty_model = create_manufacturing_uncertainty_model()
        optimizer = RobustOptimizer(
            mock_solver,
            uncertainty_model,
            max_uq_evaluations=20  # Small for testing
        )
        
        # Run short optimization
        result = optimizer.optimize(
            sample_antenna_spec,
            objective='gain',
            max_iterations=2  # Very short for testing
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.algorithm == 'robust_optimization_with_uq'
        assert result.research_data is not None
        
        # Check research data structure
        research_data = result.research_data
        assert 'robust_optimization_analysis' in research_data
        assert 'novel_contributions' in research_data
        assert 'uncertainty_quantification_study' in research_data
        
        ro_analysis = research_data['robust_optimization_analysis']
        assert 'robustness_evolution' in ro_analysis
        assert 'uncertainty_propagation_efficiency' in ro_analysis
    
    def test_robust_objective_calculation(self, mock_solver):
        """Test robust objective calculation."""
        uncertainty_model = create_manufacturing_uncertainty_model()
        optimizer = RobustOptimizer(mock_solver, uncertainty_model)
        
        # Mock UQ result
        mock_uq_result = {
            'statistics': {
                'gain': {
                    'mean': 6.0,
                    'std': 0.5,
                    'percentile_5': 5.2,
                    'min': 4.8
                }
            }
        }
        
        constraints = {'robustness_factor': 2.0}
        
        # Test different robustness measures
        optimizer.robustness_measure = 'mean_plus_std'
        robust_obj1 = optimizer._calculate_robust_objective(mock_uq_result, 'gain', constraints)
        expected1 = 6.0 - 2.0 * 0.5  # mean - k*std
        assert abs(robust_obj1 - expected1) < 0.1
        
        optimizer.robustness_measure = 'percentile'
        robust_obj2 = optimizer._calculate_robust_objective(mock_uq_result, 'gain', constraints)
        assert robust_obj2 <= 6.0  # Should be less than mean
        
        optimizer.robustness_measure = 'worst_case'
        robust_obj3 = optimizer._calculate_robust_objective(mock_uq_result, 'gain', constraints)
        assert abs(robust_obj3 - 4.8) < 0.1  # Should be minimum
    
    def test_robustness_metrics_extraction(self, mock_solver):
        """Test extraction of robustness metrics."""
        uncertainty_model = create_manufacturing_uncertainty_model()
        optimizer = RobustOptimizer(mock_solver, uncertainty_model)
        
        # Mock UQ result
        mock_uq_result = {
            'statistics': {
                'gain': {
                    'mean': 6.0,
                    'std': 0.8,
                    'min': 4.5,
                    'max': 7.5,
                    'percentile_5': 5.0,
                    'percentile_95': 7.0,
                    'cv': 0.133
                },
                's11': {
                    'mean': -15.0,
                    'std': 2.0,
                    'min': -20.0,
                    'max': -10.0
                }
            }
        }
        
        metrics = optimizer._extract_robustness_metrics(mock_uq_result)
        
        assert 'gain' in metrics.mean_performance
        assert 's11' in metrics.mean_performance
        assert metrics.mean_performance['gain'] == 6.0
        
        assert 'gain' in metrics.std_performance
        assert metrics.std_performance['gain'] == 0.8
        
        assert 'gain' in metrics.percentile_performance
        assert '5th' in metrics.percentile_performance['gain']
        assert '95th' in metrics.percentile_performance['gain']
        
        assert metrics.reliability_score > 0
        assert metrics.reliability_score <= 1.0
        
        assert 'gain' in metrics.probability_of_failure
        assert 's11' in metrics.probability_of_failure
    
    def test_robust_constraint_checking(self, mock_solver):
        """Test robust constraint checking."""
        uncertainty_model = create_manufacturing_uncertainty_model()
        optimizer = RobustOptimizer(mock_solver, uncertainty_model)
        
        # Mock robustness metrics
        mock_metrics = Mock()
        mock_metrics.reliability_score = 0.85
        mock_metrics.robust_design_margin = 0.75
        mock_metrics.mean_performance = {'gain': 6.0}
        mock_metrics.std_performance = {'gain': 1.2}
        
        constraints = {
            'reliability_threshold': 0.9,
            'min_design_margin': 0.8,
            'max_cv_gain': 0.15
        }
        
        violations = optimizer._check_robust_constraints(mock_metrics, constraints)
        
        assert 'reliability' in violations
        assert violations['reliability']  # 0.85 < 0.9
        
        assert 'design_margin' in violations
        assert violations['design_margin']  # 0.75 < 0.8
        
        assert 'cv_gain' in violations
        assert violations['cv_gain']  # CV = 1.2/6.0 = 0.2 > 0.15


class TestResearchBenchmarks:
    """Test suite for research benchmarks and validation."""
    
    def test_algorithm_integration(self, mock_solver, sample_antenna_spec):
        """Test integration between different research algorithms."""
        # Create algorithms
        mp_optimizer = MultiPhysicsOptimizer(mock_solver)
        gnn_surrogate = GraphNeuralSurrogate(hidden_dim=32, num_layers=2)
        uncertainty_model = create_manufacturing_uncertainty_model()
        robust_optimizer = RobustOptimizer(mock_solver, uncertainty_model, max_uq_evaluations=10)
        
        # Test that they can all run without errors
        algorithms = {
            'multi_physics': mp_optimizer,
            'robust_optimization': robust_optimizer
        }
        
        for name, algorithm in algorithms.items():
            try:
                result = algorithm.optimize(
                    sample_antenna_spec,
                    max_iterations=2  # Very short runs
                )
                assert isinstance(result, OptimizationResult)
                assert result.algorithm is not None
                assert result.research_data is not None
            except Exception as e:
                pytest.fail(f"Algorithm {name} failed: {str(e)}")
    
    def test_research_data_completeness(self, mock_solver, sample_antenna_spec):
        """Test that research algorithms generate complete research data."""
        # Test multi-physics optimizer
        mp_optimizer = MultiPhysicsOptimizer(mock_solver)
        mp_result = mp_optimizer.optimize(sample_antenna_spec, max_iterations=2)
        
        # Check required research data fields
        assert 'multiphysics_analysis' in mp_result.research_data
        assert 'novel_contributions' in mp_result.research_data
        assert 'optimization_methodology' in mp_result.research_data
        assert 'convergence_analysis' in mp_result.research_data
        
        # Test robust optimizer
        uncertainty_model = create_manufacturing_uncertainty_model()
        robust_optimizer = RobustOptimizer(mock_solver, uncertainty_model, max_uq_evaluations=10)
        robust_result = robust_optimizer.optimize(sample_antenna_spec, max_iterations=2)
        
        # Check required research data fields
        assert 'robust_optimization_analysis' in robust_result.research_data
        assert 'uncertainty_quantification_study' in robust_result.research_data
        assert 'robustness_methodology' in robust_result.research_data
    
    def test_computational_efficiency_tracking(self, mock_solver, sample_antenna_spec):
        """Test that algorithms track computational efficiency metrics."""
        algorithms = [
            MultiPhysicsOptimizer(mock_solver),
            RobustOptimizer(mock_solver, create_manufacturing_uncertainty_model(), max_uq_evaluations=10)
        ]
        
        for algorithm in algorithms:
            result = algorithm.optimize(sample_antenna_spec, max_iterations=2)
            
            # All algorithms should track computation time
            assert result.total_time > 0
            assert 'total_optimization_time' in result.research_data
            
            # Should have convergence tracking
            assert result.optimization_history is not None
            assert len(result.optimization_history) >= 0
    
    def test_reproducibility_features(self, mock_solver, sample_antenna_spec):
        """Test reproducibility features of research algorithms."""
        # Set seed for reproducibility testing
        np.random.seed(42)
        
        # Run same algorithm twice
        mp_optimizer = MultiPhysicsOptimizer(mock_solver)
        
        result1 = mp_optimizer.optimize(sample_antenna_spec, max_iterations=2)
        
        # Reset seed
        np.random.seed(42)
        result2 = mp_optimizer.optimize(sample_antenna_spec, max_iterations=2)
        
        # Results should be similar (allowing for some randomness)
        # This is a basic test - in practice would need more sophisticated reproducibility checks
        assert result1.algorithm == result2.algorithm
        assert len(result1.optimization_history) == len(result2.optimization_history)


# Performance and stress tests

class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    @pytest.mark.slow
    def test_multi_physics_scaling(self, mock_solver, sample_antenna_spec):
        """Test multi-physics optimization scaling with problem size."""
        # Test with different iteration counts
        iteration_counts = [5, 10, 20]
        times = []
        
        for max_iter in iteration_counts:
            optimizer = MultiPhysicsOptimizer(mock_solver)
            start_time = time.time()
            
            result = optimizer.optimize(sample_antenna_spec, max_iterations=max_iter)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Should complete successfully
            assert isinstance(result, OptimizationResult)
        
        # Time should scale reasonably (not exponentially)
        assert all(t > 0 for t in times)
        # Basic scaling check - later tests should be longer but not excessively
        if len(times) > 1:
            assert times[-1] < times[0] * 10  # No worse than 10x scaling
    
    @pytest.mark.slow
    def test_uncertainty_quantification_scaling(self):
        """Test UQ scaling with number of evaluations."""
        eval_counts = [20, 50, 100]
        times = []
        
        def simple_eval_func(params):
            return {'output': np.sum(params)}
        
        uncertainty_model = create_manufacturing_uncertainty_model()
        
        for max_evals in eval_counts:
            propagator = UncertaintyPropagator(
                method='monte_carlo',
                max_evaluations=max_evals
            )
            
            start_time = time.time()
            result = propagator.propagate_uncertainty(
                simple_eval_func,
                uncertainty_model,
                ['output']
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert result['total_evaluations'] <= max_evals
        
        # Time should scale approximately linearly
        assert all(t > 0 for t in times)
    
    def test_memory_usage_reasonable(self, mock_solver, sample_antenna_spec):
        """Test that algorithms don't use excessive memory."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run algorithms
        mp_optimizer = MultiPhysicsOptimizer(mock_solver)
        result = mp_optimizer.optimize(sample_antenna_spec, max_iterations=3)
        
        # Check memory usage didn't explode
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use more than 500MB additional memory for test runs
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f} MB"
        
        # Clean up
        del result
        del mp_optimizer


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])