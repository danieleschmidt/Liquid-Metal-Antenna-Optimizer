"""
Test suite for Generation 5 breakthrough algorithms.

Tests:
- 🧠 Neuromorphic Optimization
- 🌀 Topological Optimization  
- 🐜 Swarm Intelligence Systems
"""

import pytest
import numpy as np
import torch
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from liquid_metal_antenna.research.neuromorphic_optimization import (
    SpikeTimingPattern, NeuromorphicNeuron, SynapticPlasticity,
    NeuromorphicOptimizer, NeuromorphicAntennaOptimizer, NeuromorphicBenchmarks
)

from liquid_metal_antenna.research.topological_optimization import (
    TopologicalDescriptor, SimplexComplex, TopologicalAntennaGeometry,
    TopologicalOptimizationObjective, TopologicalOptimizer, TopologicalAntennaDesigner
)

from liquid_metal_antenna.research.swarm_intelligence import (
    SwarmAgent, SwarmCommunicationNetwork, AntColonyOptimizer,
    ParticleSwarmOptimizer, BeeColonyOptimizer, HybridSwarmOptimizer
)


class TestNeuromorphicOptimization:
    """Test neuromorphic optimization algorithms."""
    
    def test_spike_timing_pattern(self):
        """Test spike timing pattern encoding."""
        pattern = SpikeTimingPattern(
            spikes=np.array([1.0, 1.0]),
            timing=np.array([0.1, 0.5]),
            frequency=50.0,
            amplitude=1.0,
            phase=0.0
        )
        
        # Test parameter encoding
        encoded = pattern.encode_parameter(0.5, 0.0, 1.0)
        
        assert hasattr(encoded, 'spikes')
        assert hasattr(encoded, 'timing')
        assert encoded.frequency > 0
        assert len(encoded.spikes) == len(encoded.timing)
    
    def test_neuromorphic_neuron(self):
        """Test neuromorphic neuron dynamics."""
        neuron = NeuromorphicNeuron(tau_m=20.0, threshold=-55.0, reset=-70.0)
        
        # Test integration
        spike_occurred = neuron.integrate_step(input_current=100.0, dt=0.1)
        
        assert isinstance(spike_occurred, bool)
        assert neuron.voltage >= neuron.reset
        
        # Test firing rate calculation
        firing_rate = neuron.get_firing_rate()
        assert firing_rate >= 0
    
    def test_synaptic_plasticity(self):
        """Test STDP learning rule."""
        plasticity = SynapticPlasticity(a_plus=0.01, a_minus=0.01)
        
        # Test weight changes
        potentiation = plasticity.compute_weight_change(dt=5.0)  # Post after pre
        depression = plasticity.compute_weight_change(dt=-5.0)   # Pre after post
        
        assert potentiation > 0  # Should be positive (potentiation)
        assert depression < 0    # Should be negative (depression)
    
    def test_neuromorphic_optimizer(self):
        """Test neuromorphic optimization algorithm."""
        def simple_objective(x):
            return -(np.sum(x**2))  # Simple quadratic
        
        optimizer = NeuromorphicOptimizer(
            problem_dim=4,
            population_size=10,
            learning_rate=0.1
        )
        
        results = optimizer.optimize(
            objective_function=simple_objective,
            bounds=(-2.0, 2.0),
            max_generations=5,
            convergence_threshold=1e-6
        )
        
        assert 'best_solution' in results
        assert 'best_fitness' in results
        assert 'neuromorphic_metrics' in results
        assert len(results['best_solution']) == 4
        assert results['best_fitness'] <= 0  # Should be negative for quadratic
        
        # Test neuromorphic-specific metrics
        metrics = results['neuromorphic_metrics']
        assert 'avg_spikes_per_solution' in metrics
        assert 'firing_rate_diversity' in metrics
        assert 'temporal_synchronization' in metrics
    
    def test_neuromorphic_antenna_optimizer(self):
        """Test neuromorphic antenna optimization."""
        # Mock antenna spec
        mock_spec = Mock()
        mock_spec.frequency_range = (2.4e9, 5.8e9)
        
        optimizer = NeuromorphicAntennaOptimizer(mock_spec)
        
        results = optimizer.optimize_antenna_design(
            design_variables=6,
            max_generations=3
        )
        
        assert 'best_solution' in results
        assert 'antenna_analysis' in results
        assert 'neuromorphic_insights' in results
        
        # Test antenna-specific analysis
        analysis = results['antenna_analysis']
        assert 'final_gain' in analysis
        assert 'final_bandwidth' in analysis
        assert 'final_efficiency' in analysis
    
    def test_neuromorphic_benchmarks(self):
        """Test neuromorphic benchmarking."""
        # Test benchmark comparison
        results = NeuromorphicBenchmarks.benchmark_against_classical(
            problem_dim=3, n_trials=2
        )
        
        assert 'neuromorphic_mean' in results
        assert 'classical_mean' in results
        assert 'neuromorphic_advantage' in results
        assert 'significance_test' in results


class TestTopologicalOptimization:
    """Test topological optimization algorithms."""
    
    def test_topological_descriptor(self):
        """Test topological descriptor functionality."""
        descriptor1 = TopologicalDescriptor(
            betti_numbers=[1, 2, 0],
            euler_characteristic=0,
            genus=1,
            persistence_diagram=np.array([[0.1, 0.5], [0.2, 0.8]]),
            homology_groups=[]
        )
        
        descriptor2 = TopologicalDescriptor(
            betti_numbers=[1, 1, 0],
            euler_characteristic=1,
            genus=0,
            persistence_diagram=np.array([[0.15, 0.6]]),
            homology_groups=[]
        )
        
        # Test similarity computation
        similarity = descriptor1.similarity(descriptor2)
        assert 0 <= similarity <= 1
        
        # Self-similarity should be close to 1
        self_similarity = descriptor1.similarity(descriptor1)
        assert self_similarity > 0.9
    
    def test_simplex_complex(self):
        """Test simplicial complex operations."""
        complex = SimplexComplex()
        
        # Add vertices
        v1 = complex.add_vertex((0, 0, 0))
        v2 = complex.add_vertex((1, 0, 0))
        v3 = complex.add_vertex((0, 1, 0))
        v4 = complex.add_vertex((0, 0, 1))
        
        # Add edges
        complex.add_edge(v1, v2)
        complex.add_edge(v2, v3)
        complex.add_edge(v3, v1)
        complex.add_edge(v1, v4)
        
        # Add triangle
        complex.add_triangle(v1, v2, v3)
        
        assert len(complex.vertices) == 4
        assert len(complex.edges) >= 3
        assert len(complex.triangles) == 1
        
        # Test Betti numbers
        betti_numbers = complex.compute_betti_numbers()
        assert len(betti_numbers) == 3
        assert all(b >= 0 for b in betti_numbers)
        
        # Test Euler characteristic
        euler_char = complex.compute_euler_characteristic()
        assert isinstance(euler_char, int)
    
    def test_topological_antenna_geometry(self):
        """Test topological antenna geometry representation."""
        geometry = TopologicalAntennaGeometry(grid_resolution=8)
        
        # Create geometry from parameters
        params = np.random.uniform(-1, 1, 8**3)
        geometry = geometry.from_parameter_vector(params)
        
        assert geometry.voxel_grid.shape == (8, 8, 8)
        assert geometry.topological_descriptor is not None
        
        descriptor = geometry.topological_descriptor
        assert hasattr(descriptor, 'betti_numbers')
        assert hasattr(descriptor, 'euler_characteristic')
        assert hasattr(descriptor, 'genus')
    
    def test_topological_optimization_objective(self):
        """Test topological optimization objective."""
        objective = TopologicalOptimizationObjective(
            target_topology=None,
            topology_weight=0.3
        )
        
        # Create test geometry
        geometry = TopologicalAntennaGeometry(grid_resolution=6)
        params = np.random.uniform(-1, 1, 6**3)
        geometry = geometry.from_parameter_vector(params)
        
        results = objective.evaluate(geometry)
        
        assert 'total_objective' in results
        assert 'em_performance' in results
        assert 'topology_score' in results
        assert 'manufacturing_score' in results
        assert 'betti_0' in results
        assert 'euler_characteristic' in results
        
        # All scores should be reasonable
        assert -10 <= results['total_objective'] <= 10
        assert 0 <= results['em_performance'] <= 2
        assert 0 <= results['topology_score'] <= 2
    
    def test_topological_optimizer(self):
        """Test topological optimization algorithm."""
        optimizer = TopologicalOptimizer(
            grid_resolution=6,
            population_size=8
        )
        
        objective = TopologicalOptimizationObjective(topology_weight=0.2)
        
        results = optimizer.optimize(
            objective=objective,
            max_generations=3,
            convergence_threshold=1e-6
        )
        
        assert 'best_solution' in results
        assert 'best_fitness' in results
        assert 'final_topology' in results
        assert 'topology_diversity_history' in results
        
        # Test topology analysis
        final_topology = results['final_topology']
        assert 'betti_numbers' in final_topology
        assert 'euler_characteristic' in final_topology
        assert 'genus' in final_topology
    
    def test_topological_antenna_designer(self):
        """Test topological antenna designer."""
        # Mock antenna spec
        mock_spec = Mock()
        mock_spec.frequency_range = (2.4e9, 5.8e9)
        
        designer = TopologicalAntennaDesigner(mock_spec)
        
        target_topology = {
            'betti_numbers': [1, 1, 0],
            'euler_characteristic': 1,
            'genus': 0
        }
        
        results = designer.design_topology_constrained_antenna(
            target_topology=target_topology,
            grid_resolution=6,
            max_generations=3
        )
        
        assert 'best_solution' in results
        assert 'topological_analysis' in results
        assert 'design_parameters' in results
        
        # Test topological analysis
        if 'error' not in results['topological_analysis']:
            analysis = results['topological_analysis']
            assert 'topological_classification' in analysis
            assert 'geometric_properties' in analysis
            assert 'manufacturing_considerations' in analysis


class TestSwarmIntelligence:
    """Test swarm intelligence algorithms."""
    
    def test_swarm_agent(self):
        """Test swarm agent functionality."""
        agent1 = SwarmAgent(
            position=np.array([0, 0]),
            velocity=np.array([1, 1]),
            personal_best_position=np.array([0, 0]),
            personal_best_fitness=0.0,
            fitness=0.0
        )
        
        agent2 = SwarmAgent(
            position=np.array([3, 4]),
            velocity=np.array([0, 0]),
            personal_best_position=np.array([3, 4]),
            personal_best_fitness=0.0,
            fitness=0.0
        )
        
        # Test distance calculation
        distance = agent1.distance_to(agent2)
        assert distance == 5.0  # 3-4-5 triangle
        
        # Test communication
        agent1.communication_range = 6.0
        assert agent1.can_communicate_with(agent2)
        
        agent1.communication_range = 4.0
        assert not agent1.can_communicate_with(agent2)
        
        # Test energy update
        old_energy = agent1.energy
        agent1.update_energy(0.1)  # Positive improvement
        assert agent1.energy >= old_energy
    
    def test_swarm_communication_network(self):
        """Test swarm communication network."""
        network = SwarmCommunicationNetwork()
        
        # Test message broadcasting
        discovery = {
            'fitness_improvement': 0.2,
            'search_direction': np.array([1, 0])
        }
        network.broadcast_discovery(agent_id=0, discovery=discovery)
        
        # Test local information sharing
        network.share_local_information(
            agent_id=1, 
            position=np.array([1, 2]), 
            fitness=0.5
        )
        
        # Message queue should have messages
        assert not network.message_queue.empty()
        
        # Test message processing (simplified)
        mock_agents = [Mock() for _ in range(3)]
        for agent in mock_agents:
            agent.can_communicate_with = Mock(return_value=True)
            agent.learning_rate = 0.1
            agent.velocity = np.array([0, 0])
        
        network.process_messages(mock_agents)
    
    def test_ant_colony_optimizer(self):
        """Test ant colony optimization."""
        def simple_objective(x):
            return -(np.sum(x**2))  # Simple quadratic
        
        aco = AntColonyOptimizer(n_ants=10, problem_dim=3)
        
        results = aco.optimize(
            objective_function=simple_objective,
            bounds=(-2.0, 2.0),
            max_iterations=5
        )
        
        assert 'best_solution' in results
        assert 'best_fitness' in results
        assert 'pheromone_entropy_history' in results
        assert 'final_pheromone_matrix' in results
        
        assert len(results['best_solution']) == 3
        assert results['best_fitness'] <= 0  # Should be negative for quadratic
        
        # Pheromone matrix should have correct shape
        pheromone_matrix = results['final_pheromone_matrix']
        assert pheromone_matrix.shape == (3, 100)  # 3 dims, 100 discrete values
    
    def test_particle_swarm_optimizer(self):
        """Test particle swarm optimization."""
        def simple_objective(x):
            return -(np.sum(x**2))  # Simple quadratic
        
        pso = ParticleSwarmOptimizer(n_particles=10, problem_dim=3)
        
        results = pso.optimize(
            objective_function=simple_objective,
            bounds=(-2.0, 2.0),
            max_iterations=5
        )
        
        assert 'best_solution' in results
        assert 'best_fitness' in results
        assert 'role_distribution_history' in results
        assert 'communication_activity_history' in results
        
        assert len(results['best_solution']) == 3
        assert results['best_fitness'] <= 0  # Should be negative for quadratic
        
        # Check role distribution tracking
        role_history = results['role_distribution_history']
        assert len(role_history) > 0
        
        # Each entry should have role counts
        for role_dist in role_history:
            assert 'explorer' in role_dist
            assert 'exploiter' in role_dist
            assert 'scout' in role_dist
            assert 'leader' in role_dist
    
    def test_bee_colony_optimizer(self):
        """Test bee colony optimization."""
        def simple_objective(x):
            return -(np.sum(x**2))  # Simple quadratic
        
        abc = BeeColonyOptimizer(n_bees=10, problem_dim=3)
        
        results = abc.optimize(
            objective_function=simple_objective,
            bounds=(-2.0, 2.0),
            max_iterations=5
        )
        
        assert 'best_solution' in results
        assert 'best_fitness' in results
        assert 'diversity_history' in results
        assert 'final_food_sources' in results
        assert 'final_trials' in results
        
        assert len(results['best_solution']) == 3
        assert results['best_fitness'] <= 0  # Should be negative for quadratic
        
        # Check food source tracking
        food_sources = results['final_food_sources']
        trials = results['final_trials']
        assert len(food_sources) == len(trials)
        assert len(food_sources) == 5  # n_bees // 2
    
    def test_hybrid_swarm_optimizer(self):
        """Test hybrid swarm optimization."""
        def simple_objective(x):
            return -(np.sum(x**2))  # Simple quadratic
        
        hybrid = HybridSwarmOptimizer(problem_dim=3, total_agents=20)
        
        results = hybrid.optimize(
            objective_function=simple_objective,
            bounds=(-2.0, 2.0),
            max_iterations=5
        )
        
        assert 'best_solution' in results
        assert 'best_fitness' in results
        assert 'combined_fitness_history' in results
        assert 'algorithm_selection_history' in results
        assert 'individual_results' in results
        
        assert len(results['best_solution']) == 3
        assert results['best_fitness'] <= 0  # Should be negative for quadratic
        
        # Check individual algorithm results
        individual = results['individual_results']
        assert 'aco' in individual
        assert 'pso' in individual
        assert 'abc' in individual
        
        # Each should have best solution and fitness
        for alg_results in individual.values():
            assert 'best_solution' in alg_results
            assert 'best_fitness' in alg_results


class TestIntegration:
    """Integration tests for Generation 5 algorithms."""
    
    def test_comparative_performance(self):
        """Compare performance of different algorithms on same problem."""
        def test_function(x):
            # Rosenbrock function
            return -np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        problem_dim = 4
        max_iterations = 10
        bounds = (-2.0, 2.0)
        
        results = {}
        
        # Test neuromorphic
        neuro_opt = NeuromorphicOptimizer(problem_dim=problem_dim, population_size=15)
        results['neuromorphic'] = neuro_opt.optimize(
            test_function, bounds, max_generations=max_iterations
        )
        
        # Test topological (simplified test)
        topo_opt = TopologicalOptimizer(grid_resolution=6, population_size=10)
        objective = TopologicalOptimizationObjective(topology_weight=0.1)
        results['topological'] = topo_opt.optimize(
            objective, max_generations=5
        )
        
        # Test swarm
        pso = ParticleSwarmOptimizer(n_particles=15, problem_dim=problem_dim)
        results['swarm_pso'] = pso.optimize(
            test_function, bounds, max_iterations=max_iterations
        )
        
        # All algorithms should produce results
        for alg_name, result in results.items():
            assert 'best_solution' in result
            assert 'best_fitness' in result
            
            if alg_name != 'topological':  # Topological uses different problem
                assert len(result['best_solution']) == problem_dim
    
    def test_algorithm_robustness(self):
        """Test algorithm robustness to edge cases."""
        def degenerate_objective(x):
            return 0.0  # Constant function
        
        # Test neuromorphic with constant objective
        neuro_opt = NeuromorphicOptimizer(problem_dim=2, population_size=5)
        results = neuro_opt.optimize(
            degenerate_objective, 
            bounds=(-1.0, 1.0),
            max_generations=3
        )
        
        assert 'best_solution' in results
        assert results['best_fitness'] == 0.0
        
        # Test with very small bounds
        small_bounds = (-0.01, 0.01)
        results = neuro_opt.optimize(
            lambda x: -np.sum(x**2),
            bounds=small_bounds,
            max_generations=3
        )
        
        assert 'best_solution' in results
        assert all(small_bounds[0] <= xi <= small_bounds[1] for xi in results['best_solution'])
    
    def test_scalability(self):
        """Test algorithm scalability to different problem sizes."""
        def quadratic(x):
            return -np.sum(x**2)
        
        dimensions = [2, 5, 8]
        
        for dim in dimensions:
            # Test neuromorphic scaling
            neuro_opt = NeuromorphicOptimizer(problem_dim=dim, population_size=8)
            results = neuro_opt.optimize(
                quadratic,
                bounds=(-1.0, 1.0),
                max_generations=3
            )
            
            assert len(results['best_solution']) == dim
            assert 'neuromorphic_metrics' in results
            
            # Test PSO scaling
            pso = ParticleSwarmOptimizer(n_particles=8, problem_dim=dim)
            results = pso.optimize(
                quadratic,
                bounds=(-1.0, 1.0),
                max_iterations=3
            )
            
            assert len(results['best_solution']) == dim
            assert 'role_distribution_history' in results


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])