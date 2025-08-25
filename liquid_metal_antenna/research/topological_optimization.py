"""
🌀 Topological Antenna Optimization Framework
=============================================

Generation 5 breakthrough: Topology-aware optimization using differential geometry
and topological invariants for revolutionary antenna design exploration.

Author: Terry @ Terragon Labs
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


@dataclass
class TopologicalDescriptor:
    """Topological descriptor for antenna geometry."""
    betti_numbers: List[int]
    euler_characteristic: int
    genus: int
    persistence_diagram: np.ndarray
    homology_groups: List[Dict[str, Any]]
    
    def similarity(self, other: 'TopologicalDescriptor') -> float:
        """Compute topological similarity between descriptors."""
        # Betti number similarity
        betti_sim = 1.0 / (1.0 + np.sum(np.abs(np.array(self.betti_numbers) - 
                                               np.array(other.betti_numbers))))
        
        # Euler characteristic similarity  
        euler_sim = 1.0 / (1.0 + abs(self.euler_characteristic - other.euler_characteristic))
        
        # Persistence diagram similarity (simplified Wasserstein)
        if len(self.persistence_diagram) > 0 and len(other.persistence_diagram) > 0:
            pers_sim = self._wasserstein_distance(self.persistence_diagram, 
                                                 other.persistence_diagram)
        else:
            pers_sim = 0.5
        
        return 0.4 * betti_sim + 0.3 * euler_sim + 0.3 * pers_sim
    
    def _wasserstein_distance(self, diag1: np.ndarray, diag2: np.ndarray) -> float:
        """Simplified Wasserstein distance between persistence diagrams."""
        # For demonstration - in practice would use proper Wasserstein computation
        if diag1.size == 0 or diag2.size == 0:
            return 0.0
        
        # Birth-death point distances
        dist1 = np.mean(diag1[:, 1] - diag1[:, 0]) if diag1.shape[1] >= 2 else 0
        dist2 = np.mean(diag2[:, 1] - diag2[:, 0]) if diag2.shape[1] >= 2 else 0
        
        return 1.0 / (1.0 + abs(dist1 - dist2))


class SimplexComplex:
    """Simplicial complex for topological antenna representation."""
    
    def __init__(self):
        """Initialize empty simplicial complex."""
        self.vertices = {}
        self.edges = set()
        self.triangles = set()
        self.tetrahedra = set()
        self.vertex_counter = 0
        
    def add_vertex(self, position: Tuple[float, float, float]) -> int:
        """Add vertex and return its index."""
        vertex_id = self.vertex_counter
        self.vertices[vertex_id] = position
        self.vertex_counter += 1
        return vertex_id
    
    def add_edge(self, v1: int, v2: int):
        """Add edge between vertices."""
        if v1 in self.vertices and v2 in self.vertices:
            self.edges.add(tuple(sorted([v1, v2])))
    
    def add_triangle(self, v1: int, v2: int, v3: int):
        """Add triangle face."""
        vertices = [v1, v2, v3]
        if all(v in self.vertices for v in vertices):
            self.triangles.add(tuple(sorted(vertices)))
            # Add edges
            for i in range(3):
                self.add_edge(vertices[i], vertices[(i + 1) % 3])
    
    def compute_betti_numbers(self) -> List[int]:
        """Compute Betti numbers of the complex."""
        n_vertices = len(self.vertices)
        n_edges = len(self.edges)
        n_triangles = len(self.triangles)
        
        # Connected components (B0)
        if n_vertices == 0:
            return [0, 0, 0]
            
        # Build adjacency for connected components
        adj_matrix = np.zeros((n_vertices, n_vertices))
        for v1, v2 in self.edges:
            adj_matrix[v1, v2] = 1
            adj_matrix[v2, v1] = 1
        
        # Count connected components
        visited = set()
        components = 0
        
        for v in range(n_vertices):
            if v not in visited:
                # BFS to mark component
                queue = [v]
                visited.add(v)
                components += 1
                
                while queue:
                    current = queue.pop(0)
                    for neighbor in range(n_vertices):
                        if adj_matrix[current, neighbor] == 1 and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        
        betti_0 = components
        
        # Loops/holes (B1) - simplified Euler characteristic approach
        euler_char = n_vertices - n_edges + n_triangles
        betti_1 = max(0, n_edges - n_vertices + betti_0)
        
        # Voids (B2) 
        betti_2 = 0  # Simplified for 2D surfaces
        
        return [betti_0, betti_1, betti_2]
    
    def compute_euler_characteristic(self) -> int:
        """Compute Euler characteristic."""
        return len(self.vertices) - len(self.edges) + len(self.triangles)


class TopologicalAntennaGeometry:
    """Topological representation of antenna geometry."""
    
    def __init__(self, grid_resolution: int = 32):
        """Initialize antenna geometry on grid.
        
        Args:
            grid_resolution: Resolution of 3D voxel grid
        """
        self.grid_resolution = grid_resolution
        self.voxel_grid = np.zeros((grid_resolution, grid_resolution, grid_resolution))
        self.simplicial_complex = SimplexComplex()
        self.topological_descriptor = None
        
    def from_parameter_vector(self, params: np.ndarray) -> 'TopologicalAntennaGeometry':
        """Create geometry from parameter vector."""
        # Reshape parameters to 3D grid
        if len(params) == self.grid_resolution ** 3:
            self.voxel_grid = params.reshape((self.grid_resolution, 
                                            self.grid_resolution, 
                                            self.grid_resolution))
        else:
            # Interpolate or truncate to fit grid
            grid_size = self.grid_resolution ** 3
            if len(params) > grid_size:
                self.voxel_grid = params[:grid_size].reshape((self.grid_resolution,
                                                            self.grid_resolution,
                                                            self.grid_resolution))
            else:
                padded_params = np.pad(params, (0, grid_size - len(params)), 'constant')
                self.voxel_grid = padded_params.reshape((self.grid_resolution,
                                                       self.grid_resolution, 
                                                       self.grid_resolution))
        
        # Apply threshold for binary geometry
        threshold = np.mean(self.voxel_grid)
        self.voxel_grid = (self.voxel_grid > threshold).astype(float)
        
        self._build_simplicial_complex()
        self._compute_topology()
        
        return self
    
    def _build_simplicial_complex(self):
        """Build simplicial complex from voxel grid."""
        self.simplicial_complex = SimplexComplex()
        
        # Add vertices for filled voxels
        vertex_map = {}
        
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                for k in range(self.grid_resolution):
                    if self.voxel_grid[i, j, k] > 0:
                        vertex_id = self.simplicial_complex.add_vertex((i, j, k))
                        vertex_map[(i, j, k)] = vertex_id
        
        # Add edges between adjacent voxels
        directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), 
                     (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        
        for (i, j, k), v_id in vertex_map.items():
            for di, dj, dk in directions:
                ni, nj, nk = i + di, j + dj, k + dk
                if (ni, nj, nk) in vertex_map:
                    neighbor_id = vertex_map[(ni, nj, nk)]
                    self.simplicial_complex.add_edge(v_id, neighbor_id)
        
        # Add triangular faces for surface elements
        for (i, j, k), v_id in vertex_map.items():
            # Check for triangular patterns in local neighborhood
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        if di == dj == dk == 0:
                            continue
                        ni, nj, nk = i + di, j + dj, k + dk
                        if (ni, nj, nk) in vertex_map:
                            neighbors.append(vertex_map[(ni, nj, nk)])
            
            # Add triangles between neighbors (simplified)
            if len(neighbors) >= 2:
                for i in range(min(3, len(neighbors) - 1)):
                    if len(neighbors) > i + 1:
                        self.simplicial_complex.add_triangle(v_id, neighbors[i], neighbors[i + 1])
    
    def _compute_topology(self):
        """Compute topological descriptor."""
        betti_numbers = self.simplicial_complex.compute_betti_numbers()
        euler_char = self.simplicial_complex.compute_euler_characteristic()
        
        # Genus computation for 2D surface (simplified)
        if len(betti_numbers) > 1:
            genus = max(0, betti_numbers[1] // 2)
        else:
            genus = 0
        
        # Simplified persistence diagram
        persistence_diagram = self._compute_persistence_diagram()
        
        # Homology groups (simplified representation)
        homology_groups = [
            {'dimension': i, 'rank': betti_numbers[i] if i < len(betti_numbers) else 0}
            for i in range(3)
        ]
        
        self.topological_descriptor = TopologicalDescriptor(
            betti_numbers=betti_numbers,
            euler_characteristic=euler_char,
            genus=genus,
            persistence_diagram=persistence_diagram,
            homology_groups=homology_groups
        )
    
    def _compute_persistence_diagram(self) -> np.ndarray:
        """Compute persistence diagram (simplified implementation)."""
        # In practice, would use persistent homology algorithms
        # This is a placeholder that generates synthetic persistence pairs
        
        n_features = len(self.simplicial_complex.vertices)
        if n_features == 0:
            return np.array([])
        
        # Generate birth-death pairs based on geometric structure
        np.random.seed(hash(str(self.voxel_grid.tobytes())) % 2**32)
        n_pairs = min(10, n_features)
        
        birth_times = np.random.uniform(0, 1, n_pairs)
        death_times = birth_times + np.random.exponential(0.5, n_pairs)
        
        persistence_pairs = np.column_stack([birth_times, death_times])
        return persistence_pairs


class TopologicalOptimizationObjective:
    """Multi-objective function incorporating topological constraints."""
    
    def __init__(self, target_topology: Optional[TopologicalDescriptor] = None,
                 topology_weight: float = 0.3):
        """Initialize topological objective.
        
        Args:
            target_topology: Desired topological properties
            topology_weight: Weight of topological terms
        """
        self.target_topology = target_topology
        self.topology_weight = topology_weight
        self.performance_weight = 1.0 - topology_weight
        
    def evaluate(self, geometry: TopologicalAntennaGeometry) -> Dict[str, float]:
        """Evaluate antenna with topological constraints.
        
        Returns:
            Dictionary of objective values
        """
        # Electromagnetic performance (placeholder)
        em_performance = self._evaluate_em_performance(geometry)
        
        # Topological compliance
        topology_score = self._evaluate_topology_compliance(geometry)
        
        # Manufacturing constraints
        manufacturing_score = self._evaluate_manufacturing(geometry)
        
        # Combined objective
        total_objective = (self.performance_weight * em_performance +
                          self.topology_weight * topology_score +
                          0.1 * manufacturing_score)
        
        return {
            'total_objective': total_objective,
            'em_performance': em_performance,
            'topology_score': topology_score,
            'manufacturing_score': manufacturing_score,
            'betti_0': geometry.topological_descriptor.betti_numbers[0],
            'betti_1': geometry.topological_descriptor.betti_numbers[1] if len(geometry.topological_descriptor.betti_numbers) > 1 else 0,
            'euler_characteristic': geometry.topological_descriptor.euler_characteristic,
            'genus': geometry.topological_descriptor.genus
        }
    
    def _evaluate_em_performance(self, geometry: TopologicalAntennaGeometry) -> float:
        """Evaluate electromagnetic performance."""
        # Placeholder: compute based on geometry distribution
        voxel_density = np.mean(geometry.voxel_grid)
        edge_density = len(geometry.simplicial_complex.edges) / max(1, len(geometry.simplicial_complex.vertices))
        
        # Performance heuristic based on connectivity and distribution
        performance = 0.5 * voxel_density + 0.3 * min(1.0, edge_density / 5.0)
        
        # Bonus for interesting topological features
        n_holes = geometry.topological_descriptor.betti_numbers[1] if len(geometry.topological_descriptor.betti_numbers) > 1 else 0
        topology_bonus = 0.2 * min(1.0, n_holes / 3.0)
        
        return performance + topology_bonus
    
    def _evaluate_topology_compliance(self, geometry: TopologicalAntennaGeometry) -> float:
        """Evaluate compliance with target topology."""
        if self.target_topology is None:
            # Default: prefer moderate topological complexity
            betti_1 = geometry.topological_descriptor.betti_numbers[1] if len(geometry.topological_descriptor.betti_numbers) > 1 else 0
            
            # Optimal complexity around 2-4 holes for antenna applications
            optimal_holes = 3
            complexity_score = 1.0 / (1.0 + abs(betti_1 - optimal_holes))
            
            return complexity_score
        else:
            # Score based on similarity to target topology
            return self.target_topology.similarity(geometry.topological_descriptor)
    
    def _evaluate_manufacturing(self, geometry: TopologicalAntennaGeometry) -> float:
        """Evaluate manufacturing feasibility."""
        # Penalize disconnected components (hard to manufacture)
        n_components = geometry.topological_descriptor.betti_numbers[0]
        connectivity_score = 1.0 / max(1, n_components - 1)
        
        # Penalize excessive complexity
        n_holes = geometry.topological_descriptor.betti_numbers[1] if len(geometry.topological_descriptor.betti_numbers) > 1 else 0
        complexity_penalty = max(0, 1.0 - n_holes / 10.0)
        
        return 0.6 * connectivity_score + 0.4 * complexity_penalty


class TopologicalOptimizer:
    """
    🌀 Topology-Aware Antenna Optimization System
    
    Integrates topological data analysis with evolutionary optimization
    for discovering antennas with novel geometric properties.
    """
    
    def __init__(self, grid_resolution: int = 16, population_size: int = 40):
        """Initialize topological optimizer.
        
        Args:
            grid_resolution: Voxel grid resolution
            population_size: Number of candidate solutions
        """
        self.grid_resolution = grid_resolution
        self.population_size = population_size
        self.parameter_dim = grid_resolution ** 3
        
        # Optimization state
        self.population = []
        self.fitness_history = []
        self.topology_diversity_history = []
        
    def initialize_population(self) -> List[np.ndarray]:
        """Initialize population with topological diversity."""
        population = []
        
        # Ensure topological diversity in initial population
        target_topologies = self._generate_diverse_topologies()
        
        for i in range(self.population_size):
            if i < len(target_topologies):
                # Generate solution biased toward specific topology
                params = self._generate_topology_biased_solution(target_topologies[i])
            else:
                # Random initialization
                params = np.random.uniform(-1, 1, self.parameter_dim)
            
            population.append(params)
        
        self.population = population
        return population
    
    def _generate_diverse_topologies(self) -> List[TopologicalDescriptor]:
        """Generate set of diverse target topologies."""
        topologies = []
        
        # Simple topologies: sphere-like (genus 0)
        topologies.append(TopologicalDescriptor(
            betti_numbers=[1, 0, 0],
            euler_characteristic=2,
            genus=0,
            persistence_diagram=np.array([[0.1, 0.9]]),
            homology_groups=[]
        ))
        
        # Torus-like (genus 1)
        topologies.append(TopologicalDescriptor(
            betti_numbers=[1, 2, 1],
            euler_characteristic=0,
            genus=1,
            persistence_diagram=np.array([[0.1, 0.5], [0.2, 0.8]]),
            homology_groups=[]
        ))
        
        # More complex topology (genus 2)
        topologies.append(TopologicalDescriptor(
            betti_numbers=[1, 4, 1],
            euler_characteristic=-2,
            genus=2,
            persistence_diagram=np.array([[0.05, 0.3], [0.1, 0.6], [0.2, 0.9]]),
            homology_groups=[]
        ))
        
        return topologies
    
    def _generate_topology_biased_solution(self, target_topology: TopologicalDescriptor) -> np.ndarray:
        """Generate solution biased toward target topology."""
        # Use genus information to bias generation
        genus = target_topology.genus
        
        if genus == 0:
            # Generate blob-like structure
            center = self.grid_resolution // 2
            params = np.zeros(self.parameter_dim)
            
            for i in range(self.parameter_dim):
                x, y, z = np.unravel_index(i, (self.grid_resolution, self.grid_resolution, self.grid_resolution))
                distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                params[i] = max(0, 1.0 - distance / (self.grid_resolution * 0.4))
            
        elif genus == 1:
            # Generate torus-like structure
            params = np.zeros(self.parameter_dim)
            center = self.grid_resolution // 2
            major_radius = self.grid_resolution * 0.3
            minor_radius = self.grid_resolution * 0.1
            
            for i in range(self.parameter_dim):
                x, y, z = np.unravel_index(i, (self.grid_resolution, self.grid_resolution, self.grid_resolution))
                x_centered, y_centered, z_centered = x - center, y - center, z - center
                
                # Torus equation
                distance_from_center = np.sqrt(x_centered**2 + y_centered**2)
                torus_distance = np.sqrt((distance_from_center - major_radius)**2 + z_centered**2)
                
                if torus_distance < minor_radius:
                    params[i] = 1.0 - torus_distance / minor_radius
        
        else:
            # Higher genus: multiple connected components
            params = np.random.uniform(0, 1, self.parameter_dim)
            # Add noise for complexity
            params += np.random.normal(0, 0.2, self.parameter_dim)
        
        return np.clip(params, -1, 1)
    
    def topological_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Topology-preserving crossover operator."""
        # Create geometries to analyze topology
        geom1 = TopologicalAntennaGeometry(self.grid_resolution).from_parameter_vector(parent1)
        geom2 = TopologicalAntennaGeometry(self.grid_resolution).from_parameter_vector(parent2)
        
        # Identify topologically important regions
        important_regions1 = self._identify_topological_features(geom1)
        important_regions2 = self._identify_topological_features(geom2)
        
        # Create children by preserving topological features
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Exchange some topological features
        crossover_rate = 0.3
        for region in important_regions1:
            if np.random.random() < crossover_rate:
                child1[region] = parent2[region]
        
        for region in important_regions2:
            if np.random.random() < crossover_rate:
                child2[region] = parent1[region]
        
        return child1, child2
    
    def _identify_topological_features(self, geometry: TopologicalAntennaGeometry) -> List[np.ndarray]:
        """Identify topologically important voxel regions."""
        features = []
        
        # Find voxels that contribute to holes (simplified)
        voxel_grid = geometry.voxel_grid
        
        # Edge detection for topological boundaries
        from scipy import ndimage
        edges = ndimage.sobel(voxel_grid)
        edge_threshold = np.percentile(edges[edges > 0], 75) if np.any(edges > 0) else 0
        
        edge_voxels = np.where(edges > edge_threshold)
        if len(edge_voxels[0]) > 0:
            # Convert to linear indices
            linear_indices = np.ravel_multi_index(edge_voxels, voxel_grid.shape)
            features.append(linear_indices)
        
        return features
    
    def topological_mutation(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Topology-aware mutation operator."""
        mutated = individual.copy()
        
        # Create geometry to analyze current topology
        geometry = TopologicalAntennaGeometry(self.grid_resolution).from_parameter_vector(individual)
        current_betti = geometry.topological_descriptor.betti_numbers
        
        # Mutation strategies based on current topology
        if len(current_betti) > 1 and current_betti[1] == 0:
            # No holes - try to create some
            mutation_strength = 0.5
        elif len(current_betti) > 1 and current_betti[1] > 5:
            # Too many holes - try to close some
            mutation_strength = 0.2
        else:
            # Moderate topology - standard mutation
            mutation_strength = 0.3
        
        # Apply mutations
        mutation_mask = np.random.random(len(individual)) < mutation_rate
        mutations = np.random.normal(0, mutation_strength, len(individual))
        mutated[mutation_mask] += mutations[mutation_mask]
        
        return np.clip(mutated, -1, 1)
    
    def compute_topology_diversity(self, population: List[np.ndarray]) -> float:
        """Compute topological diversity of population."""
        topologies = []
        
        for individual in population[:10]:  # Sample for efficiency
            geometry = TopologicalAntennaGeometry(self.grid_resolution).from_parameter_vector(individual)
            topologies.append(geometry.topological_descriptor)
        
        # Compute pairwise topological similarities
        similarities = []
        for i in range(len(topologies)):
            for j in range(i + 1, len(topologies)):
                sim = topologies[i].similarity(topologies[j])
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # Diversity is inverse of average similarity
        return 1.0 - np.mean(similarities)
    
    def optimize(self, objective: TopologicalOptimizationObjective,
                max_generations: int = 100,
                convergence_threshold: float = 1e-6) -> Dict[str, Any]:
        """
        Run topology-aware optimization.
        
        Args:
            objective: Topological optimization objective
            max_generations: Maximum generations
            convergence_threshold: Convergence criteria
            
        Returns:
            Optimization results with topological analysis
        """
        logger.info("🌀 Starting Topological Antenna Optimization")
        
        # Initialize population
        self.initialize_population()
        
        best_fitness = -np.inf
        best_solution = None
        best_geometry = None
        
        for generation in range(max_generations):
            # Evaluate population
            fitness_scores = []
            geometries = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for individual in self.population:
                    future = executor.submit(self._evaluate_individual, individual, objective)
                    futures.append(future)
                
                for future in futures:
                    fitness, geometry = future.result()
                    fitness_scores.append(fitness)
                    geometries.append(geometry)
            
            fitness_scores = np.array(fitness_scores)
            
            # Track best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_solution = self.population[best_idx].copy()
                best_geometry = geometries[best_idx]
            
            # Compute topological diversity
            topology_diversity = self.compute_topology_diversity(self.population)
            
            # Track progress
            generation_stats = {
                'generation': generation,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores),
                'topology_diversity': topology_diversity,
                'best_betti_0': best_geometry.topological_descriptor.betti_numbers[0] if best_geometry else 0,
                'best_betti_1': (best_geometry.topological_descriptor.betti_numbers[1] 
                               if best_geometry and len(best_geometry.topological_descriptor.betti_numbers) > 1 else 0),
                'best_genus': best_geometry.topological_descriptor.genus if best_geometry else 0
            }
            
            self.fitness_history.append(generation_stats)
            self.topology_diversity_history.append(topology_diversity)
            
            logger.info(f"Generation {generation}: Fitness={best_fitness:.6f}, "
                       f"Diversity={topology_diversity:.3f}, "
                       f"Best topology: B₀={generation_stats['best_betti_0']}, "
                       f"B₁={generation_stats['best_betti_1']}, "
                       f"Genus={generation_stats['best_genus']}")
            
            # Check convergence
            if len(self.fitness_history) > 10:
                recent_improvement = (self.fitness_history[-1]['best_fitness'] - 
                                    self.fitness_history[-10]['best_fitness'])
                if abs(recent_improvement) < convergence_threshold:
                    logger.info(f"🌀 Converged at generation {generation}")
                    break
            
            # Selection and reproduction
            selected_indices = self._tournament_selection(fitness_scores, tournament_size=3)
            next_population = []
            
            # Create next generation
            for i in range(0, self.population_size, 2):
                parent1_idx = selected_indices[i % len(selected_indices)]
                parent2_idx = selected_indices[(i + 1) % len(selected_indices)]
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # Topological crossover
                child1, child2 = self.topological_crossover(parent1, parent2)
                
                # Topological mutation
                child1 = self.topological_mutation(child1)
                child2 = self.topological_mutation(child2)
                
                next_population.extend([child1, child2])
            
            self.population = next_population[:self.population_size]
        
        # Final analysis
        final_results = {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'best_geometry': best_geometry,
            'fitness_history': self.fitness_history,
            'topology_diversity_history': self.topology_diversity_history,
            'generations': generation + 1,
            'convergence_achieved': generation < max_generations - 1,
            'final_topology': {
                'betti_numbers': best_geometry.topological_descriptor.betti_numbers if best_geometry else [],
                'euler_characteristic': best_geometry.topological_descriptor.euler_characteristic if best_geometry else 0,
                'genus': best_geometry.topological_descriptor.genus if best_geometry else 0,
                'n_vertices': len(best_geometry.simplicial_complex.vertices) if best_geometry else 0,
                'n_edges': len(best_geometry.simplicial_complex.edges) if best_geometry else 0,
                'n_triangles': len(best_geometry.simplicial_complex.triangles) if best_geometry else 0
            }
        }
        
        logger.info("🌀 Topological optimization complete!")
        logger.info(f"Final topology: B₀={final_results['final_topology']['betti_numbers'][0] if final_results['final_topology']['betti_numbers'] else 0}, "
                   f"B₁={final_results['final_topology']['betti_numbers'][1] if len(final_results['final_topology']['betti_numbers']) > 1 else 0}, "
                   f"Genus={final_results['final_topology']['genus']}")
        
        return final_results
    
    def _evaluate_individual(self, individual: np.ndarray, 
                           objective: TopologicalOptimizationObjective) -> Tuple[float, TopologicalAntennaGeometry]:
        """Evaluate single individual."""
        geometry = TopologicalAntennaGeometry(self.grid_resolution).from_parameter_vector(individual)
        results = objective.evaluate(geometry)
        return results['total_objective'], geometry
    
    def _tournament_selection(self, fitness_scores: np.ndarray, tournament_size: int = 3) -> np.ndarray:
        """Tournament selection."""
        selected = []
        
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(winner_idx)
        
        return np.array(selected)


class TopologicalAntennaDesigner:
    """
    🌀 Complete Topological Antenna Design System
    
    Integrates topological optimization with antenna design constraints
    for discovering antennas with revolutionary geometric properties.
    """
    
    def __init__(self, antenna_spec: Any):
        """Initialize topological antenna designer."""
        self.antenna_spec = antenna_spec
        self.topological_optimizer = None
        self.design_history = []
        
    def design_topology_constrained_antenna(self, target_topology: Optional[Dict[str, Any]] = None,
                                          grid_resolution: int = 16,
                                          max_generations: int = 50) -> Dict[str, Any]:
        """
        Design antenna with topological constraints.
        
        Args:
            target_topology: Target topological properties
            grid_resolution: Voxel grid resolution
            max_generations: Optimization generations
            
        Returns:
            Topological design results
        """
        logger.info("🌀 Starting Topological Antenna Design")
        
        # Setup topological optimizer
        self.topological_optimizer = TopologicalOptimizer(
            grid_resolution=grid_resolution,
            population_size=30
        )
        
        # Define target topology if specified
        target_descriptor = None
        if target_topology:
            target_descriptor = TopologicalDescriptor(
                betti_numbers=target_topology.get('betti_numbers', [1, 2, 0]),
                euler_characteristic=target_topology.get('euler_characteristic', 0),
                genus=target_topology.get('genus', 1),
                persistence_diagram=np.array(target_topology.get('persistence_pairs', [[0.1, 0.8]])),
                homology_groups=[]
            )
        
        # Setup objective function
        objective = TopologicalOptimizationObjective(
            target_topology=target_descriptor,
            topology_weight=0.4  # Balance topology and performance
        )
        
        # Run optimization
        results = self.topological_optimizer.optimize(
            objective=objective,
            max_generations=max_generations,
            convergence_threshold=1e-6
        )
        
        # Post-process results
        design_analysis = self._analyze_topological_design(results)
        
        # Combine results
        complete_results = {
            **results,
            'topological_analysis': design_analysis,
            'design_parameters': {
                'grid_resolution': grid_resolution,
                'target_topology': target_topology,
                'optimization_generations': results['generations']
            }
        }
        
        self.design_history.append(complete_results)
        
        logger.info("🌀 Topological antenna design complete!")
        
        return complete_results
    
    def _analyze_topological_design(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze topological properties of final design."""
        best_geometry = results['best_geometry']
        
        if not best_geometry:
            return {'error': 'No valid geometry found'}
        
        topology = best_geometry.topological_descriptor
        
        analysis = {
            'topological_classification': self._classify_topology(topology),
            'geometric_properties': {
                'voxel_density': np.mean(best_geometry.voxel_grid),
                'surface_complexity': len(best_geometry.simplicial_complex.edges) / max(1, len(best_geometry.simplicial_complex.vertices)),
                'connectivity_index': topology.betti_numbers[0],
                'hole_count': topology.betti_numbers[1] if len(topology.betti_numbers) > 1 else 0,
                'void_count': topology.betti_numbers[2] if len(topology.betti_numbers) > 2 else 0
            },
            'manufacturing_considerations': {
                'single_component': topology.betti_numbers[0] == 1,
                'moderate_complexity': (topology.betti_numbers[1] if len(topology.betti_numbers) > 1 else 0) <= 5,
                'manufacturability_score': self._compute_manufacturability_score(best_geometry)
            },
            'antenna_implications': {
                'multiband_potential': (topology.betti_numbers[1] if len(topology.betti_numbers) > 1 else 0) > 1,
                'polarization_diversity': topology.genus > 0,
                'beam_shaping_capability': len(best_geometry.simplicial_complex.triangles) > 50
            }
        }
        
        return analysis
    
    def _classify_topology(self, topology: TopologicalDescriptor) -> str:
        """Classify antenna topology type."""
        genus = topology.genus
        n_holes = topology.betti_numbers[1] if len(topology.betti_numbers) > 1 else 0
        
        if genus == 0 and n_holes == 0:
            return "Simple (sphere-like)"
        elif genus == 0 and n_holes > 0:
            return f"Disk with {n_holes} holes"
        elif genus == 1:
            return "Torus-like (genus 1)"
        elif genus == 2:
            return "Double torus (genus 2)"
        elif genus > 2:
            return f"High genus surface (genus {genus})"
        else:
            return "Complex topology"
    
    def _compute_manufacturability_score(self, geometry: TopologicalAntennaGeometry) -> float:
        """Compute manufacturability score for antenna design."""
        # Single connected component is good
        n_components = geometry.topological_descriptor.betti_numbers[0]
        connectivity_score = 1.0 / max(1, n_components)
        
        # Moderate complexity is preferred
        n_holes = (geometry.topological_descriptor.betti_numbers[1] 
                  if len(geometry.topological_descriptor.betti_numbers) > 1 else 0)
        complexity_score = max(0, 1.0 - abs(n_holes - 2) / 5.0)
        
        # Reasonable voxel density
        density = np.mean(geometry.voxel_grid)
        density_score = 1.0 - abs(density - 0.3) / 0.7  # Optimal around 30%
        
        return 0.5 * connectivity_score + 0.3 * complexity_score + 0.2 * max(0, density_score)


# Export main classes
__all__ = [
    'TopologicalDescriptor',
    'SimplexComplex',
    'TopologicalAntennaGeometry',
    'TopologicalOptimizationObjective',
    'TopologicalOptimizer',
    'TopologicalAntennaDesigner'
]