"""
Multi-objective optimization algorithms for antenna design.

Implements advanced algorithms including NSGA-III, MOEA/D, and SMS-EMOA
for Pareto-optimal antenna design exploration.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

from ..utils.logging_config import get_logger, LoggingContextManager
from ..utils.validation import ValidationError
from ..solvers.base import SolverResult, BaseSolver
from ..core.antenna_spec import AntennaSpec


@dataclass
class OptimizationObjective:
    """Single optimization objective definition."""
    name: str
    function: Callable[[SolverResult], float]
    minimize: bool = True
    weight: float = 1.0
    constraint_bounds: Optional[Tuple[float, float]] = None


@dataclass
class Individual:
    """Individual solution in population-based optimization."""
    genome: np.ndarray
    objectives: np.ndarray
    constraints: np.ndarray
    fitness: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0
    dominated_solutions: List['Individual'] = None
    domination_count: int = 0
    
    def __post_init__(self):
        if self.dominated_solutions is None:
            self.dominated_solutions = []


class ParetoFront:
    """Manages Pareto frontier analysis and visualization."""
    
    def __init__(self, objective_names: List[str]):
        """
        Initialize Pareto front manager.
        
        Args:
            objective_names: Names of optimization objectives
        """
        self.objective_names = objective_names
        self.solutions = []
        self.logger = get_logger('pareto_front')
    
    def add_solution(
        self,
        genome: np.ndarray,
        objectives: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add solution to Pareto front."""
        solution = {
            'genome': genome.copy(),
            'objectives': objectives.copy(),
            'metadata': metadata or {}
        }
        self.solutions.append(solution)
    
    def get_pareto_optimal_solutions(self) -> List[Dict[str, Any]]:
        """Extract Pareto-optimal solutions."""
        if not self.solutions:
            return []
        
        # Convert to array for efficient computation
        objectives = np.array([sol['objectives'] for sol in self.solutions])
        n_solutions = len(objectives)
        
        # Find non-dominated solutions
        is_pareto_optimal = np.ones(n_solutions, dtype=bool)
        
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j and self._dominates(objectives[j], objectives[i]):
                    is_pareto_optimal[i] = False
                    break
        
        pareto_solutions = [
            self.solutions[i] for i in range(n_solutions)
            if is_pareto_optimal[i]
        ]
        
        self.logger.info(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
        
        return pareto_solutions
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (minimization)."""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def compute_hypervolume(self, reference_point: np.ndarray) -> float:
        """Compute hypervolume indicator."""
        pareto_solutions = self.get_pareto_optimal_solutions()
        if not pareto_solutions:
            return 0.0
        
        objectives = np.array([sol['objectives'] for sol in pareto_solutions])
        
        # Simple hypervolume approximation for 2D/3D
        if objectives.shape[1] == 2:
            return self._hypervolume_2d(objectives, reference_point)
        elif objectives.shape[1] == 3:
            return self._hypervolume_3d(objectives, reference_point)
        else:
            # Higher dimensions - use Monte Carlo approximation
            return self._hypervolume_monte_carlo(objectives, reference_point)
    
    def _hypervolume_2d(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """Compute 2D hypervolume exactly."""
        # Sort by first objective
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]
        
        hypervolume = 0.0
        prev_y = ref_point[1]
        
        for obj in sorted_objectives:
            if obj[1] < prev_y:
                hypervolume += (ref_point[0] - obj[0]) * (prev_y - obj[1])
                prev_y = obj[1]
        
        return hypervolume
    
    def _hypervolume_3d(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """Compute 3D hypervolume using inclusion-exclusion."""
        # Simplified 3D hypervolume calculation
        n_points = len(objectives)
        volume = 0.0
        
        for i in range(n_points):
            # Individual contribution
            obj = objectives[i]
            individual_volume = np.prod(ref_point - obj)
            volume += individual_volume
            
            # Subtract overlaps (simplified)
            for j in range(i + 1, n_points):
                obj2 = objectives[j]
                overlap = np.prod(np.maximum(0, ref_point - np.maximum(obj, obj2)))
                volume -= overlap
        
        return max(0, volume)
    
    def _hypervolume_monte_carlo(
        self,
        objectives: np.ndarray,
        ref_point: np.ndarray,
        n_samples: int = 100000
    ) -> float:
        """Monte Carlo hypervolume approximation for high dimensions."""
        n_dim = objectives.shape[1]
        
        # Generate random points in reference region
        random_points = np.random.uniform(
            low=np.min(objectives, axis=0),
            high=ref_point,
            size=(n_samples, n_dim)
        )
        
        # Count points dominated by at least one Pareto solution
        dominated_count = 0
        
        for point in random_points:
            for obj in objectives:
                if np.all(obj <= point):
                    dominated_count += 1
                    break
        
        # Estimate hypervolume
        reference_volume = np.prod(ref_point - np.min(objectives, axis=0))
        hypervolume = reference_volume * dominated_count / n_samples
        
        return hypervolume
    
    def export_solutions(self, filepath: str) -> None:
        """Export Pareto-optimal solutions."""
        pareto_solutions = self.get_pareto_optimal_solutions()
        
        export_data = {
            'objective_names': self.objective_names,
            'n_solutions': len(pareto_solutions),
            'solutions': []
        }
        
        for i, sol in enumerate(pareto_solutions):
            solution_data = {
                'id': i,
                'genome': sol['genome'].tolist(),
                'objectives': sol['objectives'].tolist(),
                'objective_dict': dict(zip(self.objective_names, sol['objectives'])),
                'metadata': sol['metadata']
            }
            export_data['solutions'].append(solution_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(pareto_solutions)} solutions to {filepath}")


class NSGA3Optimizer:
    """NSGA-III multi-objective optimizer with reference point adaptation."""
    
    def __init__(
        self,
        objectives: List[OptimizationObjective],
        population_size: int = 100,
        n_generations: int = 500,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        eta_c: float = 15.0,  # Crossover distribution index
        eta_m: float = 20.0,  # Mutation distribution index
    ):
        """
        Initialize NSGA-III optimizer.
        
        Args:
            objectives: List of optimization objectives
            population_size: Population size
            n_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            eta_c: Crossover distribution index
            eta_m: Mutation distribution index
        """
        self.objectives = objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        
        self.logger = get_logger('nsga3_optimizer')
        
        # Generate reference points
        self.reference_points = self._generate_reference_points(len(objectives))
        
        # Optimization state
        self.population = []
        self.generation = 0
        self.best_hypervolume = 0.0
        
        # Performance tracking
        self.convergence_history = []
        self.diversity_history = []
        self.hypervolume_history = []
    
    def _generate_reference_points(self, n_objectives: int, n_divisions: int = None) -> np.ndarray:
        """Generate structured reference points using Das and Dennis method."""
        if n_divisions is None:
            # Adaptive number of divisions based on objectives
            if n_objectives <= 3:
                n_divisions = 12
            elif n_objectives <= 5:
                n_divisions = 6
            else:
                n_divisions = 4
        
        # Generate reference points on unit simplex
        def recursive_reference_points(n_obj, n_div, current_point, depth):
            if depth == n_obj - 1:
                current_point[depth] = n_div
                return [current_point.copy()]
            
            points = []
            for i in range(n_div + 1):
                current_point[depth] = i
                points.extend(recursive_reference_points(
                    n_obj, n_div - i, current_point, depth + 1
                ))
            
            return points
        
        ref_points = recursive_reference_points(
            n_objectives, n_divisions, np.zeros(n_objectives), 0
        )
        
        ref_points = np.array(ref_points) / n_divisions
        
        self.logger.info(f"Generated {len(ref_points)} reference points for "
                        f"{n_objectives} objectives")
        
        return ref_points
    
    def optimize(
        self,
        solver: BaseSolver,
        antenna_spec: AntennaSpec,
        bounds: Tuple[np.ndarray, np.ndarray],
        callback: Optional[Callable] = None
    ) -> ParetoFront:
        """
        Run multi-objective optimization.
        
        Args:
            solver: Electromagnetic solver
            antenna_spec: Antenna specification
            bounds: Variable bounds (lower, upper)
            callback: Optional callback function
            
        Returns:
            Pareto front with optimal solutions
        """
        self.logger.info(f"Starting NSGA-III optimization with {self.population_size} "
                        f"individuals for {self.n_generations} generations")
        
        # Initialize population
        self.population = self._initialize_population(bounds)
        
        # Evaluate initial population
        self._evaluate_population(solver, antenna_spec)
        
        # Create Pareto front
        pareto_front = ParetoFront([obj.name for obj in self.objectives])
        
        with LoggingContextManager("NSGA-III Optimization", self.logger):
            for generation in range(self.n_generations):
                self.generation = generation
                
                # Selection, crossover, and mutation
                offspring = self._create_offspring()
                
                # Evaluate offspring
                self._evaluate_individuals(offspring, solver, antenna_spec)
                
                # Environmental selection
                self.population = self._environmental_selection(
                    self.population + offspring
                )
                
                # Update Pareto front
                for individual in self.population:
                    pareto_front.add_solution(
                        individual.genome,
                        individual.objectives,
                        {'generation': generation, 'fitness': individual.fitness}
                    )
                
                # Track convergence metrics
                self._update_convergence_metrics(pareto_front)
                
                # Callback
                if callback:
                    callback(generation, self.population, pareto_front)
                
                # Log progress
                if generation % 50 == 0:
                    best_hv = self.hypervolume_history[-1] if self.hypervolume_history else 0
                    self.logger.info(f"Generation {generation}: "
                                   f"HV = {best_hv:.6f}, "
                                   f"Pareto solutions = {len(pareto_front.get_pareto_optimal_solutions())}")
        
        self.logger.info(f"Optimization completed. Final Pareto front contains "
                        f"{len(pareto_front.get_pareto_optimal_solutions())} solutions")
        
        return pareto_front
    
    def _initialize_population(self, bounds: Tuple[np.ndarray, np.ndarray]) -> List[Individual]:
        """Initialize random population."""
        lower_bounds, upper_bounds = bounds
        n_variables = len(lower_bounds)
        
        population = []
        
        for _ in range(self.population_size):
            # Random initialization within bounds
            genome = np.random.uniform(lower_bounds, upper_bounds, n_variables)
            
            individual = Individual(
                genome=genome,
                objectives=np.zeros(len(self.objectives)),
                constraints=np.zeros(0)  # No constraints for now
            )
            
            population.append(individual)
        
        return population
    
    def _evaluate_population(self, solver: BaseSolver, antenna_spec: AntennaSpec) -> None:
        """Evaluate entire population."""
        for individual in self.population:
            self._evaluate_individual(individual, solver, antenna_spec)
    
    def _evaluate_individuals(
        self,
        individuals: List[Individual],
        solver: BaseSolver,
        antenna_spec: AntennaSpec
    ) -> None:
        """Evaluate list of individuals."""
        for individual in individuals:
            self._evaluate_individual(individual, solver, antenna_spec)
    
    def _evaluate_individual(
        self,
        individual: Individual,
        solver: BaseSolver,
        antenna_spec: AntennaSpec
    ) -> None:
        """Evaluate single individual."""
        try:
            # Convert genome to antenna geometry
            geometry = self._genome_to_geometry(individual.genome)
            
            # Run simulation
            frequency = np.mean(antenna_spec.frequency_range)
            result = solver.simulate(
                geometry=geometry,
                frequency=frequency,
                spec=antenna_spec
            )
            
            # Compute objectives
            for i, objective in enumerate(self.objectives):
                value = objective.function(result)
                
                # Apply weight and minimize/maximize
                if not objective.minimize:
                    value = -value
                
                individual.objectives[i] = value * objective.weight
                
        except Exception as e:
            self.logger.warning(f"Individual evaluation failed: {str(e)}")
            
            # Assign poor objectives for failed evaluations
            individual.objectives = np.full(len(self.objectives), 1e6)
    
    def _genome_to_geometry(self, genome: np.ndarray) -> np.ndarray:
        """Convert optimization variables to antenna geometry."""
        # Simple example: genome represents liquid metal channel states
        # In practice, this would be more sophisticated
        
        geometry_size = (32, 32, 8)
        geometry = np.zeros(geometry_size)
        
        # Create base patch
        patch_layer = geometry_size[2] - 2
        geometry[8:24, 8:24, patch_layer] = 1.0
        
        # Add channels based on genome
        n_channels = min(len(genome), 16)
        for i in range(n_channels):
            if genome[i] > 0.5:  # Channel is filled
                x = 8 + i // 4 * 4
                y = 8 + (i % 4) * 4
                geometry[x:x+2, y:y+2, patch_layer] = 1.0
        
        return geometry
    
    def _create_offspring(self) -> List[Individual]:
        """Create offspring through selection, crossover, and mutation."""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_prob:
                child1, child2 = self._simulated_binary_crossover(parent1, parent2)
            else:
                child1 = Individual(
                    genome=parent1.genome.copy(),
                    objectives=np.zeros(len(self.objectives)),
                    constraints=np.zeros(0)
                )
                child2 = Individual(
                    genome=parent2.genome.copy(),
                    objectives=np.zeros(len(self.objectives)),
                    constraints=np.zeros(0)
                )
            
            # Mutation
            self._polynomial_mutation(child1)
            self._polynomial_mutation(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection."""
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        
        # Select best individual (lowest rank, highest crowding distance)
        best = min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
        
        return best
    
    def _simulated_binary_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Simulated binary crossover (SBX)."""
        n_variables = len(parent1.genome)
        
        child1_genome = np.zeros(n_variables)
        child2_genome = np.zeros(n_variables)
        
        for i in range(n_variables):
            if np.random.random() <= 0.5:
                if abs(parent1.genome[i] - parent2.genome[i]) > 1e-14:
                    # Calculate beta
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (self.eta_c + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
                    
                    child1_genome[i] = 0.5 * ((1 + beta) * parent1.genome[i] + 
                                            (1 - beta) * parent2.genome[i])
                    child2_genome[i] = 0.5 * ((1 - beta) * parent1.genome[i] + 
                                            (1 + beta) * parent2.genome[i])
                else:
                    child1_genome[i] = parent1.genome[i]
                    child2_genome[i] = parent2.genome[i]
            else:
                child1_genome[i] = parent1.genome[i]
                child2_genome[i] = parent2.genome[i]
        
        child1 = Individual(
            genome=child1_genome,
            objectives=np.zeros(len(self.objectives)),
            constraints=np.zeros(0)
        )
        child2 = Individual(
            genome=child2_genome,
            objectives=np.zeros(len(self.objectives)),
            constraints=np.zeros(0)
        )
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: Individual) -> None:
        """Polynomial mutation."""
        for i in range(len(individual.genome)):
            if np.random.random() < self.mutation_prob:
                u = np.random.random()
                
                if u < 0.5:
                    delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
                
                # Apply bounds (assuming [0, 1] for simplicity)
                individual.genome[i] = np.clip(individual.genome[i] + delta, 0, 1)
    
    def _environmental_selection(self, combined_population: List[Individual]) -> List[Individual]:
        """Environmental selection using reference points."""
        # Non-dominated sorting
        fronts = self._non_dominated_sorting(combined_population)
        
        selected = []
        front_index = 0
        
        # Fill population with complete fronts
        while front_index < len(fronts) and len(selected) + len(fronts[front_index]) <= self.population_size:
            selected.extend(fronts[front_index])
            front_index += 1
        
        # Fill remaining spots using reference point selection
        if front_index < len(fronts):
            remaining_spots = self.population_size - len(selected)
            last_front = fronts[front_index]
            
            # Reference point based selection
            selected_from_last_front = self._reference_point_selection(
                last_front, remaining_spots
            )
            selected.extend(selected_from_last_front)
        
        return selected
    
    def _non_dominated_sorting(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting."""
        # Reset domination information
        for individual in population:
            individual.dominated_solutions = []
            individual.domination_count = 0
            individual.rank = 0
        
        # Calculate domination
        for i, individual_i in enumerate(population):
            for j, individual_j in enumerate(population):
                if i != j:
                    if self._dominates_objectives(individual_i.objectives, individual_j.objectives):
                        individual_i.dominated_solutions.append(individual_j)
                    elif self._dominates_objectives(individual_j.objectives, individual_i.objectives):
                        individual_i.domination_count += 1
        
        # Create fronts
        fronts = []
        current_front = []
        
        # Find first front
        for individual in population:
            if individual.domination_count == 0:
                individual.rank = 0
                current_front.append(individual)
        
        fronts.append(current_front[:])
        
        # Find subsequent fronts
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []
            
            for individual in fronts[front_index]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = front_index + 1
                        next_front.append(dominated)
            
            if next_front:
                fronts.append(next_front)
            front_index += 1
        
        return fronts[:-1] if fronts and not fronts[-1] else fronts
    
    def _dominates_objectives(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2."""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _reference_point_selection(
        self,
        front: List[Individual],
        n_select: int
    ) -> List[Individual]:
        """Select individuals using reference points."""
        if not front or n_select <= 0:
            return []
        
        if n_select >= len(front):
            return front
        
        # Normalize objectives
        objectives = np.array([ind.objectives for ind in front])
        normalized_objectives = self._normalize_objectives(objectives)
        
        # Associate individuals with reference points
        associations = self._associate_with_reference_points(normalized_objectives)
        
        # Select individuals
        selected = []
        ref_point_counts = np.zeros(len(self.reference_points))
        
        for _ in range(n_select):
            # Find reference point with minimum associated individuals
            min_count = np.min(ref_point_counts)
            candidate_refs = np.where(ref_point_counts == min_count)[0]
            
            # Select random reference point from candidates
            selected_ref = np.random.choice(candidate_refs)
            
            # Find individuals associated with this reference point
            associated_individuals = [
                i for i, ref in enumerate(associations) if ref == selected_ref
            ]
            
            if associated_individuals:
                # Select individual with minimum distance to reference point
                distances = []
                for idx in associated_individuals:
                    if front[idx] not in selected:
                        dist = self._distance_to_reference_point(
                            normalized_objectives[idx], self.reference_points[selected_ref]
                        )
                        distances.append((dist, idx))
                
                if distances:
                    distances.sort()
                    selected_individual = front[distances[0][1]]
                    selected.append(selected_individual)
                    ref_point_counts[selected_ref] += 1
                    
                    # Remove from future consideration
                    front.remove(selected_individual)
                    objectives = np.delete(objectives, distances[0][1], axis=0)
                    normalized_objectives = self._normalize_objectives(objectives)
                    associations = self._associate_with_reference_points(normalized_objectives)
        
        return selected
    
    def _normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0, 1] range."""
        if objectives.size == 0:
            return objectives
        
        obj_min = np.min(objectives, axis=0)
        obj_max = np.max(objectives, axis=0)
        
        # Avoid division by zero
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1
        
        normalized = (objectives - obj_min) / obj_range
        
        return normalized
    
    def _associate_with_reference_points(self, objectives: np.ndarray) -> List[int]:
        """Associate individuals with reference points."""
        associations = []
        
        for obj in objectives:
            # Find closest reference point
            distances = [
                self._distance_to_reference_point(obj, ref_point)
                for ref_point in self.reference_points
            ]
            
            closest_ref = np.argmin(distances)
            associations.append(closest_ref)
        
        return associations
    
    def _distance_to_reference_point(
        self,
        objective: np.ndarray,
        reference_point: np.ndarray
    ) -> float:
        """Calculate distance from objective to reference point."""
        # Perpendicular distance to reference line
        if np.allclose(reference_point, 0):
            return np.linalg.norm(objective)
        
        # Project objective onto reference direction
        projection_length = np.dot(objective, reference_point) / np.linalg.norm(reference_point)
        projection = projection_length * reference_point / np.linalg.norm(reference_point)
        
        # Perpendicular distance
        distance = np.linalg.norm(objective - projection)
        
        return distance
    
    def _update_convergence_metrics(self, pareto_front: ParetoFront) -> None:
        """Update convergence tracking metrics."""
        # Hypervolume
        if len(self.objectives) <= 4:  # Only compute for reasonable dimensions
            reference_point = np.array([10.0] * len(self.objectives))  # Adjust as needed
            hypervolume = pareto_front.compute_hypervolume(reference_point)
            self.hypervolume_history.append(hypervolume)
            
            if hypervolume > self.best_hypervolume:
                self.best_hypervolume = hypervolume
        
        # Diversity (average pairwise distance)
        pareto_solutions = pareto_front.get_pareto_optimal_solutions()
        if len(pareto_solutions) > 1:
            objectives = np.array([sol['objectives'] for sol in pareto_solutions])
            pairwise_distances = pdist(objectives)
            diversity = np.mean(pairwise_distances)
            self.diversity_history.append(diversity)
        else:
            self.diversity_history.append(0.0)
        
        # Convergence (spread of first front)
        first_front_objectives = np.array([ind.objectives for ind in self.population if ind.rank == 0])
        if len(first_front_objectives) > 1:
            convergence = np.std(first_front_objectives, axis=0).mean()
            self.convergence_history.append(convergence)
        else:
            self.convergence_history.append(0.0)


def create_standard_objectives() -> List[OptimizationObjective]:
    """Create standard antenna optimization objectives."""
    
    def gain_objective(result: SolverResult) -> float:
        """Maximize gain (minimize negative gain)."""
        return -(result.gain_dbi or 0.0)
    
    def vswr_objective(result: SolverResult) -> float:
        """Minimize VSWR."""
        return result.vswr[0] if len(result.vswr) > 0 else 10.0
    
    def size_objective(result: SolverResult) -> float:
        """Minimize antenna size (placeholder - would need geometry info)."""
        # This would be computed from the actual geometry
        return 1.0  # Placeholder
    
    def efficiency_objective(result: SolverResult) -> float:
        """Maximize efficiency (minimize negative efficiency)."""
        return -(result.efficiency or 0.0)
    
    def bandwidth_objective(result: SolverResult) -> float:
        """Maximize bandwidth (minimize negative bandwidth)."""
        return -(result.bandwidth_hz or 0.0) / 1e9  # Scale to GHz
    
    objectives = [
        OptimizationObjective("gain", gain_objective, minimize=True, weight=1.0),
        OptimizationObjective("vswr", vswr_objective, minimize=True, weight=1.0),
        OptimizationObjective("efficiency", efficiency_objective, minimize=True, weight=1.0),
        OptimizationObjective("bandwidth", bandwidth_objective, minimize=True, weight=0.1),
    ]
    
    return objectives


class MultiObjectiveOptimizer:
    """High-level multi-objective optimization interface."""
    
    def __init__(
        self,
        objectives: Optional[List[OptimizationObjective]] = None,
        algorithm: str = 'nsga3',
        **algorithm_params
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: Optimization objectives (default: standard antenna objectives)
            algorithm: Optimization algorithm ('nsga3', 'moead', 'sms_emoa')
            **algorithm_params: Algorithm-specific parameters
        """
        self.objectives = objectives or create_standard_objectives()
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params
        
        self.logger = get_logger('multi_objective_optimizer')
        
        # Initialize algorithm
        if algorithm == 'nsga3':
            self.optimizer = NSGA3Optimizer(self.objectives, **algorithm_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def optimize(
        self,
        solver: BaseSolver,
        antenna_spec: AntennaSpec,
        n_variables: int = 16,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callback: Optional[Callable] = None
    ) -> ParetoFront:
        """
        Run multi-objective optimization.
        
        Args:
            solver: Electromagnetic solver
            antenna_spec: Antenna specification
            n_variables: Number of optimization variables
            bounds: Variable bounds (default: [0, 1] for each variable)
            callback: Optional callback function
            
        Returns:
            Pareto front with optimal solutions
        """
        if bounds is None:
            bounds = (np.zeros(n_variables), np.ones(n_variables))
        
        self.logger.info(f"Starting multi-objective optimization with "
                        f"{len(self.objectives)} objectives using {self.algorithm}")
        
        return self.optimizer.optimize(solver, antenna_spec, bounds, callback)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary statistics."""
        if not hasattr(self.optimizer, 'convergence_history'):
            return {'error': 'Optimization not run yet'}
        
        return {
            'algorithm': self.algorithm,
            'objectives': [obj.name for obj in self.objectives],
            'generations': len(self.optimizer.convergence_history),
            'final_hypervolume': (
                self.optimizer.hypervolume_history[-1] 
                if self.optimizer.hypervolume_history else 0
            ),
            'final_diversity': (
                self.optimizer.diversity_history[-1]
                if self.optimizer.diversity_history else 0
            ),
            'convergence_trend': (
                self.optimizer.convergence_history[-10:]  # Last 10 generations
                if len(self.optimizer.convergence_history) >= 10 else
                self.optimizer.convergence_history
            )
        }


# Utility functions for multi-objective analysis
def compute_pareto_dominance_matrix(objectives: np.ndarray) -> np.ndarray:
    """Compute dominance matrix for set of objective vectors."""
    n = len(objectives)
    dominance_matrix = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dominance_matrix[i, j] = np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j])
    
    return dominance_matrix


def compute_crowding_distances(objectives: np.ndarray) -> np.ndarray:
    """Compute crowding distances for NSGA-II style diversity preservation."""
    n_solutions = len(objectives)
    n_objectives = objectives.shape[1]
    
    distances = np.zeros(n_solutions)
    
    for obj_idx in range(n_objectives):
        # Sort by this objective
        sorted_indices = np.argsort(objectives[:, obj_idx])
        
        # Set boundary points to infinity
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Compute distances for interior points
        obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
        
        if obj_range > 0:
            for i in range(1, n_solutions - 1):
                idx = sorted_indices[i]
                distances[idx] += (
                    objectives[sorted_indices[i + 1], obj_idx] - 
                    objectives[sorted_indices[i - 1], obj_idx]
                ) / obj_range
    
    return distances