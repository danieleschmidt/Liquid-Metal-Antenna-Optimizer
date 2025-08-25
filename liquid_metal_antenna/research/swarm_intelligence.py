"""
🐜 Swarm Intelligence Optimization Framework  
===========================================

Generation 5 breakthrough: Bio-inspired swarm algorithms with emergent collective
intelligence for massively parallel antenna optimization.

Author: Terry @ Terragon Labs
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import time
import random

logger = logging.getLogger(__name__)


@dataclass
class SwarmAgent:
    """Individual agent in swarm optimization."""
    position: np.ndarray
    velocity: np.ndarray
    personal_best_position: np.ndarray
    personal_best_fitness: float
    fitness: float
    age: int = 0
    energy: float = 1.0
    role: str = "explorer"  # explorer, exploiter, scout, leader
    communication_range: float = 1.0
    learning_rate: float = 0.1
    
    def distance_to(self, other: 'SwarmAgent') -> float:
        """Compute distance to another agent."""
        return np.linalg.norm(self.position - other.position)
    
    def can_communicate_with(self, other: 'SwarmAgent') -> bool:
        """Check if agent can communicate with another."""
        return self.distance_to(other) <= self.communication_range
    
    def update_energy(self, fitness_improvement: float):
        """Update agent energy based on performance."""
        if fitness_improvement > 0:
            self.energy = min(1.0, self.energy + 0.1 * fitness_improvement)
        else:
            self.energy = max(0.1, self.energy - 0.05)


class SwarmCommunicationNetwork:
    """Communication network for swarm agents."""
    
    def __init__(self):
        """Initialize communication network."""
        self.message_queue = Queue()
        self.global_knowledge = {}
        self.local_neighborhoods = {}
        
    def broadcast_discovery(self, agent_id: int, discovery: Dict[str, Any]):
        """Broadcast discovery to network."""
        message = {
            'type': 'discovery',
            'sender': agent_id,
            'content': discovery,
            'timestamp': time.time()
        }
        self.message_queue.put(message)
        
    def share_local_information(self, agent_id: int, position: np.ndarray, fitness: float):
        """Share local information with neighbors."""
        message = {
            'type': 'local_info',
            'sender': agent_id,
            'position': position.copy(),
            'fitness': fitness,
            'timestamp': time.time()
        }
        self.message_queue.put(message)
        
    def process_messages(self, agents: List[SwarmAgent]):
        """Process pending messages and update agent knowledge."""
        processed = 0
        
        while not self.message_queue.empty() and processed < 100:  # Limit processing
            message = self.message_queue.get()
            processed += 1
            
            if message['type'] == 'discovery':
                # Update global knowledge
                self._process_discovery_message(message, agents)
            elif message['type'] == 'local_info':
                # Update local neighborhoods
                self._process_local_info_message(message, agents)
                
    def _process_discovery_message(self, message: Dict[str, Any], agents: List[SwarmAgent]):
        """Process discovery message."""
        discovery = message['content']
        sender_id = message['sender']
        
        # Propagate significant discoveries
        if 'fitness_improvement' in discovery and discovery['fitness_improvement'] > 0.1:
            for i, agent in enumerate(agents):
                if i != sender_id and agent.can_communicate_with(agents[sender_id]):
                    # Influence agent behavior
                    influence_strength = 0.1 * agent.learning_rate
                    direction = discovery['search_direction']
                    agent.velocity = (1 - influence_strength) * agent.velocity + \
                                   influence_strength * direction
                    
    def _process_local_info_message(self, message: Dict[str, Any], agents: List[SwarmAgent]):
        """Process local information message."""
        sender_id = message['sender']
        sender_position = message['position']
        sender_fitness = message['fitness']
        
        # Update neighborhood information
        if sender_id not in self.local_neighborhoods:
            self.local_neighborhoods[sender_id] = []
            
        # Add to local neighborhood knowledge
        self.local_neighborhoods[sender_id].append({
            'position': sender_position,
            'fitness': sender_fitness,
            'timestamp': message['timestamp']
        })
        
        # Keep only recent information
        current_time = time.time()
        self.local_neighborhoods[sender_id] = [
            info for info in self.local_neighborhoods[sender_id]
            if current_time - info['timestamp'] < 60.0  # 60 second window
        ]


class AntColonyOptimizer:
    """
    🐜 Ant Colony Optimization for Antenna Design
    
    Implements stigmergy-based optimization using pheromone trails
    for parameter space exploration.
    """
    
    def __init__(self, n_ants: int = 50, problem_dim: int = 10,
                 pheromone_evaporation: float = 0.1, pheromone_constant: float = 100.0):
        """Initialize ant colony.
        
        Args:
            n_ants: Number of ants in colony
            problem_dim: Dimension of optimization problem  
            pheromone_evaporation: Pheromone evaporation rate
            pheromone_constant: Pheromone deposition constant
        """
        self.n_ants = n_ants
        self.problem_dim = problem_dim
        self.evaporation_rate = pheromone_evaporation
        self.pheromone_constant = pheromone_constant
        
        # Pheromone matrix (discretized parameter space)
        self.n_discrete_values = 100
        self.pheromone_matrix = np.ones((problem_dim, self.n_discrete_values))
        
        # Ant positions and paths
        self.ants = []
        self.best_path = None
        self.best_fitness = -np.inf
        
    def initialize_colony(self, bounds: Tuple[float, float] = (-5.0, 5.0)):
        """Initialize ant colony."""
        self.ants = []
        self.bounds = bounds
        
        for _ in range(self.n_ants):
            position = np.random.uniform(bounds[0], bounds[1], self.problem_dim)
            ant = {
                'position': position,
                'path': [],
                'fitness': 0.0,
                'age': 0
            }
            self.ants.append(ant)
            
    def construct_solution(self, ant: Dict[str, Any]) -> np.ndarray:
        """Construct solution using pheromone probabilities."""
        solution = np.zeros(self.problem_dim)
        
        for dim in range(self.problem_dim):
            # Pheromone-based probability distribution
            pheromones = self.pheromone_matrix[dim, :]
            probabilities = pheromones / np.sum(pheromones)
            
            # Select discrete value based on probabilities
            discrete_idx = np.random.choice(self.n_discrete_values, p=probabilities)
            
            # Convert to continuous value
            continuous_value = (self.bounds[0] + 
                              (discrete_idx / (self.n_discrete_values - 1)) * 
                              (self.bounds[1] - self.bounds[0]))
            
            solution[dim] = continuous_value
            ant['path'].append((dim, discrete_idx))
            
        return solution
    
    def update_pheromones(self, solutions_fitness: List[Tuple[np.ndarray, float]]):
        """Update pheromone trails based on solutions."""
        # Evaporation
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        
        # Pheromone deposition
        for solution, fitness in solutions_fitness:
            if fitness > 0:  # Only deposit for good solutions
                pheromone_amount = self.pheromone_constant / (1.0 + abs(fitness))
                
                for dim, value in enumerate(solution):
                    # Convert continuous value to discrete index
                    discrete_idx = int((value - self.bounds[0]) / 
                                     (self.bounds[1] - self.bounds[0]) * 
                                     (self.n_discrete_values - 1))
                    discrete_idx = np.clip(discrete_idx, 0, self.n_discrete_values - 1)
                    
                    self.pheromone_matrix[dim, discrete_idx] += pheromone_amount
        
        # Prevent pheromone stagnation
        self.pheromone_matrix = np.clip(self.pheromone_matrix, 0.01, 100.0)
    
    def optimize(self, objective_function: Callable[[np.ndarray], float],
                bounds: Tuple[float, float] = (-5.0, 5.0),
                max_iterations: int = 100) -> Dict[str, Any]:
        """Run ant colony optimization."""
        logger.info("🐜 Starting Ant Colony Optimization")
        
        self.initialize_colony(bounds)
        
        fitness_history = []
        pheromone_entropy_history = []
        
        for iteration in range(max_iterations):
            iteration_solutions = []
            iteration_fitness = []
            
            # Each ant constructs a solution
            for ant in self.ants:
                solution = self.construct_solution(ant)
                fitness = objective_function(solution)
                
                ant['position'] = solution
                ant['fitness'] = fitness
                ant['age'] += 1
                
                iteration_solutions.append((solution, fitness))
                iteration_fitness.append(fitness)
                
                # Update global best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_path = solution.copy()
            
            # Update pheromones
            self.update_pheromones(iteration_solutions)
            
            # Track pheromone diversity
            pheromone_entropy = -np.sum(
                self.pheromone_matrix * np.log(self.pheromone_matrix + 1e-8)
            ) / (self.problem_dim * self.n_discrete_values)
            pheromone_entropy_history.append(pheromone_entropy)
            
            # Track progress
            fitness_history.append({
                'iteration': iteration,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(iteration_fitness),
                'std_fitness': np.std(iteration_fitness),
                'pheromone_entropy': pheromone_entropy
            })
            
            if iteration % 10 == 0:
                logger.info(f"ACO Iteration {iteration}: Best={self.best_fitness:.6f}, "
                           f"Mean={np.mean(iteration_fitness):.6f}, "
                           f"Entropy={pheromone_entropy:.3f}")
        
        return {
            'best_solution': self.best_path,
            'best_fitness': self.best_fitness,
            'fitness_history': fitness_history,
            'pheromone_entropy_history': pheromone_entropy_history,
            'final_pheromone_matrix': self.pheromone_matrix.copy()
        }


class ParticleSwarmOptimizer:
    """
    🦆 Enhanced Particle Swarm Optimization
    
    Implements adaptive particle swarm with communication networks
    and role-based specialization.
    """
    
    def __init__(self, n_particles: int = 50, problem_dim: int = 10,
                 w: float = 0.729, c1: float = 1.494, c2: float = 1.494):
        """Initialize particle swarm.
        
        Args:
            n_particles: Number of particles
            problem_dim: Problem dimensionality
            w: Inertia weight
            c1: Cognitive component
            c2: Social component
        """
        self.n_particles = n_particles
        self.problem_dim = problem_dim
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive component
        self.c2 = c2  # Social component
        
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        self.communication_network = SwarmCommunicationNetwork()
        
    def initialize_swarm(self, bounds: Tuple[float, float] = (-5.0, 5.0)):
        """Initialize particle swarm."""
        self.bounds = bounds
        self.particles = []
        
        for i in range(self.n_particles):
            position = np.random.uniform(bounds[0], bounds[1], self.problem_dim)
            velocity = np.random.uniform(-1, 1, self.problem_dim)
            
            particle = SwarmAgent(
                position=position,
                velocity=velocity,
                personal_best_position=position.copy(),
                personal_best_fitness=-np.inf,
                fitness=-np.inf,
                age=0,
                energy=1.0,
                role=self._assign_role(i),
                communication_range=np.random.uniform(0.5, 2.0),
                learning_rate=np.random.uniform(0.05, 0.2)
            )
            
            self.particles.append(particle)
    
    def _assign_role(self, particle_id: int) -> str:
        """Assign role to particle based on ID and randomness."""
        if particle_id < self.n_particles * 0.6:
            return "explorer"
        elif particle_id < self.n_particles * 0.8:
            return "exploiter"
        elif particle_id < self.n_particles * 0.9:
            return "scout"
        else:
            return "leader"
    
    def update_particle_velocity_position(self, particle: SwarmAgent, 
                                        global_best: np.ndarray):
        """Update particle velocity and position with role-based behavior."""
        r1, r2 = np.random.random(self.problem_dim), np.random.random(self.problem_dim)
        
        # Base PSO velocity update
        cognitive_component = self.c1 * r1 * (particle.personal_best_position - particle.position)
        social_component = self.c2 * r2 * (global_best - particle.position)
        
        # Role-based modifications
        if particle.role == "explorer":
            # Explorers emphasize exploration
            exploration_noise = np.random.normal(0, 0.1, self.problem_dim)
            particle.velocity = (self.w * particle.velocity + 
                               0.8 * cognitive_component + 
                               0.4 * social_component + 
                               exploration_noise)
        
        elif particle.role == "exploiter":
            # Exploiters emphasize local search
            particle.velocity = (self.w * particle.velocity + 
                               0.5 * cognitive_component + 
                               1.2 * social_component)
        
        elif particle.role == "scout":
            # Scouts search distant regions
            if np.random.random() < 0.1:  # 10% chance to scout randomly
                particle.velocity = np.random.uniform(-2, 2, self.problem_dim)
            else:
                particle.velocity = (self.w * particle.velocity + 
                                   1.0 * cognitive_component + 
                                   0.8 * social_component)
        
        elif particle.role == "leader":
            # Leaders balance exploration and exploitation
            particle.velocity = (self.w * particle.velocity + 
                               self.c1 * r1 * (particle.personal_best_position - particle.position) +
                               self.c2 * r2 * (global_best - particle.position))
        
        # Velocity clamping
        max_velocity = (self.bounds[1] - self.bounds[0]) * 0.2
        particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)
        
        # Position update
        old_position = particle.position.copy()
        particle.position += particle.velocity
        
        # Boundary handling
        particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])
        
        # Communication: share movement information
        movement_info = {
            'search_direction': particle.velocity / (np.linalg.norm(particle.velocity) + 1e-8),
            'movement_magnitude': np.linalg.norm(particle.velocity),
            'fitness_improvement': particle.fitness - particle.personal_best_fitness
        }
        
        self.communication_network.broadcast_discovery(
            len(self.particles),  # Temporary ID
            movement_info
        )
    
    def adapt_parameters(self, iteration: int, max_iterations: int):
        """Adapt PSO parameters during optimization."""
        # Decrease inertia weight over time
        self.w = 0.9 - (0.9 - 0.4) * iteration / max_iterations
        
        # Adapt particle roles based on performance
        fitness_values = [p.fitness for p in self.particles]
        fitness_rank = np.argsort(fitness_values)
        
        for i, particle in enumerate(self.particles):
            rank = np.where(fitness_rank == i)[0][0]
            
            # Energy-based role adaptation
            if particle.energy > 0.8 and rank < len(self.particles) * 0.2:
                particle.role = "leader"
            elif particle.energy > 0.6:
                particle.role = "exploiter"
            elif particle.energy > 0.3:
                particle.role = "explorer"
            else:
                particle.role = "scout"
                
            # Adapt communication range based on performance
            if particle.fitness > particle.personal_best_fitness:
                particle.communication_range = min(3.0, particle.communication_range * 1.1)
            else:
                particle.communication_range = max(0.5, particle.communication_range * 0.95)
    
    def optimize(self, objective_function: Callable[[np.ndarray], float],
                bounds: Tuple[float, float] = (-5.0, 5.0),
                max_iterations: int = 100) -> Dict[str, Any]:
        """Run particle swarm optimization."""
        logger.info("🦆 Starting Enhanced Particle Swarm Optimization")
        
        self.initialize_swarm(bounds)
        
        fitness_history = []
        role_distribution_history = []
        communication_activity_history = []
        
        for iteration in range(max_iterations):
            iteration_fitness = []
            
            # Evaluate all particles
            for particle in self.particles:
                fitness = objective_function(particle.position)
                old_fitness = particle.fitness
                particle.fitness = fitness
                iteration_fitness.append(fitness)
                
                # Update personal best
                if fitness > particle.personal_best_fitness:
                    particle.personal_best_fitness = fitness
                    particle.personal_best_position = particle.position.copy()
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                
                # Update particle energy
                particle.update_energy(fitness - old_fitness)
                particle.age += 1
                
                # Share local information
                self.communication_network.share_local_information(
                    id(particle), particle.position, fitness
                )
            
            # Process communications
            self.communication_network.process_messages(self.particles)
            
            # Update particles
            for particle in self.particles:
                self.update_particle_velocity_position(particle, self.global_best_position)
            
            # Adapt parameters
            self.adapt_parameters(iteration, max_iterations)
            
            # Track statistics
            role_distribution = {role: 0 for role in ["explorer", "exploiter", "scout", "leader"]}
            for particle in self.particles:
                role_distribution[particle.role] += 1
            role_distribution_history.append(role_distribution)
            
            communication_activity = len([p for p in self.particles if p.energy > 0.5])
            communication_activity_history.append(communication_activity)
            
            fitness_history.append({
                'iteration': iteration,
                'best_fitness': self.global_best_fitness,
                'mean_fitness': np.mean(iteration_fitness),
                'std_fitness': np.std(iteration_fitness),
                'active_communicators': communication_activity,
                'inertia_weight': self.w
            })
            
            if iteration % 10 == 0:
                logger.info(f"PSO Iteration {iteration}: Best={self.global_best_fitness:.6f}, "
                           f"Mean={np.mean(iteration_fitness):.6f}, "
                           f"Active={communication_activity}/{self.n_particles}, "
                           f"w={self.w:.3f}")
        
        return {
            'best_solution': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'fitness_history': fitness_history,
            'role_distribution_history': role_distribution_history,
            'communication_activity_history': communication_activity_history,
            'final_swarm_state': {
                'particles': self.particles,
                'communication_network': self.communication_network
            }
        }


class BeeColonyOptimizer:
    """
    🐝 Artificial Bee Colony Optimization
    
    Implements bee colony optimization with employed bees, onlooker bees,
    and scout bees for comprehensive search strategy.
    """
    
    def __init__(self, n_bees: int = 50, problem_dim: int = 10,
                 max_trials: int = 100):
        """Initialize bee colony.
        
        Args:
            n_bees: Number of bees (should be even)
            problem_dim: Problem dimensionality
            max_trials: Maximum trials before abandoning food source
        """
        self.n_bees = n_bees
        self.n_employed = n_bees // 2
        self.n_onlooker = n_bees // 2
        self.problem_dim = problem_dim
        self.max_trials = max_trials
        
        self.food_sources = []  # Employed bee positions
        self.trials = []  # Trial counters for food sources
        self.best_solution = None
        self.best_fitness = -np.inf
    
    def initialize_colony(self, bounds: Tuple[float, float] = (-5.0, 5.0)):
        """Initialize bee colony."""
        self.bounds = bounds
        self.food_sources = []
        self.trials = []
        
        # Initialize employed bees (food sources)
        for _ in range(self.n_employed):
            food_source = np.random.uniform(bounds[0], bounds[1], self.problem_dim)
            self.food_sources.append(food_source)
            self.trials.append(0)
    
    def employed_bee_phase(self, objective_function: Callable[[np.ndarray], float]):
        """Employed bees explore around their food sources."""
        for i in range(self.n_employed):
            # Generate candidate solution
            candidate = self.food_sources[i].copy()
            
            # Randomly select dimension to modify
            dim = np.random.randint(0, self.problem_dim)
            
            # Select different food source randomly
            partner_idx = np.random.choice([j for j in range(self.n_employed) if j != i])
            partner = self.food_sources[partner_idx]
            
            # Generate new candidate
            phi = np.random.uniform(-1, 1)
            candidate[dim] = self.food_sources[i][dim] + phi * (
                self.food_sources[i][dim] - partner[dim]
            )
            
            # Boundary control
            candidate[dim] = np.clip(candidate[dim], self.bounds[0], self.bounds[1])
            
            # Evaluate candidate
            candidate_fitness = objective_function(candidate)
            current_fitness = objective_function(self.food_sources[i])
            
            # Greedy selection
            if candidate_fitness > current_fitness:
                self.food_sources[i] = candidate
                self.trials[i] = 0  # Reset trial counter
                
                # Update global best
                if candidate_fitness > self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate.copy()
            else:
                self.trials[i] += 1
    
    def calculate_probabilities(self, objective_function: Callable[[np.ndarray], float]) -> np.ndarray:
        """Calculate selection probabilities for onlooker bees."""
        fitness_values = np.array([objective_function(fs) for fs in self.food_sources])
        
        # Handle negative fitness values
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 1
        
        # Calculate probabilities using fitness proportionate selection
        total_fitness = np.sum(fitness_values)
        if total_fitness == 0:
            probabilities = np.ones(len(self.food_sources)) / len(self.food_sources)
        else:
            probabilities = fitness_values / total_fitness
            
        return probabilities
    
    def onlooker_bee_phase(self, objective_function: Callable[[np.ndarray], float]):
        """Onlooker bees select food sources based on probabilities."""
        probabilities = self.calculate_probabilities(objective_function)
        
        for _ in range(self.n_onlooker):
            # Select food source based on probability
            selected_idx = np.random.choice(len(self.food_sources), p=probabilities)
            
            # Generate candidate solution (same as employed bee)
            candidate = self.food_sources[selected_idx].copy()
            
            # Randomly select dimension and partner
            dim = np.random.randint(0, self.problem_dim)
            partner_idx = np.random.choice([j for j in range(self.n_employed) if j != selected_idx])
            partner = self.food_sources[partner_idx]
            
            # Generate new candidate
            phi = np.random.uniform(-1, 1)
            candidate[dim] = self.food_sources[selected_idx][dim] + phi * (
                self.food_sources[selected_idx][dim] - partner[dim]
            )
            
            # Boundary control
            candidate[dim] = np.clip(candidate[dim], self.bounds[0], self.bounds[1])
            
            # Evaluate and compare
            candidate_fitness = objective_function(candidate)
            current_fitness = objective_function(self.food_sources[selected_idx])
            
            if candidate_fitness > current_fitness:
                self.food_sources[selected_idx] = candidate
                self.trials[selected_idx] = 0
                
                if candidate_fitness > self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate.copy()
            else:
                self.trials[selected_idx] += 1
    
    def scout_bee_phase(self):
        """Scout bees replace exhausted food sources."""
        for i in range(self.n_employed):
            if self.trials[i] >= self.max_trials:
                # Replace exhausted food source with random solution
                self.food_sources[i] = np.random.uniform(
                    self.bounds[0], self.bounds[1], self.problem_dim
                )
                self.trials[i] = 0
    
    def optimize(self, objective_function: Callable[[np.ndarray], float],
                bounds: Tuple[float, float] = (-5.0, 5.0),
                max_iterations: int = 100) -> Dict[str, Any]:
        """Run bee colony optimization."""
        logger.info("🐝 Starting Artificial Bee Colony Optimization")
        
        self.initialize_colony(bounds)
        
        fitness_history = []
        food_source_diversity_history = []
        
        for iteration in range(max_iterations):
            # Employed bee phase
            self.employed_bee_phase(objective_function)
            
            # Onlooker bee phase
            self.onlooker_bee_phase(objective_function)
            
            # Scout bee phase
            self.scout_bee_phase()
            
            # Calculate statistics
            current_fitness = [objective_function(fs) for fs in self.food_sources]
            mean_fitness = np.mean(current_fitness)
            std_fitness = np.std(current_fitness)
            
            # Diversity of food sources
            if len(self.food_sources) > 1:
                distances = []
                for i in range(len(self.food_sources)):
                    for j in range(i + 1, len(self.food_sources)):
                        dist = np.linalg.norm(self.food_sources[i] - self.food_sources[j])
                        distances.append(dist)
                diversity = np.mean(distances) if distances else 0
            else:
                diversity = 0
            
            food_source_diversity_history.append(diversity)
            
            fitness_history.append({
                'iteration': iteration,
                'best_fitness': self.best_fitness,
                'mean_fitness': mean_fitness,
                'std_fitness': std_fitness,
                'diversity': diversity,
                'abandoned_sources': sum(1 for t in self.trials if t >= self.max_trials)
            })
            
            if iteration % 10 == 0:
                logger.info(f"ABC Iteration {iteration}: Best={self.best_fitness:.6f}, "
                           f"Mean={mean_fitness:.6f}, "
                           f"Diversity={diversity:.3f}, "
                           f"Abandoned={fitness_history[-1]['abandoned_sources']}")
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'fitness_history': fitness_history,
            'diversity_history': food_source_diversity_history,
            'final_food_sources': self.food_sources.copy(),
            'final_trials': self.trials.copy()
        }


class HybridSwarmOptimizer:
    """
    🌐 Hybrid Swarm Intelligence System
    
    Combines multiple swarm algorithms with dynamic algorithm selection
    and inter-swarm communication.
    """
    
    def __init__(self, problem_dim: int = 10, total_agents: int = 100):
        """Initialize hybrid swarm system.
        
        Args:
            problem_dim: Problem dimensionality
            total_agents: Total number of agents across all swarms
        """
        self.problem_dim = problem_dim
        self.total_agents = total_agents
        
        # Distribute agents across swarms
        n_aco = total_agents // 4
        n_pso = total_agents // 2
        n_abc = total_agents - n_aco - n_pso
        
        # Initialize swarm algorithms
        self.aco = AntColonyOptimizer(n_aco, problem_dim)
        self.pso = ParticleSwarmOptimizer(n_pso, problem_dim)
        self.abc = BeeColonyOptimizer(n_abc, problem_dim)
        
        # Inter-swarm communication
        self.global_best_solution = None
        self.global_best_fitness = -np.inf
        self.algorithm_performance = {'aco': [], 'pso': [], 'abc': []}
        
    def inter_swarm_migration(self, iteration: int):
        """Migrate best solutions between swarms."""
        if iteration % 20 == 0 and iteration > 0:  # Every 20 iterations
            # Collect best solutions
            swarm_best = {}
            
            if hasattr(self.aco, 'best_path') and self.aco.best_path is not None:
                swarm_best['aco'] = (self.aco.best_path.copy(), self.aco.best_fitness)
            
            if hasattr(self.pso, 'global_best_position') and self.pso.global_best_position is not None:
                swarm_best['pso'] = (self.pso.global_best_position.copy(), self.pso.global_best_fitness)
            
            if hasattr(self.abc, 'best_solution') and self.abc.best_solution is not None:
                swarm_best['abc'] = (self.abc.best_solution.copy(), self.abc.best_fitness)
            
            # Find global best
            global_best_alg = max(swarm_best.keys(), 
                                key=lambda k: swarm_best[k][1],
                                default=None)
            
            if global_best_alg:
                global_solution, global_fitness = swarm_best[global_best_alg]
                
                # Update global best
                if global_fitness > self.global_best_fitness:
                    self.global_best_fitness = global_fitness
                    self.global_best_solution = global_solution.copy()
                
                # Inject best solution into other swarms
                for alg_name in swarm_best.keys():
                    if alg_name != global_best_alg:
                        self._inject_solution(alg_name, global_solution)
    
    def _inject_solution(self, algorithm: str, solution: np.ndarray):
        """Inject solution into specified algorithm."""
        if algorithm == 'aco':
            # Replace worst ant with best solution
            if self.aco.ants:
                worst_idx = min(range(len(self.aco.ants)), 
                              key=lambda i: self.aco.ants[i]['fitness'])
                self.aco.ants[worst_idx]['position'] = solution.copy()
        
        elif algorithm == 'pso':
            # Replace worst particle with best solution  
            if self.pso.particles:
                worst_idx = min(range(len(self.pso.particles)),
                              key=lambda i: self.pso.particles[i].fitness)
                self.pso.particles[worst_idx].position = solution.copy()
                self.pso.particles[worst_idx].personal_best_position = solution.copy()
        
        elif algorithm == 'abc':
            # Replace worst food source with best solution
            if self.abc.food_sources:
                # This would require evaluation to find worst, simplified here
                worst_idx = np.random.randint(0, len(self.abc.food_sources))
                self.abc.food_sources[worst_idx] = solution.copy()
                self.abc.trials[worst_idx] = 0
    
    def adaptive_algorithm_selection(self, iteration: int) -> str:
        """Select most promising algorithm based on recent performance."""
        if iteration < 20:
            return 'all'  # Use all algorithms initially
        
        # Calculate recent performance
        recent_window = 10
        recent_performance = {}
        
        for alg_name, history in self.algorithm_performance.items():
            if len(history) >= recent_window:
                recent_improvements = np.diff(history[-recent_window:])
                recent_performance[alg_name] = np.mean(recent_improvements)
            else:
                recent_performance[alg_name] = 0
        
        # Select algorithm with best recent improvement
        if recent_performance:
            best_alg = max(recent_performance.keys(), 
                         key=lambda k: recent_performance[k])
            return best_alg
        else:
            return 'all'
    
    def optimize(self, objective_function: Callable[[np.ndarray], float],
                bounds: Tuple[float, float] = (-5.0, 5.0),
                max_iterations: int = 100) -> Dict[str, Any]:
        """Run hybrid swarm optimization."""
        logger.info("🌐 Starting Hybrid Swarm Intelligence Optimization")
        
        # Initialize all swarms
        self.aco.initialize_colony(bounds)
        self.pso.initialize_swarm(bounds)
        self.abc.initialize_colony(bounds)
        
        combined_fitness_history = []
        algorithm_selection_history = []
        
        for iteration in range(max_iterations):
            # Adaptive algorithm selection
            active_algorithm = self.adaptive_algorithm_selection(iteration)
            algorithm_selection_history.append(active_algorithm)
            
            iteration_results = {}
            
            # Run selected algorithm(s)
            if active_algorithm == 'all' or active_algorithm == 'aco':
                # Run one ACO iteration
                solutions_fitness = []
                for ant in self.aco.ants:
                    solution = self.aco.construct_solution(ant)
                    fitness = objective_function(solution)
                    ant['position'] = solution
                    ant['fitness'] = fitness
                    solutions_fitness.append((solution, fitness))
                    
                    if fitness > self.aco.best_fitness:
                        self.aco.best_fitness = fitness
                        self.aco.best_path = solution.copy()
                
                self.aco.update_pheromones(solutions_fitness)
                iteration_results['aco'] = self.aco.best_fitness
                self.algorithm_performance['aco'].append(self.aco.best_fitness)
            
            if active_algorithm == 'all' or active_algorithm == 'pso':
                # Run one PSO iteration
                for particle in self.pso.particles:
                    fitness = objective_function(particle.position)
                    particle.fitness = fitness
                    
                    if fitness > particle.personal_best_fitness:
                        particle.personal_best_fitness = fitness
                        particle.personal_best_position = particle.position.copy()
                    
                    if fitness > self.pso.global_best_fitness:
                        self.pso.global_best_fitness = fitness
                        self.pso.global_best_position = particle.position.copy()
                
                for particle in self.pso.particles:
                    self.pso.update_particle_velocity_position(particle, self.pso.global_best_position)
                
                iteration_results['pso'] = self.pso.global_best_fitness
                self.algorithm_performance['pso'].append(self.pso.global_best_fitness)
            
            if active_algorithm == 'all' or active_algorithm == 'abc':
                # Run one ABC iteration
                self.abc.employed_bee_phase(objective_function)
                self.abc.onlooker_bee_phase(objective_function)
                self.abc.scout_bee_phase()
                
                iteration_results['abc'] = self.abc.best_fitness
                self.algorithm_performance['abc'].append(self.abc.best_fitness)
            
            # Inter-swarm migration
            self.inter_swarm_migration(iteration)
            
            # Update global best
            current_best_fitness = max([
                self.aco.best_fitness if hasattr(self.aco, 'best_fitness') else -np.inf,
                self.pso.global_best_fitness if hasattr(self.pso, 'global_best_fitness') else -np.inf,
                self.abc.best_fitness if hasattr(self.abc, 'best_fitness') else -np.inf
            ])
            
            if current_best_fitness > self.global_best_fitness:
                self.global_best_fitness = current_best_fitness
                
                # Find which algorithm has the best solution
                if hasattr(self.aco, 'best_fitness') and self.aco.best_fitness == current_best_fitness:
                    self.global_best_solution = self.aco.best_path.copy()
                elif hasattr(self.pso, 'global_best_fitness') and self.pso.global_best_fitness == current_best_fitness:
                    self.global_best_solution = self.pso.global_best_position.copy()
                elif hasattr(self.abc, 'best_fitness') and self.abc.best_fitness == current_best_fitness:
                    self.global_best_solution = self.abc.best_solution.copy()
            
            combined_fitness_history.append({
                'iteration': iteration,
                'global_best_fitness': self.global_best_fitness,
                'active_algorithm': active_algorithm,
                'aco_fitness': iteration_results.get('aco', -np.inf),
                'pso_fitness': iteration_results.get('pso', -np.inf),
                'abc_fitness': iteration_results.get('abc', -np.inf)
            })
            
            if iteration % 10 == 0:
                logger.info(f"Hybrid Iteration {iteration}: Best={self.global_best_fitness:.6f}, "
                           f"Active={active_algorithm}, "
                           f"ACO={iteration_results.get('aco', 'N/A')}, "
                           f"PSO={iteration_results.get('pso', 'N/A')}, "
                           f"ABC={iteration_results.get('abc', 'N/A')}")
        
        return {
            'best_solution': self.global_best_solution,
            'best_fitness': self.global_best_fitness,
            'combined_fitness_history': combined_fitness_history,
            'algorithm_selection_history': algorithm_selection_history,
            'algorithm_performance_history': self.algorithm_performance,
            'individual_results': {
                'aco': {'best_solution': getattr(self.aco, 'best_path', None),
                       'best_fitness': getattr(self.aco, 'best_fitness', -np.inf)},
                'pso': {'best_solution': getattr(self.pso, 'global_best_position', None),
                       'best_fitness': getattr(self.pso, 'global_best_fitness', -np.inf)},
                'abc': {'best_solution': getattr(self.abc, 'best_solution', None),
                       'best_fitness': getattr(self.abc, 'best_fitness', -np.inf)}
            }
        }


# Export main classes
__all__ = [
    'SwarmAgent',
    'SwarmCommunicationNetwork',
    'AntColonyOptimizer',
    'ParticleSwarmOptimizer', 
    'BeeColonyOptimizer',
    'HybridSwarmOptimizer'
]