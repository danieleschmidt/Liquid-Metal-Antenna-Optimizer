"""
Comprehensive comparative study framework for antenna optimization algorithms.

This module provides tools for rigorous academic comparison of optimization
algorithms, statistical significance testing, and publication-ready analysis.
"""

import time
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.antenna_spec import AntennaSpec
from ..core.optimizer import OptimizationResult
from ..solvers.base import BaseSolver
from ..optimization.neural_surrogate import NeuralSurrogate
from ..utils.logging_config import get_logger
from .novel_algorithms import NovelOptimizer


@dataclass
class BenchmarkProblem:
    """Standardized benchmark problem for optimization comparison."""
    
    name: str
    spec: AntennaSpec
    objective: str
    constraints: Dict[str, Any]
    known_optimum: Optional[float]
    difficulty_rating: str  # 'easy', 'medium', 'hard', 'extreme'
    problem_class: str  # 'single_objective', 'multi_objective', 'constrained'
    description: str
    reference_paper: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class AlgorithmResult:
    """Results for a single algorithm on a benchmark problem."""
    
    algorithm_name: str
    problem_name: str
    best_objective: float
    convergence_history: List[float]
    total_time: float
    total_iterations: int
    total_evaluations: int
    convergence_achieved: bool
    statistical_metrics: Dict[str, float]
    research_data: Dict[str, Any]
    run_id: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ComparisonMetrics:
    """Statistical comparison metrics between algorithms."""
    
    algorithm_a: str
    algorithm_b: str
    metric_name: str
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    winner: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ComparativeStudy:
    """
    Framework for rigorous comparative studies of optimization algorithms.
    
    Features:
    - Standardized benchmark problems
    - Statistical significance testing
    - Multi-run analysis with confidence intervals
    - Performance profiling and complexity analysis
    - Publication-ready results export
    """
    
    def __init__(
        self,
        solvers: Dict[str, BaseSolver],
        surrogate_models: Optional[Dict[str, NeuralSurrogate]] = None,
        random_seed: int = 42
    ):
        """
        Initialize comparative study framework.
        
        Args:
            solvers: Dictionary of electromagnetic solvers
            surrogate_models: Optional surrogate models
            random_seed: Random seed for reproducibility
        """
        self.solvers = solvers
        self.surrogate_models = surrogate_models or {}
        self.random_seed = random_seed
        
        self.logger = get_logger('comparative_study')
        
        # Study configuration
        self.benchmark_problems = []
        self.algorithms = {}
        self.results_database = []
        
        # Analysis results
        self.statistical_comparisons = []
        self.performance_rankings = {}
        
        # Set random seed
        np.random.seed(random_seed)
        
        self._initialize_benchmark_problems()
        self.logger.info("Comparative study framework initialized")
    
    def _initialize_benchmark_problems(self) -> None:
        """Initialize standardized benchmark problems."""
        
        # Problem 1: Single-band patch antenna (easy)
        self.add_benchmark_problem(
            name="single_band_patch",
            spec=AntennaSpec(
                frequency_range=(2.4e9, 2.5e9),
                substrate='fr4',
                metal='galinstan',
                size_constraint=(25, 25, 1.6)
            ),
            objective='gain',
            constraints={'min_gain': 5.0, 'max_vswr': 2.0},
            known_optimum=7.5,
            difficulty_rating='easy',
            problem_class='single_objective',
            description="Single-band 2.4 GHz patch antenna optimization for maximum gain"
        )
        
        # Problem 2: Wideband antenna (medium)
        self.add_benchmark_problem(
            name="wideband_antenna",
            spec=AntennaSpec(
                frequency_range=(2.0e9, 4.0e9),
                substrate='rogers_4003c',
                metal='galinstan',
                size_constraint=(30, 30, 3.2)
            ),
            objective='bandwidth',
            constraints={'min_gain': 3.0, 'max_vswr': 3.0},
            known_optimum=None,
            difficulty_rating='medium',
            problem_class='single_objective',
            description="Wideband antenna design for maximum bandwidth with gain constraint"
        )
        
        # Problem 3: Multi-objective optimization (hard)
        self.add_benchmark_problem(
            name="multi_objective_patch",
            spec=AntennaSpec(
                frequency_range=(5.1e9, 5.9e9),
                substrate='rogers_5880',
                metal='egain',
                size_constraint=(20, 20, 2.5)
            ),
            objective='multi_objective',
            constraints={'pareto_objectives': ['gain', 'bandwidth', 'efficiency']},
            known_optimum=None,
            difficulty_rating='hard',
            problem_class='multi_objective',
            description="Multi-objective optimization balancing gain, bandwidth, and efficiency"
        )
        
        # Problem 4: Constrained reconfigurable antenna (extreme)
        self.add_benchmark_problem(
            name="constrained_reconfigurable",
            spec=AntennaSpec(
                frequency_range=(1.8e9, 6.0e9),
                substrate='rogers_4003c',
                metal='galinstan',
                size_constraint=(35, 35, 3.2)
            ),
            objective='reconfigurable_performance',
            constraints={
                'n_reconfig_states': 4,
                'min_isolation': 20.0,
                'max_switching_time': 10e-3,
                'power_consumption': 50e-3
            },
            known_optimum=None,
            difficulty_rating='extreme',
            problem_class='constrained',
            description="Reconfigurable antenna with multiple performance constraints"
        )
        
        self.logger.info(f"Initialized {len(self.benchmark_problems)} benchmark problems")
    
    def add_benchmark_problem(
        self,
        name: str,
        spec: AntennaSpec,
        objective: str,
        constraints: Dict[str, Any],
        known_optimum: Optional[float] = None,
        difficulty_rating: str = 'medium',
        problem_class: str = 'single_objective',
        description: str = '',
        reference_paper: Optional[str] = None
    ) -> None:
        """Add a benchmark problem to the study."""
        
        problem = BenchmarkProblem(
            name=name,
            spec=spec,
            objective=objective,
            constraints=constraints,
            known_optimum=known_optimum,
            difficulty_rating=difficulty_rating,
            problem_class=problem_class,
            description=description,
            reference_paper=reference_paper
        )
        
        self.benchmark_problems.append(problem)
        self.logger.info(f"Added benchmark problem: {name}")
    
    def register_algorithm(self, name: str, algorithm: NovelOptimizer) -> None:
        """Register an algorithm for comparison."""
        self.algorithms[name] = algorithm
        self.logger.info(f"Registered algorithm: {name}")
    
    def run_comparative_study(
        self,
        n_runs: int = 30,
        max_iterations: int = 100,
        parallel_execution: bool = True,
        save_results: bool = True,
        results_dir: str = "comparative_study_results"
    ) -> Dict[str, Any]:
        """
        Run comprehensive comparative study.
        
        Args:
            n_runs: Number of independent runs per algorithm-problem pair
            max_iterations: Maximum iterations per optimization run
            parallel_execution: Use parallel execution for speed
            save_results: Save results to disk
            results_dir: Directory to save results
            
        Returns:
            Comprehensive study results
        """
        self.logger.info(f"Starting comparative study: {len(self.algorithms)} algorithms, "
                        f"{len(self.benchmark_problems)} problems, {n_runs} runs each")
        
        if save_results:
            results_path = Path(results_dir)
            results_path.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        # Generate all experiment configurations
        experiments = []
        for problem in self.benchmark_problems:
            for alg_name, algorithm in self.algorithms.items():
                for run_idx in range(n_runs):
                    experiments.append({
                        'problem': problem,
                        'algorithm_name': alg_name,
                        'algorithm': algorithm,
                        'run_idx': run_idx,
                        'max_iterations': max_iterations
                    })
        
        self.logger.info(f"Generated {len(experiments)} total experiments")
        
        # Execute experiments
        if parallel_execution and len(experiments) > 1:
            results = self._run_experiments_parallel(experiments)
        else:
            results = self._run_experiments_sequential(experiments)
        
        # Store results
        self.results_database.extend(results)
        
        # Statistical analysis
        self.logger.info("Performing statistical analysis...")
        statistical_analysis = self._perform_statistical_analysis()
        
        # Performance analysis
        performance_analysis = self._analyze_performance_characteristics()
        
        # Convergence analysis
        convergence_analysis = self._analyze_convergence_patterns()
        
        # Computational complexity analysis
        complexity_analysis = self._analyze_computational_complexity()
        
        total_time = time.time() - start_time
        
        # Compile final results
        study_results = {
            'study_metadata': {
                'n_algorithms': len(self.algorithms),
                'n_problems': len(self.benchmark_problems),
                'n_runs_per_config': n_runs,
                'total_experiments': len(experiments),
                'total_study_time': total_time,
                'random_seed': self.random_seed,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'benchmark_problems': [p.to_dict() for p in self.benchmark_problems],
            'algorithm_results': [r.to_dict() for r in results],
            'statistical_analysis': statistical_analysis,
            'performance_analysis': performance_analysis,
            'convergence_analysis': convergence_analysis,
            'complexity_analysis': complexity_analysis,
            'rankings': self._generate_overall_rankings()
        }
        
        if save_results:
            self._save_study_results(study_results, results_path)
        
        self.logger.info(f"Comparative study completed in {total_time:.2f} seconds")
        
        return study_results
    
    def _run_experiments_sequential(self, experiments: List[Dict]) -> List[AlgorithmResult]:
        """Run experiments sequentially."""
        results = []
        
        for i, exp in enumerate(experiments):
            self.logger.debug(f"Running experiment {i+1}/{len(experiments)}: "
                            f"{exp['algorithm_name']} on {exp['problem'].name}")
            
            result = self._run_single_experiment(exp)
            if result:
                results.append(result)
        
        return results
    
    def _run_experiments_parallel(self, experiments: List[Dict]) -> List[AlgorithmResult]:
        """Run experiments in parallel."""
        results = []
        
        # Use thread pool for parallel execution
        max_workers = min(8, len(experiments))  # Limit concurrent workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(self._run_single_experiment, exp): exp
                for exp in experiments
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_exp)):
                exp = future_to_exp[future]
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    self.logger.debug(f"Completed {i+1}/{len(experiments)}: "
                                    f"{exp['algorithm_name']} on {exp['problem'].name}")
                
                except Exception as e:
                    self.logger.error(f"Experiment failed: {exp['algorithm_name']} "
                                    f"on {exp['problem'].name}: {str(e)}")
        
        return results
    
    def _run_single_experiment(self, experiment: Dict) -> Optional[AlgorithmResult]:
        """Run a single optimization experiment."""
        problem = experiment['problem']
        algorithm = experiment['algorithm']
        algorithm_name = experiment['algorithm_name']
        run_idx = experiment['run_idx']
        max_iterations = experiment['max_iterations']
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed + run_idx * 1000 + hash(algorithm_name + problem.name) % 1000)
        
        try:
            start_time = time.time()
            
            # Run optimization
            opt_result = algorithm.optimize(
                spec=problem.spec,
                objective=problem.objective,
                constraints=problem.constraints,
                max_iterations=max_iterations,
                target_accuracy=1e-6
            )
            
            total_time = time.time() - start_time
            
            # Calculate statistical metrics
            statistical_metrics = self._calculate_statistical_metrics(
                opt_result, problem, total_time
            )
            
            # Count evaluations (approximate)
            total_evaluations = len(opt_result.optimization_history)
            
            # Create result record
            result = AlgorithmResult(
                algorithm_name=algorithm_name,
                problem_name=problem.name,
                best_objective=opt_result.optimization_history[-1] if opt_result.optimization_history else 0.0,
                convergence_history=opt_result.optimization_history,
                total_time=total_time,
                total_iterations=opt_result.total_iterations,
                total_evaluations=total_evaluations,
                convergence_achieved=opt_result.convergence_achieved,
                statistical_metrics=statistical_metrics,
                research_data=getattr(opt_result, 'research_data', {}),
                run_id=f"{algorithm_name}_{problem.name}_{run_idx}",
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            return None
    
    def _calculate_statistical_metrics(
        self,
        opt_result: OptimizationResult,
        problem: BenchmarkProblem,
        total_time: float
    ) -> Dict[str, float]:
        """Calculate statistical metrics for a single run."""
        history = opt_result.optimization_history
        
        if not history:
            return {}
        
        metrics = {
            'final_objective': history[-1],
            'best_objective': max(history) if problem.objective in ['gain', 'efficiency'] else min(history),
            'convergence_rate': self._calculate_convergence_rate(history),
            'stability': self._calculate_stability(history),
            'efficiency': len(history) / total_time if total_time > 0 else 0,  # iterations per second
            'early_stopping_iteration': self._find_early_stopping_point(history),
            'exploration_vs_exploitation': self._measure_exploration_exploitation(history)
        }
        
        # Add problem-specific metrics
        if problem.known_optimum is not None:
            optimality_gap = abs(metrics['best_objective'] - problem.known_optimum) / abs(problem.known_optimum)
            metrics['optimality_gap'] = optimality_gap
            metrics['success_rate'] = 1.0 if optimality_gap < 0.05 else 0.0  # 5% tolerance
        
        return metrics
    
    def _calculate_convergence_rate(self, history: List[float]) -> float:
        """Calculate convergence rate (improvement per iteration)."""
        if len(history) < 2:
            return 0.0
        
        improvements = [abs(history[i] - history[i-1]) for i in range(1, len(history))]
        return np.mean(improvements) if improvements else 0.0
    
    def _calculate_stability(self, history: List[float]) -> float:
        """Calculate stability (1 / coefficient of variation of later convergence)."""
        if len(history) < 10:
            return 0.5
        
        # Use last 50% of history
        stable_region = history[len(history)//2:]
        
        if not stable_region:
            return 0.5
        
        mean_val = np.mean(stable_region)
        std_val = np.std(stable_region)
        
        if mean_val == 0:
            return 1.0 if std_val == 0 else 0.0
        
        cv = std_val / abs(mean_val)
        return 1.0 / (1.0 + cv)  # Higher stability = lower coefficient of variation
    
    def _find_early_stopping_point(self, history: List[float]) -> int:
        """Find point where algorithm could have stopped early."""
        if len(history) < 20:
            return len(history)
        
        # Look for point where improvement becomes negligible
        for i in range(10, len(history) - 5):
            recent_improvement = abs(history[i] - history[i-10])
            if recent_improvement < 1e-6:
                return i
        
        return len(history)
    
    def _measure_exploration_exploitation(self, history: List[float]) -> float:
        """Measure exploration vs exploitation balance (0 = pure exploitation, 1 = pure exploration)."""
        if len(history) < 10:
            return 0.5
        
        # Measure variance in improvements
        improvements = [abs(history[i] - history[i-1]) for i in range(1, len(history))]
        improvement_variance = np.var(improvements) if improvements else 0
        
        # Normalize to [0, 1] range (heuristic)
        exploration_measure = min(1.0, improvement_variance * 1000)
        
        return exploration_measure
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis of results."""
        if not self.results_database:
            return {}
        
        analysis = {
            'significance_tests': [],
            'effect_sizes': {},
            'confidence_intervals': {},
            'performance_distributions': {}
        }
        
        # Group results by algorithm and problem
        grouped_results = {}
        for result in self.results_database:
            key = (result.algorithm_name, result.problem_name)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Pairwise comparisons between algorithms on each problem
        problems = set(result.problem_name for result in self.results_database)
        algorithms = set(result.algorithm_name for result in self.results_database)
        
        for problem in problems:
            problem_results = {
                alg: [r.best_objective for r in grouped_results.get((alg, problem), [])]
                for alg in algorithms
                if (alg, problem) in grouped_results
            }
            
            # Skip if not enough algorithms for comparison
            if len(problem_results) < 2:
                continue
            
            # Pairwise statistical tests
            alg_pairs = [(a1, a2) for a1 in problem_results.keys() 
                        for a2 in problem_results.keys() if a1 < a2]
            
            for alg1, alg2 in alg_pairs:
                data1 = problem_results[alg1]
                data2 = problem_results[alg2]
                
                if len(data1) >= 3 and len(data2) >= 3:  # Minimum sample size
                    comparison = self._statistical_comparison(
                        data1, data2, alg1, alg2, f"{problem}_objective"
                    )
                    analysis['significance_tests'].append(comparison)
        
        # Overall performance distributions
        for alg in algorithms:
            alg_results = [r for r in self.results_database if r.algorithm_name == alg]
            if alg_results:
                objectives = [r.best_objective for r in alg_results]
                times = [r.total_time for r in alg_results]
                iterations = [r.total_iterations for r in alg_results]
                
                analysis['performance_distributions'][alg] = {
                    'objective_mean': np.mean(objectives),
                    'objective_std': np.std(objectives),
                    'objective_median': np.median(objectives),
                    'time_mean': np.mean(times),
                    'time_std': np.std(times),
                    'iterations_mean': np.mean(iterations),
                    'iterations_std': np.std(iterations),
                    'success_rate': np.mean([getattr(r, 'convergence_achieved', False) for r in alg_results])
                }
        
        return analysis
    
    def _statistical_comparison(
        self,
        data1: List[float],
        data2: List[float],
        alg1: str,
        alg2: str,
        metric: str
    ) -> ComparisonMetrics:
        """Perform statistical comparison between two datasets."""
        
        # Simple statistical tests (would use scipy.stats in practice)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1), np.std(data2)
        n1, n2 = len(data1), len(data2)
        
        # Effect size (Cohen's d approximation)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Simplified t-test approximation
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2) if pooled_std > 0 else 1
        t_stat = abs(mean1 - mean2) / se_diff if se_diff > 0 else 0
        
        # Rough p-value approximation (would use proper distribution in practice)
        df = n1 + n2 - 2
        p_value = max(0.001, min(0.999, 2 * (1 - min(0.999, t_stat / 3))))  # Very rough approximation
        
        # Confidence interval for difference
        margin_error = 1.96 * se_diff  # 95% CI approximation
        diff = mean1 - mean2
        conf_interval = (diff - margin_error, diff + margin_error)
        
        # Determine winner
        significant = p_value < 0.05
        winner = alg1 if mean1 > mean2 else alg2 if significant else None
        
        return ComparisonMetrics(
            algorithm_a=alg1,
            algorithm_b=alg2,
            metric_name=metric,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=conf_interval,
            significant=significant,
            winner=winner
        )
    
    def _analyze_performance_characteristics(self) -> Dict[str, Any]:
        """Analyze performance characteristics of algorithms."""
        if not self.results_database:
            return {}
        
        analysis = {}
        
        # Performance by problem difficulty
        difficulty_performance = {}
        for problem in self.benchmark_problems:
            problem_results = [r for r in self.results_database if r.problem_name == problem.name]
            
            if problem_results:
                difficulty_performance[problem.difficulty_rating] = {
                    alg: np.mean([r.best_objective for r in problem_results if r.algorithm_name == alg])
                    for alg in set(r.algorithm_name for r in problem_results)
                }
        
        analysis['difficulty_performance'] = difficulty_performance
        
        # Scalability analysis
        scalability_analysis = {}
        algorithms = set(r.algorithm_name for r in self.results_database)
        
        for alg in algorithms:
            alg_results = [r for r in self.results_database if r.algorithm_name == alg]
            
            if len(alg_results) >= 5:  # Need enough data points
                iterations = [r.total_iterations for r in alg_results]
                times = [r.total_time for r in alg_results]
                objectives = [r.best_objective for r in alg_results]
                
                # Time complexity (rough approximation)
                if len(iterations) > 1 and len(times) > 1:
                    try:
                        # Fit linear relationship between iterations and time
                        time_coeffs = np.polyfit(iterations, times, 1)
                        time_complexity = time_coeffs[0]  # Time per iteration
                    except:
                        time_complexity = np.mean(times) / np.mean(iterations)
                else:
                    time_complexity = 0
                
                scalability_analysis[alg] = {
                    'time_per_iteration': time_complexity,
                    'avg_iterations_to_convergence': np.mean(iterations),
                    'convergence_reliability': np.mean([r.convergence_achieved for r in alg_results])
                }
        
        analysis['scalability'] = scalability_analysis
        
        return analysis
    
    def _analyze_convergence_patterns(self) -> Dict[str, Any]:
        """Analyze convergence patterns of algorithms."""
        if not self.results_database:
            return {}
        
        analysis = {}
        algorithms = set(r.algorithm_name for r in self.results_database)
        
        for alg in algorithms:
            alg_results = [r for r in self.results_database if r.algorithm_name == alg]
            
            if alg_results:
                # Analyze convergence curves
                all_histories = [r.convergence_history for r in alg_results if r.convergence_history]
                
                if all_histories:
                    # Normalize histories to same length for comparison
                    max_length = max(len(h) for h in all_histories)
                    normalized_histories = []
                    
                    for history in all_histories:
                        if len(history) < max_length:
                            # Extend with final value
                            extended = history + [history[-1]] * (max_length - len(history))
                            normalized_histories.append(extended)
                        else:
                            normalized_histories.append(history[:max_length])
                    
                    # Calculate average convergence curve
                    avg_convergence = np.mean(normalized_histories, axis=0)
                    std_convergence = np.std(normalized_histories, axis=0)
                    
                    analysis[alg] = {
                        'avg_convergence_curve': avg_convergence.tolist(),
                        'std_convergence_curve': std_convergence.tolist(),
                        'avg_final_performance': np.mean([h[-1] for h in all_histories]),
                        'convergence_consistency': 1.0 / (1.0 + np.mean(std_convergence)),
                        'early_convergence_rate': np.mean([
                            len([i for i, val in enumerate(h) if abs(val - h[-1]) < 0.01]) / len(h)
                            for h in all_histories
                        ])
                    }
        
        return analysis
    
    def _analyze_computational_complexity(self) -> Dict[str, Any]:
        """Analyze computational complexity of algorithms."""
        if not self.results_database:
            return {}
        
        analysis = {}
        algorithms = set(r.algorithm_name for r in self.results_database)
        
        for alg in algorithms:
            alg_results = [r for r in self.results_database if r.algorithm_name == alg]
            
            if alg_results:
                times = [r.total_time for r in alg_results]
                iterations = [r.total_iterations for r in alg_results]
                evaluations = [r.total_evaluations for r in alg_results]
                
                analysis[alg] = {
                    'avg_total_time': np.mean(times),
                    'std_total_time': np.std(times),
                    'avg_iterations': np.mean(iterations),
                    'avg_evaluations': np.mean(evaluations),
                    'time_efficiency': np.mean(evaluations) / np.mean(times) if np.mean(times) > 0 else 0,
                    'iteration_efficiency': np.mean(evaluations) / np.mean(iterations) if np.mean(iterations) > 0 else 0
                }
        
        return analysis
    
    def _generate_overall_rankings(self) -> Dict[str, Any]:
        """Generate overall algorithm rankings across all metrics."""
        if not self.results_database:
            return {}
        
        algorithms = set(r.algorithm_name for r in self.results_database)
        problems = set(r.problem_name for r in self.results_database)
        
        # Multi-criteria ranking
        ranking_metrics = {
            'performance': {},  # Average objective value
            'efficiency': {},   # Performance per unit time
            'reliability': {},  # Convergence success rate
            'consistency': {}   # Low variance in performance
        }
        
        for alg in algorithms:
            alg_results = [r for r in self.results_database if r.algorithm_name == alg]
            
            if alg_results:
                objectives = [r.best_objective for r in alg_results]
                times = [r.total_time for r in alg_results]
                convergences = [r.convergence_achieved for r in alg_results]
                
                ranking_metrics['performance'][alg] = np.mean(objectives)
                ranking_metrics['efficiency'][alg] = np.mean(objectives) / np.mean(times) if np.mean(times) > 0 else 0
                ranking_metrics['reliability'][alg] = np.mean(convergences)
                ranking_metrics['consistency'][alg] = 1.0 / (1.0 + np.std(objectives) / max(abs(np.mean(objectives)), 1e-6))
        
        # Normalize metrics to [0, 1] and rank
        normalized_rankings = {}
        for metric, values in ranking_metrics.items():
            if values:
                min_val = min(values.values())
                max_val = max(values.values())
                
                if max_val > min_val:
                    normalized = {alg: (val - min_val) / (max_val - min_val) 
                                for alg, val in values.items()}
                else:
                    normalized = {alg: 1.0 for alg in values.keys()}
                
                # Rank algorithms (higher is better)
                sorted_algs = sorted(normalized.keys(), key=lambda x: normalized[x], reverse=True)
                normalized_rankings[metric] = {
                    alg: {'score': normalized[alg], 'rank': i + 1}
                    for i, alg in enumerate(sorted_algs)
                }
        
        # Overall ranking (equal weights)
        overall_scores = {}
        for alg in algorithms:
            total_score = sum(
                normalized_rankings[metric][alg]['score']
                for metric in ranking_metrics.keys()
                if alg in normalized_rankings[metric]
            )
            overall_scores[alg] = total_score / len(ranking_metrics)
        
        overall_ranking = sorted(overall_scores.keys(), key=lambda x: overall_scores[x], reverse=True)
        
        return {
            'metric_rankings': normalized_rankings,
            'overall_ranking': {
                alg: {'score': overall_scores[alg], 'rank': i + 1}
                for i, alg in enumerate(overall_ranking)
            },
            'ranking_methodology': {
                'metrics_used': list(ranking_metrics.keys()),
                'weighting': 'equal_weights',
                'normalization': 'min_max_scaling'
            }
        }
    
    def _save_study_results(self, results: Dict[str, Any], results_dir: Path) -> None:
        """Save comprehensive study results to disk."""
        
        # Save main results as JSON
        main_results_file = results_dir / "comparative_study_results.json"
        with open(main_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed results as CSV
        csv_file = results_dir / "detailed_results.csv"
        self._export_results_csv(csv_file)
        
        # Save statistical comparisons
        stats_file = results_dir / "statistical_comparisons.json"
        stats_data = {
            'comparisons': [comp.to_dict() for comp in self.statistical_comparisons],
            'methodology': 'pairwise_statistical_tests'
        }
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        # Save benchmark problems specification
        benchmarks_file = results_dir / "benchmark_problems.json"
        benchmarks_data = [problem.to_dict() for problem in self.benchmark_problems]
        with open(benchmarks_file, 'w') as f:
            json.dump(benchmarks_data, f, indent=2, default=str)
        
        self.logger.info(f"Study results saved to {results_dir}")
    
    def _export_results_csv(self, csv_file: Path) -> None:
        """Export detailed results to CSV format."""
        if not self.results_database:
            return
        
        fieldnames = [
            'algorithm_name', 'problem_name', 'run_id', 'best_objective',
            'total_time', 'total_iterations', 'total_evaluations',
            'convergence_achieved', 'timestamp'
        ]
        
        # Add statistical metrics
        if self.results_database:
            sample_metrics = self.results_database[0].statistical_metrics
            fieldnames.extend([f'stat_{key}' for key in sample_metrics.keys()])
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results_database:
                row = {
                    'algorithm_name': result.algorithm_name,
                    'problem_name': result.problem_name,
                    'run_id': result.run_id,
                    'best_objective': result.best_objective,
                    'total_time': result.total_time,
                    'total_iterations': result.total_iterations,
                    'total_evaluations': result.total_evaluations,
                    'convergence_achieved': result.convergence_achieved,
                    'timestamp': result.timestamp
                }
                
                # Add statistical metrics
                for key, value in result.statistical_metrics.items():
                    row[f'stat_{key}'] = value
                
                writer.writerow(row)
    
    def generate_publication_report(
        self,
        output_file: str = "publication_report.md",
        include_figures: bool = True
    ) -> str:
        """
        Generate publication-ready report.
        
        Args:
            output_file: Output file path
            include_figures: Include figure placeholders
            
        Returns:
            Report content as string
        """
        if not hasattr(self, 'study_results') or not self.study_results:
            self.logger.warning("No study results available for report generation")
            return ""
        
        # Generate comprehensive markdown report
        report_content = self._generate_publication_markdown()
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Publication report generated: {output_file}")
        
        return report_content
    
    def _generate_publication_markdown(self) -> str:
        """Generate publication-ready markdown report."""
        
        # This would be a comprehensive report template
        # For brevity, providing a shortened version
        
        report = """# Comparative Study of Novel Antenna Optimization Algorithms

## Abstract

This study presents a comprehensive comparison of novel optimization algorithms for liquid metal antenna design...

## 1. Introduction

Liquid metal antennas offer unique reconfigurability...

## 2. Methodology

### 2.1 Benchmark Problems

We established standardized benchmark problems...

### 2.2 Algorithms Evaluated

- Quantum-Inspired Optimization
- Differential Evolution with Surrogate Assistance  
- Hybrid Gradient-Free Sampling

### 2.3 Evaluation Metrics

Statistical significance testing with p < 0.05...

## 3. Results

### 3.1 Performance Comparison

[Results would be filled from study_results]

### 3.2 Statistical Analysis

[Statistical comparisons would be included]

## 4. Discussion

### 4.1 Algorithm Performance

### 4.2 Computational Complexity

### 4.3 Practical Implications

## 5. Conclusions

This comparative study demonstrates...

## References

[1] Novel optimization approaches...
"""
        
        return report


# Export classes
__all__ = [
    'BenchmarkProblem',
    'AlgorithmResult', 
    'ComparisonMetrics',
    'ComparativeStudy'
]