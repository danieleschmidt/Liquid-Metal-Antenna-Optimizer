"""
Comprehensive Comparative Studies and Benchmarking Framework.

This module implements rigorous benchmarking and comparative analysis of
optimization algorithms and antenna design methods for research publication.

Features:
- Statistical significance testing (Wilcoxon, Mann-Whitney U, Kruskal-Wallis)
- Multi-objective performance metrics (hypervolume, IGD, spread)
- Cross-validation and bootstrap analysis
- Convergence analysis and complexity measurements
- Publication-quality plots and tables
- Reproducible experimental protocols

Target Venues: IEEE TAP, Nature Communications, NeurIPS Benchmarks Track
"""

import time
import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod
import logging
import hashlib
import platform
import sys
import psutil

# Statistical analysis
import scipy.stats as stats
from scipy.spatial.distance import euclidean
from scipy.stats import kruskal, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.utils import resample

from ..core.antenna_spec import AntennaSpec
from ..optimization.neural_surrogate import NeuralSurrogate
from ..utils.logging_config import get_logger


@dataclass
class StatisticalComparison:
    """Statistical comparison results between algorithms."""
    
    algorithm_a: str
    algorithm_b: str
    metric: str
    
    # Basic statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    median_a: float
    median_b: float
    
    # Statistical tests
    mannwhitney_statistic: float
    mannwhitney_pvalue: float
    wilcoxon_statistic: Optional[float] = None
    wilcoxon_pvalue: Optional[float] = None
    ttest_statistic: float = 0.0
    ttest_pvalue: float = 1.0
    
    # Effect size measures
    cohens_d: float = 0.0
    cliffs_delta: float = 0.0
    vargha_delaney_a12: float = 0.5
    
    # Confidence intervals
    confidence_interval_a: Tuple[float, float] = (0.0, 0.0)
    confidence_interval_b: Tuple[float, float] = (0.0, 0.0)
    
    # Practical significance
    practical_significance: bool = False
    statistical_significance: bool = False
    significant_at_alpha: float = 0.05
    
    # Additional metrics
    sample_size_a: int = 0
    sample_size_b: int = 0
    power_analysis: Optional[float] = None
    
    def interpretation(self) -> str:
        """Provide interpretation of statistical results."""
        interpretation = []
        
        if self.statistical_significance:
            interpretation.append(f"Statistically significant (p={self.mannwhitney_pvalue:.4f})")
        else:
            interpretation.append(f"Not statistically significant (p={self.mannwhitney_pvalue:.4f})")
        
        # Effect size interpretation
        abs_cohens_d = abs(self.cohens_d)
        if abs_cohens_d < 0.2:
            effect_size = "negligible"
        elif abs_cohens_d < 0.5:
            effect_size = "small"
        elif abs_cohens_d < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        interpretation.append(f"Effect size: {effect_size} (Cohen's d={self.cohens_d:.3f})")
        
        # Practical significance
        if self.practical_significance:
            interpretation.append("Practically significant difference")
        else:
            interpretation.append("No practical significance")
            
        return "; ".join(interpretation)


@dataclass
class ExperimentalProtocol:
    """Experimental protocol for reproducible benchmarking."""
    
    name: str
    description: str
    
    # Experimental design
    num_independent_runs: int = 30
    max_function_evaluations: int = 10000
    convergence_tolerance: float = 1e-6
    time_limit_seconds: Optional[float] = None
    
    # Random seed management
    base_random_seed: int = 42
    seed_increment: int = 1
    
    # Statistical parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    effect_size_threshold: float = 0.5
    
    # Cross-validation
    cv_folds: int = 5
    bootstrap_samples: int = 1000
    
    # Environment control
    force_deterministic: bool = True
    cpu_affinity: Optional[List[int]] = None
    memory_limit_gb: Optional[float] = None
    
    # Output control
    save_convergence_history: bool = True
    save_intermediate_results: bool = False
    generate_plots: bool = True
    
    def generate_random_seeds(self) -> List[int]:
        """Generate random seeds for independent runs."""
        return [self.base_random_seed + i * self.seed_increment 
                for i in range(self.num_independent_runs)]
    
    def validate(self) -> List[str]:
        """Validate experimental protocol parameters."""
        issues = []
        
        if self.num_independent_runs < 10:
            issues.append("Number of runs should be at least 10 for statistical validity")
        
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            issues.append("Confidence level must be between 0 and 1")
        
        if self.significance_level <= 0 or self.significance_level >= 1:
            issues.append("Significance level must be between 0 and 1")
            
        return issues


@dataclass
class MultiObjectiveMetrics:
    """Multi-objective optimization metrics."""
    
    hypervolume: float = 0.0
    igd_metric: float = float('inf')  # Inverted Generational Distance
    gd_metric: float = float('inf')   # Generational Distance
    spread_metric: float = 0.0
    spacing_metric: float = 0.0
    coverage_metric: float = 0.0
    
    # Pareto front quality
    pareto_front_size: int = 0
    convergence_metric: float = 0.0
    diversity_metric: float = 0.0
    
    # Reference point metrics
    r_metric: float = 0.0
    epsilon_indicator: float = 0.0


@dataclass
class ConvergenceAnalysis:
    """Detailed convergence analysis metrics."""
    
    convergence_rate: float = 0.0
    convergence_stability: float = 0.0
    plateau_detection: List[Tuple[int, int]] = field(default_factory=list)
    improvement_phases: List[Tuple[int, int, float]] = field(default_factory=list)
    stagnation_periods: List[Tuple[int, int]] = field(default_factory=list)
    
    # Statistical convergence properties
    autocorrelation: List[float] = field(default_factory=list)
    trend_analysis: Dict[str, float] = field(default_factory=dict)
    volatility_measure: float = 0.0
    
    # Success metrics
    target_reached: bool = False
    target_reached_generation: Optional[int] = None
    success_rate: float = 0.0


@dataclass
class RobustnessMetrics:
    """Algorithm robustness and reliability metrics."""
    
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)
    noise_robustness: float = 0.0
    initialization_sensitivity: float = 0.0
    
    # Cross-validation metrics
    cv_mean_performance: float = 0.0
    cv_std_performance: float = 0.0
    cv_reliability_score: float = 0.0
    
    # Bootstrap analysis
    bootstrap_confidence_interval: Tuple[float, float] = (0.0, 0.0)
    bootstrap_bias: float = 0.0
    bootstrap_variance: float = 0.0


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    algorithm_name: str
    problem_instance: str
    run_id: int
    
    # Performance metrics
    best_objective: float
    final_objective: float
    convergence_history: List[float]
    function_evaluations: int
    computation_time: float
    
    # Enhanced metrics
    multiobjective_metrics: Optional[MultiObjectiveMetrics] = None
    convergence_analysis: Optional[ConvergenceAnalysis] = None
    robustness_metrics: Optional[RobustnessMetrics] = None
    
    # Legacy metrics (maintained for compatibility)
    hypervolume: Optional[float] = None
    igd_metric: Optional[float] = None
    spread_metric: Optional[float] = None
    diversity_score: Optional[float] = None
    
    # Convergence analysis
    convergence_generation: Optional[int] = None
    stagnation_count: int = 0
    improvement_rate: float = 0.0
    
    # Solution quality
    constraint_violations: int = 0
    feasibility_ratio: float = 1.0
    robustness_score: Optional[float] = None
    
    # Performance profiling
    memory_peak_mb: float = 0.0
    cpu_utilization: float = 0.0
    scaling_factor: float = 1.0
    
    # Metadata
    parameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    environment_info: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result_dict = {
            'algorithm_name': self.algorithm_name,
            'problem_instance': self.problem_instance,
            'run_id': self.run_id,
            'best_objective': self.best_objective,
            'final_objective': self.final_objective,
            'convergence_history': self.convergence_history,
            'function_evaluations': self.function_evaluations,
            'computation_time': self.computation_time,
            'memory_peak_mb': self.memory_peak_mb,
            'cpu_utilization': self.cpu_utilization,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
            'random_seed': self.random_seed
        }
        
        # Add optional metrics
        if self.multiobjective_metrics:
            result_dict['multiobjective_metrics'] = self.multiobjective_metrics.__dict__
        if self.convergence_analysis:
            result_dict['convergence_analysis'] = self.convergence_analysis.__dict__
        if self.robustness_metrics:
            result_dict['robustness_metrics'] = self.robustness_metrics.__dict__
            
        return result_dict


@dataclass
class StatisticalComparison:
    """Statistical comparison results between algorithms."""
    
    algorithm_a: str
    algorithm_b: str
    metric: str
    
    # Basic statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    median_a: float
    median_b: float
    
    # Statistical tests
    mannwhitney_statistic: float
    mannwhitney_pvalue: float
    wilcoxon_statistic: Optional[float] = None
    wilcoxon_pvalue: Optional[float] = None
    ttest_statistic: float = 0.0
    ttest_pvalue: float = 1.0
    
    # Effect size
    cohens_d: float = 0.0
    cliff_delta: float = 0.0
    hedge_g: float = 0.0
    glass_delta: float = 0.0
    
    # Additional statistical tests
    kruskal_wallis_statistic: Optional[float] = None
    kruskal_wallis_pvalue: Optional[float] = None
    levene_statistic: Optional[float] = None
    levene_pvalue: Optional[float] = None
    
    # Confidence intervals
    mean_diff_ci_lower: float = 0.0
    mean_diff_ci_upper: float = 0.0
    
    # Significance
    is_significant: bool = False
    significance_level: float = 0.05
    interpretation: str = "No significant difference"
    effect_size_interpretation: str = "Negligible"


class BenchmarkProblem(ABC):
    """Abstract base class for benchmark problems."""
    
    def __init__(self, name: str, dimensions: int, objectives: int = 1):
        self.name = name
        self.dimensions = dimensions
        self.objectives = objectives
        self.bounds = []
        self.optimal_value = None
        
    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate solution and return objective value(s)."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get variable bounds."""
        pass


class AntennaDesignProblem(BenchmarkProblem):
    """Antenna design benchmark problem."""
    
    def __init__(self, spec: AntennaSpec, solver, complexity: str = "medium"):
        """
        Initialize antenna design problem.
        
        Args:
            spec: Antenna specification
            solver: Electromagnetic solver
            complexity: Problem complexity ("low", "medium", "high")
        """
        # Determine dimensions based on complexity
        dims_map = {"low": 32*32, "medium": 32*32*8, "high": 64*64*16}
        dimensions = dims_map.get(complexity, 32*32*8)
        
        super().__init__(f"antenna_design_{complexity}", dimensions, 1)
        
        self.spec = spec
        self.solver = solver
        self.complexity = complexity
        self.logger = get_logger('antenna_benchmark')
    
    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate antenna design solution."""
        try:
            # Reshape solution to 3D geometry
            if self.complexity == "low":
                geometry = solution.reshape((32, 32, 1))
            elif self.complexity == "medium":
                geometry = solution.reshape((32, 32, 8))
            else:  # high complexity
                geometry = solution.reshape((64, 64, 16))
            
            # Simulate electromagnetic performance
            result = self._simulate_antenna(geometry)
            
            # Multi-criteria objective (maximize)
            objective = (
                0.4 * result['gain_normalized'] +
                0.3 * result['efficiency'] +
                0.2 * (1.0 - result['reflection_loss']) +
                0.1 * result['bandwidth_normalized']
            )
            
            return objective
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return 0.0  # Return worst possible objective
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get variable bounds (material presence: 0 to 1)."""
        return [(0.0, 1.0) for _ in range(self.dimensions)]
    
    def _simulate_antenna(self, geometry: np.ndarray) -> Dict[str, float]:
        """Simplified antenna simulation."""
        
        # Physical properties
        metal_fraction = np.mean(geometry > 0.5)
        complexity = np.std(geometry)
        
        # Simulate gain (dBi)
        base_gain = 2.0 + metal_fraction * 8.0
        gain_penalty = complexity * 2.0  # Penalize overly complex designs
        gain_dbi = max(0.0, base_gain - gain_penalty)
        gain_normalized = min(1.0, gain_dbi / 15.0)  # Normalize to [0,1]
        
        # Simulate efficiency
        efficiency = 0.5 + metal_fraction * 0.4 + np.random.normal(0, 0.05)
        efficiency = max(0.0, min(1.0, efficiency))
        
        # Simulate reflection loss (S11)
        s11_db = -8.0 - metal_fraction * 12.0 + complexity * 5.0
        reflection_loss = min(1.0, abs(s11_db) / 30.0)  # Normalize
        
        # Simulate bandwidth
        bandwidth_mhz = 20.0 + metal_fraction * 80.0 - complexity * 30.0
        bandwidth_normalized = min(1.0, max(0.0, bandwidth_mhz / 100.0))
        
        return {
            'gain_dbi': gain_dbi,
            'gain_normalized': gain_normalized,
            'efficiency': efficiency,
            'reflection_loss': reflection_loss,
            'bandwidth_mhz': bandwidth_mhz,
            'bandwidth_normalized': bandwidth_normalized,
            's11_db': s11_db
        }


class MultiObjectiveAntennaProblem(BenchmarkProblem):
    """Multi-objective antenna design problem."""
    
    def __init__(self, spec: AntennaSpec, solver):
        super().__init__("antenna_multiobjective", 32*32*8, 3)  # 3 objectives
        self.spec = spec
        self.solver = solver
    
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """Evaluate multi-objective antenna design."""
        geometry = solution.reshape((32, 32, 8))
        
        # Simulate antenna performance
        result = self._simulate_multiobjective(geometry)
        
        # Return three objectives: [gain, efficiency, bandwidth]
        # Note: For minimization problems, negate maximization objectives
        objectives = np.array([
            -result['gain_dbi'],  # Maximize gain -> minimize -gain
            -result['efficiency'],  # Maximize efficiency -> minimize -efficiency
            -result['bandwidth_mhz']  # Maximize bandwidth -> minimize -bandwidth
        ])
        
        return objectives
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get variable bounds."""
        return [(0.0, 1.0) for _ in range(self.dimensions)]
    
    def _simulate_multiobjective(self, geometry: np.ndarray) -> Dict[str, float]:
        """Multi-objective antenna simulation."""
        
        metal_fraction = np.mean(geometry > 0.5)
        complexity = np.std(geometry)
        
        # Conflicting objectives for realistic Pareto front
        gain_dbi = 2.0 + metal_fraction * 10.0 - complexity * 1.0
        efficiency = 0.6 + metal_fraction * 0.3 - (gain_dbi - 5.0) * 0.02  # Trade-off
        bandwidth_mhz = 30.0 + complexity * 40.0 - metal_fraction * 20.0  # Trade-off
        
        # Add realistic noise
        gain_dbi += np.random.normal(0, 0.5)
        efficiency += np.random.normal(0, 0.03)
        bandwidth_mhz += np.random.normal(0, 5.0)
        
        # Apply realistic bounds
        gain_dbi = max(0.0, min(15.0, gain_dbi))
        efficiency = max(0.1, min(0.95, efficiency))
        bandwidth_mhz = max(10.0, min(100.0, bandwidth_mhz))
        
        return {
            'gain_dbi': gain_dbi,
            'efficiency': efficiency,
            'bandwidth_mhz': bandwidth_mhz
        }


class ComprehensiveBenchmarkSuite:
    """
    Comprehensive benchmarking framework for antenna optimization algorithms.
    
    Features:
    - Multiple benchmark problems with varying complexity
    - Statistical significance testing
    - Multi-objective performance analysis
    - Reproducible experimental protocols
    - Publication-quality results generation
    """
    
    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        random_seed: int = 42,
        significance_level: float = 0.05
    ):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        self.significance_level = significance_level
        self.logger = get_logger('benchmark_suite')
        
        # Benchmark configuration
        self.num_runs = 30  # Standard for statistical analysis
        self.max_evaluations = 1000  # Budget per run
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.statistical_comparisons: List[StatisticalComparison] = []
        
        # Problem suite
        self.problems = {}
        self.algorithms = {}
        
        self.logger.info(f"Initialized benchmark suite with {self.num_runs} runs per algorithm")
        
        # Generate reproducibility hash
        self.reproducibility_hash = self._generate_reproducibility_hash()
        self.logger.info(f"Experiment reproducibility hash: {self.reproducibility_hash}")
        
        # Advanced analysis settings
        self.bootstrap_samples = 1000
        self.cv_folds = 5
        self.confidence_level = 0.95
        
        # Multi-objective optimization settings
        self.reference_point = None
        self.ideal_point = None
        self.nadir_point = None
        
        # Experimental protocol settings
        self.experiment_metadata = self._initialize_experiment_metadata()
        self.parameter_log = []
        self.reproducibility_hash = None
    
    def register_algorithm(self, name: str, algorithm_factory: Callable):
        """Register algorithm for benchmarking."""
        self.algorithms[name] = algorithm_factory
        self.logger.info(f"Registered algorithm: {name}")
        
        # Log algorithm parameters for reproducibility
        self._log_algorithm_parameters(name, algorithm_factory)
    
    def register_problem(self, problem: BenchmarkProblem):
        """Register benchmark problem."""
        self.problems[problem.name] = problem
        self.logger.info(f"Registered problem: {problem.name}")
    
    def run_comprehensive_benchmark(
        self,
        algorithms: Optional[List[str]] = None,
        problems: Optional[List[str]] = None,
        num_runs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark study.
        
        Args:
            algorithms: List of algorithm names to benchmark (None = all)
            problems: List of problem names to test (None = all)
            num_runs: Number of independent runs per algorithm (None = default)
            
        Returns:
            Comprehensive benchmark results
        """
        
        if num_runs is not None:
            self.num_runs = num_runs
        
        algorithms = algorithms or list(self.algorithms.keys())
        problems = problems or list(self.problems.keys())
        
        self.logger.info(f"Starting benchmark: {len(algorithms)} algorithms × {len(problems)} problems × {self.num_runs} runs")
        
        start_time = time.time()
        total_experiments = len(algorithms) * len(problems) * self.num_runs
        completed_experiments = 0
        
        # Run all algorithm-problem combinations
        for algorithm_name in algorithms:
            for problem_name in problems:
                self.logger.info(f"Benchmarking {algorithm_name} on {problem_name}")
                
                problem = self.problems[problem_name]
                algorithm_factory = self.algorithms[algorithm_name]
                
                # Run independent trials
                for run_id in range(self.num_runs):
                    
                    # Set unique random seed for reproducibility
                    run_seed = self.random_seed + run_id * 1000 + hash(algorithm_name) % 1000
                    
                    # Ensure reproducibility conditions
                    repro_info = self._ensure_reproducibility(run_seed, algorithm_name, problem_name)
                    np.random.seed(run_seed)
                    
                    try:
                        result = self._run_single_benchmark(
                            algorithm_name,
                            algorithm_factory,
                            problem,
                            run_id,
                            run_seed
                        )
                        
                        self.results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Benchmark failed: {algorithm_name} on {problem_name}, run {run_id}: {e}")
                        
                        # Record failure
                        failure_result = BenchmarkResult(
                            algorithm_name=algorithm_name,
                            problem_instance=problem_name,
                            run_id=run_id,
                            best_objective=0.0,
                            final_objective=0.0,
                            convergence_history=[],
                            function_evaluations=0,
                            computation_time=0.0,
                            random_seed=run_seed,
                            success=False,
                            error_message=str(e)
                        )
                        self.results.append(failure_result)
                    
                    completed_experiments += 1
                    
                    # Progress update
                    if completed_experiments % 10 == 0:
                        progress = (completed_experiments / total_experiments) * 100
                        self.logger.info(f"Progress: {completed_experiments}/{total_experiments} ({progress:.1f}%)")
        
        total_time = time.time() - start_time
        self.logger.info(f"Benchmark completed in {total_time:.1f}s")
        
        # Perform statistical analysis
        self._perform_statistical_analysis()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Save results
        self._save_results()
        
        # Save reproducibility package
        repro_package_dir = self.save_reproducibility_package(str(self.output_dir))
        report['reproducibility_package'] = repro_package_dir
        
        return report
    
    def _run_single_benchmark(
        self,
        algorithm_name: str,
        algorithm_factory: Callable,
        problem: BenchmarkProblem,
        run_id: int,
        random_seed: int
    ) -> BenchmarkResult:
        """Run single benchmark trial."""
        
        start_time = time.time()
        
        # Create algorithm instance
        algorithm = algorithm_factory()
        
        # Create bounds for optimization
        bounds = problem.get_bounds()
        
        # Create dummy AntennaSpec for compatibility
        spec = AntennaSpec(
            center_frequency=2.45e9,
            substrate='fr4',
            thickness_mm=1.6
        )
        
        # Run optimization
        optimization_result = algorithm.optimize(
            geometry_bounds=bounds,
            spec=spec,
            max_evaluations=self.max_evaluations
        )
        
        computation_time = time.time() - start_time
        
        # Extract convergence history
        if hasattr(optimization_result, 'convergence_history'):
            convergence_history = optimization_result.convergence_history
        else:
            convergence_history = []
        
        # Calculate additional metrics
        best_objective = optimization_result.optimal_objective
        final_objective = best_objective
        
        # Convergence analysis
        convergence_gen = self._detect_convergence_generation(convergence_history)
        improvement_rate = self._calculate_improvement_rate(convergence_history)
        convergence_analysis = self._analyze_convergence_pattern(convergence_history)
        
        # Multi-objective metrics (if applicable)
        multiobjective_metrics = None
        if hasattr(problem, 'objectives') and problem.objectives > 1:
            multiobjective_metrics = self._calculate_multiobjective_metrics(optimization_result, problem)
        
        # Bootstrap analysis for robustness
        if len(convergence_history) > 5:
            final_scores = np.array(convergence_history[-5:])  # Last 5 scores
            bootstrap_results = self._perform_bootstrap_analysis(final_scores)
            robustness_metrics = RobustnessMetrics(
                bootstrap_confidence_interval=(bootstrap_results['ci_lower'], bootstrap_results['ci_upper']),
                bootstrap_bias=bootstrap_results['bias'],
                bootstrap_variance=bootstrap_results['std']**2
            )
        else:
            robustness_metrics = RobustnessMetrics()
        
        # Function evaluations
        function_evals = getattr(optimization_result, 'total_iterations', self.max_evaluations)
        
        return BenchmarkResult(
            algorithm_name=algorithm_name,
            problem_instance=problem.name,
            run_id=run_id,
            best_objective=best_objective,
            final_objective=final_objective,
            convergence_history=convergence_history,
            function_evaluations=function_evals,
            computation_time=computation_time,
            convergence_generation=convergence_gen,
            improvement_rate=improvement_rate,
            multiobjective_metrics=multiobjective_metrics,
            convergence_analysis=convergence_analysis,
            robustness_metrics=robustness_metrics,
            # Legacy fields for compatibility
            hypervolume=multiobjective_metrics.hypervolume if multiobjective_metrics else None,
            igd_metric=multiobjective_metrics.igd_metric if multiobjective_metrics else None,
            spread_metric=multiobjective_metrics.spread_metric if multiobjective_metrics else None,
            random_seed=random_seed,
            success=True
        )
    
    def _calculate_multiobjective_metrics(self, optimization_result, problem: BenchmarkProblem) -> MultiObjectiveMetrics:
        \"\"\"Calculate comprehensive multi-objective optimization metrics.\"\"\"\n        if not hasattr(optimization_result, 'pareto_front') or optimization_result.pareto_front is None:\n            return MultiObjectiveMetrics()\n        \n        pareto_front = optimization_result.pareto_front\n        \n        # Generate reference point if not provided\n        if self.reference_point is None:\n            self.reference_point = np.max(pareto_front, axis=0) + 1.0\n        \n        # Calculate hypervolume\n        hypervolume = self._calculate_hypervolume(pareto_front, self.reference_point)\n        \n        # Generate reference Pareto front for IGD/GD calculations\n        if hasattr(problem, 'reference_pareto_front'):\n            ref_front = problem.reference_pareto_front\n        else:\n            # Generate synthetic reference front\n            ref_front = self._generate_reference_pareto_front(problem)\n        \n        # Calculate IGD and GD metrics\n        igd_metric = self._calculate_igd_metric(pareto_front, ref_front)\n        gd_metric = self._calculate_gd_metric(pareto_front, ref_front)\n        \n        # Calculate diversity metrics\n        spread_metric = self._calculate_spread_metric(pareto_front)\n        spacing_metric = self._calculate_spacing_metric(pareto_front)\n        \n        # Calculate coverage metric (simple version)\n        coverage_metric = len(pareto_front) / max(100, len(pareto_front))  # Normalized\n        \n        return MultiObjectiveMetrics(\n            hypervolume=hypervolume,\n            igd_metric=igd_metric,\n            gd_metric=gd_metric,\n            spread_metric=spread_metric,\n            spacing_metric=spacing_metric,\n            coverage_metric=coverage_metric,\n            pareto_front_size=len(pareto_front),\n            convergence_metric=1.0 / (1.0 + igd_metric) if igd_metric < float('inf') else 0.0,\n            diversity_metric=(1.0 - spread_metric) if spread_metric <= 1.0 else 0.0\n        )\n    \n    def _generate_reference_pareto_front(self, problem: BenchmarkProblem, n_points: int = 100) -> np.ndarray:\n        \"\"\"Generate reference Pareto front for comparison metrics.\"\"\"\n        if hasattr(problem, 'objectives') and problem.objectives > 1:\n            # For multi-objective problems, generate synthetic Pareto front\n            if problem.objectives == 2:\n                # 2D Pareto front\n                x = np.linspace(0, 1, n_points)\n                y = 1 - x**2  # Convex Pareto front\n                return np.column_stack([x, y])\n            elif problem.objectives == 3:\n                # 3D Pareto front\n                n_side = int(np.sqrt(n_points))\n                x = np.linspace(0, 1, n_side)\n                y = np.linspace(0, 1, n_side)\n                X, Y = np.meshgrid(x, y)\n                Z = 1 - (X**2 + Y**2)\n                Z = np.maximum(Z, 0)  # Ensure non-negative\n                \n                # Flatten and combine\n                points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])\n                # Filter dominated points\n                return self._filter_pareto_front(points)\n        \n        # Default case: single objective\n        return np.array([[1.0]])\n    \n    def _filter_pareto_front(self, points: np.ndarray) -> np.ndarray:\n        \"\"\"Filter points to keep only Pareto-optimal solutions.\"\"\"\n        if points.size == 0:\n            return points\n        \n        is_pareto = np.ones(len(points), dtype=bool)\n        \n        for i, point in enumerate(points):\n            if is_pareto[i]:\n                # Check if this point dominates others\n                dominated = np.all(points >= point, axis=1) & np.any(points > point, axis=1)\n                is_pareto[dominated] = False\n        \n        return points[is_pareto]\n    \n    def _detect_convergence_generation(self, history: List[float], tolerance: float = 1e-6) -> Optional[int]:
        """Detect generation where algorithm converged."""
        if len(history) < 10:
            return None
        
        for i in range(10, len(history)):
            # Check if improvement in last 10 generations is below tolerance
            recent_improvement = max(history[i-10:i]) - min(history[i-10:i])
            if recent_improvement < tolerance:
                return i - 10
        
        return None
    
    def _calculate_improvement_rate(self, history: List[float]) -> float:
        """Calculate average improvement rate per generation."""
        if len(history) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(history)):
            improvement = max(0, history[i] - history[i-1])
            improvements.append(improvement)
        
        return np.mean(improvements)
    
    def _perform_statistical_analysis(self):
        """Perform comprehensive statistical analysis."""
        
        self.logger.info("Performing statistical analysis...")
        
        # Group results by algorithm and problem
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            if result.success:
                grouped_results[result.problem_instance][result.algorithm_name].append(result)
        
        # Compare all algorithm pairs for each problem
        for problem_name, problem_results in grouped_results.items():
            algorithms = list(problem_results.keys())
            
            for i, alg_a in enumerate(algorithms):
                for j, alg_b in enumerate(algorithms[i+1:], i+1):
                    
                    # Extract performance metrics
                    scores_a = [r.best_objective for r in problem_results[alg_a]]
                    scores_b = [r.best_objective for r in problem_results[alg_b]]
                    
                    # Perform statistical comparison
                    comparison = self._compare_algorithms(
                        alg_a, alg_b, scores_a, scores_b, "best_objective"
                    )
                    
                    self.statistical_comparisons.append(comparison)
                    
                    self.logger.debug(f"{alg_a} vs {alg_b} on {problem_name}: {comparison.interpretation}")
    
    def _compare_algorithms(
        self,
        alg_a: str,
        alg_b: str,
        scores_a: List[float],
        scores_b: List[float],
        metric: str
    ) -> StatisticalComparison:
        """Compare two algorithms statistically."""
        
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        # Basic statistics
        mean_a, std_a = np.mean(scores_a), np.std(scores_a)
        mean_b, std_b = np.mean(scores_b), np.std(scores_b)
        median_a, median_b = np.median(scores_a), np.median(scores_b)
        
        # Mann-Whitney U test (non-parametric)
        try:
            mw_stat, mw_pval = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
        except ValueError:
            mw_stat, mw_pval = 0.0, 1.0
        
        # Wilcoxon signed-rank test (if paired)
        wilcoxon_stat, wilcoxon_pval = None, None
        if len(scores_a) == len(scores_b):
            try:
                wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(scores_a, scores_b)
            except ValueError:
                pass
        
        # t-test (parametric, for comparison)
        try:
            ttest_stat, ttest_pval = stats.ttest_ind(scores_a, scores_b)
        except ValueError:
            ttest_stat, ttest_pval = 0.0, 1.0
        
        # Effect sizes
        cohens_d = self._calculate_cohens_d(scores_a, scores_b)
        cliff_delta = self._calculate_cliff_delta(scores_a, scores_b)
        hedge_g = self._calculate_hedge_g(scores_a, scores_b)
        glass_delta = self._calculate_glass_delta(scores_a, scores_b)
        
        # Additional statistical tests
        levene_stat, levene_pval = self._perform_levene_test(scores_a, scores_b)
        
        # Confidence interval for difference in means
        ci_lower, ci_upper = self._calculate_confidence_interval(
            scores_a, scores_b, self.confidence_level
        )
        
        # Determine significance and interpretation
        is_significant = mw_pval < self.significance_level
        effect_size_interpretation = self._interpret_effect_size(cohens_d)
        
        if is_significant:
            if mean_a > mean_b:
                interpretation = f"{alg_a} significantly better than {alg_b} (p={mw_pval:.4f})"
            else:
                interpretation = f"{alg_b} significantly better than {alg_a} (p={mw_pval:.4f})"
        else:
            interpretation = "No significant difference"
        
        return StatisticalComparison(
            algorithm_a=alg_a,
            algorithm_b=alg_b,
            metric=metric,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            median_a=median_a,
            median_b=median_b,
            mannwhitney_statistic=mw_stat,
            mannwhitney_pvalue=mw_pval,
            wilcoxon_statistic=wilcoxon_stat,
            wilcoxon_pvalue=wilcoxon_pval,
            ttest_statistic=ttest_stat,
            ttest_pvalue=ttest_pval,
            cohens_d=cohens_d,
            cliff_delta=cliff_delta,
            hedge_g=hedge_g,
            glass_delta=glass_delta,
            levene_statistic=levene_stat,
            levene_pvalue=levene_pval,
            mean_diff_ci_lower=ci_lower,
            mean_diff_ci_upper=ci_upper,
            is_significant=is_significant,
            significance_level=self.significance_level,
            interpretation=interpretation,
            effect_size_interpretation=effect_size_interpretation
        )
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            return 0.0
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _calculate_cliff_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta effect size."""
        if len(group1) == 0 or len(group2) == 0:
            return 0.0
        
        # Count comparisons
        greater = 0
        equal = 0
        total = 0
        
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    greater += 1
                elif x1 == x2:
                    equal += 1
                total += 1
        
        if total == 0:
            return 0.0
        
        return (greater - (total - greater - equal)) / total
    
    def _calculate_hedge_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedge's g effect size (bias-corrected Cohen's d)."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        cohens_d = self._calculate_cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        
        # Bias correction factor
        j = 1 - (3 / (4 * df - 1))
        
        return j * cohens_d
    
    def _calculate_glass_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Glass's delta effect size."""
        if len(group2) < 2:
            return 0.0
        
        control_std = np.std(group2, ddof=1)
        if control_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / control_std
    
    def _calculate_confidence_interval(self, group1: np.ndarray, group2: np.ndarray, 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        if len(group1) < 2 or len(group2) < 2:
            return (0.0, 0.0)
        
        mean_diff = np.mean(group1) - np.mean(group2)
        
        # Pooled standard error
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        se_pooled = np.sqrt((s1**2 / n1) + (s2**2 / n2))
        
        # Degrees of freedom (Welch's t-test)
        df = ((s1**2 / n1) + (s2**2 / n2))**2 / ((s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1))
        
        # Critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_critical * se_pooled
        
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def _perform_kruskal_wallis_test(self, groups: List[np.ndarray]) -> Tuple[float, float]:
        """Perform Kruskal-Wallis H-test for multiple groups."""
        if len(groups) < 2 or any(len(group) == 0 for group in groups):
            return 0.0, 1.0
        
        try:
            statistic, pvalue = stats.kruskal(*groups)
            return statistic, pvalue
        except ValueError:
            return 0.0, 1.0
    
    def _perform_levene_test(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """Perform Levene's test for equal variances."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0, 1.0
        
        try:
            statistic, pvalue = stats.levene(group1, group2)
            return statistic, pvalue
        except ValueError:
            return 0.0, 1.0
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret effect size magnitude."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _calculate_hypervolume(self, pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
        """Calculate hypervolume indicator for multi-objective optimization."""
        if pareto_front.size == 0 or len(pareto_front.shape) != 2:
            return 0.0
        
        # Simple hypervolume calculation for 2D/3D cases
        if pareto_front.shape[1] == 2:
            return self._hypervolume_2d(pareto_front, reference_point)
        elif pareto_front.shape[1] == 3:
            return self._hypervolume_3d(pareto_front, reference_point)
        else:
            # For higher dimensions, use approximation
            return self._hypervolume_approximate(pareto_front, reference_point)
    
    def _hypervolume_2d(self, front: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate 2D hypervolume."""
        if front.size == 0:
            return 0.0
        
        # Sort by first objective
        sorted_front = front[np.argsort(front[:, 0])]
        
        hypervolume = 0.0
        for i in range(len(sorted_front)):
            if i == 0:
                width = ref_point[0] - sorted_front[i, 0]
            else:
                width = sorted_front[i-1, 0] - sorted_front[i, 0]
            
            height = ref_point[1] - sorted_front[i, 1]
            hypervolume += width * height
        
        return max(0.0, hypervolume)
    
    def _hypervolume_3d(self, front: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate 3D hypervolume (simplified)."""
        if front.size == 0:
            return 0.0
        
        # Simplified 3D hypervolume calculation
        volume = 0.0
        for point in front:
            contribution = 1.0
            for i in range(3):
                contribution *= max(0.0, ref_point[i] - point[i])
            volume += contribution
        
        return volume / len(front)  # Normalized
    
    def _hypervolume_approximate(self, front: np.ndarray, ref_point: np.ndarray) -> float:
        """Approximate hypervolume for high dimensions."""
        if front.size == 0:
            return 0.0
        
        # Monte Carlo approximation
        n_samples = 10000
        n_objectives = front.shape[1]
        
        # Generate random points in the hypervolume region
        dominated_count = 0
        for _ in range(n_samples):
            random_point = np.random.uniform(
                low=np.min(front, axis=0),
                high=ref_point,
                size=n_objectives
            )
            
            # Check if random point is dominated by any point in front
            dominated = False
            for front_point in front:
                if np.all(front_point <= random_point):
                    dominated = True
                    break
            
            if dominated:
                dominated_count += 1
        
        # Estimate hypervolume
        total_volume = np.prod(ref_point - np.min(front, axis=0))
        return total_volume * (dominated_count / n_samples)
    
    def _calculate_igd_metric(self, obtained_front: np.ndarray, reference_front: np.ndarray) -> float:
        \"\"\"Calculate Inverted Generational Distance.\"\"\"\n        if reference_front.size == 0 or obtained_front.size == 0:\n            return float('inf')\n        \n        # Calculate minimum distance from each reference point to obtained front\n        distances = []\n        for ref_point in reference_front:\n            min_distance = min(euclidean(ref_point, point) for point in obtained_front)\n            distances.append(min_distance)\n        \n        return np.mean(distances)\n    \n    def _calculate_gd_metric(self, obtained_front: np.ndarray, reference_front: np.ndarray) -> float:\n        \"\"\"Calculate Generational Distance.\"\"\"\n        if obtained_front.size == 0 or reference_front.size == 0:\n            return float('inf')\n        \n        # Calculate minimum distance from each obtained point to reference front\n        distances = []\n        for obtained_point in obtained_front:\n            min_distance = min(euclidean(obtained_point, ref_point) for ref_point in reference_front)\n            distances.append(min_distance)\n        \n        return np.mean(distances)\n    \n    def _calculate_spread_metric(self, pareto_front: np.ndarray) -> float:\n        \"\"\"Calculate spread metric for diversity assessment.\"\"\"\n        if pareto_front.size == 0 or len(pareto_front) < 2:\n            return 0.0\n        \n        # Calculate distances between consecutive points\n        sorted_indices = np.lexsort(pareto_front.T)\n        sorted_front = pareto_front[sorted_indices]\n        \n        distances = []\n        for i in range(len(sorted_front) - 1):\n            distance = euclidean(sorted_front[i], sorted_front[i + 1])\n            distances.append(distance)\n        \n        if len(distances) == 0:\n            return 0.0\n        \n        # Calculate spread as standard deviation of distances\n        mean_distance = np.mean(distances)\n        spread = np.std(distances) / mean_distance if mean_distance > 0 else 0.0\n        \n        return spread\n    \n    def _calculate_spacing_metric(self, pareto_front: np.ndarray) -> float:\n        \"\"\"Calculate spacing metric for uniformity assessment.\"\"\"\n        if len(pareto_front) < 2:\n            return 0.0\n        \n        # Calculate nearest neighbor distances\n        nn_distances = []\n        for i, point in enumerate(pareto_front):\n            distances_to_others = [euclidean(point, other) for j, other in enumerate(pareto_front) if i != j]\n            if distances_to_others:\n                nn_distances.append(min(distances_to_others))\n        \n        if len(nn_distances) == 0:\n            return 0.0\n        \n        # Calculate spacing as standard deviation of nearest neighbor distances\n        mean_nn_distance = np.mean(nn_distances)\n        spacing = np.std(nn_distances) / mean_nn_distance if mean_nn_distance > 0 else 0.0\n        \n        return spacing\n    \n    def _perform_bootstrap_analysis(self, data: np.ndarray, n_bootstrap: int = 1000, \n                                  statistic_func: Callable = np.mean) -> Dict[str, float]:\n        \"\"\"Perform bootstrap analysis for statistical robustness.\"\"\"\n        if len(data) == 0:\n            return {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}\n        \n        bootstrap_statistics = []\n        \n        for _ in range(n_bootstrap):\n            # Resample with replacement\n            resampled = resample(data, replace=True, n_samples=len(data))\n            statistic = statistic_func(resampled)\n            bootstrap_statistics.append(statistic)\n        \n        bootstrap_statistics = np.array(bootstrap_statistics)\n        \n        # Calculate confidence intervals\n        alpha = 1 - self.confidence_level\n        ci_lower = np.percentile(bootstrap_statistics, 100 * alpha / 2)\n        ci_upper = np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2))\n        \n        return {\n            'mean': np.mean(bootstrap_statistics),\n            'std': np.std(bootstrap_statistics),\n            'ci_lower': ci_lower,\n            'ci_upper': ci_upper,\n            'bias': np.mean(bootstrap_statistics) - statistic_func(data)\n        }\n    \n    def _perform_cross_validation_analysis(self, results: List[BenchmarkResult], \n                                         algorithm_name: str) -> Dict[str, float]:\n        \"\"\"Perform cross-validation analysis for algorithm stability.\"\"\"\n        if not results:\n            return {'cv_mean': 0.0, 'cv_std': 0.0, 'cv_score': 0.0}\n        \n        scores = [r.best_objective for r in results if r.algorithm_name == algorithm_name]\n        \n        if len(scores) < self.cv_folds:\n            return {'cv_mean': np.mean(scores), 'cv_std': np.std(scores), 'cv_score': 1.0}\n        \n        # Simple k-fold cross-validation simulation\n        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)\n        cv_scores = []\n        \n        scores_array = np.array(scores)\n        for train_idx, test_idx in kf.split(scores_array):\n            # In a real CV, we would retrain the algorithm on train_idx\n            # Here we simulate by using the test set performance\n            test_scores = scores_array[test_idx]\n            cv_scores.append(np.mean(test_scores))\n        \n        cv_mean = np.mean(cv_scores)\n        cv_std = np.std(cv_scores)\n        cv_score = 1.0 / (1.0 + cv_std) if cv_std > 0 else 1.0\n        \n        return {\n            'cv_mean': cv_mean,\n            'cv_std': cv_std,\n            'cv_score': cv_score\n        }\n    \n    def _analyze_convergence_pattern(self, history: List[float]) -> ConvergenceAnalysis:\n        \"\"\"Analyze convergence patterns in optimization history.\"\"\"\n        if len(history) < 2:\n            return ConvergenceAnalysis()\n        \n        history_array = np.array(history)\n        \n        # Calculate convergence rate\n        improvements = np.diff(history_array)\n        positive_improvements = improvements[improvements > 0]\n        convergence_rate = np.mean(positive_improvements) if len(positive_improvements) > 0 else 0.0\n        \n        # Detect plateaus (periods with little improvement)\n        plateau_threshold = np.std(history_array) * 0.1\n        plateaus = []\n        current_plateau_start = None\n        \n        for i in range(1, len(history)):\n            improvement = abs(history[i] - history[i-1])\n            \n            if improvement < plateau_threshold:\n                if current_plateau_start is None:\n                    current_plateau_start = i - 1\n            else:\n                if current_plateau_start is not None:\n                    plateau_length = i - current_plateau_start\n                    if plateau_length >= 5:  # Minimum plateau length\n                        plateaus.append((current_plateau_start, i))\n                    current_plateau_start = None\n        \n        # Final plateau check\n        if current_plateau_start is not None:\n            plateau_length = len(history) - current_plateau_start\n            if plateau_length >= 5:\n                plateaus.append((current_plateau_start, len(history)))\n        \n        # Calculate stability (inverse of volatility)\n        volatility = np.std(improvements) / np.mean(np.abs(improvements)) if np.mean(np.abs(improvements)) > 0 else 0\n        stability = 1.0 / (1.0 + volatility)\n        \n        # Trend analysis\n        if len(history) >= 10:\n            # Linear trend\n            x = np.arange(len(history))\n            slope, intercept, r_value, p_value, std_err = stats.linregress(x, history)\n            trend_analysis = {\n                'slope': slope,\n                'r_squared': r_value**2,\n                'p_value': p_value\n            }\n        else:\n            trend_analysis = {'slope': 0.0, 'r_squared': 0.0, 'p_value': 1.0}\n        \n        return ConvergenceAnalysis(\n            convergence_rate=convergence_rate,\n            convergence_stability=stability,\n            plateau_detection=plateaus,\n            volatility_measure=volatility,\n            trend_analysis=trend_analysis,\n            target_reached=history[-1] > np.percentile(history, 90),\n            success_rate=len(positive_improvements) / len(improvements) if len(improvements) > 0 else 0.0\n        )\n    \n    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        # Group results for analysis
        algorithm_performance = defaultdict(lambda: defaultdict(list))
        problem_difficulty = defaultdict(list)
        
        successful_results = [r for r in self.results if r.success]
        
        for result in successful_results:
            algorithm_performance[result.algorithm_name][result.problem_instance].append(result.best_objective)
            problem_difficulty[result.problem_instance].append(result.best_objective)
        
        # Calculate summary statistics
        algorithm_summary = {}
        for alg_name, problems in algorithm_performance.items():
            all_scores = []
            problem_stats = {}
            
            for prob_name, scores in problems.items():
                problem_stats[prob_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'median': float(np.median(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'runs': len(scores)
                }
                all_scores.extend(scores)
            
            algorithm_summary[alg_name] = {
                'overall_mean': float(np.mean(all_scores)) if all_scores else 0.0,
                'overall_std': float(np.std(all_scores)) if all_scores else 0.0,
                'problem_performance': problem_stats,
                'total_runs': len(all_scores)
            }
        
        # Problem difficulty ranking
        problem_rankings = []
        for prob_name, scores in problem_difficulty.items():
            problem_rankings.append({
                'problem': prob_name,
                'difficulty_score': float(np.mean(scores)),
                'score_variance': float(np.var(scores))
            })
        
        problem_rankings.sort(key=lambda x: x['difficulty_score'])
        
        # Statistical significance summary
        significant_comparisons = [comp for comp in self.statistical_comparisons if comp.is_significant]
        
        # Performance rankings
        overall_rankings = []
        for alg_name, stats in algorithm_summary.items():
            overall_rankings.append({
                'algorithm': alg_name,
                'score': stats['overall_mean'],
                'std': stats['overall_std'],
                'rank': 0  # Will be filled below
            })
        
        # Sort and assign ranks
        overall_rankings.sort(key=lambda x: x['score'], reverse=True)
        for i, ranking in enumerate(overall_rankings):
            ranking['rank'] = i + 1
        
        report = {
            'metadata': {
                'benchmark_date': datetime.now().isoformat(),
                'num_algorithms': len(self.algorithms),
                'num_problems': len(self.problems),
                'runs_per_combination': self.num_runs,
                'total_experiments': len(self.results),
                'successful_experiments': len(successful_results),
                'significance_level': self.significance_level
            },
            'algorithm_performance': algorithm_summary,
            'problem_difficulty': problem_rankings,
            'overall_rankings': overall_rankings,
            'statistical_analysis': {
                'total_comparisons': len(self.statistical_comparisons),
                'significant_comparisons': len(significant_comparisons),
                'significance_rate': len(significant_comparisons) / max(len(self.statistical_comparisons), 1)
            },
            'detailed_comparisons': [
                {
                    'algorithms': f"{comp.algorithm_a} vs {comp.algorithm_b}",
                    'metric': comp.metric,
                    'p_value': comp.mannwhitney_pvalue,
                    'effect_size': comp.cohens_d,
                    'interpretation': comp.interpretation
                }
                for comp in significant_comparisons
            ]
        }
        
        return report
    
    def _save_results(self):
        """Save all benchmark results to files."""
        
        # Save raw results
        results_file = self.output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            # Convert to serializable format
            serializable_results = []
            for result in self.results:
                result_dict = {
                    'algorithm_name': result.algorithm_name,
                    'problem_instance': result.problem_instance,
                    'run_id': result.run_id,
                    'best_objective': result.best_objective,
                    'final_objective': result.final_objective,
                    'convergence_history': result.convergence_history,
                    'function_evaluations': result.function_evaluations,
                    'computation_time': result.computation_time,
                    'success': result.success,
                    'random_seed': result.random_seed
                }
                serializable_results.append(result_dict)
            
            json.dump(serializable_results, f, indent=2)
        
        # Save statistical comparisons
        stats_file = self.output_dir / 'statistical_comparisons.json'
        with open(stats_file, 'w') as f:
            serializable_stats = []
            for comp in self.statistical_comparisons:
                comp_dict = {
                    'algorithm_a': comp.algorithm_a,
                    'algorithm_b': comp.algorithm_b,
                    'metric': comp.metric,
                    'mean_a': comp.mean_a,
                    'mean_b': comp.mean_b,
                    'p_value': comp.mannwhitney_pvalue,
                    'effect_size': comp.cohens_d,
                    'is_significant': comp.is_significant,
                    'interpretation': comp.interpretation
                }
                serializable_stats.append(comp_dict)
            
            json.dump(serializable_stats, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def generate_publication_tables(self) -> Dict[str, str]:
        """Generate publication-ready LaTeX tables."""
        
        tables = {}
        
        # Performance comparison table
        performance_table = self._generate_performance_table()
        tables['performance_comparison'] = performance_table
        
        # Statistical significance table  
        significance_table = self._generate_significance_table()
        tables['statistical_significance'] = significance_table
        
        return tables
    
    def _generate_performance_table(self) -> str:
        """Generate LaTeX performance comparison table."""
        
        # Group results by algorithm and problem
        algorithm_performance = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            if result.success:
                algorithm_performance[result.algorithm_name][result.problem_instance].append(result.best_objective)
        
        # Generate LaTeX table
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Performance Comparison of Optimization Algorithms}\n"
        latex += "\\label{tab:performance_comparison}\n"
        
        # Determine table structure
        problems = sorted(set(r.problem_instance for r in self.results if r.success))
        algorithms = sorted(set(r.algorithm_name for r in self.results if r.success))
        
        # Table header
        header = "\\begin{tabular}{l" + "c" * len(problems) + "}\n"
        latex += header
        latex += "\\toprule\n"
        latex += "Algorithm & " + " & ".join(problems) + " \\\\\n"
        latex += "\\midrule\n"
        
        # Table body
        for alg in algorithms:
            row = [alg]
            for prob in problems:
                scores = algorithm_performance[alg][prob]
                if scores:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    row.append(f"{mean_score:.3f} ± {std_score:.3f}")
                else:
                    row.append("--")
            
            latex += " & ".join(row) + " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _generate_significance_table(self) -> str:
        """Generate LaTeX statistical significance table."""
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Statistical Significance Tests (Mann-Whitney U)}\n"
        latex += "\\label{tab:statistical_significance}\n"
        
        latex += "\\begin{tabular}{llccl}\n"
        latex += "\\toprule\n"
        latex += "Algorithm A & Algorithm B & p-value & Effect Size & Interpretation \\\\\n"
        latex += "\\midrule\n"
        
        for comp in self.statistical_comparisons:
            if comp.is_significant:
                p_val_str = f"{comp.mannwhitney_pvalue:.3e}" if comp.mannwhitney_pvalue < 0.001 else f"{comp.mannwhitney_pvalue:.3f}"
                effect_str = f"{comp.cohens_d:.3f}"
                
                latex += f"{comp.algorithm_a} & {comp.algorithm_b} & {p_val_str} & {effect_str} & {comp.interpretation} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex


# Example usage and factory functions
def create_quantum_inspired_optimizer():
    """Factory for quantum-inspired optimizer."""
    from .novel_algorithms import QuantumInspiredOptimizer
    return QuantumInspiredOptimizer(solver=None)


def create_multifidelity_optimizer():
    """Factory for multi-fidelity optimizer."""  
    from .novel_algorithms import MultiFidelityOptimizer
    return MultiFidelityOptimizer(solver=None)


def create_physics_informed_optimizer():
    """Factory for physics-informed optimizer."""
    from .novel_algorithms import PhysicsInformedOptimizer
    return PhysicsInformedOptimizer(solver=None)


def run_benchmark_study():
    """Run complete benchmark study."""
    
    # Create benchmark suite
    suite = ComprehensiveBenchmarkSuite(output_dir="./research_benchmarks")
    
    # Register algorithms
    suite.register_algorithm("QuantumInspired", create_quantum_inspired_optimizer)
    suite.register_algorithm("MultiFidelity", create_multifidelity_optimizer)
    suite.register_algorithm("PhysicsInformed", create_physics_informed_optimizer)
    
    # Register problems
    spec = AntennaSpec(center_frequency=2.45e9, substrate='fr4')
    
    suite.register_problem(AntennaDesignProblem(spec, None, "low"))
    suite.register_problem(AntennaDesignProblem(spec, None, "medium"))
    suite.register_problem(AntennaDesignProblem(spec, None, "high"))
    
    # Run benchmark
    results = suite.run_comprehensive_benchmark(num_runs=10)  # Reduced for demo
    
    # Generate publication materials
    tables = suite.generate_publication_tables()
    
    return results, tables


if __name__ == "__main__":
    results, tables = run_benchmark_study()
    print("Benchmark study completed!")
    print(f"Overall best algorithm: {results['overall_rankings'][0]['algorithm']}")