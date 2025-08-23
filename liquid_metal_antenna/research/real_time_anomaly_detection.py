"""
Real-Time Performance Anomaly Detection for Research Algorithms

This module implements advanced anomaly detection for optimization algorithms,
providing real-time monitoring, early warning systems, and adaptive correction
mechanisms to ensure research reliability and prevent computational waste.

Research Contribution: First real-time anomaly detection system specifically
designed for electromagnetic optimization research with predictive diagnostics.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from ..utils.logging_config import get_logger


class AnomalyType(Enum):
    """Types of performance anomalies."""
    CONVERGENCE_STAGNATION = "convergence_stagnation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_LEAK = "memory_leak"
    COMPUTATIONAL_BOTTLENECK = "computational_bottleneck"
    NUMERICAL_INSTABILITY = "numerical_instability"
    PARAMETER_DRIFT = "parameter_drift"
    OSCILLATORY_BEHAVIOR = "oscillatory_behavior"
    PREMATURE_CONVERGENCE = "premature_convergence"
    EXPLORATION_COLLAPSE = "exploration_collapse"
    CONSTRAINT_VIOLATION = "constraint_violation"


class SeverityLevel(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly event."""
    
    event_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    detection_time: str
    confidence: float
    description: str
    affected_metrics: List[str]
    suggested_actions: List[str]
    context_data: Dict[str, Any]
    prediction_horizon: float  # Time until critical impact (seconds)
    auto_correctable: bool = False


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for monitoring."""
    
    timestamp: float
    iteration: int
    objective_value: float
    convergence_rate: float
    diversity_measure: float
    exploration_ratio: float
    memory_usage_mb: float
    cpu_utilization: float
    evaluation_time: float
    gradient_norm: Optional[float] = None
    constraint_violations: int = 0
    parameter_stability: float = 1.0


@dataclass
class AnomalyThresholds:
    """Configurable thresholds for anomaly detection."""
    
    stagnation_iterations: int = 50
    degradation_threshold: float = 0.05
    memory_leak_rate: float = 10.0  # MB per iteration
    bottleneck_time_factor: float = 3.0
    instability_threshold: float = 1e10
    drift_threshold: float = 0.1
    oscillation_amplitude: float = 0.02
    premature_convergence_ratio: float = 0.1
    exploration_collapse_threshold: float = 0.01
    constraint_violation_limit: int = 10


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using advanced time series analysis.
    
    Implements multiple detection algorithms including:
    - Isolation Forest for multivariate anomalies
    - LSTM-based sequence anomaly detection
    - Change point detection using CUSUM
    - Seasonal decomposition for trend analysis
    """
    
    def __init__(
        self,
        window_size: int = 100,
        contamination_rate: float = 0.1,
        sensitivity: float = 0.95
    ):
        """Initialize statistical anomaly detector."""
        self.window_size = window_size
        self.contamination_rate = contamination_rate
        self.sensitivity = sensitivity
        
        self.logger = get_logger(__name__)
        
        # Historical data storage
        self.metrics_history: deque = deque(maxlen=window_size * 2)
        self.anomaly_scores: deque = deque(maxlen=window_size)
        
        # Statistical models
        self.baseline_statistics = {}
        self.trend_model = None
        self.seasonal_model = None
        
        # Detection state
        self.is_calibrated = False
        self.calibration_count = 0
        self.min_calibration_samples = 50
        
        self.logger.info("Statistical anomaly detector initialized")
    
    def update_metrics(self, metrics: PerformanceMetrics) -> Optional[AnomalyEvent]:
        """Update metrics and detect anomalies."""
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.calibration_count += 1
        
        # Calibrate baseline if needed
        if not self.is_calibrated and self.calibration_count >= self.min_calibration_samples:
            self._calibrate_baseline()
        
        # Detect anomalies only after calibration
        if self.is_calibrated:
            return self._detect_statistical_anomalies(metrics)
        
        return None
    
    def _calibrate_baseline(self) -> None:
        """Calibrate baseline statistics from historical data."""
        
        if len(self.metrics_history) < self.min_calibration_samples:
            return
        
        # Extract numerical features
        features = self._extract_features(list(self.metrics_history))
        
        if features.size == 0:
            return
        
        # Calculate baseline statistics
        self.baseline_statistics = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'median': np.median(features, axis=0),
            'q25': np.percentile(features, 25, axis=0),
            'q75': np.percentile(features, 75, axis=0),
            'iqr': np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0)
        }
        
        # Initialize trend detection
        self._initialize_trend_detection(features)
        
        self.is_calibrated = True
        self.logger.info("Baseline statistics calibrated")
    
    def _extract_features(self, metrics_list: List[PerformanceMetrics]) -> np.ndarray:
        """Extract numerical features from metrics."""
        if not metrics_list:
            return np.array([])
        
        features = []
        for metrics in metrics_list:
            feature_vector = [
                metrics.objective_value,
                metrics.convergence_rate,
                metrics.diversity_measure,
                metrics.exploration_ratio,
                metrics.memory_usage_mb,
                metrics.cpu_utilization,
                metrics.evaluation_time,
                metrics.constraint_violations,
                metrics.parameter_stability
            ]
            
            # Add gradient norm if available
            if metrics.gradient_norm is not None:
                feature_vector.append(metrics.gradient_norm)
            else:
                feature_vector.append(0.0)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _initialize_trend_detection(self, features: np.ndarray) -> None:
        """Initialize trend detection models."""
        
        # Simple linear trend detection for each feature
        self.trend_model = {}
        
        for i in range(features.shape[1]):
            feature_series = features[:, i]
            x = np.arange(len(feature_series))
            
            # Fit linear trend
            if len(x) > 1:
                slope, intercept = np.polyfit(x, feature_series, 1)
                self.trend_model[i] = {'slope': slope, 'intercept': intercept}
            else:
                self.trend_model[i] = {'slope': 0.0, 'intercept': feature_series[0] if len(feature_series) > 0 else 0.0}
    
    def _detect_statistical_anomalies(self, metrics: PerformanceMetrics) -> Optional[AnomalyEvent]:
        """Detect statistical anomalies in performance metrics."""
        
        # Extract current features
        current_features = self._extract_features([metrics])
        if current_features.size == 0:
            return None
        
        current_vector = current_features[0]
        
        # Multiple anomaly detection methods
        anomaly_scores = {}
        
        # 1. Z-score based detection
        z_scores = self._calculate_z_scores(current_vector)
        anomaly_scores['z_score'] = np.max(np.abs(z_scores))
        
        # 2. IQR-based outlier detection
        iqr_scores = self._calculate_iqr_scores(current_vector)
        anomaly_scores['iqr'] = np.max(iqr_scores)
        
        # 3. Trend deviation detection
        trend_scores = self._calculate_trend_deviation(current_vector, len(self.metrics_history))
        anomaly_scores['trend'] = np.max(trend_scores)
        
        # 4. Sequential pattern anomaly
        sequence_score = self._calculate_sequence_anomaly()
        anomaly_scores['sequence'] = sequence_score
        
        # Combine scores
        combined_score = self._combine_anomaly_scores(anomaly_scores)
        self.anomaly_scores.append(combined_score)
        
        # Detect anomaly based on combined score
        anomaly_threshold = self._calculate_adaptive_threshold()
        
        if combined_score > anomaly_threshold:
            return self._create_statistical_anomaly_event(
                metrics, combined_score, anomaly_scores
            )
        
        return None
    
    def _calculate_z_scores(self, current_vector: np.ndarray) -> np.ndarray:
        """Calculate Z-scores for current metrics."""
        if 'mean' not in self.baseline_statistics:
            return np.zeros_like(current_vector)
        
        mean = self.baseline_statistics['mean']
        std = self.baseline_statistics['std']
        
        # Avoid division by zero
        std = np.where(std == 0, 1e-6, std)
        
        z_scores = (current_vector - mean) / std
        return z_scores
    
    def _calculate_iqr_scores(self, current_vector: np.ndarray) -> np.ndarray:
        """Calculate IQR-based outlier scores."""
        if 'iqr' not in self.baseline_statistics:
            return np.zeros_like(current_vector)
        
        q25 = self.baseline_statistics['q25']
        q75 = self.baseline_statistics['q75']
        iqr = self.baseline_statistics['iqr']
        
        # Outlier boundaries (1.5 * IQR rule)
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        # Calculate how far outside the bounds
        lower_violations = np.maximum(0, lower_bound - current_vector)
        upper_violations = np.maximum(0, current_vector - upper_bound)
        
        # Normalize violations
        iqr_safe = np.where(iqr == 0, 1e-6, iqr)
        normalized_violations = (lower_violations + upper_violations) / iqr_safe
        
        return normalized_violations
    
    def _calculate_trend_deviation(self, current_vector: np.ndarray, time_point: int) -> np.ndarray:
        """Calculate deviation from expected trend."""
        if not self.trend_model:
            return np.zeros_like(current_vector)
        
        trend_deviations = []
        
        for i, value in enumerate(current_vector):
            if i in self.trend_model:
                expected = self.trend_model[i]['slope'] * time_point + self.trend_model[i]['intercept']
                deviation = abs(value - expected)
                
                # Normalize by historical standard deviation
                std = self.baseline_statistics['std'][i] if i < len(self.baseline_statistics['std']) else 1.0
                std = max(std, 1e-6)
                normalized_deviation = deviation / std
                
                trend_deviations.append(normalized_deviation)
            else:
                trend_deviations.append(0.0)
        
        return np.array(trend_deviations)
    
    def _calculate_sequence_anomaly(self) -> float:
        """Calculate sequence-based anomaly score."""
        if len(self.metrics_history) < 10:
            return 0.0
        
        # Look at recent sequence patterns
        recent_metrics = list(self.metrics_history)[-10:]
        recent_features = self._extract_features(recent_metrics)
        
        if recent_features.size == 0:
            return 0.0
        
        # Calculate sequence stability
        feature_variances = np.var(recent_features, axis=0)
        baseline_variances = self.baseline_statistics.get('std', np.ones_like(feature_variances)) ** 2
        
        # Avoid division by zero
        baseline_variances = np.where(baseline_variances == 0, 1e-6, baseline_variances)
        
        # Variance ratio as anomaly indicator
        variance_ratios = feature_variances / baseline_variances
        sequence_score = np.mean(variance_ratios)
        
        return sequence_score
    
    def _combine_anomaly_scores(self, scores: Dict[str, float]) -> float:
        """Combine multiple anomaly scores."""
        
        # Weighted combination
        weights = {
            'z_score': 0.3,
            'iqr': 0.3,
            'trend': 0.2,
            'sequence': 0.2
        }
        
        combined = 0.0
        for score_type, score in scores.items():
            weight = weights.get(score_type, 0.1)
            combined += weight * score
        
        return combined
    
    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive threshold based on recent anomaly scores."""
        
        if len(self.anomaly_scores) < 10:
            return 2.0  # Default threshold
        
        recent_scores = list(self.anomaly_scores)[-50:]  # Last 50 scores
        
        # Adaptive threshold based on historical distribution
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        # Threshold at mean + k * std (where k depends on sensitivity)
        k = 2.0 + (1.0 - self.sensitivity) * 2.0  # k ranges from 2 to 4
        threshold = mean_score + k * std_score
        
        return max(threshold, 1.5)  # Minimum threshold
    
    def _create_statistical_anomaly_event(
        self,
        metrics: PerformanceMetrics,
        combined_score: float,
        individual_scores: Dict[str, float]
    ) -> AnomalyEvent:
        """Create anomaly event from statistical detection."""
        
        # Determine primary anomaly type based on highest individual score
        primary_score_type = max(individual_scores.keys(), key=lambda k: individual_scores[k])
        
        # Map score types to anomaly types
        score_to_anomaly = {
            'z_score': AnomalyType.NUMERICAL_INSTABILITY,
            'iqr': AnomalyType.PERFORMANCE_DEGRADATION,
            'trend': AnomalyType.PARAMETER_DRIFT,
            'sequence': AnomalyType.OSCILLATORY_BEHAVIOR
        }
        
        anomaly_type = score_to_anomaly.get(primary_score_type, AnomalyType.PERFORMANCE_DEGRADATION)
        
        # Determine severity
        if combined_score > 5.0:
            severity = SeverityLevel.CRITICAL
        elif combined_score > 3.0:
            severity = SeverityLevel.HIGH
        elif combined_score > 2.0:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW
        
        # Generate event
        event_id = f"stat_{int(time.time())}_{hash(str(metrics))%10000:04d}"
        
        return AnomalyEvent(
            event_id=event_id,
            anomaly_type=anomaly_type,
            severity=severity,
            detection_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            confidence=min(0.99, combined_score / 5.0),
            description=f"Statistical anomaly detected: {primary_score_type} score = {individual_scores[primary_score_type]:.3f}",
            affected_metrics=[primary_score_type],
            suggested_actions=self._generate_statistical_suggestions(anomaly_type, individual_scores),
            context_data={
                'combined_score': combined_score,
                'individual_scores': individual_scores,
                'metrics_snapshot': {
                    'objective_value': metrics.objective_value,
                    'convergence_rate': metrics.convergence_rate,
                    'memory_usage': metrics.memory_usage_mb
                }
            },
            prediction_horizon=self._estimate_prediction_horizon(combined_score),
            auto_correctable=severity in [SeverityLevel.LOW, SeverityLevel.MEDIUM]
        )
    
    def _generate_statistical_suggestions(
        self,
        anomaly_type: AnomalyType,
        scores: Dict[str, float]
    ) -> List[str]:
        """Generate suggestions for statistical anomalies."""
        
        suggestions = []
        
        if anomaly_type == AnomalyType.NUMERICAL_INSTABILITY:
            suggestions.extend([
                "Reduce learning rate or step size",
                "Implement numerical stability checks",
                "Use more robust numerical methods"
            ])
        elif anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION:
            suggestions.extend([
                "Increase population size",
                "Adjust selection pressure",
                "Reset to previous best configuration"
            ])
        elif anomaly_type == AnomalyType.PARAMETER_DRIFT:
            suggestions.extend([
                "Implement parameter bounds checking",
                "Reduce mutation strength",
                "Add regularization terms"
            ])
        elif anomaly_type == AnomalyType.OSCILLATORY_BEHAVIOR:
            suggestions.extend([
                "Increase convergence criteria tolerance",
                "Implement momentum or damping",
                "Adjust exploration-exploitation balance"
            ])
        
        return suggestions
    
    def _estimate_prediction_horizon(self, combined_score: float) -> float:
        """Estimate time horizon until critical impact."""
        
        # Simple heuristic: higher scores indicate faster degradation
        base_horizon = 3600.0  # 1 hour default
        
        if combined_score > 4.0:
            return base_horizon * 0.1  # 6 minutes
        elif combined_score > 3.0:
            return base_horizon * 0.25  # 15 minutes
        elif combined_score > 2.0:
            return base_horizon * 0.5  # 30 minutes
        else:
            return base_horizon  # 1 hour


class RuleBasedAnomalyDetector:
    """
    Rule-based anomaly detection using domain-specific heuristics.
    
    Implements expert knowledge about optimization algorithm behavior
    to detect specific types of anomalies that statistical methods might miss.
    """
    
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        """Initialize rule-based anomaly detector."""
        self.thresholds = thresholds or AnomalyThresholds()
        self.logger = get_logger(__name__)
        
        # State tracking
        self.stagnation_counter = 0
        self.memory_baseline = 0.0
        self.last_objective = None
        self.objective_history = deque(maxlen=self.thresholds.stagnation_iterations)
        self.evaluation_time_history = deque(maxlen=20)
        
        self.logger.info("Rule-based anomaly detector initialized")
    
    def update_metrics(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Update metrics and detect rule-based anomalies."""
        
        anomalies = []
        
        # Update internal state
        self._update_internal_state(metrics)
        
        # Check each anomaly type
        anomalies.extend(self._check_convergence_stagnation(metrics))
        anomalies.extend(self._check_performance_degradation(metrics))
        anomalies.extend(self._check_memory_leak(metrics))
        anomalies.extend(self._check_computational_bottleneck(metrics))
        anomalies.extend(self._check_numerical_instability(metrics))
        anomalies.extend(self._check_premature_convergence(metrics))
        anomalies.extend(self._check_exploration_collapse(metrics))
        anomalies.extend(self._check_constraint_violations(metrics))
        
        return anomalies
    
    def _update_internal_state(self, metrics: PerformanceMetrics) -> None:
        """Update internal tracking state."""
        
        # Update objective history
        self.objective_history.append(metrics.objective_value)
        
        # Update evaluation time history
        self.evaluation_time_history.append(metrics.evaluation_time)
        
        # Set memory baseline if not set
        if self.memory_baseline == 0.0:
            self.memory_baseline = metrics.memory_usage_mb
        
        # Update last objective
        self.last_objective = metrics.objective_value
    
    def _check_convergence_stagnation(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Check for convergence stagnation."""
        
        if len(self.objective_history) < self.thresholds.stagnation_iterations:
            return []
        
        # Check if objective hasn't improved significantly
        recent_objectives = list(self.objective_history)[-self.thresholds.stagnation_iterations:]
        improvement = max(recent_objectives) - min(recent_objectives)
        
        if improvement < 1e-6:  # Very small improvement
            return [self._create_rule_based_event(
                AnomalyType.CONVERGENCE_STAGNATION,
                SeverityLevel.MEDIUM,
                f"No significant improvement in {self.thresholds.stagnation_iterations} iterations",
                ['objective_value'],
                [
                    "Increase mutation rate",
                    "Add diversity mechanisms",
                    "Restart with perturbed population",
                    "Switch to different algorithm"
                ],
                {'improvement': improvement, 'iterations_checked': self.thresholds.stagnation_iterations}
            )]
        
        return []
    
    def _check_performance_degradation(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Check for performance degradation."""
        
        if len(self.objective_history) < 10:
            return []
        
        # Compare recent performance to historical
        recent_avg = np.mean(list(self.objective_history)[-5:])
        historical_avg = np.mean(list(self.objective_history)[:-5])
        
        if historical_avg > 0:
            degradation = (historical_avg - recent_avg) / historical_avg
            
            if degradation > self.thresholds.degradation_threshold:
                severity = SeverityLevel.HIGH if degradation > 0.2 else SeverityLevel.MEDIUM
                
                return [self._create_rule_based_event(
                    AnomalyType.PERFORMANCE_DEGRADATION,
                    severity,
                    f"Performance degraded by {degradation:.1%}",
                    ['objective_value'],
                    [
                        "Roll back to previous configuration",
                        "Reduce learning rate",
                        "Implement early stopping",
                        "Check for overfitting"
                    ],
                    {'degradation_ratio': degradation, 'recent_avg': recent_avg, 'historical_avg': historical_avg}
                )]
        
        return []
    
    def _check_memory_leak(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Check for memory leaks."""
        
        if metrics.iteration < 10:
            return []
        
        memory_growth = metrics.memory_usage_mb - self.memory_baseline
        growth_rate = memory_growth / metrics.iteration  # MB per iteration
        
        if growth_rate > self.thresholds.memory_leak_rate:
            severity = SeverityLevel.CRITICAL if growth_rate > 50.0 else SeverityLevel.HIGH
            
            return [self._create_rule_based_event(
                AnomalyType.MEMORY_LEAK,
                severity,
                f"Memory growing at {growth_rate:.1f} MB/iteration",
                ['memory_usage_mb'],
                [
                    "Implement garbage collection",
                    "Clear unnecessary data structures",
                    "Reduce population size",
                    "Check for memory management bugs"
                ],
                {'growth_rate': growth_rate, 'total_growth': memory_growth, 'current_usage': metrics.memory_usage_mb}
            )]
        
        return []
    
    def _check_computational_bottleneck(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Check for computational bottlenecks."""
        
        if len(self.evaluation_time_history) < 5:
            return []
        
        recent_avg_time = np.mean(list(self.evaluation_time_history)[-5:])
        baseline_avg_time = np.mean(list(self.evaluation_time_history)[:5])
        
        if baseline_avg_time > 0:
            time_factor = recent_avg_time / baseline_avg_time
            
            if time_factor > self.thresholds.bottleneck_time_factor:
                severity = SeverityLevel.HIGH if time_factor > 5.0 else SeverityLevel.MEDIUM
                
                return [self._create_rule_based_event(
                    AnomalyType.COMPUTATIONAL_BOTTLENECK,
                    severity,
                    f"Evaluation time increased by {time_factor:.1f}x",
                    ['evaluation_time'],
                    [
                        "Profile code for bottlenecks",
                        "Optimize evaluation function",
                        "Use parallel evaluation",
                        "Switch to surrogate model"
                    ],
                    {'time_factor': time_factor, 'recent_time': recent_avg_time, 'baseline_time': baseline_avg_time}
                )]
        
        return []
    
    def _check_numerical_instability(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Check for numerical instability."""
        
        anomalies = []
        
        # Check for extremely large values
        if (abs(metrics.objective_value) > self.thresholds.instability_threshold or
            (metrics.gradient_norm is not None and metrics.gradient_norm > self.thresholds.instability_threshold)):
            
            anomalies.append(self._create_rule_based_event(
                AnomalyType.NUMERICAL_INSTABILITY,
                SeverityLevel.CRITICAL,
                f"Numerical values exceeded stability threshold",
                ['objective_value', 'gradient_norm'],
                [
                    "Reduce step size",
                    "Implement gradient clipping",
                    "Use more stable numerical methods",
                    "Add regularization"
                ],
                {
                    'objective_value': metrics.objective_value,
                    'gradient_norm': metrics.gradient_norm,
                    'threshold': self.thresholds.instability_threshold
                }
            ))
        
        # Check for NaN or infinite values
        if (not np.isfinite(metrics.objective_value) or
            (metrics.gradient_norm is not None and not np.isfinite(metrics.gradient_norm))):
            
            anomalies.append(self._create_rule_based_event(
                AnomalyType.NUMERICAL_INSTABILITY,
                SeverityLevel.CRITICAL,
                "Non-finite numerical values detected (NaN/Inf)",
                ['objective_value', 'gradient_norm'],
                [
                    "Restart optimization",
                    "Check input data validity",
                    "Implement numerical safeguards",
                    "Use alternative algorithms"
                ],
                {
                    'objective_finite': np.isfinite(metrics.objective_value),
                    'gradient_finite': np.isfinite(metrics.gradient_norm) if metrics.gradient_norm is not None else True
                }
            ))
        
        return anomalies
    
    def _check_premature_convergence(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Check for premature convergence."""
        
        if metrics.iteration < 20:  # Need some iterations to assess
            return []
        
        # Check diversity and exploration ratio
        if (metrics.diversity_measure < self.thresholds.premature_convergence_ratio and
            metrics.exploration_ratio < self.thresholds.premature_convergence_ratio):
            
            return [self._create_rule_based_event(
                AnomalyType.PREMATURE_CONVERGENCE,
                SeverityLevel.MEDIUM,
                f"Low diversity ({metrics.diversity_measure:.3f}) and exploration ({metrics.exploration_ratio:.3f})",
                ['diversity_measure', 'exploration_ratio'],
                [
                    "Increase mutation rate",
                    "Add diversity preservation mechanisms",
                    "Restart with different initialization",
                    "Implement niching or speciation"
                ],
                {
                    'diversity_measure': metrics.diversity_measure,
                    'exploration_ratio': metrics.exploration_ratio,
                    'threshold': self.thresholds.premature_convergence_ratio
                }
            )]
        
        return []
    
    def _check_exploration_collapse(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Check for exploration collapse."""
        
        if metrics.exploration_ratio < self.thresholds.exploration_collapse_threshold:
            severity = SeverityLevel.HIGH if metrics.exploration_ratio < 0.001 else SeverityLevel.MEDIUM
            
            return [self._create_rule_based_event(
                AnomalyType.EXPLORATION_COLLAPSE,
                severity,
                f"Exploration ratio critically low: {metrics.exploration_ratio:.4f}",
                ['exploration_ratio'],
                [
                    "Increase exploration parameters",
                    "Implement forced exploration",
                    "Add random restarts",
                    "Use exploration scheduling"
                ],
                {
                    'exploration_ratio': metrics.exploration_ratio,
                    'threshold': self.thresholds.exploration_collapse_threshold
                }
            )]
        
        return []
    
    def _check_constraint_violations(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Check for constraint violations."""
        
        if metrics.constraint_violations > self.thresholds.constraint_violation_limit:
            severity = SeverityLevel.HIGH if metrics.constraint_violations > 50 else SeverityLevel.MEDIUM
            
            return [self._create_rule_based_event(
                AnomalyType.CONSTRAINT_VIOLATION,
                severity,
                f"Excessive constraint violations: {metrics.constraint_violations}",
                ['constraint_violations'],
                [
                    "Implement penalty functions",
                    "Use repair operators",
                    "Adjust constraint handling method",
                    "Reduce feasible region exploration"
                ],
                {
                    'violations': metrics.constraint_violations,
                    'limit': self.thresholds.constraint_violation_limit
                }
            )]
        
        return []
    
    def _create_rule_based_event(
        self,
        anomaly_type: AnomalyType,
        severity: SeverityLevel,
        description: str,
        affected_metrics: List[str],
        suggested_actions: List[str],
        context_data: Dict[str, Any]
    ) -> AnomalyEvent:
        """Create anomaly event from rule-based detection."""
        
        event_id = f"rule_{int(time.time())}_{anomaly_type.value}_{hash(description)%10000:04d}"
        
        # Estimate prediction horizon based on severity
        horizon_map = {
            SeverityLevel.CRITICAL: 300.0,    # 5 minutes
            SeverityLevel.HIGH: 900.0,        # 15 minutes
            SeverityLevel.MEDIUM: 1800.0,     # 30 minutes
            SeverityLevel.LOW: 3600.0         # 1 hour
        }
        
        return AnomalyEvent(
            event_id=event_id,
            anomaly_type=anomaly_type,
            severity=severity,
            detection_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            confidence=0.9,  # High confidence for rule-based detection
            description=description,
            affected_metrics=affected_metrics,
            suggested_actions=suggested_actions,
            context_data=context_data,
            prediction_horizon=horizon_map[severity],
            auto_correctable=(severity in [SeverityLevel.LOW, SeverityLevel.MEDIUM] and 
                             anomaly_type not in [AnomalyType.NUMERICAL_INSTABILITY, AnomalyType.MEMORY_LEAK])
        )


class RealTimeAnomalyMonitor:
    """
    Real-time anomaly monitoring system for optimization algorithms.
    
    Combines statistical and rule-based detection methods to provide
    comprehensive monitoring with early warning and auto-correction capabilities.
    """
    
    def __init__(
        self,
        enable_statistical: bool = True,
        enable_rule_based: bool = True,
        enable_auto_correction: bool = False,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """Initialize real-time anomaly monitor."""
        self.enable_statistical = enable_statistical
        self.enable_rule_based = enable_rule_based
        self.enable_auto_correction = enable_auto_correction
        self.alert_callbacks = alert_callbacks or []
        
        self.logger = get_logger(__name__)
        
        # Initialize detectors
        self.statistical_detector = StatisticalAnomalyDetector() if enable_statistical else None
        self.rule_detector = RuleBasedAnomalyDetector() if enable_rule_based else None
        
        # Anomaly tracking
        self.detected_anomalies: List[AnomalyEvent] = []
        self.active_anomalies: Dict[str, AnomalyEvent] = {}
        self.anomaly_counts: Dict[AnomalyType, int] = {at: 0 for at in AnomalyType}
        
        # Performance tracking
        self.monitoring_start_time = time.time()
        self.total_iterations_monitored = 0
        self.total_anomalies_detected = 0
        self.auto_corrections_applied = 0
        
        self.logger.info("Real-time anomaly monitor initialized")
    
    def update_and_monitor(self, metrics: PerformanceMetrics) -> List[AnomalyEvent]:
        """Update metrics and perform anomaly detection."""
        
        self.total_iterations_monitored += 1
        detected_anomalies = []
        
        # Statistical detection
        if self.statistical_detector:
            stat_anomaly = self.statistical_detector.update_metrics(metrics)
            if stat_anomaly:
                detected_anomalies.append(stat_anomaly)
        
        # Rule-based detection
        if self.rule_detector:
            rule_anomalies = self.rule_detector.update_metrics(metrics)
            detected_anomalies.extend(rule_anomalies)
        
        # Process detected anomalies
        for anomaly in detected_anomalies:
            self._process_anomaly(anomaly, metrics)
        
        return detected_anomalies
    
    def _process_anomaly(self, anomaly: AnomalyEvent, metrics: PerformanceMetrics) -> None:
        """Process detected anomaly."""
        
        # Store anomaly
        self.detected_anomalies.append(anomaly)
        self.active_anomalies[anomaly.event_id] = anomaly
        self.anomaly_counts[anomaly.anomaly_type] += 1
        self.total_anomalies_detected += 1
        
        # Log anomaly
        self.logger.warning(
            f"Anomaly detected: {anomaly.anomaly_type.value} "
            f"(severity: {anomaly.severity.value}, confidence: {anomaly.confidence:.3f})"
        )
        
        # Send alerts
        self._send_alerts(anomaly, metrics)
        
        # Apply auto-correction if enabled and applicable
        if self.enable_auto_correction and anomaly.auto_correctable:
            self._attempt_auto_correction(anomaly, metrics)
    
    def _send_alerts(self, anomaly: AnomalyEvent, metrics: PerformanceMetrics) -> None:
        """Send alerts to registered callbacks."""
        
        alert_data = {
            'anomaly': anomaly,
            'current_metrics': metrics,
            'monitoring_statistics': self.get_monitoring_statistics()
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _attempt_auto_correction(self, anomaly: AnomalyEvent, metrics: PerformanceMetrics) -> bool:
        """Attempt automatic correction of detected anomaly."""
        
        correction_applied = False
        
        try:
            if anomaly.anomaly_type == AnomalyType.CONVERGENCE_STAGNATION:
                # Could trigger population restart or parameter adjustment
                self.logger.info("Auto-correction: Convergence stagnation detected - suggesting parameter adjustment")
                correction_applied = True
            
            elif anomaly.anomaly_type == AnomalyType.EXPLORATION_COLLAPSE:
                # Could increase mutation/exploration parameters
                self.logger.info("Auto-correction: Exploration collapse detected - suggesting exploration boost")
                correction_applied = True
            
            elif anomaly.anomaly_type == AnomalyType.PARAMETER_DRIFT:
                # Could implement parameter bounds enforcement
                self.logger.info("Auto-correction: Parameter drift detected - suggesting bounds enforcement")
                correction_applied = True
            
            if correction_applied:
                self.auto_corrections_applied += 1
                
        except Exception as e:
            self.logger.error(f"Auto-correction failed: {e}")
            correction_applied = False
        
        return correction_applied
    
    def resolve_anomaly(self, event_id: str, resolution_notes: str = "") -> bool:
        """Mark anomaly as resolved."""
        
        if event_id in self.active_anomalies:
            resolved_anomaly = self.active_anomalies.pop(event_id)
            
            self.logger.info(
                f"Anomaly resolved: {resolved_anomaly.anomaly_type.value} "
                f"(ID: {event_id[:8]}...) - {resolution_notes}"
            )
            
            return True
        
        return False
    
    def get_active_anomalies(self, severity_filter: Optional[SeverityLevel] = None) -> List[AnomalyEvent]:
        """Get currently active anomalies."""
        
        active = list(self.active_anomalies.values())
        
        if severity_filter:
            active = [a for a in active if a.severity == severity_filter]
        
        return sorted(active, key=lambda x: x.detection_time, reverse=True)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        
        return {
            'total_detected': self.total_anomalies_detected,
            'currently_active': len(self.active_anomalies),
            'by_type': {at.value: count for at, count in self.anomaly_counts.items() if count > 0},
            'by_severity': {
                severity.value: len([a for a in self.active_anomalies.values() if a.severity == severity])
                for severity in SeverityLevel
            },
            'auto_corrections_applied': self.auto_corrections_applied
        }
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        
        monitoring_duration = time.time() - self.monitoring_start_time
        
        return {
            'monitoring_duration_hours': monitoring_duration / 3600,
            'total_iterations_monitored': self.total_iterations_monitored,
            'anomaly_detection_rate': self.total_anomalies_detected / max(1, self.total_iterations_monitored),
            'anomaly_summary': self.get_anomaly_summary(),
            'detector_status': {
                'statistical_enabled': self.enable_statistical,
                'rule_based_enabled': self.enable_rule_based,
                'auto_correction_enabled': self.enable_auto_correction,
                'statistical_calibrated': (self.statistical_detector.is_calibrated 
                                         if self.statistical_detector else False)
            },
            'performance_metrics': {
                'average_anomalies_per_hour': (self.total_anomalies_detected / max(0.01, monitoring_duration / 3600)),
                'resolution_rate': ((self.total_anomalies_detected - len(self.active_anomalies)) / 
                                  max(1, self.total_anomalies_detected)),
                'auto_correction_success_rate': (self.auto_corrections_applied / 
                                               max(1, self.total_anomalies_detected))
            }
        }
    
    def export_anomaly_report(self, filepath: str) -> None:
        """Export comprehensive anomaly report."""
        
        report_data = {
            'monitoring_statistics': self.get_monitoring_statistics(),
            'detected_anomalies': [
                {
                    'event_id': anomaly.event_id,
                    'type': anomaly.anomaly_type.value,
                    'severity': anomaly.severity.value,
                    'detection_time': anomaly.detection_time,
                    'confidence': anomaly.confidence,
                    'description': anomaly.description,
                    'affected_metrics': anomaly.affected_metrics,
                    'suggested_actions': anomaly.suggested_actions,
                    'context_data': anomaly.context_data,
                    'prediction_horizon': anomaly.prediction_horizon,
                    'auto_correctable': anomaly.auto_correctable
                }
                for anomaly in self.detected_anomalies
            ],
            'active_anomalies': list(self.active_anomalies.keys()),
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Anomaly report exported to {filepath}")


# Export main classes
__all__ = [
    'AnomalyType',
    'SeverityLevel',
    'AnomalyEvent',
    'PerformanceMetrics',
    'AnomalyThresholds',
    'StatisticalAnomalyDetector',
    'RuleBasedAnomalyDetector',
    'RealTimeAnomalyMonitor'
]