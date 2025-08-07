"""
Bayesian optimization for expensive antenna simulations.

Implements Gaussian Process-based optimization with acquisition functions
for efficient exploration of design space.
"""

import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import pickle

from ..utils.logging_config import get_logger, LoggingContextManager
from ..utils.validation import ValidationError
from ..solvers.base import SolverResult, BaseSolver
from ..core.antenna_spec import AntennaSpec


@dataclass
class BayesianResult:
    """Result from Bayesian optimization."""
    best_objective: float
    best_geometry: np.ndarray
    best_result: SolverResult
    n_evaluations: int
    optimization_time: float
    convergence_history: List[float]
    acquisition_history: List[Tuple[np.ndarray, float]]


class GaussianProcess:
    """Gaussian Process surrogate model for Bayesian optimization."""
    
    def __init__(
        self,
        kernel: str = 'rbf',
        length_scale: Union[float, np.ndarray] = 1.0,
        noise_level: float = 1e-6,
        optimize_hyperparameters: bool = True
    ):
        """
        Initialize Gaussian Process.
        
        Args:
            kernel: Kernel type ('rbf', 'matern32', 'matern52')
            length_scale: Kernel length scale parameter
            noise_level: Observation noise level
            optimize_hyperparameters: Whether to optimize GP hyperparameters
        """
        self.kernel = kernel
        self.length_scale = length_scale if isinstance(length_scale, np.ndarray) else np.array([length_scale])
        self.noise_level = noise_level
        self.optimize_hyperparameters = optimize_hyperparameters
        
        self.logger = get_logger('gaussian_process')
        
        # Training data
        self.X_train = None
        self.y_train = None
        
        # Fitted model parameters
        self.K_inv = None
        self.alpha = None
        self.log_marginal_likelihood = None
    
    def kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of points."""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == 'matern32':
            return self._matern32_kernel(X1, X2)
        elif self.kernel == 'matern52':
            return self._matern52_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        # Compute pairwise squared distances
        X1_expanded = X1[:, np.newaxis, :]  # (n1, 1, d)
        X2_expanded = X2[np.newaxis, :, :]  # (1, n2, d)
        
        diff = X1_expanded - X2_expanded  # (n1, n2, d)
        
        # Scale by length scales
        scaled_diff = diff / self.length_scale[np.newaxis, np.newaxis, :]
        
        # Squared distances
        sq_distances = np.sum(scaled_diff ** 2, axis=2)
        
        return np.exp(-0.5 * sq_distances)
    
    def _matern32_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matérn 3/2 kernel."""
        # Compute pairwise distances
        X1_expanded = X1[:, np.newaxis, :]
        X2_expanded = X2[np.newaxis, :, :]
        
        diff = X1_expanded - X2_expanded
        scaled_diff = diff / self.length_scale[np.newaxis, np.newaxis, :]
        
        distances = np.sqrt(np.sum(scaled_diff ** 2, axis=2))
        sqrt3_distances = np.sqrt(3) * distances
        
        return (1 + sqrt3_distances) * np.exp(-sqrt3_distances)
    
    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matérn 5/2 kernel."""
        # Compute pairwise distances
        X1_expanded = X1[:, np.newaxis, :]
        X2_expanded = X2[np.newaxis, :, :]
        
        diff = X1_expanded - X2_expanded
        scaled_diff = diff / self.length_scale[np.newaxis, np.newaxis, :]
        
        distances = np.sqrt(np.sum(scaled_diff ** 2, axis=2))
        sqrt5_distances = np.sqrt(5) * distances
        
        return (1 + sqrt5_distances + (5/3) * distances ** 2) * np.exp(-sqrt5_distances)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Gaussian Process to training data.
        
        Args:
            X: Training inputs (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Ensure length_scale has correct dimensionality
        if len(self.length_scale) == 1 and X.shape[1] > 1:
            self.length_scale = np.full(X.shape[1], self.length_scale[0])
        
        # Optimize hyperparameters if requested
        if self.optimize_hyperparameters and len(X) > 5:
            self._optimize_hyperparameters()
        
        # Compute kernel matrix and inverse
        K = self.kernel_function(X, X)
        K += self.noise_level * np.eye(len(X))  # Add noise term
        
        try:
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ y
            
            # Compute log marginal likelihood
            sign, log_det = np.linalg.slogdet(K)
            self.log_marginal_likelihood = (
                -0.5 * y @ self.alpha - 
                0.5 * log_det - 
                0.5 * len(X) * np.log(2 * np.pi)
            )
            
        except np.linalg.LinAlgError:
            self.logger.warning("Kernel matrix inversion failed, adding regularization")
            
            # Add regularization and retry
            K += 1e-3 * np.eye(len(X))
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ y
    
    def predict(self, X_test: np.ndarray, return_std: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions at test points.
        
        Args:
            X_test: Test inputs (n_test, n_features)
            return_std: Whether to return prediction uncertainty
            
        Returns:
            Predictions (and standard deviations if requested)
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Compute kernel between test and training points
        K_test = self.kernel_function(X_test, self.X_train)
        
        # Mean prediction
        mean = K_test @ self.alpha
        
        if not return_std:
            return mean
        
        # Variance prediction
        K_test_test = self.kernel_function(X_test, X_test)
        variance = np.diag(K_test_test) - np.sum(K_test @ self.K_inv * K_test, axis=1)
        variance = np.maximum(variance, 1e-10)  # Numerical stability
        
        std = np.sqrt(variance)
        
        return mean, std
    
    def _optimize_hyperparameters(self) -> None:
        """Optimize GP hyperparameters using maximum likelihood."""
        def negative_log_marginal_likelihood(params):
            # Unpack parameters
            length_scales = params[:-1]
            noise_level = params[-1]
            
            # Ensure positive values
            length_scales = np.exp(length_scales)
            noise_level = np.exp(noise_level)
            
            # Temporarily update parameters
            old_length_scale = self.length_scale.copy()
            old_noise_level = self.noise_level
            
            self.length_scale = length_scales
            self.noise_level = noise_level
            
            try:
                # Compute kernel matrix
                K = self.kernel_function(self.X_train, self.X_train)
                K += noise_level * np.eye(len(self.X_train))
                
                # Compute log marginal likelihood
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
                
                log_marginal_likelihood = (
                    -0.5 * self.y_train @ alpha -
                    np.sum(np.log(np.diag(L))) -
                    0.5 * len(self.X_train) * np.log(2 * np.pi)
                )
                
                # Restore parameters
                self.length_scale = old_length_scale
                self.noise_level = old_noise_level
                
                return -log_marginal_likelihood
                
            except np.linalg.LinAlgError:
                # Restore parameters
                self.length_scale = old_length_scale
                self.noise_level = old_noise_level
                return 1e6  # Return large value for failed evaluation
        
        # Initial parameters (log-transformed)
        initial_params = np.concatenate([
            np.log(self.length_scale),
            [np.log(self.noise_level)]
        ])
        
        # Optimize
        try:
            result = minimize(
                negative_log_marginal_likelihood,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            if result.success:
                # Update parameters with optimized values
                optimized_params = result.x
                self.length_scale = np.exp(optimized_params[:-1])
                self.noise_level = np.exp(optimized_params[-1])
                
                self.logger.info(f"Optimized GP hyperparameters: "
                               f"length_scales={self.length_scale}, "
                               f"noise_level={self.noise_level:.6f}")
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed: {str(e)}")


class AcquisitionFunction:
    """Base class for acquisition functions."""
    
    def __init__(self, gp: GaussianProcess, kappa: float = 2.576):
        """
        Initialize acquisition function.
        
        Args:
            gp: Trained Gaussian Process model
            kappa: Exploration parameter
        """
        self.gp = gp
        self.kappa = kappa
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate acquisition function at points X."""
        raise NotImplementedError


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate UCB at points X."""
        mean, std = self.gp.predict(X, return_std=True)
        return mean + self.kappa * std


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    
    def __init__(self, gp: GaussianProcess, xi: float = 0.01):
        """
        Initialize EI acquisition function.
        
        Args:
            gp: Trained Gaussian Process model
            xi: Exploration parameter
        """
        super().__init__(gp)
        self.xi = xi
        
        # Current best objective value
        if gp.y_train is not None:
            self.f_best = np.max(gp.y_train)  # For maximization
        else:
            self.f_best = 0.0
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate EI at points X."""
        mean, std = self.gp.predict(X, return_std=True)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-9)
        
        # Expected improvement calculation
        z = (mean - self.f_best - self.xi) / std
        ei = (mean - self.f_best - self.xi) * norm.cdf(z) + std * norm.pdf(z)
        
        return ei
    
    def update_best(self, f_new: float) -> None:
        """Update best observed value."""
        if f_new > self.f_best:
            self.f_best = f_new


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function."""
    
    def __init__(self, gp: GaussianProcess, xi: float = 0.01):
        """
        Initialize PI acquisition function.
        
        Args:
            gp: Trained Gaussian Process model
            xi: Exploration parameter
        """
        super().__init__(gp)
        self.xi = xi
        
        # Current best objective value
        if gp.y_train is not None:
            self.f_best = np.max(gp.y_train)  # For maximization
        else:
            self.f_best = 0.0
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate PI at points X."""
        mean, std = self.gp.predict(X, return_std=True)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-9)
        
        # Probability of improvement calculation
        z = (mean - self.f_best - self.xi) / std
        pi = norm.cdf(z)
        
        return pi
    
    def update_best(self, f_new: float) -> None:
        """Update best observed value."""
        if f_new > self.f_best:
            self.f_best = f_new


class BayesianOptimizer:
    """Bayesian optimizer for expensive antenna simulations."""
    
    def __init__(
        self,
        objective_function: Optional[Callable] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        acquisition_function: str = 'ei',
        kernel: str = 'matern52',
        n_initial_points: int = 5,
        optimize_gp_hyperparameters: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            objective_function: Function to optimize
            bounds: Variable bounds (lower, upper)
            acquisition_function: Acquisition function ('ucb', 'ei', 'pi')
            kernel: GP kernel type
            n_initial_points: Number of initial random evaluations
            optimize_gp_hyperparameters: Whether to optimize GP hyperparameters
            random_state: Random seed
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.acquisition_function_type = acquisition_function
        self.kernel = kernel
        self.n_initial_points = n_initial_points
        self.optimize_gp_hyperparameters = optimize_gp_hyperparameters
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.logger = get_logger('bayesian_optimizer')
        
        # Initialize Gaussian Process
        self.gp = GaussianProcess(
            kernel=kernel,
            optimize_hyperparameters=optimize_gp_hyperparameters
        )
        
        # Optimization history
        self.X_evaluated = []
        self.y_evaluated = []
        self.acquisition_values = []
    
    def optimize(
        self,
        solver: BaseSolver,
        antenna_spec: AntennaSpec,
        n_calls: int = 50,
        callback: Optional[Callable] = None
    ) -> BayesianResult:
        """
        Run Bayesian optimization.
        
        Args:
            solver: Electromagnetic solver
            antenna_spec: Antenna specification
            n_calls: Maximum number of function evaluations
            callback: Optional callback function
            
        Returns:
            Optimization result
        """
        self.logger.info(f"Starting Bayesian optimization with {n_calls} evaluations")
        
        start_time = time.time()
        convergence_history = []
        
        # Set up objective function if not provided
        if self.objective_function is None:
            def antenna_objective(geometry: np.ndarray) -> float:
                try:
                    frequency = np.mean(antenna_spec.frequency_range)
                    result = solver.simulate(
                        geometry=self._decode_geometry(geometry),
                        frequency=frequency,
                        spec=antenna_spec
                    )
                    
                    # Multi-objective combination (can be customized)
                    gain_score = result.gain_dbi or 0.0
                    efficiency_score = (result.efficiency or 0.0) * 10  # Scale efficiency
                    vswr_penalty = -max(0, (result.vswr[0] if len(result.vswr) > 0 else 2.0) - 2.0) * 5
                    
                    objective = gain_score + efficiency_score + vswr_penalty
                    
                    return objective
                    
                except Exception as e:
                    self.logger.warning(f"Objective evaluation failed: {str(e)}")
                    return -100.0  # Poor objective for failed evaluations
            
            self.objective_function = antenna_objective
        
        # Set default bounds if not provided
        if self.bounds is None:
            n_variables = 16  # Default number of design variables
            self.bounds = (np.zeros(n_variables), np.ones(n_variables))
        
        # Initial random evaluations
        n_dims = len(self.bounds[0])
        
        with LoggingContextManager("Initial Sampling", self.logger):
            for i in range(min(self.n_initial_points, n_calls)):
                # Random point within bounds
                x_next = np.random.uniform(self.bounds[0], self.bounds[1], n_dims)
                
                # Evaluate objective
                y_next = self.objective_function(x_next)
                
                # Store evaluation
                self.X_evaluated.append(x_next.copy())
                self.y_evaluated.append(y_next)
                
                convergence_history.append(max(self.y_evaluated))
                
                self.logger.info(f"Initial evaluation {i+1}/{self.n_initial_points}: "
                               f"objective = {y_next:.4f}")
                
                if callback:
                    callback(i, x_next, y_next, "initial")
        
        # Bayesian optimization iterations
        remaining_calls = n_calls - len(self.X_evaluated)
        
        with LoggingContextManager("Bayesian Optimization", self.logger):
            for iteration in range(remaining_calls):
                # Fit Gaussian Process
                X_train = np.array(self.X_evaluated)
                y_train = np.array(self.y_evaluated)
                self.gp.fit(X_train, y_train)
                
                # Create acquisition function
                acquisition_func = self._create_acquisition_function()
                
                # Find next point to evaluate
                x_next = self._optimize_acquisition(acquisition_func)
                
                # Evaluate objective at new point
                y_next = self.objective_function(x_next)
                
                # Store evaluation
                self.X_evaluated.append(x_next.copy())
                self.y_evaluated.append(y_next)
                
                # Update acquisition function if needed
                if hasattr(acquisition_func, 'update_best'):
                    acquisition_func.update_best(y_next)
                
                # Track convergence
                best_so_far = max(self.y_evaluated)
                convergence_history.append(best_so_far)
                
                # Store acquisition value
                acq_value = float(acquisition_func(x_next.reshape(1, -1))[0])
                self.acquisition_values.append((x_next.copy(), acq_value))
                
                self.logger.info(f"BO iteration {iteration+1}/{remaining_calls}: "
                               f"objective = {y_next:.4f}, "
                               f"best = {best_so_far:.4f}, "
                               f"acquisition = {acq_value:.4f}")
                
                if callback:
                    callback(
                        len(self.X_evaluated), x_next, y_next, 
                        "optimization", acquisition_value=acq_value
                    )
        
        # Find best result
        best_idx = np.argmax(self.y_evaluated)
        best_objective = self.y_evaluated[best_idx]
        best_geometry_encoded = self.X_evaluated[best_idx]
        best_geometry = self._decode_geometry(best_geometry_encoded)
        
        # Get final result
        frequency = np.mean(antenna_spec.frequency_range)
        best_result = solver.simulate(
            geometry=best_geometry,
            frequency=frequency,
            spec=antenna_spec
        )
        
        optimization_time = time.time() - start_time
        
        self.logger.info(f"Bayesian optimization completed in {optimization_time:.2f}s. "
                        f"Best objective: {best_objective:.4f}")
        
        return BayesianResult(
            best_objective=best_objective,
            best_geometry=best_geometry,
            best_result=best_result,
            n_evaluations=len(self.X_evaluated),
            optimization_time=optimization_time,
            convergence_history=convergence_history,
            acquisition_history=self.acquisition_values
        )
    
    def _create_acquisition_function(self) -> AcquisitionFunction:
        """Create acquisition function."""
        if self.acquisition_function_type == 'ucb':
            return UpperConfidenceBound(self.gp, kappa=2.576)
        elif self.acquisition_function_type == 'ei':
            return ExpectedImprovement(self.gp, xi=0.01)
        elif self.acquisition_function_type == 'pi':
            return ProbabilityOfImprovement(self.gp, xi=0.01)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function_type}")
    
    def _optimize_acquisition(self, acquisition_func: AcquisitionFunction) -> np.ndarray:
        """Find point that maximizes acquisition function."""
        n_dims = len(self.bounds[0])
        
        # Multi-start optimization
        n_starts = min(10, 5 * n_dims)
        best_x = None
        best_acq = -np.inf
        
        for _ in range(n_starts):
            # Random starting point
            x0 = np.random.uniform(self.bounds[0], self.bounds[1], n_dims)
            
            # Minimize negative acquisition (to maximize acquisition)
            try:
                result = minimize(
                    fun=lambda x: -acquisition_func(x.reshape(1, -1))[0],
                    x0=x0,
                    bounds=list(zip(self.bounds[0], self.bounds[1])),
                    method='L-BFGS-B',
                    options={'maxiter': 100}
                )
                
                if result.success:
                    acq_value = acquisition_func(result.x.reshape(1, -1))[0]
                    if acq_value > best_acq:
                        best_acq = acq_value
                        best_x = result.x
                        
            except Exception as e:
                self.logger.warning(f"Acquisition optimization failed: {str(e)}")
                continue
        
        # Fallback to random point if optimization failed
        if best_x is None:
            best_x = np.random.uniform(self.bounds[0], self.bounds[1], n_dims)
            self.logger.warning("Acquisition optimization failed, using random point")
        
        return best_x
    
    def _decode_geometry(self, encoded: np.ndarray) -> np.ndarray:
        """Convert encoded variables to antenna geometry."""
        # Simple decoding: create liquid metal antenna geometry
        geometry_size = (32, 32, 8)
        geometry = np.zeros(geometry_size)
        
        # Base patch
        patch_layer = geometry_size[2] - 2
        geometry[8:24, 8:24, patch_layer] = 1.0
        
        # Liquid metal channels based on encoded variables
        n_channels = min(len(encoded), 16)
        for i in range(n_channels):
            if encoded[i] > 0.5:  # Channel is filled
                x = 8 + (i % 4) * 4
                y = 8 + (i // 4) * 4
                
                # Add channel with some randomness
                channel_width = max(1, int(2 + encoded[i] * 2))
                channel_height = max(1, int(2 + encoded[i] * 2))
                
                x_end = min(x + channel_width, geometry_size[0])
                y_end = min(y + channel_height, geometry_size[1])
                
                geometry[x:x_end, y:y_end, patch_layer] = 1.0
        
        return geometry
    
    def suggest_next_point(self) -> np.ndarray:
        """Suggest next point to evaluate (for interactive optimization)."""
        if len(self.X_evaluated) == 0:
            # Return random point for first evaluation
            n_dims = len(self.bounds[0])
            return np.random.uniform(self.bounds[0], self.bounds[1], n_dims)
        
        # Fit GP and find next point
        X_train = np.array(self.X_evaluated)
        y_train = np.array(self.y_evaluated)
        self.gp.fit(X_train, y_train)
        
        acquisition_func = self._create_acquisition_function()
        x_next = self._optimize_acquisition(acquisition_func)
        
        return x_next
    
    def tell(self, x: np.ndarray, y: float) -> None:
        """Add evaluation result to optimizer."""
        self.X_evaluated.append(x.copy())
        self.y_evaluated.append(y)
    
    def get_best_result(self) -> Tuple[np.ndarray, float]:
        """Get current best point and objective value."""
        if not self.y_evaluated:
            raise ValueError("No evaluations available")
        
        best_idx = np.argmax(self.y_evaluated)
        return self.X_evaluated[best_idx].copy(), self.y_evaluated[best_idx]
    
    def save_state(self, filepath: str) -> None:
        """Save optimizer state to file."""
        state = {
            'X_evaluated': self.X_evaluated,
            'y_evaluated': self.y_evaluated,
            'acquisition_values': self.acquisition_values,
            'bounds': self.bounds,
            'acquisition_function_type': self.acquisition_function_type,
            'kernel': self.kernel,
            'n_initial_points': self.n_initial_points
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Optimizer state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load optimizer state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.X_evaluated = state['X_evaluated']
        self.y_evaluated = state['y_evaluated']
        self.acquisition_values = state.get('acquisition_values', [])
        self.bounds = state['bounds']
        self.acquisition_function_type = state['acquisition_function_type']
        self.kernel = state['kernel']
        self.n_initial_points = state['n_initial_points']
        
        self.logger.info(f"Optimizer state loaded from {filepath}")
    
    def get_gp_predictions(
        self, 
        X_test: np.ndarray, 
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get GP predictions at test points."""
        if len(self.X_evaluated) == 0:
            raise ValueError("No training data available")
        
        # Fit GP with current data
        X_train = np.array(self.X_evaluated)
        y_train = np.array(self.y_evaluated)
        self.gp.fit(X_train, y_train)
        
        return self.gp.predict(X_test, return_std=return_std)


# Utility functions for Bayesian optimization analysis
def plot_convergence(convergence_history: List[float], title: str = "Bayesian Optimization Convergence") -> None:
    """Plot convergence history."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Objective Value')
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


def analyze_acquisition_history(acquisition_history: List[Tuple[np.ndarray, float]]) -> Dict[str, Any]:
    """Analyze acquisition function values over optimization."""
    if not acquisition_history:
        return {}
    
    acq_values = [item[1] for item in acquisition_history]
    
    return {
        'mean_acquisition': np.mean(acq_values),
        'std_acquisition': np.std(acq_values),
        'min_acquisition': np.min(acq_values),
        'max_acquisition': np.max(acq_values),
        'final_acquisition': acq_values[-1] if acq_values else 0,
        'acquisition_trend': np.polyfit(range(len(acq_values)), acq_values, 1)[0] if len(acq_values) > 1 else 0
    }