"""CMA-ES optimizer for liquid metal antenna geometry."""

import numpy as np
from .antenna_geometry import AntennaGeometry
from .neural_surrogate import NeuralSurrogate
from .em_solver import EMSolver


class DifferentiableOptimizer:
    """
    Simplified CMA-ES that optimises antenna geometry toward a target frequency.

    The optimizer uses a NeuralSurrogate (trained internally on EMSolver samples)
    for fast function evaluations, and maintains a Gaussian distribution over the
    parameter space (length, width).

    Parameters
    ----------
    em_solver : EMSolver, optional
        Pre-configured EM solver.  A default instance is created if not provided.
    surrogate : NeuralSurrogate, optional
        Pre-trained surrogate.  If None, one is trained on EMSolver samples.
    pop_size : int
        CMA-ES population size (default 20).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        em_solver: EMSolver = None,
        surrogate: NeuralSurrogate = None,
        pop_size: int = 20,
        random_state=42,
    ):
        self.em_solver = em_solver or EMSolver()
        self.surrogate = surrogate
        self.pop_size = pop_size
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_surrogate(self):
        """Generate EM solver samples and train a fresh surrogate."""
        lengths = np.linspace(0.01, 0.1, 20)
        widths = np.linspace(0.01, 0.12, 20)
        Ls, Ws = np.meshgrid(lengths, widths)
        Ls = Ls.ravel()
        Ws = Ws.ravel()
        X = np.column_stack([Ls, Ws])
        y = np.array(
            [
                self.em_solver.resonant_frequency(AntennaGeometry(l, w))
                for l, w in zip(Ls, Ws)
            ]
        )
        surrogate = NeuralSurrogate(hidden_size=32, n_epochs=800, random_state=0)
        surrogate.train(X, y)
        return surrogate

    def _evaluate(self, params: np.ndarray, target_freq: float) -> float:
        """Return cost = (predicted_freq - target_freq)^2."""
        length = float(np.clip(params[0], 1e-4, 1.0))
        width = float(np.clip(params[1], 1e-4, 1.0))
        pred = self.surrogate.predict(np.array([[length, width]]))[0]
        return (pred - target_freq) ** 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        target_freq: float,
        n_iterations: int = 50,
        initial_length: float = 0.03,
        initial_width: float = 0.04,
    ) -> AntennaGeometry:
        """
        Find the antenna geometry closest to target_freq using simplified CMA-ES.

        Parameters
        ----------
        target_freq : float
            Desired resonant frequency in Hz.
        n_iterations : int
            Number of CMA-ES generations.
        initial_length : float
            Starting guess for patch length in metres.
        initial_width : float
            Starting guess for patch width in metres.

        Returns
        -------
        AntennaGeometry
            Best geometry found.
        """
        if self.surrogate is None:
            self.surrogate = self._train_surrogate()

        n_params = 2
        mean = np.array([initial_length, initial_width])
        sigma = 0.01  # initial step size
        C = np.eye(n_params)  # covariance matrix

        # CMA-ES hyper-parameters
        lam = self.pop_size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights**2).sum()

        c_sigma = (mu_eff + 2) / (n_params + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n_params + 1)) - 1) + c_sigma
        cc = (4 + mu_eff / n_params) / (n_params + 4 + 2 * mu_eff / n_params)
        c1 = 2 / ((n_params + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n_params + 2) ** 2 + mu_eff))

        p_sigma = np.zeros(n_params)
        p_c = np.zeros(n_params)
        chi_n = np.sqrt(n_params) * (1 - 1 / (4 * n_params) + 1 / (21 * n_params**2))

        best_params = mean.copy()
        best_cost = self._evaluate(mean, target_freq)

        for _ in range(n_iterations):
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n_params)
                L = np.eye(n_params)

            # Sample population
            z_samples = self.rng.standard_normal((lam, n_params))
            x_samples = mean + sigma * (z_samples @ L.T)
            x_samples = np.clip(x_samples, 1e-4, 1.0)

            # Evaluate and rank
            costs = np.array([self._evaluate(x, target_freq) for x in x_samples])
            idx = np.argsort(costs)

            # Track global best
            if costs[idx[0]] < best_cost:
                best_cost = costs[idx[0]]
                best_params = x_samples[idx[0]].copy()

            # Update mean
            old_mean = mean.copy()
            mean = weights @ x_samples[idx[:mu]]

            # Evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * (mean - old_mean) / sigma
            h_sig = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (_ + 1)))
                / chi_n
                < 1.4 + 2 / (n_params + 1)
            )
            p_c = (1 - cc) * p_c + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * (
                mean - old_mean
            ) / sigma

            # Covariance update
            artmp = (x_samples[idx[:mu]] - old_mean) / sigma
            C_mu = np.einsum("i,ij,ik->jk", weights, artmp, artmp)
            C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * C_mu

            # Step-size adaptation
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 1.0)

        length = float(np.clip(best_params[0], 1e-4, 1.0))
        width = float(np.clip(best_params[1], 1e-4, 1.0))
        return AntennaGeometry(length, width)
