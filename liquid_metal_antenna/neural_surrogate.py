"""Two-layer MLP surrogate model for antenna resonant frequency prediction."""

import numpy as np


class NeuralSurrogate:
    """
    Lightweight 2-layer MLP (numpy-only) trained on EM solver samples.

    Architecture: input → hidden (ReLU) → output (linear)

    Parameters
    ----------
    hidden_size : int
        Number of neurons in the hidden layer (default 32).
    learning_rate : float
        Initial learning rate for Adam optimizer (default 1e-3).
    n_epochs : int
        Training epochs (default 500).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        learning_rate: float = 1e-3,
        n_epochs: int = 500,
        random_state=42,
    ):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(random_state)
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _relu(self, x):
        return np.maximum(0.0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _forward(self, X):
        h_pre = X @ self.W1 + self.b1  # (n, hidden)
        h = self._relu(h_pre)
        out = h @ self.W2 + self.b2  # (n, 1)
        return h_pre, h, out

    def _init_weights(self, n_features):
        scale1 = np.sqrt(2.0 / n_features)
        scale2 = np.sqrt(2.0 / self.hidden_size)
        self.W1 = self.rng.standard_normal((n_features, self.hidden_size)) * scale1
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = self.rng.standard_normal((self.hidden_size, 1)) * scale2
        self.b2 = np.zeros(1)
        # Adam moments
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)
        self._t = 0

    def _adam_update(self, param, grad, m, v, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**self._t)
        v_hat = v / (1 - beta2**self._t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        return param, m, v

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the surrogate on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features (e.g., [length, width]).
        y : np.ndarray, shape (n_samples,) or (n_samples, 1)
            Target resonant frequencies.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n_samples, n_features = X.shape

        # Normalise inputs for better convergence
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        Xn = (X - self.X_mean) / self.X_std
        yn = (y - self.y_mean) / self.y_std

        self._init_weights(n_features)

        for epoch in range(self.n_epochs):
            h_pre, h, out = self._forward(Xn)

            # MSE loss
            diff = out - yn
            loss = (diff**2).mean()  # noqa: F841 (kept for potential logging)

            # Backprop
            d_out = 2 * diff / n_samples  # (n, 1)
            dW2 = h.T @ d_out
            db2 = d_out.sum(axis=0)
            d_h = d_out @ self.W2.T * self._relu_grad(h_pre)
            dW1 = Xn.T @ d_h
            db1 = d_h.sum(axis=0)

            # Adam updates (increment _t once per epoch, not per parameter)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            self._t += 1
            for attr, grad, m_attr, v_attr in [
                ("W1", dW1, "mW1", "vW1"),
                ("b1", db1, "mb1", "vb1"),
                ("W2", dW2, "mW2", "vW2"),
                ("b2", db2, "mb2", "vb2"),
            ]:
                p = getattr(self, attr)
                m = getattr(self, m_attr)
                v = getattr(self, v_attr)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**self._t)
                v_hat = v / (1 - beta2**self._t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                setattr(self, attr, p)
                setattr(self, m_attr, m)
                setattr(self, v_attr, v)

        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict resonant frequency for input feature matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if not self._fitted:
            raise RuntimeError("Call train() before predict().")
        X = np.asarray(X, dtype=float)
        Xn = (X - self.X_mean) / self.X_std
        _, _, out = self._forward(Xn)
        return (out * self.y_std + self.y_mean).ravel()
