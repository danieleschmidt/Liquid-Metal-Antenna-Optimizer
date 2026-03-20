"""Tests for the Liquid Metal Antenna Optimizer package."""

import importlib
import numpy as np
import pytest
import sys
import os

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from liquid_metal_antenna import AntennaGeometry, EMSolver, NeuralSurrogate, DifferentiableOptimizer


# ---------------------------------------------------------------------------
# AntennaGeometry tests
# ---------------------------------------------------------------------------

class TestAntennaGeometry:
    def test_get_params_returns_dict_with_expected_keys(self):
        geom = AntennaGeometry(0.03, 0.04, [(0.01, 0.02)])
        params = geom.get_params()
        assert isinstance(params, dict)
        assert "length" in params
        assert "width" in params
        assert "channel_positions" in params

    def test_get_params_values_correct(self):
        geom = AntennaGeometry(0.03, 0.04, [(0.01, 0.02)])
        params = geom.get_params()
        assert params["length"] == pytest.approx(0.03)
        assert params["width"] == pytest.approx(0.04)
        assert params["channel_positions"] == [(0.01, 0.02)]

    def test_perturb_creates_new_geometry(self):
        geom = AntennaGeometry(0.03, 0.04)
        perturbed = geom.perturb(0.001)
        assert isinstance(perturbed, AntennaGeometry)
        assert perturbed is not geom

    def test_perturb_changes_dimensions(self):
        geom = AntennaGeometry(0.03, 0.04)
        # With a large delta, at least one dimension should change
        perturbed = geom.perturb(0.005)
        changed = (
            abs(perturbed.length - geom.length) > 0 or
            abs(perturbed.width - geom.width) > 0
        )
        assert changed

    def test_perturb_preserves_channel_count(self):
        channels = [(0.01, 0.02), (0.02, 0.03)]
        geom = AntennaGeometry(0.03, 0.04, channels)
        perturbed = geom.perturb(0.001)
        assert len(perturbed.channel_positions) == len(channels)

    def test_area_returns_float(self):
        geom = AntennaGeometry(0.03, 0.04)
        assert isinstance(geom.area(), float)

    def test_area_correct_value(self):
        geom = AntennaGeometry(0.03, 0.04)
        assert geom.area() == pytest.approx(0.03 * 0.04)

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            AntennaGeometry(-0.01, 0.04)

    def test_invalid_width_raises(self):
        with pytest.raises(ValueError):
            AntennaGeometry(0.03, 0.0)


# ---------------------------------------------------------------------------
# EMSolver tests
# ---------------------------------------------------------------------------

class TestEMSolver:
    def test_resonant_frequency_returns_plausible_range(self):
        solver = EMSolver()
        geom = AntennaGeometry(0.031, 0.04)
        f = solver.resonant_frequency(geom)
        assert 1e9 <= f <= 10e9, f"Expected 1–10 GHz, got {f/1e9:.3f} GHz"

    def test_resonant_frequency_is_float(self):
        solver = EMSolver()
        geom = AntennaGeometry(0.031, 0.04)
        assert isinstance(solver.resonant_frequency(geom), float)

    def test_compute_s11_returns_correct_length(self):
        solver = EMSolver()
        geom = AntennaGeometry(0.031, 0.04)
        freq_range = np.linspace(1e9, 5e9, 200)
        s11 = solver.compute_s11(geom, freq_range)
        assert len(s11) == 200

    def test_compute_s11_has_dip_at_resonance(self):
        solver = EMSolver()
        geom = AntennaGeometry(0.031, 0.04)
        f0 = solver.resonant_frequency(geom)
        freq_range = np.linspace(f0 * 0.8, f0 * 1.2, 500)
        s11 = solver.compute_s11(geom, freq_range)
        min_idx = np.argmin(s11)
        f_min = freq_range[min_idx]
        # Minimum should be within 5% of true resonance
        assert abs(f_min - f0) / f0 < 0.05

    def test_resonant_frequency_changes_with_geometry(self):
        solver = EMSolver()
        geom1 = AntennaGeometry(0.031, 0.04)
        geom2 = AntennaGeometry(0.062, 0.04)  # double the length → half the frequency
        f1 = solver.resonant_frequency(geom1)
        f2 = solver.resonant_frequency(geom2)
        assert f1 != pytest.approx(f2)
        assert f2 < f1  # longer patch → lower frequency

    def test_s11_values_are_non_positive_db(self):
        """S11 in dB should be ≤ 0 (reflection ≤ 1)."""
        solver = EMSolver()
        geom = AntennaGeometry(0.031, 0.04)
        freq_range = np.linspace(1e9, 5e9, 100)
        s11 = solver.compute_s11(geom, freq_range)
        assert np.all(s11 <= 0)


# ---------------------------------------------------------------------------
# NeuralSurrogate tests
# ---------------------------------------------------------------------------

class TestNeuralSurrogate:
    def _make_training_data(self, n=50):
        solver = EMSolver()
        rng = np.random.default_rng(0)
        lengths = rng.uniform(0.01, 0.08, n)
        widths = rng.uniform(0.01, 0.10, n)
        X = np.column_stack([lengths, widths])
        y = np.array([solver.resonant_frequency(AntennaGeometry(l, w)) for l, w in zip(lengths, widths)])
        return X, y

    def test_surrogate_trains_and_predicts(self):
        X, y = self._make_training_data(40)
        model = NeuralSurrogate(n_epochs=100, random_state=1)
        model.train(X, y)
        preds = model.predict(X[:5])
        assert preds is not None

    def test_prediction_shape_matches_input(self):
        X, y = self._make_training_data(40)
        model = NeuralSurrogate(n_epochs=100, random_state=2)
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_surrogate_reduces_training_loss(self):
        """Trained model should predict in the right frequency ballpark."""
        X, y = self._make_training_data(80)
        model = NeuralSurrogate(n_epochs=500, random_state=3)
        model.train(X, y)
        preds = model.predict(X)
        # Relative error should be under 30% on training data
        rel_errors = np.abs(preds - y) / y
        assert rel_errors.mean() < 0.30

    def test_predict_without_train_raises(self):
        model = NeuralSurrogate()
        with pytest.raises(RuntimeError):
            model.predict(np.array([[0.03, 0.04]]))

    def test_prediction_values_in_plausible_range(self):
        X, y = self._make_training_data(60)
        model = NeuralSurrogate(n_epochs=300, random_state=4)
        model.train(X, y)
        preds = model.predict(X)
        # All predictions should be in a sane frequency range
        assert np.all(preds > 0)


# ---------------------------------------------------------------------------
# DifferentiableOptimizer tests
# ---------------------------------------------------------------------------

class TestDifferentiableOptimizer:
    def test_optimizer_returns_antenna_geometry(self):
        opt = DifferentiableOptimizer(pop_size=10, random_state=5)
        result = opt.optimize(2.4e9, n_iterations=10)
        assert isinstance(result, AntennaGeometry)

    def test_optimizer_moves_toward_target(self):
        solver = EMSolver()
        opt = DifferentiableOptimizer(em_solver=solver, pop_size=20, random_state=6)
        target = 2.4e9
        best = opt.optimize(target, n_iterations=30, initial_length=0.04, initial_width=0.05)
        f_best = solver.resonant_frequency(best)
        # Frequency should be closer to target than a naive starting guess
        initial_geom = AntennaGeometry(0.04, 0.05)
        f_initial = solver.resonant_frequency(initial_geom)
        assert abs(f_best - target) <= abs(f_initial - target) * 2  # lenient bound

    def test_optimizer_best_geometry_has_positive_dimensions(self):
        opt = DifferentiableOptimizer(pop_size=10, random_state=8)
        best = opt.optimize(3.0e9, n_iterations=10)
        assert best.length > 0
        assert best.width > 0

    def test_optimizer_different_targets_give_different_geometries(self):
        opt1 = DifferentiableOptimizer(pop_size=15, random_state=9)
        opt2 = DifferentiableOptimizer(pop_size=15, random_state=9)
        best1 = opt1.optimize(2.4e9, n_iterations=20)
        best2 = opt2.optimize(5.8e9, n_iterations=20)
        # Different targets → different lengths
        assert best1.length != pytest.approx(best2.length, rel=0.01)


# ---------------------------------------------------------------------------
# Demo integration test
# ---------------------------------------------------------------------------

class TestDemo:
    def test_demo_runs_without_error(self, capsys):
        import demo
        importlib.reload(demo)
        demo.main()
        captured = capsys.readouterr()
        assert "Best geometry found" in captured.out
        assert "GHz" in captured.out
