import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path to import example.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from example import experiment, list_experiments


class TestExperiment:
    """Test the experiment function for numerical integration."""

    def test_deterministic_method_accuracy(self):
        """Test deterministic method produces accurate integral estimate."""
        result = experiment(seed=None, method="deterministic", N=1000)
        assert "estimate" in result
        assert "true_val" in result
        assert "error" in result
        assert result["true_val"] == 1 / 3
        assert result["error"] < 1e-6
        np.testing.assert_allclose(result["estimate"], 1 / 3, atol=1e-6)

    def test_deterministic_convergence(self):
        """Test that deterministic method error decreases with N."""
        result_10 = experiment(seed=None, method="deterministic", N=10)
        result_100 = experiment(seed=None, method="deterministic", N=100)
        result_1000 = experiment(seed=None, method="deterministic", N=1000)

        assert result_10["error"] > result_100["error"]
        assert result_100["error"] > result_1000["error"]

    def test_stochastic_method_seeded(self):
        """Test stochastic method with seed produces reproducible results."""
        result1 = experiment(seed=3000, method="stochastic", N=100)
        result2 = experiment(seed=3000, method="stochastic", N=100)

        assert result1["estimate"] == result2["estimate"]
        assert result1["error"] == result2["error"]

    def test_stochastic_method_reasonable(self):
        """Test stochastic method produces reasonable estimates."""
        result = experiment(seed=3000, method="stochastic", N=1000)

        assert 0 < result["estimate"] < 1
        assert result["error"] < 0.1
        assert result["true_val"] == 1 / 3

    def test_stochastic_different_seeds(self):
        """Test different seeds produce different estimates."""
        result1 = experiment(seed=3000, method="stochastic", N=100)
        result2 = experiment(seed=3001, method="stochastic", N=100)

        assert result1["estimate"] != result2["estimate"]

    @pytest.mark.parametrize("N", [10, 100, 1000])
    def test_deterministic_n_values(self, N):
        """Test deterministic method works for different N values."""
        result = experiment(seed=None, method="deterministic", N=N)
        assert 0.3 < result["estimate"] < 0.35
        assert result["error"] < 0.01

    def test_known_deterministic_values(self):
        """Test against known deterministic output values."""
        result = experiment(seed=None, method="deterministic", N=10)
        np.testing.assert_allclose(result["estimate"], 0.335391, atol=1e-6)
        np.testing.assert_allclose(result["error"], 2.057613e-03, atol=1e-8)

        result = experiment(seed=None, method="deterministic", N=100)
        np.testing.assert_allclose(result["estimate"], 0.333350, atol=1e-6)
        np.testing.assert_allclose(result["error"], 1.700507e-05, atol=1e-10)

    def test_known_stochastic_values(self):
        """Test against known stochastic output values with fixed seeds."""
        result = experiment(seed=3000, method="stochastic", N=10)
        np.testing.assert_allclose(result["estimate"], 0.332873, atol=1e-6)
        np.testing.assert_allclose(result["error"], 4.603193e-04, atol=1e-8)

        result = experiment(seed=3001, method="stochastic", N=1000)
        np.testing.assert_allclose(result["estimate"], 0.329567, atol=1e-6)
        np.testing.assert_allclose(result["error"], 3.766277e-03, atol=1e-8)


class TestListExperiments:
    """Test the list_experiments function for configuration generation."""

    def test_experiment_count(self):
        """Test that correct number of experiments is generated."""
        xps = list_experiments()
        assert len(xps) == 9

    def test_parameter_coverage(self):
        """Test all combinations of method and N are present."""
        xps = list_experiments()

        # Extract unique values
        methods = {x["method"] for x in xps}
        n_values = {x["N"] for x in xps}

        assert methods == {"stochastic", "deterministic"}
        assert n_values == {10, 100, 1000}

    def test_seed_handling_deterministic(self):
        """Test deterministic experiments have seed=None."""
        xps = list_experiments()
        deterministic = [x for x in xps if x["method"] == "deterministic"]

        assert len(deterministic) == 3
        assert all(x["seed"] is None for x in deterministic)

    def test_seed_handling_stochastic(self):
        """Test stochastic experiments have correct seeds."""
        xps = list_experiments()
        stochastic = [x for x in xps if x["method"] == "stochastic"]

        assert len(stochastic) == 6
        seeds = {x["seed"] for x in stochastic}
        assert seeds == {3000, 3001}

    def test_no_duplicates(self):
        """Test that there are no duplicate experiments."""
        xps = list_experiments()
        # Convert to tuples for comparison
        xps_tuples = [tuple(sorted(x.items())) for x in xps]
        assert len(xps_tuples) == len(set(xps_tuples))

    def test_all_experiments_valid(self):
        """Test that all generated experiments can be run."""
        xps = list_experiments()
        for kwargs in xps:
            result = experiment(**kwargs)
            assert "estimate" in result
            assert "error" in result

    def test_experiment_keys(self):
        """Test that all experiments have required keys."""
        xps = list_experiments()
        required_keys = {"method", "N", "seed"}

        for xp in xps:
            assert set(xp.keys()) == required_keys


class TestIntegration:
    """Integration tests running the full example workflow."""

    def test_run_all_experiments(self):
        """Test running all experiments from list_experiments."""
        xps = list_experiments()
        results = [experiment(**kwargs) for kwargs in xps]

        assert len(results) == 9
        assert all("estimate" in r for r in results)
        assert all("error" in r for r in results)

    def test_deterministic_more_accurate(self):
        """Test that deterministic is generally more accurate than stochastic for same N."""
        xps = list_experiments()
        results = [experiment(**kwargs) for kwargs in xps]

        # Compare at N=1000
        det_1000 = [
            r["error"]
            for xp, r in zip(xps, results, strict=True)
            if xp["method"] == "deterministic" and xp["N"] == 1000
        ][0]
        stoch_1000 = [
            r["error"]
            for xp, r in zip(xps, results, strict=True)
            if xp["method"] == "stochastic" and xp["N"] == 1000
        ]

        assert det_1000 < min(stoch_1000)

    def test_error_bounds(self):
        """Test all errors are within reasonable bounds."""
        xps = list_experiments()
        results = [experiment(**kwargs) for kwargs in xps]

        for xp, result in zip(xps, results, strict=True):
            assert 0 <= result["error"] < 0.1, f"Error too large for {xp}"

        # Deterministic should be very accurate for N=1000
        det_1000 = [
            (xp, r)
            for xp, r in zip(xps, results, strict=True)
            if xp["method"] == "deterministic" and xp["N"] == 1000
        ][0]
        assert det_1000[1]["error"] < 1e-6
