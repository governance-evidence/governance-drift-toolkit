"""Tests for score distribution shift monitors."""

import numpy as np
import pytest

from drift.monitors.score_distribution import compute_ks_test, compute_psi
from drift.types import MonitorCategory
from tests.fixtures.distributions import shifted_scores, stable_scores


class TestComputePsi:
    def test_stable_distribution(self):
        ref, cur = stable_scores()
        result = compute_psi(ref, cur)
        assert result.category == MonitorCategory.SCORE_DISTRIBUTION
        assert result.monitor_name == "PSI"
        assert result.statistic < 0.25
        assert result.triggered is False

    def test_shifted_distribution(self):
        ref, cur = shifted_scores(shift=0.5)
        result = compute_psi(ref, cur)
        assert result.statistic > 0.25
        assert result.triggered is True

    def test_custom_threshold(self):
        ref, cur = shifted_scores(shift=0.1)
        result = compute_psi(ref, cur, threshold=0.01)
        assert result.threshold == 0.01
        assert result.triggered is True

    def test_custom_bins(self):
        ref, cur = stable_scores()
        result = compute_psi(ref, cur, n_bins=20)
        assert result.details["n_bins"] == 20

    def test_empty_reference(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_psi(np.array([]), np.array([1.0, 2.0]))

    def test_empty_current(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_psi(np.array([1.0, 2.0]), np.array([]))

    def test_too_few_bins(self):
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            compute_psi(np.array([1.0]), np.array([1.0]), n_bins=1)

    def test_p_value_is_none(self):
        ref, cur = stable_scores()
        result = compute_psi(ref, cur)
        assert result.p_value is None

    def test_psi_nonnegative(self):
        ref, cur = stable_scores()
        result = compute_psi(ref, cur)
        assert result.statistic >= 0.0


class TestComputeKsTest:
    def test_stable_distribution(self):
        ref, cur = stable_scores()
        result = compute_ks_test(ref, cur)
        assert result.category == MonitorCategory.SCORE_DISTRIBUTION
        assert result.monitor_name == "KS-test"
        assert result.p_value is not None
        # Same distribution -- p_value should be high
        assert result.p_value > 0.01

    def test_shifted_distribution(self):
        ref, cur = shifted_scores(shift=0.5)
        result = compute_ks_test(ref, cur)
        assert result.p_value is not None
        assert result.p_value < 0.05
        assert result.triggered is True

    def test_custom_alpha(self):
        ref, cur = stable_scores()
        result = compute_ks_test(ref, cur, alpha=0.001)
        assert result.threshold == 0.001

    def test_empty_inputs(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_ks_test(np.array([]), np.array([1.0]))
