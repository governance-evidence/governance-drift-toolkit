"""Tests for uncertainty and confidence monitors."""

import numpy as np
import pytest

from drift.monitors.uncertainty import compute_confidence_drift, compute_prediction_entropy
from drift.types import MonitorCategory
from tests.fixtures.distributions import high_confidence_probs, low_confidence_probs


class TestComputePredictionEntropy:
    def test_high_confidence(self):
        probs = high_confidence_probs()
        result = compute_prediction_entropy(probs)
        assert result.category == MonitorCategory.UNCERTAINTY
        assert result.monitor_name == "Prediction-Entropy"
        # High confidence -> low entropy -> not triggered
        assert result.statistic < 0.7
        assert result.triggered is False

    def test_low_confidence(self):
        probs = low_confidence_probs()
        result = compute_prediction_entropy(probs)
        # Low confidence -> high entropy -> triggered
        assert result.statistic > 0.7
        assert result.triggered is True

    def test_custom_threshold(self):
        probs = high_confidence_probs()
        result = compute_prediction_entropy(probs, threshold=0.01)
        assert result.threshold == 0.01

    def test_multiclass(self):
        rng = np.random.default_rng(42)
        # 3-class with uniform-ish probabilities -> high entropy
        probs = rng.dirichlet([1, 1, 1], size=500)
        result = compute_prediction_entropy(probs)
        assert result.details["n_classes"] == 3

    def test_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_prediction_entropy(np.array([]))

    def test_invalid_probabilities(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            compute_prediction_entropy(np.array([-0.1, 0.5, 0.8]))

    def test_boundary_probabilities(self):
        # Probabilities at 0 and 1 should not crash (clipping handles them)
        probs = np.array([0.0, 1.0, 0.5, 0.5])
        result = compute_prediction_entropy(probs)
        assert result.statistic >= 0.0


class TestComputeConfidenceDrift:
    def test_same_distribution(self):
        ref = high_confidence_probs(seed=42)
        cur = high_confidence_probs(seed=43)
        result = compute_confidence_drift(ref, cur)
        assert result.category == MonitorCategory.UNCERTAINTY
        assert result.p_value is not None

    def test_different_distributions(self):
        ref = high_confidence_probs()
        cur = low_confidence_probs()
        result = compute_confidence_drift(ref, cur)
        assert result.triggered is True
        assert result.p_value is not None
        assert result.p_value < 0.05

    def test_custom_alpha(self):
        ref = high_confidence_probs()
        cur = high_confidence_probs(seed=43)
        result = compute_confidence_drift(ref, cur, alpha=0.001)
        assert result.threshold == 0.001

    def test_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_confidence_drift(np.array([]), np.array([1.0]))
