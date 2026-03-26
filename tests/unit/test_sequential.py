"""Tests for e-value sequential testing."""

import pytest

from drift.sequential import DriftEValueAccumulator


class TestDriftEValueAccumulator:
    def test_defaults(self):
        acc = DriftEValueAccumulator()
        assert acc.threshold == 0.5
        assert acc.alpha == 0.05
        assert acc.e_value == 1.0  # exp(0) = 1
        assert acc.rejected is False
        assert acc.observations == 0

    def test_observe_below_threshold(self):
        acc = DriftEValueAccumulator()
        rejected = acc.observe(0.1)
        assert rejected is False
        assert acc.observations == 1

    def test_observe_above_threshold(self):
        acc = DriftEValueAccumulator()
        acc.observe(0.9)
        assert acc.e_value > 1.0

    def test_accumulation_to_rejection(self):
        acc = DriftEValueAccumulator(threshold=0.5, alpha=0.05)
        # Repeatedly observe high scores -> should eventually reject
        for _ in range(50):
            if acc.observe(0.95):
                break
        assert acc.rejected is True
        assert acc.e_value >= 1.0 / 0.05

    def test_low_scores_no_rejection(self):
        acc = DriftEValueAccumulator()
        for _ in range(10):
            acc.observe(0.3)
        assert acc.rejected is False

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            DriftEValueAccumulator(threshold=0.0)

    def test_invalid_threshold_one(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            DriftEValueAccumulator(threshold=1.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            DriftEValueAccumulator(alpha=0.0)

    def test_invalid_log_e_value(self):
        with pytest.raises(ValueError, match="log_e_value must be finite"):
            DriftEValueAccumulator(log_e_value=float("inf"))

    def test_invalid_score(self):
        acc = DriftEValueAccumulator()
        with pytest.raises(ValueError, match="Score must be in"):
            acc.observe(1.5)

    def test_nonfinite_score(self):
        acc = DriftEValueAccumulator()
        with pytest.raises(ValueError, match="Score must be in"):
            acc.observe(float("nan"))

    def test_boundary_scores(self):
        acc = DriftEValueAccumulator()
        acc.observe(0.0)
        acc.observe(1.0)
        assert acc.observations == 2
