"""Tests for per-feature drift monitors."""

import numpy as np
import pytest

from drift.monitors.feature_drift import compute_feature_kl, compute_feature_psi
from drift.types import MonitorCategory
from tests.fixtures.distributions import shifted_features, stable_features


class TestComputeFeaturePsi:
    def test_stable_features(self):
        ref, cur = stable_features()
        result = compute_feature_psi(ref, cur)
        assert result.category == MonitorCategory.FEATURE_DRIFT
        assert result.monitor_name == "Feature-PSI"
        assert result.triggered is False

    def test_shifted_features(self):
        ref, cur = shifted_features(shift=2.0)
        result = compute_feature_psi(ref, cur)
        assert result.triggered is True
        assert result.details["max_feature"] == "f0"

    def test_custom_feature_names(self):
        ref, cur = stable_features()
        result = compute_feature_psi(ref, cur, feature_names=["a", "b", "c"])
        assert "a" in result.details["per_feature_psi"]

    def test_1d_input(self):
        ref, cur = stable_features(n_features=1)
        ref_1d = ref.ravel()
        cur_1d = cur.ravel()
        result = compute_feature_psi(ref_1d, cur_1d)
        assert result.details["n_features"] == 1

    def test_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_feature_psi(np.array([]), np.array([1.0]))

    def test_feature_count_mismatch(self):
        with pytest.raises(ValueError, match="Feature count mismatch"):
            compute_feature_psi(np.ones((10, 3)), np.ones((10, 4)))

    def test_feature_names_length_mismatch(self):
        ref, cur = stable_features()
        with pytest.raises(ValueError, match="feature_names length"):
            compute_feature_psi(ref, cur, feature_names=["a", "b"])

    def test_too_few_bins(self):
        ref, cur = stable_features()
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            compute_feature_psi(ref, cur, n_bins=1)


class TestComputeFeatureKl:
    def test_stable_features(self):
        ref, cur = stable_features()
        result = compute_feature_kl(ref, cur)
        assert result.category == MonitorCategory.FEATURE_DRIFT
        assert result.monitor_name == "Feature-KL"

    def test_shifted_features(self):
        ref, cur = shifted_features(shift=2.0)
        result = compute_feature_kl(ref, cur, threshold=0.01)
        assert result.triggered is True

    def test_1d_input(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(size=500)
        cur = rng.normal(size=500)
        result = compute_feature_kl(ref, cur)
        assert result.details["n_features"] == 1

    def test_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_feature_kl(np.array([]), np.array([1.0]))

    def test_feature_count_mismatch(self):
        with pytest.raises(ValueError, match="Feature count mismatch"):
            compute_feature_kl(np.ones((10, 3)), np.ones((10, 4)))

    def test_feature_names_length_mismatch(self):
        ref, cur = stable_features()
        with pytest.raises(ValueError, match="feature_names length"):
            compute_feature_kl(ref, cur, feature_names=["a"])

    def test_too_few_bins(self):
        ref, cur = stable_features()
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            compute_feature_kl(ref, cur, n_bins=1)
