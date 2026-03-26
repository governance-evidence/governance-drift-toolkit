"""Tests for configuration factories."""

from drift.config import credit_scoring_config, default_config, fraud_detection_config
from drift.types import MonitorCategory


class TestDefaultConfig:
    def test_equal_weights(self):
        config = default_config()
        weights = list(config.weights.values())
        assert all(abs(w - 1.0 / 7) < 1e-6 for w in weights)

    def test_all_categories_present(self):
        config = default_config()
        assert set(config.weights.keys()) == set(MonitorCategory)


class TestFraudDetectionConfig:
    def test_weights_sum(self):
        config = fraud_detection_config()
        assert abs(sum(config.weights.values()) - 1.0) < 1e-6

    def test_score_distribution_weight(self):
        config = fraud_detection_config()
        assert config.weights[MonitorCategory.SCORE_DISTRIBUTION] == 0.20

    def test_lower_thresholds(self):
        config = fraud_detection_config()
        assert config.alert_thresholds.warning == 0.2


class TestCreditScoringConfig:
    def test_weights_sum(self):
        config = credit_scoring_config()
        assert abs(sum(config.weights.values()) - 1.0) < 1e-6

    def test_feature_drift_weight(self):
        config = credit_scoring_config()
        assert config.weights[MonitorCategory.FEATURE_DRIFT] == 0.20

    def test_outcome_maturity_weight(self):
        config = credit_scoring_config()
        assert config.weights[MonitorCategory.OUTCOME_MATURITY] == 0.20
