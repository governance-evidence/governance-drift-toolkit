"""Tests for composite alert logic."""

import pytest

from drift.composite import compute_composite_alert
from drift.config import credit_scoring_config
from drift.sequential import DriftEValueAccumulator
from drift.types import (
    AlertSeverity,
    AlertThresholds,
    DriftConfig,
    MonitorCategory,
)
from tests.conftest import make_monitor_result as _result


class TestComputeCompositeAlert:
    def test_no_triggers(self):
        config = DriftConfig()
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, False),
            _result(MonitorCategory.FEATURE_DRIFT, False),
        ]
        alert = compute_composite_alert(results, config)
        assert alert.triggered_monitors == 0
        assert alert.weighted_score == 0.0
        assert alert.severity == AlertSeverity.LOW

    def test_all_triggers(self):
        config = DriftConfig()
        results = [_result(cat, True) for cat in MonitorCategory]
        alert = compute_composite_alert(results, config)
        assert alert.triggered_monitors == 7
        assert abs(alert.weighted_score - 1.0) < 1e-6
        assert alert.severity == AlertSeverity.CRITICAL

    def test_partial_triggers(self):
        config = DriftConfig()
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, True),
            _result(MonitorCategory.FEATURE_DRIFT, True),
            _result(MonitorCategory.UNCERTAINTY, False),
        ]
        alert = compute_composite_alert(results, config)
        assert alert.triggered_monitors == 2
        assert alert.active_monitors == 3

    def test_too_few_monitors(self):
        config = DriftConfig(minimum_active_monitors=3)
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, True),
            _result(MonitorCategory.FEATURE_DRIFT, True),
        ]
        with pytest.raises(ValueError, match="Need >= 3 active monitors"):
            compute_composite_alert(results, config)

    def test_harmful_shift_suppression(self):
        config = DriftConfig()
        results = [_result(cat, True) for cat in MonitorCategory]
        alert = compute_composite_alert(results, config, sufficiency_score=0.9)
        assert alert.harmful_shift_suppressed is True
        assert alert.severity == AlertSeverity.LOW

    def test_harmful_shift_not_suppressed(self):
        config = DriftConfig()
        results = [_result(cat, True) for cat in MonitorCategory]
        alert = compute_composite_alert(results, config, sufficiency_score=0.5)
        assert alert.harmful_shift_suppressed is False

    def test_sequential_testing(self):
        config = DriftConfig()
        acc = DriftEValueAccumulator(threshold=0.5, alpha=0.05)
        results = [_result(cat, True) for cat in MonitorCategory]
        alert = compute_composite_alert(results, config, e_value_accumulator=acc)
        assert alert.e_value is not None
        assert alert.e_value > 0

    def test_sequential_rejection_escalates_to_critical(self):
        # Use lower thresholds so partial triggers give MEDIUM, not CRITICAL
        config = DriftConfig(
            alert_thresholds=AlertThresholds(warning=0.2, alert=0.4, critical=0.9),
        )
        acc = DriftEValueAccumulator(threshold=0.3, alpha=0.05)
        # Pre-reject by observing high scores directly
        for _ in range(50):
            acc.observe(0.95)
        assert acc.rejected is True
        # Now use a single triggered monitor -> weighted_score ~1/7 ≈ 0.14 -> LOW
        # But sequential test already rejected -> should escalate to CRITICAL
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, True),
            _result(MonitorCategory.FEATURE_DRIFT, False),
        ]
        alert = compute_composite_alert(results, config, e_value_accumulator=acc)
        assert alert.severity == AlertSeverity.CRITICAL

    def test_adversarial_aware_redistribution(self):
        config = DriftConfig()
        # SCORE_DISTRIBUTION not triggered, but 2+ others ARE triggered
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, False),
            _result(MonitorCategory.FEATURE_DRIFT, True),
            _result(MonitorCategory.UNCERTAINTY, True),
            _result(MonitorCategory.CROSS_MODEL, True, name="Cross-Model-Disagreement"),
        ]
        alert = compute_composite_alert(results, config)
        # Should have redistributed SCORE_DISTRIBUTION weight to CROSS_MODEL
        assert alert.triggered_monitors == 3

    def test_message_content(self):
        config = DriftConfig()
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, True, name="PSI"),
            _result(MonitorCategory.FEATURE_DRIFT, False, name="Feature-KL"),
        ]
        alert = compute_composite_alert(results, config)
        assert "PSI" in alert.message

    def test_no_triggered_message(self):
        config = DriftConfig()
        results = [_result(MonitorCategory.SCORE_DISTRIBUTION, False)]
        alert = compute_composite_alert(results, config)
        assert "No monitors triggered" in alert.message

    def test_multiple_monitors_same_category(self):
        config = DriftConfig()
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, False, name="PSI"),
            _result(MonitorCategory.SCORE_DISTRIBUTION, True, name="KS-test"),
        ]
        alert = compute_composite_alert(results, config)
        # Second monitor triggers the category
        assert alert.triggered_monitors == 1

    def test_credit_scoring_four_signal_weighting(self):
        config = credit_scoring_config()
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, False, name="PSI"),
            _result(MonitorCategory.FEATURE_DRIFT, True, name="Feature-PSI"),
            _result(MonitorCategory.UNCERTAINTY, True, name="Entropy"),
            _result(MonitorCategory.CROSS_MODEL, False, name="ConfKS"),
        ]
        alert = compute_composite_alert(results, config)
        assert abs(alert.weighted_score - (7 / 12)) < 1e-9

    def test_credit_scoring_redistribution_when_cross_model_triggered(self):
        config = credit_scoring_config()
        results = [
            _result(MonitorCategory.SCORE_DISTRIBUTION, False, name="PSI"),
            _result(MonitorCategory.FEATURE_DRIFT, True, name="Feature-PSI"),
            _result(MonitorCategory.UNCERTAINTY, True, name="Entropy"),
            _result(MonitorCategory.CROSS_MODEL, True, name="Cross-Model-Disagreement"),
        ]
        alert = compute_composite_alert(results, config)
        assert abs(alert.weighted_score - 1.0) < 1e-9
