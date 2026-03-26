"""Tests for core data types and enums."""

from datetime import UTC, datetime

import pytest

from drift.types import (
    AlertSeverity,
    AlertThresholds,
    CompositeAlert,
    DriftConfig,
    GovernanceResponse,
    MonitorCategory,
    MonitorResult,
    ResponseAction,
)


class TestAlertThresholds:
    def test_defaults(self):
        t = AlertThresholds()
        assert t.warning == 0.3
        assert t.alert == 0.5
        assert t.critical == 0.7

    def test_custom(self):
        t = AlertThresholds(warning=0.2, alert=0.4, critical=0.6)
        assert t.warning == 0.2

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="warning < alert < critical"):
            AlertThresholds(warning=0.5, alert=0.3, critical=0.7)

    def test_warning_zero(self):
        with pytest.raises(ValueError, match="warning < alert < critical"):
            AlertThresholds(warning=0.0, alert=0.5, critical=0.7)

    def test_critical_above_one(self):
        with pytest.raises(ValueError, match="warning < alert < critical"):
            AlertThresholds(warning=0.3, alert=0.5, critical=1.1)

    def test_nonfinite(self):
        with pytest.raises(ValueError, match="must be finite"):
            AlertThresholds(warning=float("nan"), alert=0.5, critical=0.7)

    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.8, AlertSeverity.CRITICAL),
            (0.5, AlertSeverity.HIGH),
            (0.3, AlertSeverity.MEDIUM),
            (0.1, AlertSeverity.LOW),
        ],
    )
    def test_classify(self, score, expected):
        t = AlertThresholds()
        assert t.classify(score) == expected


class TestDriftConfig:
    def test_defaults(self):
        config = DriftConfig()
        assert len(config.weights) == 7
        assert abs(sum(config.weights.values()) - 1.0) < 1e-6
        assert config.minimum_active_monitors == 1

    def test_frozen_weights(self):
        config = DriftConfig()
        with pytest.raises(TypeError):
            config.weights[MonitorCategory.SCORE_DISTRIBUTION] = 0.5  # type: ignore[index]

    def test_missing_category(self):
        weights = dict.fromkeys(list(MonitorCategory)[:6], 1.0 / 6)
        with pytest.raises(ValueError, match="MonitorCategory"):
            DriftConfig(weights=weights)

    def test_weights_not_summing(self):
        weights = dict.fromkeys(MonitorCategory, 0.5)
        with pytest.raises(ValueError, match="sum to 1.0"):
            DriftConfig(weights=weights)

    def test_negative_weight(self):
        weights = dict.fromkeys(MonitorCategory, 1.0 / 7)
        weights[MonitorCategory.SCORE_DISTRIBUTION] = -0.1
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            DriftConfig(weights=weights)

    def test_min_active_zero(self):
        with pytest.raises(ValueError, match="minimum_active_monitors"):
            DriftConfig(minimum_active_monitors=0)

    def test_suppression_threshold_invalid(self):
        with pytest.raises(ValueError, match="sufficiency_suppression_threshold"):
            DriftConfig(sufficiency_suppression_threshold=0.0)

    def test_alpha_invalid(self):
        with pytest.raises(ValueError, match="e_value_alpha"):
            DriftConfig(e_value_alpha=1.0)


class TestMonitorResult:
    def test_valid(self):
        r = MonitorResult(
            monitor_name="test",
            category=MonitorCategory.SCORE_DISTRIBUTION,
            statistic=0.3,
            p_value=0.01,
            threshold=0.25,
            triggered=True,
            timestamp=datetime.now(tz=UTC),
        )
        assert r.triggered is True
        assert r.statistic == 0.3

    def test_none_p_value(self):
        r = MonitorResult(
            monitor_name="test",
            category=MonitorCategory.FEATURE_DRIFT,
            statistic=0.1,
            p_value=None,
            threshold=0.25,
            triggered=False,
            timestamp=datetime.now(tz=UTC),
        )
        assert r.p_value is None

    def test_invalid_statistic(self):
        with pytest.raises(ValueError, match="statistic must be finite"):
            MonitorResult(
                monitor_name="test",
                category=MonitorCategory.UNCERTAINTY,
                statistic=float("inf"),
                p_value=None,
                threshold=0.5,
                triggered=False,
                timestamp=datetime.now(tz=UTC),
            )

    def test_invalid_p_value(self):
        with pytest.raises(ValueError, match="p_value must be in"):
            MonitorResult(
                monitor_name="test",
                category=MonitorCategory.UNCERTAINTY,
                statistic=0.5,
                p_value=1.5,
                threshold=0.5,
                triggered=False,
                timestamp=datetime.now(tz=UTC),
            )

    def test_nonfinite_p_value(self):
        with pytest.raises(ValueError, match="p_value must be in"):
            MonitorResult(
                monitor_name="test",
                category=MonitorCategory.UNCERTAINTY,
                statistic=0.5,
                p_value=float("nan"),
                threshold=0.5,
                triggered=False,
                timestamp=datetime.now(tz=UTC),
            )

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold must be finite"):
            MonitorResult(
                monitor_name="test",
                category=MonitorCategory.UNCERTAINTY,
                statistic=0.5,
                p_value=None,
                threshold=float("inf"),
                triggered=False,
                timestamp=datetime.now(tz=UTC),
            )

    def test_frozen_details(self):
        r = MonitorResult(
            monitor_name="test",
            category=MonitorCategory.SCORE_DISTRIBUTION,
            statistic=0.1,
            p_value=None,
            threshold=0.25,
            triggered=False,
            timestamp=datetime.now(tz=UTC),
            details={"key": "value"},
        )
        with pytest.raises(TypeError):
            r.details["new_key"] = "new_value"  # type: ignore[index]


class TestEnums:
    def test_monitor_categories(self):
        assert len(MonitorCategory) == 7

    def test_alert_severity(self):
        assert len(AlertSeverity) == 4

    def test_response_action(self):
        assert len(ResponseAction) == 5
        assert ResponseAction.ROLLBACK.value == "rollback"


class TestCompositeAlert:
    def test_creation(self):
        r = MonitorResult(
            monitor_name="test",
            category=MonitorCategory.SCORE_DISTRIBUTION,
            statistic=0.3,
            p_value=None,
            threshold=0.25,
            triggered=True,
            timestamp=datetime.now(tz=UTC),
        )
        alert = CompositeAlert(
            severity=AlertSeverity.HIGH,
            active_monitors=1,
            triggered_monitors=1,
            weighted_score=0.5,
            harmful_shift_suppressed=False,
            e_value=None,
            monitor_results=(r,),
            timestamp=datetime.now(tz=UTC),
            message="test alert",
        )
        assert alert.severity == AlertSeverity.HIGH
        assert alert.active_monitors == 1


class TestGovernanceResponse:
    def test_creation(self):
        r = MonitorResult(
            monitor_name="test",
            category=MonitorCategory.SCORE_DISTRIBUTION,
            statistic=0.3,
            p_value=None,
            threshold=0.25,
            triggered=True,
            timestamp=datetime.now(tz=UTC),
        )
        alert = CompositeAlert(
            severity=AlertSeverity.HIGH,
            active_monitors=1,
            triggered_monitors=1,
            weighted_score=0.5,
            harmful_shift_suppressed=False,
            e_value=None,
            monitor_results=(r,),
            timestamp=datetime.now(tz=UTC),
            message="test",
        )
        resp = GovernanceResponse(
            action=ResponseAction.ESCALATE,
            alert=alert,
            reason="test reason",
            timestamp=datetime.now(tz=UTC),
        )
        assert resp.action == ResponseAction.ESCALATE
