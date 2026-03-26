"""Tests for governance response chain."""

import pytest

from drift.response import determine_response, escalation_chain
from drift.types import AlertSeverity, DriftConfig, ResponseAction
from tests.conftest import make_composite_alert


class TestDetermineResponse:
    @pytest.mark.parametrize(
        ("severity", "expected_action"),
        [
            (AlertSeverity.LOW, ResponseAction.MONITOR),
            (AlertSeverity.MEDIUM, ResponseAction.ALERT),
            (AlertSeverity.HIGH, ResponseAction.ESCALATE),
            (AlertSeverity.CRITICAL, ResponseAction.FALLBACK),
        ],
    )
    def test_severity_to_action(self, severity, expected_action):
        config = DriftConfig()
        resp = determine_response(make_composite_alert(severity), config)
        assert resp.action == expected_action

    def test_suppressed_always_monitor(self):
        config = DriftConfig()
        resp = determine_response(
            make_composite_alert(AlertSeverity.CRITICAL, suppressed=True), config
        )
        assert resp.action == ResponseAction.MONITOR
        assert "adequate" in resp.reason

    def test_e_value_rollback(self):
        config = DriftConfig(e_value_alpha=0.05)
        # e_value >= 1/0.05 = 20 -> ROLLBACK
        resp = determine_response(make_composite_alert(AlertSeverity.HIGH, e_value=25.0), config)
        assert resp.action == ResponseAction.ROLLBACK
        assert "Sequential test" in resp.reason

    def test_e_value_below_threshold(self):
        config = DriftConfig(e_value_alpha=0.05)
        resp = determine_response(make_composite_alert(AlertSeverity.HIGH, e_value=5.0), config)
        assert resp.action == ResponseAction.ESCALATE


class TestEscalationChain:
    def test_multiple_alerts(self):
        config = DriftConfig()
        alerts = [
            make_composite_alert(AlertSeverity.LOW),
            make_composite_alert(AlertSeverity.MEDIUM),
            make_composite_alert(AlertSeverity.HIGH),
        ]
        responses = escalation_chain(alerts, config)
        assert len(responses) == 3
        assert responses[0].action == ResponseAction.MONITOR
        assert responses[1].action == ResponseAction.ALERT
        assert responses[2].action == ResponseAction.ESCALATE

    def test_empty_chain(self):
        config = DriftConfig()
        responses = escalation_chain([], config)
        assert responses == []
