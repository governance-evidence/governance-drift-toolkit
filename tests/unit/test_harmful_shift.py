"""Tests for harmful-shift filter."""

import pytest

from drift.harmful_shift import apply_suppression, is_harmful_shift
from drift.types import AlertSeverity
from tests.conftest import make_composite_alert


class TestIsHarmfulShift:
    @pytest.mark.parametrize(
        ("score", "threshold", "expected"),
        [
            (0.5, 0.8, True),  # below default -> harmful
            (0.9, 0.8, False),  # above default -> benign
            (0.8, 0.8, False),  # at threshold -> benign
            (0.6, 0.7, True),  # custom threshold
        ],
    )
    def test_harmful_shift_classification(self, score, threshold, expected):
        assert is_harmful_shift(score, threshold=threshold) is expected


class TestApplySuppression:
    def test_harmful_returns_unchanged(self):
        alert = make_composite_alert()
        result = apply_suppression(alert, sufficiency_score=0.5)
        assert result.severity == AlertSeverity.HIGH
        assert result.harmful_shift_suppressed is False

    def test_benign_suppresses(self):
        alert = make_composite_alert()
        result = apply_suppression(alert, sufficiency_score=0.9)
        assert result.severity == AlertSeverity.LOW
        assert result.harmful_shift_suppressed is True
        assert "Suppressed" in result.message

    def test_at_threshold_suppresses(self):
        alert = make_composite_alert()
        result = apply_suppression(alert, sufficiency_score=0.8)
        assert result.harmful_shift_suppressed is True

    def test_custom_threshold(self):
        alert = make_composite_alert()
        result = apply_suppression(alert, sufficiency_score=0.6, threshold=0.5)
        assert result.harmful_shift_suppressed is True
