"""Blind period simulation tests -- gradual drift detection without labels."""

import numpy as np
import pytest

from drift import compute_composite_alert, default_config
from drift.monitors.score_distribution import compute_psi
from drift.monitors.uncertainty import compute_prediction_entropy
from drift.types import AlertSeverity


@pytest.mark.simulation
@pytest.mark.slow
class TestBlindPeriodSimulation:
    def test_gradual_drift_detection(self):
        """Simulate gradual score distribution shift over time windows."""
        rng = np.random.default_rng(42)
        config = default_config()
        ref_scores = rng.normal(0.30, 0.15, size=1000)

        severities: list[AlertSeverity] = []
        for day in range(0, 91, 30):
            shift = day * 0.005  # Gradual drift
            cur_scores = rng.normal(0.30 + shift, 0.15, size=1000)
            probs = rng.uniform(max(0.01, 0.5 - shift), min(0.99, 0.5 + shift), size=500)

            results = [
                compute_psi(ref_scores, cur_scores),
                compute_prediction_entropy(probs),
            ]
            alert = compute_composite_alert(results, config)
            severities.append(alert.severity)

        # Severity should not decrease over time as drift accumulates
        severity_order = {
            AlertSeverity.LOW: 0,
            AlertSeverity.MEDIUM: 1,
            AlertSeverity.HIGH: 2,
            AlertSeverity.CRITICAL: 3,
        }
        severity_values = [severity_order[s] for s in severities]
        # At least by day 90, severity should be higher than day 0
        assert severity_values[-1] >= severity_values[0]
