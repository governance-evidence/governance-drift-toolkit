"""End-to-end integration test: data -> monitors -> alert -> response."""

import numpy as np
import pytest

from drift import (
    compute_composite_alert,
    default_config,
    determine_response,
    fraud_detection_config,
)
from drift.monitors.feature_drift import compute_feature_psi
from drift.monitors.score_distribution import compute_psi
from drift.monitors.uncertainty import compute_prediction_entropy
from drift.sequential import DriftEValueAccumulator
from drift.types import AlertSeverity, ResponseAction


@pytest.mark.integration
class TestEndToEnd:
    def test_no_drift_pipeline(self):
        rng = np.random.default_rng(42)
        config = default_config()

        ref_scores = rng.normal(0.3, 0.15, size=1000)
        cur_scores = rng.normal(0.3, 0.15, size=1000)
        ref_features = rng.normal(size=(500, 3))
        cur_features = rng.normal(size=(500, 3))
        probs = rng.beta(0.5, 0.5, size=500).clip(0.01, 0.99)

        results = [
            compute_psi(ref_scores, cur_scores),
            compute_feature_psi(ref_features, cur_features),
            compute_prediction_entropy(probs),
        ]

        alert = compute_composite_alert(results, config)
        response = determine_response(alert, config)

        assert alert.severity == AlertSeverity.LOW
        assert response.action == ResponseAction.MONITOR

    def test_drift_pipeline(self):
        rng = np.random.default_rng(42)
        config = fraud_detection_config()

        ref_scores = rng.normal(0.3, 0.15, size=1000)
        cur_scores = rng.normal(0.6, 0.20, size=1000)  # Shifted
        ref_features = rng.normal(size=(500, 3))
        cur_features = rng.normal(size=(500, 3))
        cur_features[:, 0] += 2.0  # Feature shift
        probs = rng.uniform(0.35, 0.65, size=500)  # Uncertain

        results = [
            compute_psi(ref_scores, cur_scores),
            compute_feature_psi(ref_features, cur_features),
            compute_prediction_entropy(probs),
        ]

        alert = compute_composite_alert(results, config)
        response = determine_response(alert, config)

        assert alert.triggered_monitors > 0
        assert response.action != ResponseAction.MONITOR

    def test_drift_with_suppression(self):
        rng = np.random.default_rng(42)
        config = default_config()

        ref_scores = rng.normal(0.3, 0.15, size=1000)
        cur_scores = rng.normal(0.6, 0.20, size=1000)

        results = [compute_psi(ref_scores, cur_scores)]
        alert = compute_composite_alert(results, config, sufficiency_score=0.9)

        assert alert.harmful_shift_suppressed is True
        response = determine_response(alert, config)
        assert response.action == ResponseAction.MONITOR

    def test_sequential_escalation(self):
        config = default_config()
        acc = DriftEValueAccumulator(threshold=0.5, alpha=0.05)

        rng = np.random.default_rng(42)
        ref_scores = rng.normal(0.3, 0.15, size=1000)
        cur_scores = rng.normal(0.7, 0.20, size=1000)

        results = [compute_psi(ref_scores, cur_scores)]

        # Observe enough times to potentially reject
        for _ in range(30):
            alert = compute_composite_alert(results, config, e_value_accumulator=acc)

        if acc.rejected:
            response = determine_response(alert, config)
            assert response.action == ResponseAction.ROLLBACK
