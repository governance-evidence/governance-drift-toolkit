"""Fraud detection monitoring example.

Demonstrates the full Governance Drift Toolkit pipeline: generate synthetic data, run
three proxy monitors, compute a composite alert, and determine the
governance response.
"""

import numpy as np

from drift import compute_composite_alert, determine_response, fraud_detection_config
from drift.monitors.feature_drift import compute_feature_psi
from drift.monitors.score_distribution import compute_psi
from drift.monitors.uncertainty import compute_prediction_entropy
from drift.sequential import DriftEValueAccumulator

config = fraud_detection_config()
rng = np.random.default_rng(42)

# Reference window (stable)
ref_scores = rng.normal(0.30, 0.15, size=1000)
ref_features = rng.normal(size=(500, 4))

# Current window (drifted)
cur_scores = rng.normal(0.45, 0.20, size=1000)
cur_features = rng.normal(size=(500, 4))
cur_features[:, 0] += 1.5  # Feature 0 shifted

# Uncertain predictions
probs = rng.uniform(0.30, 0.70, size=500)

# Run monitors
results = [
    compute_psi(ref_scores, cur_scores),
    compute_feature_psi(
        ref_features, cur_features, feature_names=["amount", "velocity", "distance", "time_delta"]
    ),
    compute_prediction_entropy(probs),
]

# Composite alert without suppression
alert = compute_composite_alert(results, config)
response = determine_response(alert, config)

print("=== Fraud Detection Monitoring ===")
print()
for r in results:
    print(f"  {r.monitor_name}: stat={r.statistic:.4f}, triggered={r.triggered}")
print()
print(f"  Composite: severity={alert.severity.value}, score={alert.weighted_score:.3f}")
print(f"  Response:  action={response.action.value}")
print(f"  Reason:    {response.reason}")

# With harmful-shift suppression (sufficiency still high)
print()
print("=== With Harmful-Shift Suppression (sufficiency=0.85) ===")
alert_suppressed = compute_composite_alert(results, config, sufficiency_score=0.85)
response_suppressed = determine_response(alert_suppressed, config)
print(f"  Suppressed: {alert_suppressed.harmful_shift_suppressed}")
print(f"  Response:   {response_suppressed.action.value}")

# Sequential testing over multiple windows
print()
print("=== Sequential Testing ===")
acc = DriftEValueAccumulator(threshold=0.5, alpha=0.05)
for window in range(1, 6):
    shift = window * 0.1
    cur = rng.normal(0.30 + shift, 0.15, size=1000)
    r = [compute_psi(ref_scores, cur)]
    a = compute_composite_alert(r, config, e_value_accumulator=acc)
    print(f"  Window {window}: e_value={acc.e_value:.2f}, rejected={acc.rejected}")
