"""Credit scoring monitoring example.

Demonstrates monitoring with 90-day label delay, feature drift detection,
and harmful-shift suppression for benign distributional changes.
"""

import numpy as np

from drift import compute_composite_alert, credit_scoring_config, determine_response
from drift.monitors.feature_drift import compute_feature_kl, compute_feature_psi
from drift.monitors.uncertainty import compute_confidence_drift

config = credit_scoring_config()
rng = np.random.default_rng(42)

feature_names = ["income", "debt_ratio", "credit_age", "num_inquiries"]
ref_features = rng.normal(size=(500, 4))
ref_confidences = rng.beta(5, 1, size=500).clip(0.5, 0.99)

print("=== Credit Scoring: 90-Day Label Delay ===")
print()

for month in range(1, 5):
    # Gradual feature shift over months
    shift = month * 0.3
    cur_features = rng.normal(size=(500, 4))
    cur_features[:, 0] += shift  # Income distribution shifting

    # Confidence degradation
    cur_confidences = rng.beta(5 - month * 0.5, 1 + month * 0.3, size=500).clip(0.1, 0.99)

    results = [
        compute_feature_psi(ref_features, cur_features, feature_names=feature_names),
        compute_feature_kl(ref_features, cur_features, feature_names=feature_names),
        compute_confidence_drift(ref_confidences, cur_confidences),
    ]

    # Simulate sufficiency staying high initially, then degrading
    sufficiency = max(0.4, 0.9 - month * 0.1)

    alert = compute_composite_alert(results, config, sufficiency_score=sufficiency)
    response = determine_response(alert, config)

    print(f"  Month {month}: sufficiency={sufficiency:.2f}")
    for r in results:
        status = "TRIGGERED" if r.triggered else "ok"
        print(f"    {r.monitor_name}: {r.statistic:.4f} [{status}]")
    print(f"    -> severity={alert.severity.value}, suppressed={alert.harmful_shift_suppressed}")
    print(f"    -> response={response.action.value}: {response.reason}")
    print()
