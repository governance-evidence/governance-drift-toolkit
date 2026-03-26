# Proxy Monitors

## 1. Score Distribution Shift

**Functions:** `compute_psi()`, `compute_ks_test()`

PSI formula: `PSI = sum((p_i - q_i) * ln(p_i / q_i))`

Industry heuristic: PSI > 0.2 indicates significant shift.

## 2. Feature Drift

**Functions:** `compute_feature_psi()`, `compute_feature_kl()`

Per-feature PSI or KL divergence. Reports max across all features.

## 3. Uncertainty

**Functions:** `compute_prediction_entropy()`, `compute_confidence_drift()`

Normalized prediction entropy: `H = -(p*log2(p) + (1-p)*log2(1-p))`

## 4-7. Not Yet Implemented

Cross-model disagreement, operational process proxies, outcome-maturity
modeling, and proxy ground truth are defined but not implemented in Phase 1.
