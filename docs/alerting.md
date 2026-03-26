# Composite Alert Logic

## Weighted Combination

```text
weighted_score = sum(w_i * triggered_i) / sum(w_i)
```

Only active (provided) monitor categories contribute to the score.

## Severity Classification

| Score Range | Severity |
|------------|----------|
| < warning | LOW |
| >= warning | MEDIUM |
| >= alert | HIGH |
| >= critical | CRITICAL |

Default thresholds: warning=0.3, alert=0.5, critical=0.7.

## Adversarial-Aware Redistribution

When score distribution is NOT triggered but 2+ other categories ARE:
the score distribution weight is redistributed to cross-model disagreement.
This prevents adversarial evasion that preserves the score distribution.

## Harmful-Shift Filter

If an sufficiency score (from the Evidence Sufficiency Calculator) is provided and remains above the suppression
threshold (default 0.8), the alert is suppressed to LOW severity.

## Sequential Testing

E-value accumulator provides anytime-valid rejection without fixed-window
bias. When the accumulated e-value exceeds 1/alpha, severity is escalated
to CRITICAL regardless of the weighted score.
