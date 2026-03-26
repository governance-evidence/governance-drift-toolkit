# Industry Baselines for Threshold Calibration

## PSI Thresholds

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.10 | No significant change |
| 0.10 - 0.25 | Moderate change, investigate |
| > 0.25 | Significant change, action required |

Industry standard (fraud/credit): PSI > 0.2 triggers review.

## KS Test

Two-sample KS test at alpha = 0.05. Rejection indicates statistically
significant distribution shift.

## Label Delay Windows

| Domain | Typical Delay |
|--------|---------------|
| Fraud (card-present) | 45-120 days (chargeback window) |
| Fraud (CNP) | 90-180 days |
| Credit scoring | 90-360 days |
| AML | 30-90 days |

Stripe: 120-day dispute window.

## Reference Architectures

- Adyen: Ghost/Challenger/Principal model architecture
- Industry standard: minimum 4 proxy metrics active simultaneously
- Sequential testing: e-value alpha = 0.05 (anytime-valid at 95% confidence)
