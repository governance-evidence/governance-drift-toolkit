"""Monitor configuration factories for common governance contexts."""

from __future__ import annotations

from drift.types import AlertThresholds, DriftConfig, MonitorCategory


def default_config() -> DriftConfig:
    """Create default configuration with equal weights across all monitors.

    Returns
    -------
    DriftConfig
        Configuration with 1/7 weight per category.
    """
    return DriftConfig()


def fraud_detection_config() -> DriftConfig:
    """Create configuration optimized for fraud detection systems.

    Higher weight on score distribution and operational proxies.
    Lower alert thresholds for faster escalation.

    Returns
    -------
    DriftConfig
        Fraud-detection-optimized configuration.
    """
    return DriftConfig(
        weights={
            MonitorCategory.SCORE_DISTRIBUTION: 0.20,
            MonitorCategory.FEATURE_DRIFT: 0.10,
            MonitorCategory.UNCERTAINTY: 0.10,
            MonitorCategory.CROSS_MODEL: 0.20,
            MonitorCategory.OPERATIONAL: 0.20,
            MonitorCategory.OUTCOME_MATURITY: 0.10,
            MonitorCategory.PROXY_GROUND_TRUTH: 0.10,
        },
        minimum_active_monitors=1,
        alert_thresholds=AlertThresholds(warning=0.2, alert=0.4, critical=0.6),
        sufficiency_suppression_threshold=0.8,
        e_value_alpha=0.05,
    )


def credit_scoring_config() -> DriftConfig:
    """Create configuration optimized for credit scoring systems.

    Higher weight on feature drift and outcome maturity (longer label delay).

    Returns
    -------
    DriftConfig
        Credit-scoring-optimized configuration.
    """
    return DriftConfig(
        weights={
            MonitorCategory.SCORE_DISTRIBUTION: 0.10,
            MonitorCategory.FEATURE_DRIFT: 0.20,
            MonitorCategory.UNCERTAINTY: 0.15,
            MonitorCategory.CROSS_MODEL: 0.15,
            MonitorCategory.OPERATIONAL: 0.10,
            MonitorCategory.OUTCOME_MATURITY: 0.20,
            MonitorCategory.PROXY_GROUND_TRUTH: 0.10,
        },
        minimum_active_monitors=1,
        alert_thresholds=AlertThresholds(warning=0.25, alert=0.45, critical=0.65),
        sufficiency_suppression_threshold=0.8,
        e_value_alpha=0.05,
    )
