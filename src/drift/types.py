"""Core data types for governance drift monitoring.

All types are frozen dataclasses -- immutable value objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime


class MonitorCategory(Enum):
    """Seven proxy metric categories for governance drift detection."""

    SCORE_DISTRIBUTION = "score_distribution"
    FEATURE_DRIFT = "feature_drift"
    UNCERTAINTY = "uncertainty"
    CROSS_MODEL = "cross_model"
    OPERATIONAL = "operational"
    OUTCOME_MATURITY = "outcome_maturity"
    PROXY_GROUND_TRUTH = "proxy_ground_truth"


class AlertSeverity(Enum):
    """Composite alert severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseAction(Enum):
    """Governance response chain actions."""

    MONITOR = "monitor"
    ALERT = "alert"
    ESCALATE = "escalate"
    FALLBACK = "fallback"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class AlertThresholds:
    """Thresholds for composite alert severity classification.

    Attributes
    ----------
    warning : float
        Minimum weighted score for MEDIUM severity (default 0.3).
    alert : float
        Minimum weighted score for HIGH severity (default 0.5).
    critical : float
        Minimum weighted score for CRITICAL severity (default 0.7).
    """

    warning: float = 0.3
    alert: float = 0.5
    critical: float = 0.7

    def __post_init__(self) -> None:
        for name, val in [
            ("warning", self.warning),
            ("alert", self.alert),
            ("critical", self.critical),
        ]:
            if not math.isfinite(val):
                msg = f"{name} must be finite, got {val}"
                raise ValueError(msg)
        if not 0.0 < self.warning < self.alert < self.critical <= 1.0:
            msg = (
                f"Need 0 < warning < alert < critical <= 1, "
                f"got {self.warning}, {self.alert}, {self.critical}"
            )
            raise ValueError(msg)

    def classify(self, score: float) -> AlertSeverity:
        """Classify a weighted score into an alert severity."""
        if score >= self.critical:
            return AlertSeverity.CRITICAL
        if score >= self.alert:
            return AlertSeverity.HIGH
        if score >= self.warning:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW


_DEFAULT_WEIGHTS: dict[MonitorCategory, float] = dict.fromkeys(MonitorCategory, 1.0 / 7)


@dataclass(frozen=True)
class DriftConfig:
    """Configuration for governance drift monitoring.

    Attributes
    ----------
    weights : Mapping[MonitorCategory, float]
        Per-category weights. Must sum to 1.0.
    minimum_active_monitors : int
        Minimum monitors required for composite alerting (default 1).
    alert_thresholds : AlertThresholds
        Severity classification thresholds.
    sufficiency_suppression_threshold : float
        Sufficiency score above which harmful-shift filter suppresses alerts
        (default 0.8).
    e_value_alpha : float
        Significance level for e-value sequential testing (default 0.05).
    """

    weights: Mapping[MonitorCategory, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    minimum_active_monitors: int = 1
    alert_thresholds: AlertThresholds = field(default_factory=AlertThresholds)
    sufficiency_suppression_threshold: float = 0.8
    e_value_alpha: float = 0.05

    def __post_init__(self) -> None:
        # Validate weights keys
        expected = set(MonitorCategory)
        got = set(self.weights.keys())
        if got != expected:
            msg = f"Weights must have all MonitorCategory keys, got {got}"
            raise ValueError(msg)
        # Validate weight values
        for cat, w in self.weights.items():
            if not math.isfinite(w) or not 0.0 <= w <= 1.0:
                msg = f"Weight for {cat.value} must be in [0, 1], got {w}"
                raise ValueError(msg)
        total = sum(self.weights.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
            msg = f"Weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        # Validate other fields
        if self.minimum_active_monitors < 1:
            msg = f"minimum_active_monitors must be >= 1, got {self.minimum_active_monitors}"
            raise ValueError(msg)
        if (
            not math.isfinite(self.sufficiency_suppression_threshold)
            or not 0.0 < self.sufficiency_suppression_threshold <= 1.0
        ):
            msg = (
                f"sufficiency_suppression_threshold must be in (0, 1], "
                f"got {self.sufficiency_suppression_threshold}"
            )
            raise ValueError(msg)
        if not math.isfinite(self.e_value_alpha) or not 0.0 < self.e_value_alpha < 1.0:
            msg = f"e_value_alpha must be in (0, 1), got {self.e_value_alpha}"
            raise ValueError(msg)
        # Freeze the mapping
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


@dataclass(frozen=True)
class MonitorResult:
    """Result from a single proxy monitor evaluation.

    Attributes
    ----------
    monitor_name : str
        Human-readable monitor name.
    category : MonitorCategory
        Which of the seven categories this monitor belongs to.
    statistic : float
        Computed test statistic (e.g. PSI value, KL divergence).
    p_value : float | None
        P-value if applicable (None for heuristic monitors).
    threshold : float
        Threshold used for triggering.
    triggered : bool
        Whether the monitor detected drift.
    timestamp : datetime
        When the evaluation was performed.
    details : Mapping[str, object]
        Additional monitor-specific details.
    """

    monitor_name: str
    category: MonitorCategory
    statistic: float
    p_value: float | None
    threshold: float
    triggered: bool
    timestamp: datetime
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.statistic):
            msg = f"statistic must be finite, got {self.statistic}"
            raise ValueError(msg)
        if self.p_value is not None and (
            not math.isfinite(self.p_value) or not 0.0 <= self.p_value <= 1.0
        ):
            msg = f"p_value must be in [0, 1] or None, got {self.p_value}"
            raise ValueError(msg)
        if not math.isfinite(self.threshold):
            msg = f"threshold must be finite, got {self.threshold}"
            raise ValueError(msg)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class CompositeAlert:
    """Composite alert from multi-signal combination.

    Attributes
    ----------
    severity : AlertSeverity
        Classified severity level.
    active_monitors : int
        Number of monitors that provided results.
    triggered_monitors : int
        Number of monitors that detected drift.
    weighted_score : float
        Weighted combination of triggered monitors.
    harmful_shift_suppressed : bool
        Whether the harmful-shift filter suppressed this alert.
    e_value : float | None
        Accumulated e-value from sequential testing.
    monitor_results : tuple[MonitorResult, ...]
        Individual monitor results.
    timestamp : datetime
        When the composite alert was computed.
    message : str
        Human-readable alert description.
    """

    severity: AlertSeverity
    active_monitors: int
    triggered_monitors: int
    weighted_score: float
    harmful_shift_suppressed: bool
    e_value: float | None
    monitor_results: tuple[MonitorResult, ...]
    timestamp: datetime
    message: str


@dataclass(frozen=True)
class GovernanceResponse:
    """Governance response to a composite alert.

    Attributes
    ----------
    action : ResponseAction
        Recommended governance action.
    alert : CompositeAlert
        The alert that triggered this response.
    reason : str
        Explanation for the recommended action.
    timestamp : datetime
        When the response was determined.
    """

    action: ResponseAction
    alert: CompositeAlert
    reason: str
    timestamp: datetime
