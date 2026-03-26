"""Label-free monitoring of governance evidence degradation in risk decision systems."""

from __future__ import annotations

from drift.composite import compute_composite_alert
from drift.config import credit_scoring_config, default_config, fraud_detection_config
from drift.harmful_shift import apply_suppression, is_harmful_shift
from drift.response import determine_response, escalation_chain
from drift.sequential import DriftEValueAccumulator
from drift.types import (
    AlertSeverity,
    AlertThresholds,
    CompositeAlert,
    DriftConfig,
    GovernanceResponse,
    MonitorCategory,
    MonitorResult,
    ResponseAction,
)

__version__ = "0.1.0"

__all__ = [
    "AlertSeverity",
    "AlertThresholds",
    "CompositeAlert",
    "DriftConfig",
    "DriftEValueAccumulator",
    "GovernanceResponse",
    "MonitorCategory",
    "MonitorResult",
    "ResponseAction",
    "apply_suppression",
    "compute_composite_alert",
    "credit_scoring_config",
    "default_config",
    "determine_response",
    "escalation_chain",
    "fraud_detection_config",
    "is_harmful_shift",
]
