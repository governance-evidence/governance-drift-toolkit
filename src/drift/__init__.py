"""Label-free monitoring of governance evidence degradation in risk decision systems."""

from __future__ import annotations

from importlib.metadata import version as _metadata_version

from drift.composite import compute_composite_alert
from drift.config import credit_scoring_config, default_config, fraud_detection_config
from drift.harmful_shift import apply_suppression, is_harmful_shift
from drift.proxy_sufficiency import (
    ProxySufficiencyResult,
    compute_proxy_sufficiency,
    estimate_dimensions,
    normalize_proxy,
)
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

# Single source of truth: the version lives in pyproject.toml alone. Keeping a
# literal here let it drift from the packaged version twice — at the 0.3.0 tag
# and again at 0.5.1 — and each time PyPI rejected the upload as a duplicate.
__version__ = _metadata_version("governance-drift-toolkit")

__all__ = [
    "AlertSeverity",
    "AlertThresholds",
    "CompositeAlert",
    "DriftConfig",
    "DriftEValueAccumulator",
    "GovernanceResponse",
    "MonitorCategory",
    "MonitorResult",
    "ProxySufficiencyResult",
    "ResponseAction",
    "apply_suppression",
    "compute_composite_alert",
    "compute_proxy_sufficiency",
    "credit_scoring_config",
    "default_config",
    "determine_response",
    "escalation_chain",
    "estimate_dimensions",
    "fraud_detection_config",
    "is_harmful_shift",
    "normalize_proxy",
]
