"""Governance response chain for drift alerts.

Maps composite alert severity to governance response actions:
Monitor -> Alert -> Escalate -> Fallback -> Rollback.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from drift.types import (
    AlertSeverity,
    GovernanceResponse,
    ResponseAction,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from drift.types import CompositeAlert, DriftConfig


def determine_response(
    alert: CompositeAlert,
    config: DriftConfig,
) -> GovernanceResponse:
    """Determine the appropriate governance response for a composite alert.

    Parameters
    ----------
    alert : CompositeAlert
        The composite alert to respond to.
    config : DriftConfig
        Monitoring configuration.

    Returns
    -------
    GovernanceResponse
        Recommended governance action with reason.
    """
    if alert.harmful_shift_suppressed:
        return GovernanceResponse(
            action=ResponseAction.MONITOR,
            alert=alert,
            reason="Drift detected but governance evidence remains adequate",
            timestamp=datetime.now(tz=UTC),
        )

    # E-value sequential test rejection -> ROLLBACK
    if (
        alert.e_value is not None
        and config.e_value_alpha > 0
        and alert.e_value >= 1.0 / config.e_value_alpha
    ):
        return GovernanceResponse(
            action=ResponseAction.ROLLBACK,
            alert=alert,
            reason=(
                f"Sequential test rejected H0 (e-value={alert.e_value:.1f} "
                f">= {1.0 / config.e_value_alpha:.1f})"
            ),
            timestamp=datetime.now(tz=UTC),
        )

    severity_to_action: dict[AlertSeverity, tuple[ResponseAction, str]] = {
        AlertSeverity.LOW: (
            ResponseAction.MONITOR,
            "Low severity -- continue monitoring",
        ),
        AlertSeverity.MEDIUM: (
            ResponseAction.ALERT,
            "Medium severity -- notify governance owner",
        ),
        AlertSeverity.HIGH: (
            ResponseAction.ESCALATE,
            "High severity -- require human review of automated decisions",
        ),
        AlertSeverity.CRITICAL: (
            ResponseAction.FALLBACK,
            "Critical severity -- switch to conservative decision rules",
        ),
    }

    action, reason = severity_to_action[alert.severity]
    return GovernanceResponse(
        action=action,
        alert=alert,
        reason=reason,
        timestamp=datetime.now(tz=UTC),
    )


def escalation_chain(
    alerts: Sequence[CompositeAlert],
    config: DriftConfig,
) -> list[GovernanceResponse]:
    """Process a sequence of alerts through the governance response chain.

    Parameters
    ----------
    alerts : sequence of CompositeAlert
        Alerts to process, typically in chronological order.
    config : DriftConfig
        Monitoring configuration.

    Returns
    -------
    list of GovernanceResponse
        One response per alert.
    """
    return [determine_response(alert, config) for alert in alerts]
