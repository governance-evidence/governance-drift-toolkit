"""Shared pytest fixtures for the governance drift toolkit test suite."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from drift.types import (
    AlertSeverity,
    CompositeAlert,
    DriftConfig,
    MonitorCategory,
    MonitorResult,
)


def make_monitor_result(
    category: MonitorCategory = MonitorCategory.SCORE_DISTRIBUTION,
    triggered: bool = True,
    name: str = "test",
    statistic: float = 0.3,
) -> MonitorResult:
    """Create a MonitorResult for testing."""
    return MonitorResult(
        monitor_name=name,
        category=category,
        statistic=statistic if triggered else 0.1,
        p_value=None,
        threshold=0.25,
        triggered=triggered,
        timestamp=datetime.now(tz=UTC),
    )


def make_composite_alert(
    severity: AlertSeverity = AlertSeverity.HIGH,
    suppressed: bool = False,
    e_value: float | None = None,
) -> CompositeAlert:
    """Create a CompositeAlert for testing."""
    r = make_monitor_result()
    return CompositeAlert(
        severity=severity,
        active_monitors=1,
        triggered_monitors=1,
        weighted_score=0.5,
        harmful_shift_suppressed=suppressed,
        e_value=e_value,
        monitor_results=(r,),
        timestamp=datetime.now(tz=UTC),
        message="test alert",
    )


@pytest.fixture
def default_config() -> DriftConfig:
    """Default DriftConfig with equal weights."""
    return DriftConfig()
