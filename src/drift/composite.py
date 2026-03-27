"""Composite alert logic for multi-signal governance drift monitoring.

Combines results from multiple proxy monitors into a single governance
alert with severity classification, harmful-shift filtering, and
optional sequential testing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from drift.harmful_shift import apply_suppression
from drift.types import (
    AlertSeverity,
    CompositeAlert,
    DriftConfig,
    MonitorCategory,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from drift.sequential import DriftEValueAccumulator
    from drift.types import MonitorResult


def compute_composite_alert(
    results: Sequence[MonitorResult],
    config: DriftConfig,
    *,
    sufficiency_score: float | None = None,
    e_value_accumulator: DriftEValueAccumulator | None = None,
) -> CompositeAlert:
    """Compute a composite governance drift alert from monitor results.

    Parameters
    ----------
    results : sequence of MonitorResult
        Individual monitor results.
    config : DriftConfig
        Monitoring configuration with weights and thresholds.
    sufficiency_score : float, optional
        Current sufficiency score (from the Evidence Sufficiency Calculator)
        for harmful-shift filtering.
    e_value_accumulator : DriftEValueAccumulator, optional
        Accumulator for sequential testing.

    Returns
    -------
    CompositeAlert
        Composite alert with severity and response recommendation.

    Raises
    ------
    ValueError
        If fewer monitors than minimum_active_monitors are provided.
    """
    if len(results) < config.minimum_active_monitors:
        msg = f"Need >= {config.minimum_active_monitors} active monitors, got {len(results)}"
        raise ValueError(msg)

    # Group by category -- use first result per category for weighting
    category_triggered: dict[MonitorCategory, bool] = {}
    for r in results:
        if r.category not in category_triggered:
            category_triggered[r.category] = r.triggered
        else:
            # If any monitor in a category triggers, the category triggers
            category_triggered[r.category] = category_triggered[r.category] or r.triggered

    # Build effective weights (only for active categories)
    active_categories = set(category_triggered.keys())
    weights = {cat: config.weights[cat] for cat in active_categories}

    # Adversarial-aware: if SCORE_DISTRIBUTION not triggered but 2+ others are,
    # redistribute SCORE_DISTRIBUTION weight to CROSS_MODEL
    score_dist = MonitorCategory.SCORE_DISTRIBUTION
    cross_model = MonitorCategory.CROSS_MODEL
    triggered_count = sum(1 for t in category_triggered.values() if t)

    has_genuine_cross_model_monitor = any(
        r.category == cross_model and "confidence" not in r.monitor_name.lower() for r in results
    )

    if (
        score_dist in category_triggered
        and not category_triggered[score_dist]
        and triggered_count >= 2
        and cross_model in weights
        and has_genuine_cross_model_monitor
    ):
        redistributed = weights[score_dist]
        weights[score_dist] = 0.0
        weights[cross_model] = weights[cross_model] + redistributed

    # Compute weighted score
    total_weight = sum(weights.values())
    if total_weight > 0:
        weighted_score = (
            sum(
                weights[cat] * (1.0 if triggered else 0.0)
                for cat, triggered in category_triggered.items()
            )
            / total_weight
        )
    else:
        weighted_score = 0.0  # pragma: no cover

    # Classify severity
    severity = config.alert_thresholds.classify(weighted_score)

    # Sequential testing
    e_value: float | None = None
    if e_value_accumulator is not None:
        e_value_accumulator.observe(weighted_score)
        e_value = e_value_accumulator.e_value
        # If sequential test rejects, escalate to CRITICAL
        if e_value_accumulator.rejected and severity != AlertSeverity.CRITICAL:
            severity = AlertSeverity.CRITICAL

    # Build message
    triggered_names = [r.monitor_name for r in results if r.triggered]
    if triggered_names:
        msg = f"{len(triggered_names)} monitors triggered: {', '.join(triggered_names)}"
    else:
        msg = "No monitors triggered"

    alert = CompositeAlert(
        severity=severity,
        active_monitors=len(results),
        triggered_monitors=sum(1 for r in results if r.triggered),
        weighted_score=weighted_score,
        harmful_shift_suppressed=False,
        e_value=e_value,
        monitor_results=tuple(results),
        timestamp=datetime.now(tz=UTC),
        message=msg,
    )

    # Apply harmful-shift filter
    if sufficiency_score is not None:
        alert = apply_suppression(
            alert,
            sufficiency_score,
            threshold=config.sufficiency_suppression_threshold,
        )

    return alert
