"""Harmful-shift filter for governance-aware alert suppression.

Applies the harmful-shift principle (Amoukou et al., NeurIPS 2024):
alert only on shifts that degrade governance evidence sufficiency,
not on benign distributional changes.

A drift alert is suppressed when the sufficiency score
(from the Evidence Sufficiency Calculator) remains
above the configured threshold, meaning governance evidence is still
adequate despite the detected distributional shift.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from drift.types import AlertSeverity, CompositeAlert


def is_harmful_shift(sufficiency_score: float, *, threshold: float = 0.8) -> bool:
    """Determine whether a detected shift is governance-harmful.

    Parameters
    ----------
    sufficiency_score : float
        Current evidence sufficiency score from Evidence Sufficiency Calculator in [0, 1].
    threshold : float
        Sufficiency score above which drift is considered benign
        (default 0.8).

    Returns
    -------
    bool
        True if the shift IS harmful (sufficiency below threshold).
    """
    return sufficiency_score < threshold


def apply_suppression(
    alert: CompositeAlert,
    sufficiency_score: float,
    *,
    threshold: float = 0.8,
) -> CompositeAlert:
    """Suppress a composite alert if the shift is not governance-harmful.

    If sufficiency remains above threshold, the alert severity is demoted
    to LOW and marked as suppressed. Otherwise the alert is returned unchanged.

    Parameters
    ----------
    alert : CompositeAlert
        The composite alert to potentially suppress.
    sufficiency_score : float
        Current evidence sufficiency score from Evidence Sufficiency Calculator.
    threshold : float
        Suppression threshold (default 0.8).

    Returns
    -------
    CompositeAlert
        Original alert if harmful, suppressed alert if benign.
    """
    if is_harmful_shift(sufficiency_score, threshold=threshold):
        return alert

    return dataclasses.replace(
        alert,
        severity=AlertSeverity.LOW,
        harmful_shift_suppressed=True,
        message=(
            f"Suppressed: drift detected but sufficiency {sufficiency_score:.2f} "
            f">= {threshold:.2f} -- governance evidence remains adequate"
        ),
        timestamp=datetime.now(tz=UTC),
    )
