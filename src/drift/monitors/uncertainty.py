"""Uncertainty and confidence monitoring (entropy, confidence drift).

Detects calibration degradation and model uncertainty increase.
Misses confident-but-wrong predictions under concept drift.

WARNING: CBPE-style monitoring breaks under concept drift
(NannyML's own documentation acknowledges this limitation).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from drift.types import MonitorCategory, MonitorResult

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


def compute_prediction_entropy(
    probabilities: NDArray[np.floating[Any]],
    *,
    threshold: float = 0.7,
) -> MonitorResult:
    """Compute mean normalized prediction entropy as uncertainty measure.

    For binary classification, normalized entropy is::

        H = -(p * log2(p) + (1 - p) * log2(1 - p))

    High mean entropy indicates the model is uncertain, which degrades
    evidence reliability.

    Parameters
    ----------
    probabilities : ndarray
        Prediction probabilities in [0, 1]. For binary: 1-D array of
        positive-class probabilities. For multiclass: 2-D (n_samples, n_classes).
    threshold : float
        Mean entropy above which drift is triggered (default 0.7).

    Returns
    -------
    MonitorResult
        Result with mean normalized entropy.

    Raises
    ------
    ValueError
        If probabilities are empty or contain values outside [0, 1].
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        msg = "Probabilities array must be non-empty"
        raise ValueError(msg)

    if probs.ndim == 1:
        # Binary classification: convert to 2-class
        if np.any(probs < 0) or np.any(probs > 1):
            msg = "Probabilities must be in [0, 1]"
            raise ValueError(msg)
        probs_2d = np.column_stack([1 - probs, probs])
    else:
        probs_2d = probs

    n_classes = probs_2d.shape[1]
    max_entropy = np.log2(n_classes) if n_classes > 1 else 1.0

    # Clip to avoid log(0)
    clipped = np.clip(probs_2d, 1e-10, 1.0)
    entropy_per_sample = -np.sum(clipped * np.log2(clipped), axis=1)
    normalized = entropy_per_sample / max_entropy
    mean_entropy = float(np.mean(normalized))

    return MonitorResult(
        monitor_name="Prediction-Entropy",
        category=MonitorCategory.UNCERTAINTY,
        statistic=mean_entropy,
        p_value=None,
        threshold=threshold,
        triggered=mean_entropy > threshold,
        timestamp=datetime.now(tz=UTC),
        details={
            "n_samples": probs_2d.shape[0],
            "n_classes": n_classes,
            "std_entropy": float(np.std(normalized)),
        },
    )


def compute_confidence_drift(
    reference_confidences: NDArray[np.floating[Any]],
    current_confidences: NDArray[np.floating[Any]],
    *,
    alpha: float = 0.05,
) -> MonitorResult:
    """Detect systematic confidence shifts using KS test.

    Compares the distribution of model confidence scores (max class
    probability) between reference and current windows. A significant
    shift suggests calibration degradation.

    Parameters
    ----------
    reference_confidences : ndarray
        Reference confidence scores (max class probabilities).
    current_confidences : ndarray
        Current confidence scores.
    alpha : float
        Significance level for KS test (default 0.05).

    Returns
    -------
    MonitorResult
        Result with KS statistic and p-value.

    Raises
    ------
    ValueError
        If inputs are empty.
    """
    ref = np.asarray(reference_confidences, dtype=np.float64).ravel()
    cur = np.asarray(current_confidences, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        msg = "Reference and current arrays must be non-empty"
        raise ValueError(msg)

    ks_stat, p_value = stats.ks_2samp(ref, cur)

    return MonitorResult(
        monitor_name="Confidence-Drift",
        category=MonitorCategory.UNCERTAINTY,
        statistic=float(ks_stat),
        p_value=float(p_value),
        threshold=alpha,
        triggered=float(p_value) < alpha,
        timestamp=datetime.now(tz=UTC),
        details={
            "ref_mean": float(np.mean(ref)),
            "cur_mean": float(np.mean(cur)),
            "ref_size": ref.size,
            "cur_size": cur.size,
        },
    )
