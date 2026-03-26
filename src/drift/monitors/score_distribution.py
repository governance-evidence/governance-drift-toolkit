"""Score distribution shift monitoring (PSI and KS tests).

Detects P(X) changes reflected in the prediction score distribution.
Strong early warning for covariate shift; misses adversarial drift
that preserves the score distribution.

PSI formula::

    PSI = sum((p_i - q_i) * ln(p_i / q_i))

Industry heuristic: PSI > 0.2 indicates significant shift.
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


def _bin_proportions(
    data: NDArray[np.floating[Any]],
    edges: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Compute bin proportions with epsilon smoothing to avoid log(0).

    Parameters
    ----------
    data : ndarray
        1-D array of values to bin.
    edges : ndarray
        Bin edges (n_bins + 1 values).

    Returns
    -------
    ndarray
        Bin proportions (sum to ~1.0), smoothed with epsilon.
    """
    counts = np.histogram(data, bins=edges)[0].astype(np.float64)
    epsilon = 1e-8
    counts += epsilon
    result: NDArray[np.float64] = counts / counts.sum()
    return result


def compute_psi(
    reference: NDArray[np.floating[Any]],
    current: NDArray[np.floating[Any]],
    *,
    n_bins: int = 10,
    threshold: float = 0.25,
) -> MonitorResult:
    """Compute Population Stability Index between reference and current scores.

    Parameters
    ----------
    reference : ndarray
        Reference (baseline) score distribution.
    current : ndarray
        Current (production) score distribution.
    n_bins : int
        Number of equal-width bins (default 10).
    threshold : float
        PSI value above which drift is triggered (default 0.25).

    Returns
    -------
    MonitorResult
        Result with PSI statistic and trigger status.

    Raises
    ------
    ValueError
        If inputs are empty or n_bins < 2.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        msg = "Reference and current arrays must be non-empty"
        raise ValueError(msg)
    if n_bins < 2:
        msg = f"n_bins must be >= 2, got {n_bins}"
        raise ValueError(msg)

    # Use reference distribution to define bin edges
    edges = np.linspace(float(ref.min()), float(ref.max()), n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    p = _bin_proportions(ref, edges)
    q = _bin_proportions(cur, edges)

    psi_value = float(np.sum((p - q) * np.log(p / q)))

    return MonitorResult(
        monitor_name="PSI",
        category=MonitorCategory.SCORE_DISTRIBUTION,
        statistic=psi_value,
        p_value=None,
        threshold=threshold,
        triggered=psi_value > threshold,
        timestamp=datetime.now(tz=UTC),
        details={"n_bins": n_bins, "ref_size": ref.size, "cur_size": cur.size},
    )


def compute_ks_test(
    reference: NDArray[np.floating[Any]],
    current: NDArray[np.floating[Any]],
    *,
    alpha: float = 0.05,
) -> MonitorResult:
    """Compute two-sample Kolmogorov-Smirnov test for score distribution shift.

    Parameters
    ----------
    reference : ndarray
        Reference (baseline) score distribution.
    current : ndarray
        Current (production) score distribution.
    alpha : float
        Significance level for triggering (default 0.05).

    Returns
    -------
    MonitorResult
        Result with KS statistic, p-value, and trigger status.

    Raises
    ------
    ValueError
        If inputs are empty.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        msg = "Reference and current arrays must be non-empty"
        raise ValueError(msg)

    ks_stat, p_value = stats.ks_2samp(ref, cur)

    return MonitorResult(
        monitor_name="KS-test",
        category=MonitorCategory.SCORE_DISTRIBUTION,
        statistic=float(ks_stat),
        p_value=float(p_value),
        threshold=alpha,
        triggered=float(p_value) < alpha,
        timestamp=datetime.now(tz=UTC),
        details={"ref_size": ref.size, "cur_size": cur.size},
    )
