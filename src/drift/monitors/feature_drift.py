"""Per-feature drift monitoring (PSI and KL divergence).

Detects covariate shift in input feature space. Strong signal for
P(X) changes; misses real concept drift P(Y|X) when features are stable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as sp_stats

from drift.types import MonitorCategory, MonitorResult

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray


def _per_feature_psi(
    ref_col: NDArray[np.float64],
    cur_col: NDArray[np.float64],
    n_bins: int,
) -> float:
    """Compute PSI for a single feature column."""
    edges = np.linspace(float(ref_col.min()), float(ref_col.max()), n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    epsilon = 1e-8
    ref_counts = np.histogram(ref_col, bins=edges)[0].astype(np.float64) + epsilon
    cur_counts = np.histogram(cur_col, bins=edges)[0].astype(np.float64) + epsilon
    p = ref_counts / ref_counts.sum()
    q = cur_counts / cur_counts.sum()
    return float(np.sum((p - q) * np.log(p / q)))


def compute_feature_psi(
    reference: NDArray[np.floating[Any]],
    current: NDArray[np.floating[Any]],
    *,
    feature_names: Sequence[str] | None = None,
    n_bins: int = 10,
    threshold: float = 0.25,
) -> MonitorResult:
    """Compute PSI per input feature, reporting max PSI as aggregate statistic.

    Parameters
    ----------
    reference : ndarray
        Reference feature matrix (n_samples, n_features).
    current : ndarray
        Current feature matrix (n_samples, n_features).
    feature_names : sequence of str, optional
        Names for each feature column.
    n_bins : int
        Number of bins for PSI computation (default 10).
    threshold : float
        Max PSI above which drift is triggered (default 0.25).

    Returns
    -------
    MonitorResult
        Result with max feature PSI and per-feature details.

    Raises
    ------
    ValueError
        If inputs are empty, have different feature counts, or n_bins < 2.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)
    if ref.size == 0 or cur.size == 0:
        msg = "Reference and current arrays must be non-empty"
        raise ValueError(msg)
    if ref.shape[1] != cur.shape[1]:
        msg = f"Feature count mismatch: {ref.shape[1]} vs {cur.shape[1]}"
        raise ValueError(msg)
    if n_bins < 2:
        msg = f"n_bins must be >= 2, got {n_bins}"
        raise ValueError(msg)

    n_features = ref.shape[1]
    names = (
        list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
    )
    if len(names) != n_features:
        msg = f"feature_names length {len(names)} != n_features {n_features}"
        raise ValueError(msg)

    psi_values: dict[str, float] = {}
    for i, name in enumerate(names):
        psi_values[name] = _per_feature_psi(ref[:, i], cur[:, i], n_bins)

    max_psi = max(psi_values.values())
    max_feature = max(psi_values, key=psi_values.__getitem__)

    return MonitorResult(
        monitor_name="Feature-PSI",
        category=MonitorCategory.FEATURE_DRIFT,
        statistic=max_psi,
        p_value=None,
        threshold=threshold,
        triggered=max_psi > threshold,
        timestamp=datetime.now(tz=UTC),
        details={
            "per_feature_psi": psi_values,
            "max_feature": max_feature,
            "n_features": n_features,
        },
    )


def compute_feature_kl(
    reference: NDArray[np.floating[Any]],
    current: NDArray[np.floating[Any]],
    *,
    feature_names: Sequence[str] | None = None,
    n_bins: int = 10,
    threshold: float = 0.1,
) -> MonitorResult:
    """Compute KL divergence per input feature, reporting max as aggregate.

    Parameters
    ----------
    reference : ndarray
        Reference feature matrix (n_samples, n_features).
    current : ndarray
        Current feature matrix (n_samples, n_features).
    feature_names : sequence of str, optional
        Names for each feature column.
    n_bins : int
        Number of bins for distribution estimation (default 10).
    threshold : float
        Max KL divergence above which drift is triggered (default 0.1).

    Returns
    -------
    MonitorResult
        Result with max feature KL divergence and per-feature details.

    Raises
    ------
    ValueError
        If inputs are empty, have different feature counts, or n_bins < 2.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)
    if ref.size == 0 or cur.size == 0:
        msg = "Reference and current arrays must be non-empty"
        raise ValueError(msg)
    if ref.shape[1] != cur.shape[1]:
        msg = f"Feature count mismatch: {ref.shape[1]} vs {cur.shape[1]}"
        raise ValueError(msg)
    if n_bins < 2:
        msg = f"n_bins must be >= 2, got {n_bins}"
        raise ValueError(msg)

    n_features = ref.shape[1]
    names = (
        list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
    )
    if len(names) != n_features:
        msg = f"feature_names length {len(names)} != n_features {n_features}"
        raise ValueError(msg)

    epsilon = 1e-8
    kl_values: dict[str, float] = {}
    for i, name in enumerate(names):
        edges = np.linspace(float(ref[:, i].min()), float(ref[:, i].max()), n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_counts = np.histogram(ref[:, i], bins=edges)[0].astype(np.float64) + epsilon
        cur_counts = np.histogram(cur[:, i], bins=edges)[0].astype(np.float64) + epsilon
        p = ref_counts / ref_counts.sum()
        q = cur_counts / cur_counts.sum()
        kl_values[name] = float(sp_stats.entropy(p, q))

    max_kl = max(kl_values.values())
    max_feature = max(kl_values, key=kl_values.__getitem__)

    return MonitorResult(
        monitor_name="Feature-KL",
        category=MonitorCategory.FEATURE_DRIFT,
        statistic=max_kl,
        p_value=None,
        threshold=threshold,
        triggered=max_kl > threshold,
        timestamp=datetime.now(tz=UTC),
        details={
            "per_feature_kl": kl_values,
            "max_feature": max_feature,
            "n_features": n_features,
        },
    )
