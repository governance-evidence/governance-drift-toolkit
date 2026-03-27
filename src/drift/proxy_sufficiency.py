"""Continuous proxy sufficiency estimation (Section 4.4).

Maps raw proxy monitor statistics to continuous dimension estimates,
then computes S_proxy(t) using the DA-05 sufficiency formula:

    S_proxy(t) = A(t) * [w_c*C + w_f*F + w_r*R_proxy + w_p*P_proxy]

where C and F are deterministic (observed directly from system metadata
and timestamps), and R_proxy and P_proxy are estimated from proxy
monitor signals via the coverage matrix (Table 6).

Dimension estimation per Section 4.4:

    D_i(t) = sum_j(w_ij * P_j(t)) / sum_j(w_ij)

where P_j(t) = max(0, 1 - raw_divergence / cap) is the health-mapped
proxy signal bounded to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from drift.types import MonitorCategory

if TYPE_CHECKING:
    from collections.abc import Mapping


# ---------------------------------------------------------------------------
# Coverage matrix (Table 6) — which categories estimate which dimensions
# ---------------------------------------------------------------------------

# Weights: Strong=1.0, Moderate=0.5, Weak=0.25, None=0 (absent)
COVERAGE_WEIGHTS: Mapping[MonitorCategory, Mapping[str, float]] = MappingProxyType(
    {
        MonitorCategory.SCORE_DISTRIBUTION: MappingProxyType(
            {"reliability": 0.25, "representativeness": 1.0}
        ),
        MonitorCategory.FEATURE_DRIFT: MappingProxyType({"representativeness": 1.0}),
        MonitorCategory.UNCERTAINTY: MappingProxyType({"reliability": 0.5}),
        # Categories 4-7: not yet implemented, weights defined for
        # future extensibility
        MonitorCategory.CROSS_MODEL: MappingProxyType({"reliability": 1.0}),
        MonitorCategory.OPERATIONAL: MappingProxyType({"reliability": 1.0}),
        MonitorCategory.OUTCOME_MATURITY: MappingProxyType(
            {"completeness": 0.5, "reliability": 1.0}
        ),
        MonitorCategory.PROXY_GROUND_TRUTH: MappingProxyType(
            {"completeness": 0.5, "reliability": 0.5}
        ),
    }
)

PROXY_ESTIMATED_DIMENSIONS = frozenset({"reliability", "representativeness"})
# Dimensions estimated from proxy signals. Completeness and freshness are
# deterministic and come directly from system metadata and timestamps.


@dataclass(frozen=True)
class ProxySufficiencyResult:
    """Result of continuous proxy sufficiency estimation.

    Attributes
    ----------
    s_proxy : float
        Composite proxy sufficiency score S_proxy(t) in [0, 1].
    gate : float
        Decision-readiness gate A(t) in [0, 1].
    r_proxy : float
        Proxy-estimated reliability R_proxy(t) in [0, 1].
    p_proxy : float
        Proxy-estimated representativeness P_proxy(t) in [0, 1].
    completeness : float
        Deterministic completeness C(t) in [0, 1].
    freshness : float
        Deterministic freshness F(t) in [0, 1].
    status : str
        Governance status: 'sufficient', 'degraded', or 'insufficient'.
    proxy_signals : dict[str, float]
        Per-category P_j(t) values for traceability.
    """

    s_proxy: float
    gate: float
    r_proxy: float
    p_proxy: float
    completeness: float
    freshness: float
    status: str
    proxy_signals: dict[str, float]


def normalize_proxy(raw_stat: float, cap: float) -> float:
    """Map raw divergence statistic to health signal P_j(t) in [0, 1].

    Parameters
    ----------
    raw_stat : float
        Raw divergence metric (e.g., PSI, KS statistic). Higher = worse.
    cap : float
        Normalization cap — the 99th percentile of the metric's observed
        range during the initial labeled period. Must be > 0.

    Returns
    -------
    float
        Health-mapped signal: 1.0 = no divergence, 0.0 = extreme divergence.

    Raises
    ------
    ValueError
        If cap <= 0.
    """
    if cap <= 0:
        msg = f"cap must be > 0, got {cap}"
        raise ValueError(msg)
    return max(0.0, 1.0 - raw_stat / cap)


def estimate_dimensions(
    proxy_signals: Mapping[MonitorCategory, float],
    coverage: Mapping[MonitorCategory, Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Estimate evidence dimensions from proxy signals (Section 4.4).

    Computes D_i(t) = sum_j(w_ij * P_j(t)) / sum_j(w_ij) for each
    proxy-estimated dimension (reliability, representativeness).

    Parameters
    ----------
    proxy_signals : mapping
        P_j(t) health values keyed by MonitorCategory. Only categories
        present in this mapping are used for estimation.
    coverage : mapping, optional
        Coverage weight matrix. Defaults to COVERAGE_WEIGHTS (Table 6).

    Returns
    -------
    dict[str, float]
        Estimated dimension values keyed by dimension name.
        Only returns proxy-estimated dimensions (reliability,
        representativeness). Missing dimensions (no proxy coverage)
        default to 1.0 (optimistic — no evidence of degradation).
    """
    if coverage is None:
        coverage = COVERAGE_WEIGHTS

    # Accumulate weighted signals per dimension
    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}

    for cat, p_j in proxy_signals.items():
        if cat not in coverage:
            continue
        for dim, w_ij in coverage[cat].items():
            if dim not in PROXY_ESTIMATED_DIMENSIONS:
                continue
            numerators[dim] = numerators.get(dim, 0.0) + w_ij * p_j
            denominators[dim] = denominators.get(dim, 0.0) + w_ij

    result: dict[str, float] = {}
    for dim in PROXY_ESTIMATED_DIMENSIONS:
        denom = denominators.get(dim, 0.0)
        if denom > 0:
            result[dim] = numerators.get(dim, 0.0) / denom
        else:
            # No proxy coverage for this dimension — assume healthy
            result[dim] = 1.0

    return result


def compute_proxy_sufficiency(
    proxy_signals: Mapping[MonitorCategory, float],
    completeness: float,
    freshness: float,
    *,
    weights: Mapping[str, float] | None = None,
    tau_c: float = 0.6,
    tau_r: float = 0.55,
    sufficient_threshold: float = 0.8,
    degraded_threshold: float = 0.5,
    coverage: Mapping[MonitorCategory, Mapping[str, float]] | None = None,
) -> ProxySufficiencyResult:
    """Compute continuous proxy sufficiency S_proxy(t).

    Implements the full Section 4.4 pipeline: normalize proxy signals,
    estimate R_proxy and P_proxy via coverage matrix, compute gate A(t),
    and produce S_proxy(t) = A(t) * weighted_sum.

    Parameters
    ----------
    proxy_signals : mapping
        P_j(t) health values keyed by MonitorCategory.
    completeness : float
        Deterministic C(t) from system metadata.
    freshness : float
        Deterministic F(t) from timestamps.
    weights : mapping, optional
        Dimension weights {completeness, freshness, reliability,
        representativeness}. Default: equal weights (0.25 each).
    tau_c : float
        Completeness gate threshold (default 0.6).
    tau_r : float
        Reliability gate threshold on normalized [0,1] scale (default 0.55).
        Calibrated to ~80% of mean baseline R_proxy so the gate fires
        when reliability health degrades significantly below normal range.
    sufficient_threshold : float
        S_proxy >= this → 'sufficient' (default 0.8).
    degraded_threshold : float
        S_proxy >= this → 'degraded', below → 'insufficient' (default 0.5).
    coverage : mapping, optional
        Coverage matrix. Defaults to COVERAGE_WEIGHTS.

    Returns
    -------
    ProxySufficiencyResult
        Continuous sufficiency score with full traceability.
    """
    if weights is None:
        weights = {
            "completeness": 0.25,
            "freshness": 0.25,
            "reliability": 0.25,
            "representativeness": 0.25,
        }

    # Estimate dimensions from proxy signals
    dims = estimate_dimensions(proxy_signals, coverage)
    r_proxy = dims.get("reliability", 1.0)
    p_proxy = dims.get("representativeness", 1.0)

    # Decision-readiness gate A(t) = min(1, C/tau_c) * min(1, R/tau_r)
    gate = min(1.0, completeness / tau_c) * min(1.0, r_proxy / tau_r)

    # Composite: S(t) = A(t) * [w_c*C + w_f*F + w_r*R + w_p*P]
    weighted_sum = (
        weights.get("completeness", 0.25) * completeness
        + weights.get("freshness", 0.25) * freshness
        + weights.get("reliability", 0.25) * r_proxy
        + weights.get("representativeness", 0.25) * p_proxy
    )
    s_proxy = gate * weighted_sum

    # Classify status
    if s_proxy >= sufficient_threshold:
        status = "sufficient"
    elif s_proxy >= degraded_threshold:
        status = "degraded"
    else:
        status = "insufficient"

    return ProxySufficiencyResult(
        s_proxy=s_proxy,
        gate=gate,
        r_proxy=r_proxy,
        p_proxy=p_proxy,
        completeness=completeness,
        freshness=freshness,
        status=status,
        proxy_signals={cat.name.lower(): val for cat, val in proxy_signals.items()},
    )
