"""Evidence Sufficiency Calculator bridge.

Provides bidirectional integration with the sufficiency calculator:
the drift toolkit feeds proxy values to the sufficiency calculator,
which provides sufficiency scores back for harmful-shift filtering.

Requires optional ``sufficiency`` dependency:
``pip install -e ".[sufficiency]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike


def get_sufficiency_score(
    dimensions: Mapping[str, float],
    *,
    reference_scores: ArrayLike | None = None,
    production_scores: ArrayLike | None = None,
    y_true: ArrayLike | None = None,
    y_pred: ArrayLike | None = None,
) -> float:
    """Compute sufficiency score using the evidence sufficiency calculator.

    Parameters
    ----------
    dimensions : mapping
        Dimension proxy values:
        - ``completeness``: fraction of labeled observations in [0, 1]
        - ``freshness_days``: days since last model update
        - ``reliability``: pre-computed reliability score in [0, 1]
          (used when ``y_true``/``y_pred`` not available)
        - ``representativeness``: pre-computed representativeness in [0, 1]
          (used when ``reference_scores``/``production_scores`` not available)
    reference_scores : array-like, optional
        Reference (training) score distribution for representativeness.
    production_scores : array-like, optional
        Current production score distribution for representativeness.
    y_true : array-like, optional
        Ground truth labels for reliability (bootstrap calibration).
    y_pred : array-like, optional
        Predicted probabilities for reliability.

    Returns
    -------
    float
        Composite sufficiency score S(t).

    Raises
    ------
    ImportError
        If evidence-sufficiency-calc is not installed.
    """
    try:
        from sufficiency import compute_sufficiency, default_config
        from sufficiency.dimensions.completeness import compute_completeness
        from sufficiency.dimensions.freshness import compute_freshness
        from sufficiency.dimensions.reliability import compute_reliability
        from sufficiency.dimensions.representativeness import compute_representativeness
        from sufficiency.types import DimensionScore
    except ImportError as exc:
        msg = (
            "evidence-sufficiency-calc is required for sufficiency integration. "
            "Install with: pip install -e '.[sufficiency]'"
        )
        raise ImportError(msg) from exc

    config = default_config()

    # Completeness
    completeness = compute_completeness(
        labeled_count=int(dimensions.get("completeness", 0.85) * 10000),
        total_count=10000,
    )

    # Freshness
    freshness = compute_freshness(
        delta_t_days=dimensions.get("freshness_days", 7.0),
        lambda_rate=config.lambda_freshness,
    )

    # Reliability: from raw data if available, else from pre-computed score
    if y_true is not None and y_pred is not None:
        reliability = compute_reliability(y_true=y_true, y_pred=y_pred)
    else:
        reliability = DimensionScore(
            value=dimensions.get("reliability", 0.85),
            label="reliability",
        )

    # Representativeness: from distributions if available, else pre-computed
    if reference_scores is not None and production_scores is not None:
        representativeness = compute_representativeness(
            reference=reference_scores,
            production=production_scores,
            ks_cap=config.ks_cap,
        )
    else:
        representativeness = DimensionScore(
            value=dimensions.get("representativeness", 0.85),
            label="representativeness",
        )

    dim_scores = {
        "completeness": completeness,
        "freshness": freshness,
        "reliability": reliability,
        "representativeness": representativeness,
    }

    result = compute_sufficiency(dim_scores, config)
    return float(result.composite)
