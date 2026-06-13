"""Exercise the evidence-sufficiency-calc bridge against the real package.

Skipped when the ``sufficiency`` extra is not installed; the dedicated CI job
installs the sibling package so contract drift surfaces here, not at user
runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sufficiency", reason="requires the sufficiency extra")

from integrations.evidence_sufficiency import get_sufficiency_score


def test_bridge_composes_real_sufficiency_score_from_raw_data() -> None:
    rng = np.random.default_rng(42)
    score = get_sufficiency_score(
        {"completeness": 0.9, "freshness_days": 3.0},
        reference_scores=rng.normal(0.5, 0.1, 500),
        production_scores=rng.normal(0.5, 0.1, 500),
        y_true=np.array([0, 1, 1, 0, 1, 0, 1, 1] * 25),
        y_pred=np.array([0, 1, 0, 0, 1, 0, 1, 1] * 25),
    )
    assert 0.0 <= score <= 1.0


def test_bridge_composes_real_sufficiency_score_from_precomputed() -> None:
    score = get_sufficiency_score(
        {
            "completeness": 0.95,
            "freshness_days": 1.0,
            "reliability": 0.9,
            "representativeness": 0.92,
        }
    )
    assert 0.0 <= score <= 1.0
