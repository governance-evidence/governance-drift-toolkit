"""Tests for continuous proxy sufficiency estimation (Section 4.4)."""

from __future__ import annotations

import pytest

from drift.proxy_sufficiency import (
    COVERAGE_WEIGHTS,
    PROXY_ESTIMATED_DIMENSIONS,
    ProxySufficiencyResult,
    compute_proxy_sufficiency,
    estimate_dimensions,
    normalize_proxy,
)
from drift.types import MonitorCategory


class TestNormalizeProxy:
    def test_zero_divergence(self) -> None:
        assert normalize_proxy(0.0, 0.5) == 1.0

    def test_at_cap(self) -> None:
        assert normalize_proxy(0.5, 0.5) == 0.0

    def test_beyond_cap(self) -> None:
        assert normalize_proxy(1.0, 0.5) == 0.0

    def test_partial(self) -> None:
        assert normalize_proxy(0.25, 0.5) == pytest.approx(0.5)

    def test_small_divergence(self) -> None:
        assert normalize_proxy(0.1, 1.0) == pytest.approx(0.9)

    def test_cap_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="cap must be > 0"):
            normalize_proxy(0.1, 0.0)

    def test_negative_cap(self) -> None:
        with pytest.raises(ValueError, match="cap must be > 0"):
            normalize_proxy(0.1, -1.0)


class TestEstimateDimensions:
    def test_single_category_reliability(self) -> None:
        signals = {MonitorCategory.UNCERTAINTY: 0.8}
        dims = estimate_dimensions(signals)
        # UNCERTAINTY covers reliability with weight 0.5
        # D_reliability = 0.5 * 0.8 / 0.5 = 0.8
        assert dims["reliability"] == pytest.approx(0.8)

    def test_single_category_representativeness(self) -> None:
        signals = {MonitorCategory.FEATURE_DRIFT: 0.6}
        dims = estimate_dimensions(signals)
        # FEATURE_DRIFT covers representativeness with weight 1.0
        assert dims["representativeness"] == pytest.approx(0.6)

    def test_multiple_categories(self) -> None:
        signals = {
            MonitorCategory.SCORE_DISTRIBUTION: 0.9,
            MonitorCategory.FEATURE_DRIFT: 0.7,
            MonitorCategory.UNCERTAINTY: 0.5,
        }
        dims = estimate_dimensions(signals)

        # Reliability: SCORE_DIST(0.25*0.9) + UNC(0.5*0.5) / (0.25+0.5)
        expected_r = (0.25 * 0.9 + 0.5 * 0.5) / (0.25 + 0.5)
        assert dims["reliability"] == pytest.approx(expected_r)

        # Representativeness: SCORE_DIST(1.0*0.9) + FEAT(1.0*0.7) / (1.0+1.0)
        expected_p = (1.0 * 0.9 + 1.0 * 0.7) / (1.0 + 1.0)
        assert dims["representativeness"] == pytest.approx(expected_p)

    def test_no_coverage_defaults_to_one(self) -> None:
        # Empty signals → no coverage → default 1.0 (optimistic)
        dims = estimate_dimensions({})
        assert dims["reliability"] == 1.0
        assert dims["representativeness"] == 1.0

    def test_custom_coverage_matrix(self) -> None:
        custom = {
            MonitorCategory.SCORE_DISTRIBUTION: {"reliability": 1.0},
        }
        signals = {MonitorCategory.SCORE_DISTRIBUTION: 0.7}
        dims = estimate_dimensions(signals, coverage=custom)
        assert dims["reliability"] == pytest.approx(0.7)
        # No representativeness coverage → default 1.0
        assert dims["representativeness"] == 1.0

    def test_returns_only_proxy_dimensions(self) -> None:
        signals = {MonitorCategory.OUTCOME_MATURITY: 0.5}
        dims = estimate_dimensions(signals)
        # Should not return completeness even though OUTCOME_MATURITY covers it
        assert set(dims.keys()) == PROXY_ESTIMATED_DIMENSIONS

    def test_unknown_category_ignored(self) -> None:
        # If a category is not in the coverage matrix, ignore it
        signals = {MonitorCategory.SCORE_DISTRIBUTION: 0.8}
        custom = {}  # empty coverage
        dims = estimate_dimensions(signals, coverage=custom)
        assert dims["reliability"] == 1.0
        assert dims["representativeness"] == 1.0


class TestComputeProxySufficiency:
    def test_perfect_signals(self) -> None:
        signals = {
            MonitorCategory.SCORE_DISTRIBUTION: 1.0,
            MonitorCategory.FEATURE_DRIFT: 1.0,
            MonitorCategory.UNCERTAINTY: 1.0,
        }
        result = compute_proxy_sufficiency(
            signals,
            completeness=1.0,
            freshness=1.0,
        )
        assert isinstance(result, ProxySufficiencyResult)
        assert result.s_proxy == pytest.approx(1.0)
        assert result.gate == pytest.approx(1.0)
        assert result.status == "sufficient"

    def test_zero_signals(self) -> None:
        signals = {
            MonitorCategory.SCORE_DISTRIBUTION: 0.0,
            MonitorCategory.FEATURE_DRIFT: 0.0,
            MonitorCategory.UNCERTAINTY: 0.0,
        }
        result = compute_proxy_sufficiency(
            signals,
            completeness=1.0,
            freshness=1.0,
        )
        # R_proxy = 0, P_proxy = 0
        # gate = min(1, 1/0.6) * min(1, 0/0.15) = 1.0 * 0 = 0
        assert result.s_proxy == pytest.approx(0.0)
        assert result.gate == pytest.approx(0.0)
        assert result.status == "insufficient"

    def test_degraded_status(self) -> None:
        signals = {
            MonitorCategory.SCORE_DISTRIBUTION: 0.6,
            MonitorCategory.FEATURE_DRIFT: 0.6,
            MonitorCategory.UNCERTAINTY: 0.6,
        }
        result = compute_proxy_sufficiency(
            signals,
            completeness=0.8,
            freshness=0.7,
        )
        assert 0.5 <= result.s_proxy < 0.8
        assert result.status == "degraded"

    def test_low_freshness(self) -> None:
        signals = {
            MonitorCategory.SCORE_DISTRIBUTION: 1.0,
            MonitorCategory.FEATURE_DRIFT: 1.0,
            MonitorCategory.UNCERTAINTY: 1.0,
        }
        result = compute_proxy_sufficiency(
            signals,
            completeness=1.0,
            freshness=0.1,
        )
        # All dimensions 1.0 except freshness 0.1
        assert result.s_proxy < 1.0

    def test_custom_weights(self) -> None:
        signals = {MonitorCategory.UNCERTAINTY: 1.0}
        result = compute_proxy_sufficiency(
            signals,
            completeness=1.0,
            freshness=1.0,
            weights={
                "completeness": 0.20,
                "freshness": 0.30,
                "reliability": 0.30,
                "representativeness": 0.20,
            },
        )
        assert result.s_proxy > 0

    def test_custom_thresholds(self) -> None:
        signals = {MonitorCategory.UNCERTAINTY: 0.5}
        result = compute_proxy_sufficiency(
            signals,
            completeness=0.8,
            freshness=0.5,
            tau_c=0.5,
            tau_r=0.10,
        )
        assert isinstance(result.s_proxy, float)

    def test_gate_suppression(self) -> None:
        signals = {MonitorCategory.UNCERTAINTY: 0.1}
        result = compute_proxy_sufficiency(
            signals,
            completeness=0.3,
            freshness=1.0,
            tau_c=0.6,
            tau_r=0.15,
        )
        # Low reliability → gate < 1
        assert result.gate < 1.0

    def test_proxy_signals_in_result(self) -> None:
        signals = {
            MonitorCategory.SCORE_DISTRIBUTION: 0.9,
            MonitorCategory.FEATURE_DRIFT: 0.7,
        }
        result = compute_proxy_sufficiency(
            signals,
            completeness=1.0,
            freshness=1.0,
        )
        assert "score_distribution" in result.proxy_signals
        assert "feature_drift" in result.proxy_signals

    def test_empty_signals(self) -> None:
        result = compute_proxy_sufficiency(
            {},
            completeness=1.0,
            freshness=1.0,
        )
        # No proxy signals → dimensions default to 1.0
        assert result.r_proxy == 1.0
        assert result.p_proxy == 1.0
        assert result.s_proxy > 0


class TestCoverageWeights:
    def test_all_categories_present(self) -> None:
        for cat in MonitorCategory:
            assert cat in COVERAGE_WEIGHTS

    def test_weights_positive(self) -> None:
        for cat, dim_weights in COVERAGE_WEIGHTS.items():
            for dim, w in dim_weights.items():
                assert w > 0, f"{cat}.{dim} weight must be > 0"

    def test_weights_at_most_one(self) -> None:
        for cat, dim_weights in COVERAGE_WEIGHTS.items():
            for dim, w in dim_weights.items():
                assert w <= 1.0, f"{cat}.{dim} weight must be <= 1.0"


class TestProxySufficiencyResult:
    def test_frozen(self) -> None:
        result = ProxySufficiencyResult(
            s_proxy=0.5,
            gate=1.0,
            r_proxy=0.5,
            p_proxy=0.5,
            completeness=1.0,
            freshness=1.0,
            status="degraded",
            proxy_signals={},
        )
        with pytest.raises(AttributeError):
            result.s_proxy = 0.9  # type: ignore[misc]
