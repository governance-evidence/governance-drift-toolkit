"""Property-based tests for mathematical invariants.

Uses hypothesis to verify that statistical properties hold across
randomly generated inputs, complementing the example-based unit tests.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from drift.composite import compute_composite_alert
from drift.monitors.score_distribution import compute_psi
from drift.sequential import DriftEValueAccumulator
from drift.types import DriftConfig, MonitorCategory
from tests.conftest import make_monitor_result

# -- Strategies ------------------------------------------------------------ #

finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
score_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=50, max_value=200),
    elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)


# -- PSI properties -------------------------------------------------------- #


class TestPSIProperties:
    @given(ref=score_arrays, cur=score_arrays)
    @settings(max_examples=30, deadline=2000)
    def test_psi_nonnegative(self, ref, cur):
        """PSI is always >= 0 for any valid distributions."""
        result = compute_psi(ref, cur)
        assert result.statistic >= 0.0

    @given(data=score_arrays)
    @settings(max_examples=20, deadline=2000)
    def test_identical_distributions_low_psi(self, data):
        """PSI of identical arrays should be near zero."""
        result = compute_psi(data, data.copy())
        assert result.statistic < 0.1
        assert result.triggered is False


# -- Composite alert properties -------------------------------------------- #


class TestCompositeAlertProperties:
    @given(
        triggered_flags=st.lists(st.booleans(), min_size=7, max_size=7),
    )
    @settings(max_examples=30, deadline=2000)
    def test_weighted_score_bounded(self, triggered_flags):
        """Weighted score is always in [0, 1]."""
        config = DriftConfig()
        categories = list(MonitorCategory)
        results = [
            make_monitor_result(cat, triggered=flag)
            for cat, flag in zip(categories, triggered_flags, strict=True)
        ]
        alert = compute_composite_alert(results, config)
        assert 0.0 <= alert.weighted_score <= 1.0

    @given(
        triggered_flags=st.lists(st.booleans(), min_size=7, max_size=7),
    )
    @settings(max_examples=30, deadline=2000)
    def test_triggered_count_consistent(self, triggered_flags):
        """Number of triggered monitors matches category count."""
        config = DriftConfig()
        categories = list(MonitorCategory)
        results = [
            make_monitor_result(cat, triggered=flag)
            for cat, flag in zip(categories, triggered_flags, strict=True)
        ]
        alert = compute_composite_alert(results, config)
        assert alert.triggered_monitors <= alert.active_monitors


# -- Suppression properties ------------------------------------------------ #


class TestSuppressionProperties:
    @given(
        sufficiency=st.floats(min_value=0.81, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=20, deadline=2000)
    def test_high_sufficiency_always_suppresses(self, sufficiency):
        """When sufficiency > threshold (0.8), suppression always occurs."""
        config = DriftConfig()
        categories = list(MonitorCategory)
        results = [make_monitor_result(cat, triggered=True) for cat in categories]
        alert = compute_composite_alert(results, config, sufficiency_score=sufficiency)
        assert alert.harmful_shift_suppressed is True

    @given(
        sufficiency=st.floats(min_value=0.0, max_value=0.79, allow_nan=False),
    )
    @settings(max_examples=20, deadline=2000)
    def test_low_sufficiency_never_suppresses(self, sufficiency):
        """When sufficiency < threshold, suppression never occurs."""
        config = DriftConfig()
        categories = list(MonitorCategory)
        results = [make_monitor_result(cat, triggered=True) for cat in categories]
        alert = compute_composite_alert(results, config, sufficiency_score=sufficiency)
        assert alert.harmful_shift_suppressed is False


# -- E-value properties ---------------------------------------------------- #


class TestEValueProperties:
    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=5,
            max_size=20,
        ),
    )
    @settings(max_examples=30, deadline=2000)
    def test_e_value_always_positive(self, scores):
        """E-value is always > 0 regardless of observed scores."""
        acc = DriftEValueAccumulator(threshold=0.5, alpha=0.05)
        for s in scores:
            acc.observe(s)
        assert acc.e_value > 0

    @given(
        n_obs=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=20, deadline=2000)
    def test_constant_high_scores_eventually_reject(self, n_obs):
        """Consistently high scores should accumulate e-value above 1."""
        acc = DriftEValueAccumulator(threshold=0.3, alpha=0.05)
        for _ in range(n_obs):
            acc.observe(0.95)
        # With score=0.95 >> threshold=0.3, e-value should grow
        assert acc.e_value > 1.0
