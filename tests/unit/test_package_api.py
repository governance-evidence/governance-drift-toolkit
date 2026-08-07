"""Tests for public API exports."""

import tomllib
from pathlib import Path

import drift


class TestPublicApi:
    def test_version_matches_declared_source(self):
        """The packaged version must equal the one declared in pyproject.toml.

        `drift.__version__` now derives from installed metadata, so comparing the
        two would be tautological. The pair that can actually diverge is the
        declared version and the installed one, which is what a stale editable
        install or an unbumped release produces.
        """
        declared = tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        )["project"]["version"]
        assert drift.__version__ == declared, (
            f"installed {drift.__version__} but pyproject.toml declares {declared}; "
            "run `pip install -e . --no-deps` after a version bump"
        )

    def test_all_exports(self):
        expected = {
            "AlertSeverity",
            "AlertThresholds",
            "CompositeAlert",
            "DriftConfig",
            "DriftEValueAccumulator",
            "GovernanceResponse",
            "MonitorCategory",
            "MonitorResult",
            "ProxySufficiencyResult",
            "ResponseAction",
            "apply_suppression",
            "compute_composite_alert",
            "compute_proxy_sufficiency",
            "credit_scoring_config",
            "default_config",
            "determine_response",
            "escalation_chain",
            "estimate_dimensions",
            "fraud_detection_config",
            "is_harmful_shift",
            "normalize_proxy",
        }
        assert set(drift.__all__) == expected

    def test_all_importable(self):
        for name in drift.__all__:
            assert hasattr(drift, name), f"{name} not importable from drift"
