"""Tests for public API exports."""

import drift


class TestPublicApi:
    def test_version(self):
        assert drift.__version__ == "0.1.0"

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
