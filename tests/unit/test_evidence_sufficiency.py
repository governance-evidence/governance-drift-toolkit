"""Tests for evidence sufficiency calculator bridge."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from integrations.evidence_sufficiency import get_sufficiency_score


class TestGetSufficiencyScore:
    def test_import_error_without_dependency(self):
        """Raises ImportError when sufficiency package is not installed."""
        with pytest.raises(ImportError, match="evidence-sufficiency-calc"):
            get_sufficiency_score({"completeness": 0.85, "freshness_days": 7.0})

    def test_success_with_all_four_dimensions(self):
        """Returns a score when all four dimensions are computed."""
        mock_sufficiency = ModuleType("sufficiency")
        mock_dims = ModuleType("sufficiency.dimensions")
        mock_comp = ModuleType("sufficiency.dimensions.completeness")
        mock_fresh = ModuleType("sufficiency.dimensions.freshness")
        mock_rel = ModuleType("sufficiency.dimensions.reliability")
        mock_repr = ModuleType("sufficiency.dimensions.representativeness")
        mock_types = ModuleType("sufficiency.types")

        mock_config = MagicMock()
        mock_config.lambda_freshness = 0.02
        mock_config.ks_cap = 0.30

        mock_result = MagicMock()
        mock_result.composite = 0.75

        mock_dim_score = MagicMock()
        mock_dim_score.value = 0.85

        mock_sufficiency.compute_sufficiency = MagicMock(return_value=mock_result)
        mock_sufficiency.default_config = MagicMock(return_value=mock_config)
        mock_comp.compute_completeness = MagicMock(return_value="comp_score")
        mock_fresh.compute_freshness = MagicMock(return_value="fresh_score")
        mock_rel.compute_reliability = MagicMock(return_value="rel_score")
        mock_repr.compute_representativeness = MagicMock(return_value="repr_score")
        mock_types.DimensionScore = MagicMock(return_value=mock_dim_score)

        injected = {
            "sufficiency": mock_sufficiency,
            "sufficiency.dimensions": mock_dims,
            "sufficiency.dimensions.completeness": mock_comp,
            "sufficiency.dimensions.freshness": mock_fresh,
            "sufficiency.dimensions.reliability": mock_rel,
            "sufficiency.dimensions.representativeness": mock_repr,
            "sufficiency.types": mock_types,
        }

        with patch.dict(sys.modules, injected):
            # Without raw data: uses pre-computed reliability/representativeness
            score = get_sufficiency_score(
                {
                    "completeness": 0.9,
                    "freshness_days": 5.0,
                    "reliability": 0.85,
                    "representativeness": 0.80,
                }
            )
            assert score == 0.75

            call_args = mock_sufficiency.compute_sufficiency.call_args[0][0]
            assert set(call_args.keys()) == {
                "completeness",
                "freshness",
                "reliability",
                "representativeness",
            }

            mock_comp.compute_completeness.assert_called_once_with(
                labeled_count=9000, total_count=10000
            )
            mock_fresh.compute_freshness.assert_called_once_with(delta_t_days=5.0, lambda_rate=0.02)

    def test_success_with_raw_reliability_and_representativeness_inputs(self):
        """Computes reliability and representativeness from raw arrays when provided."""
        mock_sufficiency = ModuleType("sufficiency")
        mock_dims = ModuleType("sufficiency.dimensions")
        mock_comp = ModuleType("sufficiency.dimensions.completeness")
        mock_fresh = ModuleType("sufficiency.dimensions.freshness")
        mock_rel = ModuleType("sufficiency.dimensions.reliability")
        mock_repr = ModuleType("sufficiency.dimensions.representativeness")
        mock_types = ModuleType("sufficiency.types")

        mock_config = MagicMock()
        mock_config.lambda_freshness = 0.02
        mock_config.ks_cap = 0.30

        mock_result = MagicMock()
        mock_result.composite = 0.81

        mock_sufficiency.compute_sufficiency = MagicMock(return_value=mock_result)
        mock_sufficiency.default_config = MagicMock(return_value=mock_config)
        mock_comp.compute_completeness = MagicMock(return_value="comp_score")
        mock_fresh.compute_freshness = MagicMock(return_value="fresh_score")
        mock_rel.compute_reliability = MagicMock(return_value="rel_score")
        mock_repr.compute_representativeness = MagicMock(return_value="repr_score")
        mock_types.DimensionScore = MagicMock()

        injected = {
            "sufficiency": mock_sufficiency,
            "sufficiency.dimensions": mock_dims,
            "sufficiency.dimensions.completeness": mock_comp,
            "sufficiency.dimensions.freshness": mock_fresh,
            "sufficiency.dimensions.reliability": mock_rel,
            "sufficiency.dimensions.representativeness": mock_repr,
            "sufficiency.types": mock_types,
        }

        with patch.dict(sys.modules, injected):
            score = get_sufficiency_score(
                {"completeness": 0.9, "freshness_days": 5.0},
                reference_scores=[0.1, 0.2, 0.3],
                production_scores=[0.2, 0.3, 0.4],
                y_true=[0, 1, 0],
                y_pred=[0.2, 0.8, 0.3],
            )

        assert score == 0.81
        mock_rel.compute_reliability.assert_called_once_with(
            y_true=[0, 1, 0],
            y_pred=[0.2, 0.8, 0.3],
        )
        mock_repr.compute_representativeness.assert_called_once_with(
            reference=[0.1, 0.2, 0.3],
            production=[0.2, 0.3, 0.4],
            ks_cap=0.30,
        )
        mock_types.DimensionScore.assert_not_called()
