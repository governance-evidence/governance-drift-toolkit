"""Tests for Decision Event Schema reader."""

import numpy as np
import pytest

from integrations.decision_event_schema import extract_features, extract_scores


class TestExtractScores:
    def test_basic(self):
        events = [{"score": 0.5}, {"score": 0.8}, {"score": 0.2}]
        result = extract_scores(events)
        assert result.shape == (3,)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [0.5, 0.8, 0.2])

    def test_custom_key(self):
        events = [{"risk": 0.1}, {"risk": 0.9}]
        result = extract_scores(events, score_key="risk")
        np.testing.assert_allclose(result, [0.1, 0.9])

    def test_empty_events(self):
        with pytest.raises(ValueError, match="non-empty"):
            extract_scores([])

    def test_missing_key(self):
        with pytest.raises(KeyError):
            extract_scores([{"other": 1.0}])


class TestExtractFeatures:
    def test_basic(self):
        events = [
            {"a": 1.0, "b": 2.0},
            {"a": 3.0, "b": 4.0},
        ]
        result = extract_features(events, feature_keys=["a", "b"])
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_empty_events(self):
        with pytest.raises(ValueError, match="non-empty"):
            extract_features([], feature_keys=["a"])

    def test_missing_key(self):
        with pytest.raises(KeyError):
            extract_features([{"a": 1.0}], feature_keys=["a", "b"])
