"""Factory functions for generating synthetic test distributions.

Each factory accepts a ``seed`` parameter for deterministic reproducibility.
Default parameters model realistic risk-scoring distributions:

- Scores center at 0.3 (typical low-risk baseline) with std=0.15
- Features use standard normal (mean=0, std=1) per dimension
- Probability arrays model calibrated model outputs in (0, 1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def stable_scores(n: int = 1000, seed: int = 42) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference and current score arrays from the same distribution (no drift)."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(0.3, 0.15, size=n)
    cur = rng.normal(0.3, 0.15, size=n)
    return ref, cur


def shifted_scores(
    n: int = 1000, shift: float = 0.2, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference and current score arrays with a mean shift (simulated drift)."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(0.3, 0.15, size=n)
    cur = rng.normal(0.3 + shift, 0.15, size=n)
    return ref, cur


def stable_features(
    n: int = 500, n_features: int = 3, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference and current feature matrices from the same distribution."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(n, n_features))
    cur = rng.normal(size=(n, n_features))
    return ref, cur


def shifted_features(
    n: int = 500, n_features: int = 3, shift: float = 1.0, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference and current features with a shift on the first feature."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(n, n_features))
    cur = rng.normal(size=(n, n_features))
    cur[:, 0] += shift
    return ref, cur


def low_confidence_probs(n: int = 500, seed: int = 42) -> NDArray[np.float64]:
    """Uniform probabilities in [0.35, 0.65] — high uncertainty regime."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.35, 0.65, size=n)


def high_confidence_probs(n: int = 500, seed: int = 42) -> NDArray[np.float64]:
    """Beta(0.5, 0.5) probabilities clipped to [0.01, 0.99] — bimodal confidence."""
    rng = np.random.default_rng(seed)
    probs = rng.beta(0.5, 0.5, size=n)
    return np.clip(probs, 0.01, 0.99)
