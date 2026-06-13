"""Shared histogram helpers for distribution-drift monitors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

_EPSILON = 1e-8


def binned_proportions(
    data: NDArray[np.floating[Any]],
    edges: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Bin *data* by *edges* and return epsilon-smoothed proportions.

    The epsilon smoothing keeps downstream ``log`` ratios finite when a bin
    is empty; proportions sum to ~1.0.
    """
    counts = np.histogram(data, bins=edges)[0].astype(np.float64)
    counts += _EPSILON
    result: NDArray[np.float64] = counts / counts.sum()
    return result
