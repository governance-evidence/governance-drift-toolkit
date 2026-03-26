"""Decision Event Schema reader.

Extracts monitor inputs from decision event (Decision Event Schema) streams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def extract_scores(
    events: list[dict[str, Any]],
    *,
    score_key: str = "score",
) -> NDArray[np.float64]:
    """Extract prediction scores from a list of decision events.

    Parameters
    ----------
    events : list of dict
        Decision events conforming to the Decision Event Schema.
    score_key : str
        Key within each event containing the prediction score (default "score").

    Returns
    -------
    ndarray
        1-D array of extracted scores.

    Raises
    ------
    ValueError
        If events is empty.
    KeyError
        If score_key is missing from any event.
    """
    if not events:
        msg = "Events list must be non-empty"
        raise ValueError(msg)
    return np.array([e[score_key] for e in events], dtype=np.float64)


def extract_features(
    events: list[dict[str, Any]],
    *,
    feature_keys: list[str],
) -> NDArray[np.float64]:
    """Extract feature matrix from a list of decision events.

    Parameters
    ----------
    events : list of dict
        Decision events conforming to the Decision Event Schema.
    feature_keys : list of str
        Keys to extract as columns.

    Returns
    -------
    ndarray
        2-D feature matrix (n_events, n_features).

    Raises
    ------
    ValueError
        If events is empty.
    """
    if not events:
        msg = "Events list must be non-empty"
        raise ValueError(msg)
    return np.array([[e[k] for k in feature_keys] for e in events], dtype=np.float64)
