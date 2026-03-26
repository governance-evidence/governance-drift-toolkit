"""E-value sequential testing for anytime-valid drift alerting.

Sequential testing avoids fixed-window bias by accumulating evidence
against H0 ("no harmful governance drift") over time. An e-value
exceeding 1/alpha provides an anytime-valid rejection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DriftEValueAccumulator:
    """Accumulate e-values for sequential governance drift testing.

    Attributes
    ----------
    threshold : float
        Score threshold; observations above this contribute evidence
        against H0 (default 0.5).
    alpha : float
        Significance level (default 0.05). Rejection at e_value >= 1/alpha.
    log_e_value : float
        Log of accumulated e-value (internal state).
    """

    threshold: float = 0.5
    alpha: float = 0.05
    log_e_value: float = 0.0
    _observations: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 < self.threshold < 1.0:
            msg = f"threshold must be in (0, 1), got {self.threshold}"
            raise ValueError(msg)
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            msg = f"alpha must be in (0, 1), got {self.alpha}"
            raise ValueError(msg)
        if not math.isfinite(self.log_e_value):
            msg = f"log_e_value must be finite, got {self.log_e_value}"
            raise ValueError(msg)

    def observe(self, score: float) -> bool:
        """Observe a composite weighted score and accumulate evidence.

        Parameters
        ----------
        score : float
            Composite weighted score in [0, 1].

        Returns
        -------
        bool
            True if e-value has reached rejection threshold.
        """
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            msg = f"Score must be in [0, 1], got {score}"
            raise ValueError(msg)

        self._observations += 1
        # One-sided e-value: accumulate evidence that the process mean
        # exceeds the threshold. Scores above threshold yield lr > 1
        # (evidence for drift); scores below yield lr < 1 (decay).
        # Using a simple betting fraction approach.
        bet = 0.5  # conservative betting fraction
        lr = 1.0 + bet * (score - self.threshold) / max(self.threshold, 1e-10)

        self.log_e_value += math.log(max(lr, 1e-300))
        return self.rejected

    @property
    def e_value(self) -> float:
        """Current accumulated e-value."""
        return math.exp(min(self.log_e_value, 700))  # Prevent overflow

    @property
    def rejected(self) -> bool:
        """Whether H0 has been rejected at the configured alpha level."""
        return self.e_value >= 1.0 / self.alpha

    @property
    def observations(self) -> int:
        """Number of observations accumulated."""
        return self._observations
