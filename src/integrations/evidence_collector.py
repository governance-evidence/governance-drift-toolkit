"""Evidence Collector SDK reader.

Defines the Protocol interface for reading evidence streams from the
evidence-collector-sdk. The drift toolkit consumes batches of decision
events through this interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from typing import Any


@runtime_checkable
class EvidenceStreamReader(Protocol):
    """Protocol for reading evidence streams from the evidence collector.

    Implementors will connect to the evidence collector SDK
    and yield decision event batches.
    """

    def read_batch(self, *, batch_size: int = 100) -> list[dict[str, Any]]:
        """Read a batch of evidence events.

        Parameters
        ----------
        batch_size : int
            Maximum events to return per call.

        Returns
        -------
        list of dict
            Batch of decision events.
        """
        ...  # pragma: no cover

    def close(self) -> None:
        """Close the stream connection."""
        ...  # pragma: no cover
