"""Flink pipeline connector.

STUB: Defines the Protocol interface for connecting drift toolkit monitors
to a Flink streaming pipeline for continuous monitoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from typing import Any


@runtime_checkable
class FlinkMonitorSink(Protocol):
    """Protocol for sinking monitor results to a Flink pipeline.

    Implementors connect to the Flink deployment and publish
    monitoring results for downstream consumption.
    """

    def publish_result(self, result: dict[str, Any]) -> None:
        """Publish a monitor result to the Flink pipeline.

        Parameters
        ----------
        result : dict
            Serialized MonitorResult or CompositeAlert.
        """
        ...  # pragma: no cover

    def close(self) -> None:
        """Close the pipeline connection."""
        ...  # pragma: no cover
