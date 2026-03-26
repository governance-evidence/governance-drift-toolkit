"""Tests for Flink pipeline connector Protocol."""

from integrations.flink import FlinkMonitorSink


class TestFlinkMonitorSinkProtocol:
    def test_protocol_is_runtime_checkable(self):
        assert (
            hasattr(FlinkMonitorSink, "__protocol_attrs__")
            or hasattr(FlinkMonitorSink, "__abstractmethods__")
            or callable(getattr(FlinkMonitorSink, "__instancecheck__", None))
        )

    def test_conforming_class_is_instance(self):
        class FakeSink:
            def publish_result(self, result: dict) -> None:
                pass

            def close(self) -> None:
                pass

        assert isinstance(FakeSink(), FlinkMonitorSink)

    def test_non_conforming_class_is_not_instance(self):
        class BadSink:
            pass

        assert not isinstance(BadSink(), FlinkMonitorSink)
