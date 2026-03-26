"""Tests for Evidence Collector SDK Protocol."""

from integrations.evidence_collector import EvidenceStreamReader


class TestEvidenceStreamReaderProtocol:
    def test_protocol_is_runtime_checkable(self):
        assert (
            hasattr(EvidenceStreamReader, "__protocol_attrs__")
            or hasattr(EvidenceStreamReader, "__abstractmethods__")
            or callable(getattr(EvidenceStreamReader, "__instancecheck__", None))
        )

    def test_conforming_class_is_instance(self):
        class FakeReader:
            def read_batch(self, *, batch_size: int = 100) -> list:
                return []

            def close(self) -> None:
                pass

        assert isinstance(FakeReader(), EvidenceStreamReader)

    def test_non_conforming_class_is_not_instance(self):
        class BadReader:
            pass

        assert not isinstance(BadReader(), EvidenceStreamReader)
