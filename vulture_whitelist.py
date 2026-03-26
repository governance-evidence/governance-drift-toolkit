"""Vulture whitelist -- false positives for dead code detection.

Protocol method parameters and __all__ exports appear unused to vulture
but are part of the public API contract. This file must define matching
names so vulture treats them as used.
"""

# Protocol parameters (structural subtyping -- used by implementors)
# integrations/evidence_collector.py EvidenceStreamReader.read_batch
batch_size: int
