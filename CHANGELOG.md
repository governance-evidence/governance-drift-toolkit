# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-06-13

Zenodo release: [10.5281/zenodo.20673692](https://doi.org/10.5281/zenodo.20673692).

### Fixed

- `drift.__version__` now matches the released package version (it had stayed at 0.1.0 after the v0.2.x releases); the
  public-API test now pins `__version__` against the installed package metadata so the two can no longer drift apart.
- `get_sufficiency_score` no longer crashes on the pre-computed reliability/representativeness fallback paths against
  evidence-sufficiency-calc v0.2.x: `DimensionScore` is now constructed with the required point-estimate confidence
  bounds. Caught by the new live bridge test.

### Added

- Live integration tests for the evidence-sufficiency-calc bridge (`tests/integration/test_sufficiency_live.py`),
  skipped when the extra is absent, plus a CI job that installs the sibling package from its pinned release tag and
  exercises the real import surface — sibling API drift now fails in CI instead of at user runtime.
- Release workflow publishing to PyPI via trusted publishing on tag push.
- Shared `drift.monitors._common.binned_proportions` helper consolidating the epsilon-smoothed histogram logic
  duplicated across the PSI and KL monitors; monitor outputs are numerically unchanged.

### Changed

- The mocked ImportError test for the sufficiency bridge is now hermetic (forces the import failure) instead of
  depending on the package being absent from the environment.

## [0.2.1] - 2026-03-27

Zenodo release: [10.5281/zenodo.19248601](https://doi.org/10.5281/zenodo.19248601).

Detailed change notes pending; see GitHub release notes for the interim summary.

## [0.2.0] - 2026-03-27

Zenodo release: [10.5281/zenodo.19244915](https://doi.org/10.5281/zenodo.19244915).

Detailed change notes pending; see GitHub release notes for the interim summary.

## [0.1.0] - 2026-03-26

Initial public release. Zenodo: [10.5281/zenodo.19236418](https://doi.org/10.5281/zenodo.19236418).

### Added

- Seven proxy monitor categories for label-free drift detection.
- Composite alerting with harmful-shift suppression (Amoukou-inspired).
- Governance response chain: Monitor -> Alert -> Escalate -> Fallback -> Rollback.
- E-value sequential testing utilities for anytime-valid inference.
- Apache-2.0 license.
- CITATION.cff for academic citation.
