# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Corrected the package version, which stayed at `0.5.0` through the 0.5.1 tag,
  so the release built a 0.5.0 wheel and PyPI rejected it as a duplicate. This
  is the second occurrence: the same thing happened at the 0.3.0 tag and was
  fixed by editing the number rather than the mechanism that let it drift.
- The version now lives in `pyproject.toml` alone. `drift.__version__` derives
  from installed metadata, so the two can no longer disagree, and the package
  test compares the installed version against the declared one instead of
  against itself.
- The release workflow refuses to build when the tag and the declared version
  disagree, naming both. The failure now arrives in seconds from this
  repository rather than as a `400 File already exists` from PyPI after a
  successful build.

- `results/ieee_cis_robustness/SUMMARY.md` is rebuilt from the current result
  JSON. It had been written by hand and fell two releases behind: its simulator
  checkpoints were pre-reseed values that no longer match the accompanying
  manuscript, and its Total column still summed the three-condition aggregate
  the manuscript withdrew. The stored JSON was correct throughout; only this
  human-readable rendering was stale.
- `normalize_proxy` no longer documents its cap as a percentile of the reference
  range. The cap is the larger of a per-metric floor and five times the maximum
  the metric takes across pairwise comparisons of thirds of the reference
  window, and in the reported instantiation three of four caps take their floor.
- The reproduction guard names the release its frozen values belong to instead
  of reporting `MISMATCH vs v31`, which had been wrong since the 0.4.0 refresh.
- Docstrings cite the accompanying manuscript's current table numbers. The
  coverage matrix is its Table 2, not Table 6.

## [0.5.0] - 2026-08-02

Zenodo release: [10.5281/zenodo.21762798](https://doi.org/10.5281/zenodo.21762798).

### Changed

- Schedule perturbation magnitude by calendar day instead of by window index.
  Indexing by window number made the injected process depend on the window
  scheme: the 45-day scheme reached only the third magnitude while the 15-day
  scheme reached the last one twice as early in calendar time, so window
  geometry was confounded with perturbation strength. The anchors reproduce the
  30-day base scheme exactly, and its published values are unchanged.
- Cross-fit the whole reference pipeline when estimating reference F1. Median
  imputation and scaling were previously fitted over the entire reference window
  before the folds, so a held-out record contributed to the statistics used to
  score it. Imputation, scaling and the classifier are now refitted inside each
  fold. The estimates move by at most 0.001.
- Correct the condition documentation to match the accompanying manuscript. The
  feature-perturbation condition is no longer described as preserving the
  conditional, and the label-flip condition is no longer called pure concept
  drift; the summary output states that its zero separation follows from the
  construction.

### Notes

- Package metadata is realigned. The v0.4.0 tag carried `version = "0.3.1"`,
  left its changes under `Unreleased`, and shipped a release record naming
  drift 0.2.2. Nothing in that tag is mutated; this release supersedes it as
  the reproducing archive.
- Published result values are unchanged from v0.4.0 apart from the 45-day
  scheme's reference F1, which moves from 0.132 to 0.131.

## [0.4.0] - 2026-07-31

Zenodo release: [10.5281/zenodo.21717036](https://doi.org/10.5281/zenodo.21717036).

### Changed

- Fit median imputation statistics on the reference window of each window scheme and apply them
  forward. They were previously fitted on the full 182-day span before the temporal split, which let
  monitoring windows inform the values used to fit and calibrate the reference pipeline.
- Estimate `ref_f1` by five-fold cross-fitting within the reference window instead of scoring the
  fitted model on its own training records. The in-sample value is retained as `ref_f1_insample` and
  is not used downstream.
- Renamed `results/ieee_cis_robustness_v32/` to `results/ieee_cis_robustness/`. The directory no
  longer encodes a manuscript revision; release identity is carried by `RELEASE.md` and the Zenodo
  version DOI.
- Refreshed the reproduction guard to the corrected demo baseline.

### Added

- `reference_medians` and `apply_medians` in `examples/ieee_cis_demo.py`, so an evaluation can fit
  imputation on a chosen reference window and apply it forward.
- `ref_f1_insample` in every signal table.

### Fixed

- Aggregate `with_cf_gap_mean` and `without_cf_gap_mean` as means of absolute per-window
  differences. The signed mean let errors of opposite sign cancel.

### Notes

- All published result values changed; results produced against 0.3.1 no longer reproduce. The
  separation boundary survives the correction: the headline separation over the two empirical
  conditions is unchanged and four low-signal grid rows move down. Reference F1 falls from 0.133 to
  0.126 for the logistic-regression reference and from 0.228 to 0.171 for the gradient-boosting
  reference.

## [0.3.1] - 2026-07-23

Zenodo release: [10.5281/zenodo.21501987](https://doi.org/10.5281/zenodo.21501987).

### Fixed

- Corrected the package version, which had stayed at `0.2.2` in `pyproject.toml` and
  `drift.__version__` through the 0.3.0 tag. The 0.3.0 build therefore produced a 0.2.2 wheel and
  PyPI rejected it as a duplicate, so 0.3.0 never reached PyPI. The library code, study entry point,
  and published results are identical in 0.3.0 and 0.3.1, and the Zenodo archive of 0.3.0 remains a
  valid snapshot of them.

## [0.3.0] - 2026-07-23

Zenodo release: [10.5281/zenodo.21501897](https://doi.org/10.5281/zenodo.21501897).

### Added

- `examples/ieee_cis_robustness.py`: a bounded, predefined robustness grid over the IEEE-CIS demo
  covering evidence-dimension weights, operational thresholds, proxy-group and shared-component
  ablations, window schemes (15-day, 45-day, and 30-day with 15-day stride), injection seeds, and a
  second model class (`HistGradientBoostingClassifier`). Raw per-window signals are computed once per
  window-scheme/model/seed configuration and stored, so every weight, threshold, and ablation variant
  re-aggregates auditable inputs instead of recomputing them.
- A reproduction guard in that runner: the base configuration must reproduce the published
  `ieee_cis_demo.py` detection counts and final-window values before any variant is allowed to run.
- `results/ieee_cis_robustness_v32/`: published signal tables, per-variant summary JSON, a
  human-readable `SUMMARY.md`, and a `RELEASE.md` recording release identity, environment, dataset
  access, and reproduction commands for the results shipped with this tag.

### Changed

- Raised the mypy type-checking target to Python 3.12. numpy 2.5 ships stubs written with PEP 695
  `type` statements, which mypy refuses to parse at an older target, so `mypy src/` failed against a
  fresh dependency resolution. Runtime support is unchanged (`requires-python >= 3.11`), and 3.11
  stays covered by the ruff target version and the CI test matrix.

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
