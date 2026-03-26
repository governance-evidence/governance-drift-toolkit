# Governance Drift Toolkit

[![CI](https://github.com/governance-evidence/governance-drift-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/governance-evidence/governance-drift-toolkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/governance-evidence/governance-drift-toolkit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19236418.svg)](https://doi.org/10.5281/zenodo.19236418)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/governance-evidence/governance-drift-toolkit/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/governance-evidence/governance-drift-toolkit)](https://github.com/governance-evidence/governance-drift-toolkit/releases)

A Python toolkit for label-free monitoring of governance evidence degradation
in risk decision systems. Answers: *"Is our governance evidence still sufficient,
even though we can't see the ground truth yet?"*

The toolkit combines proxy drift monitors, composite alerting, harmful-shift
suppression, and a governance response chain for delayed-label environments
such as fraud detection, credit scoring, and related risk systems.

## Install

### From a Package Index

Use this when the package is published to your package index:

```bash
pip install governance-drift-toolkit
```

### From GitHub

Use this before package-index publication, or when installing directly from source control:

```bash
pip install git+https://github.com/governance-evidence/governance-drift-toolkit.git
```

Optional sufficiency integration:

```bash
pip install "governance-drift-toolkit[sufficiency]"
```

### For Contributors

Clone the repository, create a local virtual environment, and install development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For dataset-backed demos, install the extra demo dependencies:

```bash
pip install -e ".[dev,demo]"
```

Dataset download and local directory setup for the demos are documented in
[docs/demo_datasets.md](docs/demo_datasets.md).

## Quick Start

The core package depends only on NumPy and SciPy. A minimal monitoring pass looks like this:

```python
import numpy as np
from drift import (
    compute_composite_alert,
    determine_response,
    fraud_detection_config,
)
from drift.monitors.score_distribution import compute_psi
from drift.monitors.feature_drift import compute_feature_psi
from drift.monitors.uncertainty import compute_prediction_entropy

config = fraud_detection_config()
rng = np.random.default_rng(42)

ref_scores = rng.normal(0.30, 0.15, size=1000)
cur_scores = rng.normal(0.45, 0.20, size=1000)

results = [
    compute_psi(ref_scores, cur_scores),
    compute_feature_psi(
        rng.normal(size=(500, 3)),
        rng.normal(0.5, 1.0, size=(500, 3)),
    ),
    compute_prediction_entropy(rng.uniform(0.1, 0.9, size=500)),
]

alert = compute_composite_alert(results, config)
response = determine_response(alert, config)
print(f"Alert: {alert.severity.value}, Response: {response.action.value}")
```

See [docs/deployment.md](docs/deployment.md) for installation modes and
[docs/alerting.md](docs/alerting.md) for the composite alert logic.

For dataset-backed demos, see [docs/demo_datasets.md](docs/demo_datasets.md).

## Seven Proxy Monitors

| # | Category | Detects | Misses |
| --- | ---------- | --------- | -------- |
| 1 | Score Distribution Shift | P(X) changes in scores | Adversarial drift preserving scores |
| 2 | Feature Drift | Covariate shift in inputs | Concept drift with stable features |
| 3 | Uncertainty | Calibration degradation | Confident-but-wrong predictions |
| 4 | Cross-Model Disagreement | Adversarial evasion | Correlated model failures |
| 5 | Operational Process | Behavioral changes | Fast-onset drift |
| 6 | Outcome-Maturity | Cohort-based drift | Novel patterns |
| 7 | Proxy Ground Truth | Pattern changes pre-labels | Social engineering |

## Governance Response Chain

Monitor -> Alert -> Escalate -> Fallback -> Rollback

## Related Projects

This toolkit is part of the [governance-evidence](https://github.com/governance-evidence) toolkit:

| Repository | Role | DOI |
| ---------- | ---- | --- |
| [decision-event-schema](https://github.com/governance-evidence/decision-event-schema) | Schema for events this toolkit monitors | [10.5281/zenodo.18923178](https://doi.org/10.5281/zenodo.18923178) |
| [evidence-sufficiency-calc](https://github.com/governance-evidence/evidence-sufficiency-calc) | Sufficiency scoring — bidirectional integration with this toolkit | Pending |
| [evidence-collector-sdk](https://github.com/governance-evidence/evidence-collector-sdk) | Collects evidence streams that feed into this toolkit | Pending |

## License

Apache-2.0
