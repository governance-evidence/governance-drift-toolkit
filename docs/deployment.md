# Deployment Guide

## Standalone Usage

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

### For Contributors

For local development from a clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For dataset-backed demos:

```bash
pip install -e ".[dev,demo]"
```

Dataset download and local directory setup for the demo scripts are documented
in [demo_datasets.md](demo_datasets.md).

The core package depends only on NumPy and SciPy. A minimal deployment path looks like this:

```python
from drift import compute_composite_alert, determine_response, default_config
from drift.monitors.score_distribution import compute_psi

config = default_config()
results = [compute_psi(reference_scores, current_scores)]
alert = compute_composite_alert(results, config)
response = determine_response(alert, config)
```

## Integration with the Evidence Sufficiency Calculator

Install the optional sufficiency bridge when you want harmful-shift filtering
to be informed by evidence sufficiency scores:

```bash
pip install "governance-drift-toolkit[sufficiency]"
```

Pass the sufficiency score from the Evidence Sufficiency Calculator for harmful-shift filtering:

```python
alert = compute_composite_alert(results, config, sufficiency_score=0.85)
```

## Production Monitoring

For production use, connect monitors to a streaming or batch pipeline
using the integration stubs in `src/integrations/`.

Current integration surfaces:

- Decision Event Schema input extraction
- Evidence Collector protocol interface
- Evidence Sufficiency Calculator bridge
- Flink sink protocol stub

This repository currently provides the monitoring primitives and integration
interfaces rather than a full deployment stack. Full production deployment
documentation will be available with Paper 15.
