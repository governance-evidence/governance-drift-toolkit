# Demo Datasets

This page documents the external datasets used by the demo scripts.
The datasets are not redistributed in this repository. You must download
them separately and place them under the local `data/` directory.

## Requirements

Install the demo dependencies first:

```bash
pip install -e ".[dev,demo]"
```

If you only need the runtime demo dependencies and not the full contributor
tooling, use:

```bash
pip install -e ".[demo]"
```

## IEEE-CIS Fraud Detection

Used by [examples/ieee_cis_demo.py](../examples/ieee_cis_demo.py).

Expected local layout:

```text
data/
  ieee_cis/
    train_transaction.csv
```

Download from Kaggle competition:

```bash
kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/ieee_cis/
unzip data/ieee_cis/train_transaction.csv.zip -d data/ieee_cis/
```

Manual download source:

```text
https://www.kaggle.com/c/ieee-fraud-detection/data
```

Run the demo:

```bash
python examples/ieee_cis_demo.py
```

## Lending Club

Used by [examples/lending_club_demo.py](../examples/lending_club_demo.py).

Expected local layout:

```text
data/
  lending_club/
    accepted_2007_to_2018Q4.csv
```

Download from Kaggle dataset:

```bash
kaggle datasets download -d wordsforthewise/lending-club -p data/lending_club/
cd data/lending_club
unzip lending-club.zip "accepted_2007_to_2018Q4.csv.gz"
gunzip accepted_2007_to_2018Q4.csv.gz
```

Run the demo:

```bash
python examples/lending_club_demo.py
```

## Notes

- Both datasets are external and may require Kaggle authentication and acceptance of dataset terms.
- The repository ignores downloaded dataset files under `data/`.
- The demos expect the exact filenames shown above.
