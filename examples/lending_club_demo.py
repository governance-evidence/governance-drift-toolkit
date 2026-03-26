"""Lending Club — governance drift monitoring (credit scoring domain).

Cross-domain companion to ieee_cis_demo.py. Demonstrates the Governance Drift Toolkit on credit
scoring data with natural temporal structure (2008-2018) and real label
delay (loans mature over 36-60 months).

Paper 15 uses this dataset to validate Governance Drift Toolkit generalizability beyond
fraud detection (Paper 14 / IEEE-CIS).

Prerequisites
-------------
1. Install demo dependencies::

       pip install -e ".[demo]"

2. Download Lending Club dataset::

       kaggle datasets download -d wordsforthewise/lending-club -p data/lending_club/
       cd data/lending_club && unzip lending-club.zip "accepted_2007_to_2018Q4.csv.gz"
       gunzip accepted_2007_to_2018Q4.csv.gz

3. Run::

       python examples/lending_club_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/lending_club/accepted_2007_to_2018Q4.csv")
WINDOW_QUARTERS = 4  # quarters per window (1 year)
SEED = 42

# Numeric features for credit scoring drift monitoring
FEATURE_COLS = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
    "revol_bal",
    "total_acc",
    "open_acc",
    "pub_rec",
    "installment",
]

# Drift injection parameters
COVARIATE_SHIFT_FEATURES = ["annual_inc", "dti", "revol_util"]
COVARIATE_SIGMAS = [0.3, 0.6, 1.0, 1.5, 2.0]
CONCEPT_FLIP_RATES = [0.03, 0.06, 0.10, 0.15, 0.25]
MIXED_SIGMAS = [0.2, 0.4, 0.7, 1.0, 1.5]
MIXED_FLIPS = [0.02, 0.04, 0.08, 0.12, 0.20]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_and_prepare() -> tuple[list[pd.DataFrame], list[str]]:
    """Load Lending Club data, create binary label, split by year."""
    import pandas as pd

    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        print("Download: kaggle datasets download -d wordsforthewise/lending-club")
        sys.exit(1)

    print("Loading Lending Club accepted loans ...")
    df = pd.read_csv(
        DATA_PATH,
        usecols=["issue_d", "loan_status", *FEATURE_COLS],
        low_memory=False,
    )

    # Binary label: default (Charged Off / Default / Late 31-120) vs paid
    default_statuses = {"Charged Off", "Default", "Late (31-120 days)"}
    df = df[df["loan_status"].isin({"Fully Paid", *default_statuses})].copy()
    df["is_default"] = df["loan_status"].isin(default_statuses).astype(int)
    df = df.drop(columns=["loan_status"])

    # Parse issue date
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df = df.dropna(subset=["issue_d"])
    df = df.sort_values("issue_d")

    # Fill NaN with median
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Split into yearly windows (Q1-Q4 = 1 year)
    df["year"] = df["issue_d"].dt.year
    years = sorted(df["year"].unique())
    windows = []
    window_labels = []
    for y in years:
        wdf = df[df["year"] == y].copy()
        if len(wdf) > 1000:  # skip tiny early years
            windows.append(wdf)
            window_labels.append(str(y))

    total = sum(len(w) for w in windows)
    print(
        f"  {total:,} loans across {len(windows)} yearly windows ({window_labels[0]}-{window_labels[-1]})"
    )
    default_rate = sum(w["is_default"].sum() for w in windows) / total
    print(f"  Overall default rate: {default_rate:.2%}")
    return windows, window_labels


# ---------------------------------------------------------------------------
# Drift injection
# ---------------------------------------------------------------------------


def _inject_covariate(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    sigma = COVARIATE_SIGMAS[min(idx, len(COVARIATE_SIGMAS) - 1)]
    for col in COVARIATE_SHIFT_FEATURES:
        out[col] = out[col] + rng.normal(0, sigma * out[col].std(), len(out))
    return out


def _inject_concept(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    flip_rate = CONCEPT_FLIP_RATES[min(idx, len(CONCEPT_FLIP_RATES) - 1)]
    n_flip = int(len(out) * flip_rate)
    flip_idx = rng.choice(out.index, size=n_flip, replace=False)
    out.loc[flip_idx, "is_default"] = 1 - out.loc[flip_idx, "is_default"]
    return out


def _inject_mixed(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> pd.DataFrame:
    sigma = MIXED_SIGMAS[min(idx, len(MIXED_SIGMAS) - 1)]
    flip_rate = MIXED_FLIPS[min(idx, len(MIXED_FLIPS) - 1)]
    out = df.copy()
    for col in COVARIATE_SHIFT_FEATURES:
        out[col] = out[col] + rng.normal(0, sigma * out[col].std(), len(out))
    n_flip = int(len(out) * flip_rate)
    flip_idx = rng.choice(out.index, size=n_flip, replace=False)
    out.loc[flip_idx, "is_default"] = 1 - out.loc[flip_idx, "is_default"]
    return out


# ---------------------------------------------------------------------------
# Monitor execution
# ---------------------------------------------------------------------------

_WIDTH = 115


def _run_scenario(
    title: str,
    windows: list[pd.DataFrame],
    window_labels: list[str],
    model: object,
    scaler: object,
    ref_probs: object,
    ref_features: object,
    inject_fn: object | None = None,
) -> list[dict]:
    """Run drift monitors across yearly windows."""
    from drift import compute_composite_alert, credit_scoring_config, determine_response
    from drift.monitors.feature_drift import compute_feature_psi
    from drift.monitors.score_distribution import compute_psi
    from drift.monitors.uncertainty import compute_confidence_drift, compute_prediction_entropy
    from drift.sequential import DriftEValueAccumulator

    config = credit_scoring_config()
    acc = DriftEValueAccumulator(threshold=0.5, alpha=0.05)
    rng = np.random.default_rng(SEED)

    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")
    print(
        f"{'Year':>6} | {'Loans':>8} | {'Def%':>5} | "
        f"{'PSI':>7} | {'FeatPSI':>7} | {'Entropy':>7} | {'ConfKS':>7} | "
        f"{'Score':>5} | {'Severity':>8} | {'Response':>10} | {'E-val':>7}"
    )
    print(f"{'-' * _WIDTH}")

    rows = []
    for i, raw_df in enumerate(windows[1:]):
        current_df = inject_fn(raw_df, i, rng) if inject_fn is not None else raw_df

        x_cur = scaler.transform(current_df[FEATURE_COLS].values)  # type: ignore[union-attr]
        cur_probs = model.predict_proba(x_cur)[:, 1]  # type: ignore[union-attr]
        cur_features = x_cur.astype(np.float64)

        psi_r = compute_psi(ref_probs, cur_probs)
        feat_r = compute_feature_psi(ref_features, cur_features, feature_names=FEATURE_COLS)
        entr_r = compute_prediction_entropy(cur_probs)
        conf_r = compute_confidence_drift(ref_probs, cur_probs)

        sufficiency = max(0.4, 0.95 - (i + 1) * 0.05)
        alert = compute_composite_alert(
            [psi_r, feat_r, entr_r, conf_r],
            config,
            sufficiency_score=sufficiency,
            e_value_accumulator=acc,
        )
        response = determine_response(alert, config)

        row = {
            "year": window_labels[i + 1],
            "n_loans": len(current_df),
            "default_rate": current_df["is_default"].mean(),
            "psi": psi_r.statistic,
            "feat_psi": feat_r.statistic,
            "entropy": entr_r.statistic,
            "conf_ks": conf_r.statistic,
            "composite": alert.weighted_score,
            "severity": alert.severity.value,
            "response": response.action.value,
            "e_value": acc.e_value,
            "scenario": title,
            "psi_triggered": psi_r.triggered,
            "feat_triggered": feat_r.triggered,
        }
        rows.append(row)

        print(
            f"{window_labels[i + 1]:>6} | {len(current_df):>8,} | {current_df['is_default'].mean():>4.2f} | "
            f"{psi_r.statistic:>7.4f} | {feat_r.statistic:>7.4f} | "
            f"{entr_r.statistic:>7.4f} | {conf_r.statistic:>7.4f} | "
            f"{alert.weighted_score:>5.3f} | {alert.severity.value:>8} | "
            f"{response.action.value:>10} | {acc.e_value:>7.2f}"
        )

    # Sequential verdict
    print(f"\n  Sequential: e_value={acc.e_value:.4f}, rejected={acc.rejected}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Governance Drift Toolkit on Lending Club with four evaluation scenarios."""
    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("Missing dependencies. Install with: pip install -e '.[demo]'")
        sys.exit(1)

    from sklearn.preprocessing import StandardScaler

    windows, labels = _load_and_prepare()

    # Train on first yearly window
    ref_df = windows[0]
    scaler = StandardScaler()
    x_ref = scaler.fit_transform(ref_df[FEATURE_COLS].values)
    y_ref = ref_df["is_default"].values

    model = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")
    model.fit(x_ref, y_ref)
    ref_probs = model.predict_proba(x_ref)[:, 1]
    ref_features = x_ref.astype(np.float64)
    print(f"  Reference: {labels[0]}, {len(ref_df):,} loans, default_rate={y_ref.mean():.4f}")

    # Run all scenarios
    all_rows: list[dict] = []
    for title, fn in [
        ("Baseline (natural temporal drift)", None),
        ("Covariate Drift P(X) — income/debt shift", _inject_covariate),
        ("Mixed Drift P(X)+P(Y|X) — economic regime change", _inject_mixed),
        ("Pure Concept Drift P(Y|X) — default pattern change", _inject_concept),
    ]:
        rows = _run_scenario(title, windows, labels, model, scaler, ref_probs, ref_features, fn)
        all_rows.extend(rows)

    # Summary
    print(f"\n{'=' * _WIDTH}")
    print("  Summary: Structural Conditions (cf. Paper 15 / Paper 14 Table 10)")
    print(f"{'=' * _WIDTH}")

    import pandas as pd

    summary = pd.DataFrame(all_rows)
    for scenario, sdf in summary.groupby("scenario", sort=False):
        n_detected = (sdf["feat_psi"] > 0.25).sum()
        n_win = len(sdf)
        print(f"  {scenario:55s}: {n_detected}/{n_win} windows with Feature PSI > 0.25")

    print()
    print("  Cross-domain comparison:")
    print("    IEEE-CIS (fraud):   Covariate=100%, Mixed=100%, Pure P(Y|X)=0%")
    print("    Lending Club (credit): see results above")
    print("  If detection rates match → the toolkit is domain-agnostic (Paper 15 thesis)")


if __name__ == "__main__":
    main()
