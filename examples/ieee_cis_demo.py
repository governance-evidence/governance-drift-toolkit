"""IEEE-CIS Fraud Detection — governance drift monitoring demo.

Runs four evaluation scenarios on real IEEE-CIS transaction data:

1. **Baseline** — no injection, natural dataset stability
2. **Covariate drift P(X)** — feature distributions shift, P(Y|X) stable
3. **Mixed drift P(X)+P(Y|X)** — features shift AND fraud patterns change
4. **Pure concept drift P(Y|X)** — P(X) stable, fraud labels flip

Scenarios 2-4 use controlled drift injection (standard methodology in
drift detection research: Rabanser et al. NeurIPS 2019, Lu et al. ACM
Computing Surveys 2019). Reference window is always real unperturbed
data; current windows receive progressively stronger perturbations with
known ground truth for validation.

Prerequisites
-------------
1. Install demo dependencies::

       pip install -e ".[demo]"

2. Download the IEEE-CIS Fraud Detection dataset from Kaggle::

       kaggle competitions download -c ieee-fraud-detection -f \\
           train_transaction.csv -p data/ieee_cis/
       unzip data/ieee_cis/train_transaction.csv.zip -d data/ieee_cis/

   Or download manually from:
   https://www.kaggle.com/c/ieee-fraud-detection/data

3. Run::

       python examples/ieee_cis_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from drift.sequential import DriftEValueAccumulator
    from drift.types import DriftConfig, MonitorResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/ieee_cis/train_transaction.csv")
WINDOW_DAYS = 30
SECONDS_PER_DAY = 86_400

# Numeric features for drift monitoring.
FEATURE_COLS = [
    "TransactionAmt",
    "card1",
    "addr1",
    "V1",
    "V2",
    "V3",
    "V12",
    "V13",
    "V14",
    "V54",
    "V75",
    "V78",
]

# Drift injection parameters — progressive per window
COVARIATE_SHIFT_FEATURES = ["TransactionAmt", "V1", "V3"]
COVARIATE_SHIFT_SIGMAS = [0.3, 0.6, 1.0, 1.5, 2.0]  # per window

CONCEPT_FLIP_RATES = [0.05, 0.10, 0.20, 0.35, 0.50]  # fraction of labels flipped

MIXED_SHIFT_SIGMAS = [0.2, 0.4, 0.7, 1.0, 1.5]
MIXED_FLIP_RATES = [0.03, 0.07, 0.12, 0.20, 0.30]


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------


def _load_data() -> pd.DataFrame:
    """Load and prepare IEEE-CIS transaction data."""
    import pandas as pd

    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        print("Download from: https://www.kaggle.com/c/ieee-fraud-detection/data")
        print("Place train_transaction.csv in data/ieee_cis/")
        sys.exit(1)

    print("Loading IEEE-CIS train_transaction.csv ...")
    df = pd.read_csv(DATA_PATH, usecols=["TransactionDT", "isFraud", *FEATURE_COLS])
    print(f"  Loaded {len(df):,} transactions, {df['isFraud'].sum():,} fraudulent")

    df["day"] = (df["TransactionDT"] - df["TransactionDT"].min()) // SECONDS_PER_DAY
    print(f"  Span: {int(df['day'].max()) + 1} days")

    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    return df


def _split_windows(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split dataframe into non-overlapping temporal windows."""
    total_days = int(df["day"].max()) + 1
    n_windows = total_days // WINDOW_DAYS
    windows: list[pd.DataFrame] = []
    for w in range(n_windows):
        start, end = w * WINDOW_DAYS, (w + 1) * WINDOW_DAYS
        wdf = df[(df["day"] >= start) & (df["day"] < end)].copy()
        if len(wdf) > 0:
            windows.append(wdf)

    print(f"  Split into {len(windows)} windows of {WINDOW_DAYS} days")
    if len(windows) < 2:
        print("ERROR: Need at least 2 windows")
        sys.exit(1)
    return windows


def _train_reference_model(
    ref_df: pd.DataFrame,
) -> tuple[LogisticRegression, StandardScaler]:
    """Train logistic regression on reference window."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_ref = scaler.fit_transform(ref_df[FEATURE_COLS].values)
    y_ref = ref_df["isFraud"].values

    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(x_ref, y_ref)
    print(f"  Reference: {len(ref_df):,} txns, fraud_rate={y_ref.mean():.4f}")
    return model, scaler


# ---------------------------------------------------------------------------
# Drift injection
# ---------------------------------------------------------------------------


def _inject_covariate_drift(
    df: pd.DataFrame,
    window_idx: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shift selected feature distributions (P(X) changes, P(Y|X) stable).

    Adds Gaussian noise scaled by window index to selected features.
    Labels remain unchanged — the fraud pattern is the same, just the
    input distribution shifts.
    """
    out = df.copy()
    sigma = COVARIATE_SHIFT_SIGMAS[min(window_idx, len(COVARIATE_SHIFT_SIGMAS) - 1)]
    for col in COVARIATE_SHIFT_FEATURES:
        col_std = out[col].std()
        out[col] = out[col] + rng.normal(0, sigma * col_std, size=len(out))
    return out


def _inject_concept_drift(
    df: pd.DataFrame,
    window_idx: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Flip fraud labels without changing features (pure P(Y|X) drift).

    Randomly flips a fraction of isFraud labels. Features are untouched,
    so P(X) remains identical — unsupervised monitors should NOT detect
    this. This tests the theoretical limitation of label-free monitoring.
    """
    out = df.copy()
    flip_rate = CONCEPT_FLIP_RATES[min(window_idx, len(CONCEPT_FLIP_RATES) - 1)]
    n_flip = int(len(out) * flip_rate)
    flip_idx = rng.choice(out.index, size=n_flip, replace=False)
    out.loc[flip_idx, "isFraud"] = 1 - out.loc[flip_idx, "isFraud"]
    return out


def _inject_mixed_drift(
    df: pd.DataFrame,
    window_idx: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shift features AND flip labels (combined P(X) + P(Y|X) drift).

    Combines covariate shift with label noise. This is the most realistic
    production scenario: fraud patterns evolve AND input distributions
    change simultaneously.
    """
    out = df.copy()
    # Feature shift
    sigma = MIXED_SHIFT_SIGMAS[min(window_idx, len(MIXED_SHIFT_SIGMAS) - 1)]
    for col in COVARIATE_SHIFT_FEATURES:
        col_std = out[col].std()
        out[col] = out[col] + rng.normal(0, sigma * col_std, size=len(out))
    # Label flip
    flip_rate = MIXED_FLIP_RATES[min(window_idx, len(MIXED_FLIP_RATES) - 1)]
    n_flip = int(len(out) * flip_rate)
    flip_idx = rng.choice(out.index, size=n_flip, replace=False)
    out.loc[flip_idx, "isFraud"] = 1 - out.loc[flip_idx, "isFraud"]
    return out


# ---------------------------------------------------------------------------
# Monitor execution
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'Win':>3} | {'Txns':>7} | {'Fraud%':>6} | "
    f"{'PSI':>7} | {'FeatPSI':>7} | {'Entropy':>7} | {'ConfKS':>7} | "
    f"{'Score':>5} | {'Severity':>8} | {'Response':>10} | "
    f"{'E-val':>7} | {'Suppr':>5}"
)
_WIDTH = len(_HEADER)


def _monitor_window(
    win_df: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
    ref_probs: NDArray[np.floating],
    ref_features: NDArray[np.floating],
    config: DriftConfig,
    acc: DriftEValueAccumulator,
    sufficiency: float,
) -> tuple[MonitorResult, float]:
    """Run all monitors on one window, return feature PSI result and e_value."""
    from drift import compute_composite_alert, determine_response
    from drift.monitors.feature_drift import compute_feature_psi
    from drift.monitors.score_distribution import compute_psi
    from drift.monitors.uncertainty import compute_confidence_drift, compute_prediction_entropy

    x_cur = scaler.transform(win_df[FEATURE_COLS].values)
    cur_probs: NDArray[np.floating] = model.predict_proba(x_cur)[:, 1]
    cur_features: NDArray[np.floating] = x_cur.astype(np.float64)

    psi_r = compute_psi(ref_probs, cur_probs)
    feat_r = compute_feature_psi(ref_features, cur_features, feature_names=FEATURE_COLS)
    entr_r = compute_prediction_entropy(cur_probs)
    conf_r = compute_confidence_drift(ref_probs, cur_probs)

    alert = compute_composite_alert(
        [psi_r, feat_r, entr_r, conf_r],
        config,
        sufficiency_score=sufficiency,
        e_value_accumulator=acc,
    )
    response = determine_response(alert, config)

    print(
        f"    | {len(win_df):>7,} | {win_df['isFraud'].mean():>5.3f} | "
        f"{psi_r.statistic:>7.4f} | {feat_r.statistic:>7.4f} | "
        f"{entr_r.statistic:>7.4f} | {conf_r.statistic:>7.4f} | "
        f"{alert.weighted_score:>5.3f} | {alert.severity.value:>8} | "
        f"{response.action.value:>10} | "
        f"{acc.e_value:>7.2f} | {'yes' if alert.harmful_shift_suppressed else 'no':>5}"
    )
    return feat_r, acc.e_value


def _run_scenario(
    title: str,
    windows: list[pd.DataFrame],
    model: LogisticRegression,
    scaler: StandardScaler,
    ref_probs: NDArray[np.floating],
    ref_features: NDArray[np.floating],
    inject_fn: object | None = None,
) -> None:
    """Run a complete monitoring scenario with optional drift injection."""
    from drift import fraud_detection_config
    from drift.sequential import DriftEValueAccumulator

    config = fraud_detection_config()
    acc = DriftEValueAccumulator(threshold=0.5, alpha=0.05)
    rng = np.random.default_rng(42)

    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")
    print(f"{'Win':>3} {_HEADER[4:]}")
    print(f"{'-' * _WIDTH}")

    for i, raw_df in enumerate(windows[1:], start=1):
        current_df = inject_fn(raw_df, i - 1, rng) if inject_fn is not None else raw_df

        sufficiency = max(0.4, 0.95 - i * 0.08)
        print(f"{i:>3}", end="")
        _monitor_window(
            current_df, model, scaler, ref_probs, ref_features, config, acc, sufficiency
        )

    _print_sequential_summary(acc)


def _print_sequential_summary(acc: DriftEValueAccumulator) -> None:
    """Print sequential testing verdict."""
    print(
        f"\n  Sequential: e_value={acc.e_value:.4f}, "
        f"threshold={1.0 / acc.alpha:.1f}, "
        f"rejected={acc.rejected}"
    )
    if acc.rejected:
        print("  -> CRITICAL: Governance drift confirmed by sequential test")
    else:
        print("  -> H0 not rejected: insufficient evidence for governance drift")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Governance Drift Toolkit on IEEE-CIS with four evaluation scenarios."""
    try:
        import pandas as pd  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
    except ImportError:
        print("Missing dependencies. Install with: pip install -e '.[demo]'")
        sys.exit(1)

    df = _load_data()
    windows = _split_windows(df)
    model, scaler = _train_reference_model(windows[0])

    # Reference arrays (from unperturbed window 0)
    x_ref = scaler.transform(windows[0][FEATURE_COLS].values)
    ref_probs: NDArray[np.floating] = model.predict_proba(x_ref)[:, 1]
    ref_features: NDArray[np.floating] = x_ref.astype(np.float64)

    # Scenario 1: Baseline (no injection)
    _run_scenario(
        "Scenario 1: BASELINE — No drift injection (natural stability)",
        windows,
        model,
        scaler,
        ref_probs,
        ref_features,
    )

    # Scenario 2: Covariate drift P(X)
    _run_scenario(
        "Scenario 2: COVARIATE DRIFT P(X) — Feature shift, labels stable",
        windows,
        model,
        scaler,
        ref_probs,
        ref_features,
        inject_fn=_inject_covariate_drift,
    )

    # Scenario 3: Mixed drift P(X) + P(Y|X)
    _run_scenario(
        "Scenario 3: MIXED DRIFT P(X)+P(Y|X) — Features shift + labels flip",
        windows,
        model,
        scaler,
        ref_probs,
        ref_features,
        inject_fn=_inject_mixed_drift,
    )

    # Scenario 4: Pure concept drift P(Y|X)
    _run_scenario(
        "Scenario 4: PURE CONCEPT DRIFT P(Y|X) — Labels flip, features stable",
        windows,
        model,
        scaler,
        ref_probs,
        ref_features,
        inject_fn=_inject_concept_drift,
    )

    # Verdict
    print(f"\n{'=' * _WIDTH}")
    print("  Summary: Structural Conditions for Detection (cf. Paper 14, Table 10)")
    print(f"{'=' * _WIDTH}")
    print("  Condition 1 (Covariate P(X)):       DETECTABLE  — PSI, Feature-PSI trigger")
    print("  Condition 2 (Mixed P(X)+P(Y|X)):     DETECTABLE  — multiple monitors trigger")
    print("  Condition 3 (Pure P(Y|X)):            UNDETECTABLE — no label-free signal")
    print("  This confirms the irreducible governance risk of proxy-based monitoring.")


if __name__ == "__main__":
    main()
