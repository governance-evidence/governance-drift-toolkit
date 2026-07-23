"""IEEE-CIS robustness study for the delayed-label evidence sufficiency work.

Companion to ``ieee_cis_demo.py``. The demo establishes the headline
detectability boundary; this script tests how far that boundary survives
perturbation of the evaluation's own design choices. Results back the
sensitivity and ablation analysis reported in the accompanying manuscript
(preprint arXiv:2604.15740).

Implements a bounded variant grid that was fixed before any variant ran;
nothing was added or removed after results were seen:

- R1 evidence-dimension weight sensitivity (aggregation-only)
- R2 operational threshold sensitivity (aggregation-only)
- R3 proxy-group ablation, leave-one-out and solo (aggregation-only)
- R4 coverage-component ablation: shared deterministic C/F removed
- R5 window-scheme sensitivity: 15d, 45d, 30d with 15d stride
- R6 second model class: HistGradientBoostingClassifier
- R7 injection-seed stability: seeds 42, 7, 2026

Raw per-window signals are computed once per (scheme, model, seed)
configuration and stored; weight/threshold/ablation variants re-aggregate
the stored signals so every variant is auditable from the signal tables.

Run from the repository root::

    .venv/bin/python examples/ieee_cis_robustness.py

Outputs JSON + markdown under results/ieee_cis_robustness_v32/.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ieee_cis_demo as demo  # shared config, feature list, and drift injectors

from drift.monitors.feature_drift import compute_feature_psi
from drift.monitors.score_distribution import compute_psi
from drift.monitors.uncertainty import (
    compute_confidence_drift,
    compute_prediction_entropy,
)

RESULTS_DIR = Path("results/ieee_cis_robustness_v32")

BASE_WEIGHTS = {
    "completeness": 0.20,
    "freshness": 0.30,
    "reliability": 0.30,
    "representativeness": 0.20,
}
BASE_TAUS = {"tau_c": 0.6, "tau_r_actual": 0.15, "tau_r_proxy": 0.55}
BASE_DELTA = 0.05
LAMBDA_FRESHNESS = 0.02
LABEL_DECAY_PER_DAY = 0.004  # 0.12 per 30-day window in the v31 base scheme
KS_CAP = 0.30

SCENARIOS = ("baseline", "covariate", "mixed", "concept")
INJECTORS = {
    "baseline": None,
    "covariate": demo._inject_covariate_drift,
    "mixed": demo._inject_mixed_drift,
    "concept": demo._inject_concept_drift,
}

# Coverage matrix restricted to the three implemented categories
# (mirrors drift.proxy_sufficiency.COVERAGE_WEIGHTS).
COVERAGE = {
    "scr": {"reliability": 0.25, "representativeness": 1.0},
    "fea": {"representativeness": 1.0},
    "unc": {"reliability": 0.5},
}


# ---------------------------------------------------------------------------
# Signal computation (one pass per scheme/model/seed configuration)
# ---------------------------------------------------------------------------


@dataclass
class WindowSignals:
    scenario: str
    window: int
    start_day: int
    n_txns: int
    fraud_rate: float
    psi: float
    fpsi: float
    ent_delta: float
    conf_ks: float
    completeness: float
    freshness: float
    f1_labeled: float
    repr_ks: float


@dataclass
class ConfigSignals:
    config_id: str
    scheme: str
    model_class: str
    seed: int
    window_days: int
    stride_days: int
    n_windows_monitored: int
    ref_txns: int
    ref_fraud_rate: float
    ref_f1: float
    caps: dict[str, float]
    rows: list[WindowSignals] = field(default_factory=list)


def _split_windows_scheme(df, window_days: int, stride_days: int):
    """Split into windows of window_days starting every stride_days.

    Window 0 (the reference) always starts at day 0. Monitored windows
    start at stride_days, 2*stride_days, ... while fully inside the span.
    """
    total_days = int(df["day"].max()) + 1
    starts = list(range(0, total_days - window_days + 1, stride_days))
    windows = []
    for s in starts:
        wdf = df[(df["day"] >= s) & (df["day"] < s + window_days)].copy()
        if len(wdf) > 0:
            windows.append((s, wdf))
    return windows


def _make_model(model_class: str, seed: int):
    if model_class == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=1000, random_state=seed, solver="lbfgs", class_weight="balanced"
        )
    if model_class == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    msg = f"unknown model_class {model_class}"
    raise ValueError(msg)


def compute_config_signals(
    df,
    config_id: str,
    scheme: str,
    window_days: int,
    stride_days: int,
    model_class: str,
    seed: int,
) -> ConfigSignals:
    """Compute raw per-window signals for one configuration."""
    from scipy.stats import ks_2samp
    from sklearn.metrics import f1_score as sk_f1
    from sklearn.preprocessing import StandardScaler

    cols = demo.FEATURE_COLS
    windows = _split_windows_scheme(df, window_days, stride_days)
    _ref_start, ref_df = windows[0]
    monitored = windows[1:]

    scaler = StandardScaler()
    x_ref = scaler.fit_transform(ref_df[cols].values)
    y_ref = ref_df["isFraud"].values
    model = _make_model(model_class, 42)  # model fit is not part of R7 seed variation
    model.fit(x_ref, y_ref)

    ref_probs = model.predict_proba(x_ref)[:, 1]
    ref_preds = (ref_probs > 0.5).astype(int)
    ref_f1 = float(sk_f1(y_ref, ref_preds))
    ref_features = x_ref.astype(np.float64)
    ref_entropy = compute_prediction_entropy(ref_probs).statistic

    # Cap calibration: identical procedure to the v31 demo (thirds of ref).
    n = len(ref_probs)
    third = n // 3
    idx = [slice(0, third), slice(third, 2 * third), slice(2 * third, None)]
    subs_p = [ref_probs[s] for s in idx]
    subs_f = [ref_features[s] for s in idx]
    psi_vals, fpsi_vals, ent_vals, ks_vals = [], [], [], []
    for a, b in combinations(range(3), 2):
        psi_vals.append(compute_psi(subs_p[a], subs_p[b]).statistic)
        fpsi_vals.append(compute_feature_psi(subs_f[a], subs_f[b], feature_names=cols).statistic)
        ent_vals.append(
            abs(
                compute_prediction_entropy(subs_p[a]).statistic
                - compute_prediction_entropy(subs_p[b]).statistic
            )
        )
        ks_vals.append(compute_confidence_drift(subs_p[a], subs_p[b]).statistic)
    caps = {
        "psi": max(max(psi_vals) * 5.0, 0.50),
        "fpsi": max(max(fpsi_vals) * 5.0, 1.0),
        "entropy": max(max(ent_vals) * 5.0, 0.15),
        "conf": max(max(ks_vals) * 5.0, 0.10),
    }

    cfg = ConfigSignals(
        config_id=config_id,
        scheme=scheme,
        model_class=model_class,
        seed=seed,
        window_days=window_days,
        stride_days=stride_days,
        n_windows_monitored=len(monitored),
        ref_txns=len(ref_df),
        ref_fraud_rate=float(y_ref.mean()),
        ref_f1=ref_f1,
        caps=caps,
    )

    ref_scores_full = model.predict_proba(scaler.transform(ref_df[cols].values))[:, 1]

    for scenario in SCENARIOS:
        inject = INJECTORS[scenario]
        rng = np.random.default_rng(seed)
        for i, (start_day, raw_df) in enumerate(monitored, start=1):
            cur_df = inject(raw_df, i - 1, rng) if inject is not None else raw_df

            x_cur = scaler.transform(cur_df[cols].values)
            cur_probs = model.predict_proba(x_cur)[:, 1]

            psi_stat = compute_psi(ref_probs, cur_probs).statistic
            fpsi_stat = compute_feature_psi(
                ref_features, x_cur.astype(np.float64), feature_names=cols
            ).statistic
            ent_stat = compute_prediction_entropy(cur_probs).statistic
            conf_stat = compute_confidence_drift(ref_probs, cur_probs).statistic

            # Deterministic components and actual dimensions (day-based;
            # reproduces the v31 inline formulas for the 30d base scheme).
            completeness = max(0.3, 1.0 - LABEL_DECAY_PER_DAY * start_day)
            freshness = float(np.exp(-LAMBDA_FRESHNESS * start_day))
            n_labeled = int(len(cur_df) * completeness)
            y_true = cur_df["isFraud"].values[:n_labeled]
            y_pred = (cur_probs[:n_labeled] > 0.5).astype(int)
            f1 = float(sk_f1(y_true, y_pred)) if len(y_true) > 10 else 0.1
            ks_stat = float(ks_2samp(ref_scores_full, cur_probs).statistic)

            cfg.rows.append(
                WindowSignals(
                    scenario=scenario,
                    window=i,
                    start_day=int(start_day),
                    n_txns=len(cur_df),
                    fraud_rate=float(cur_df["isFraud"].mean()),
                    psi=float(psi_stat),
                    fpsi=float(fpsi_stat),
                    ent_delta=float(abs(ent_stat - ref_entropy)),
                    conf_ks=float(conf_stat),
                    completeness=float(completeness),
                    freshness=float(freshness),
                    f1_labeled=f1,
                    repr_ks=ks_stat,
                )
            )
    return cfg


# ---------------------------------------------------------------------------
# Aggregation (variants re-aggregate stored signals)
# ---------------------------------------------------------------------------


def aggregate(
    cfg: ConfigSignals,
    *,
    weights: dict[str, float] | None = None,
    tau_c: float = 0.6,
    tau_r_actual: float = 0.15,
    tau_r_proxy: float = 0.55,
    categories: tuple[str, ...] = ("scr", "fea", "unc"),
    drop_shared_components: bool = False,
) -> dict[str, list[dict]]:
    """Compute per-window S_proxy and S_actual for one aggregation variant."""
    weights = dict(BASE_WEIGHTS if weights is None else weights)

    # Not dict.fromkeys: every scenario needs its own list instance.
    out: dict[str, list[dict]] = {s: [] for s in SCENARIOS}
    for row in cfg.rows:
        p = {
            "scr": max(0.0, 1.0 - row.psi / cfg.caps["psi"]),
            "fea": max(0.0, 1.0 - row.fpsi / cfg.caps["fpsi"]),
            "unc": min(
                max(0.0, 1.0 - row.ent_delta / cfg.caps["entropy"]),
                max(0.0, 1.0 - row.conf_ks / cfg.caps["conf"]),
            ),
        }
        num: dict[str, float] = {}
        den: dict[str, float] = {}
        for cat in categories:
            for dim, w in COVERAGE[cat].items():
                num[dim] = num.get(dim, 0.0) + w * p[cat]
                den[dim] = den.get(dim, 0.0) + w
        r_proxy = (
            num.get("reliability", 0.0) / den["reliability"] if den.get("reliability") else 1.0
        )
        p_proxy = (
            num.get("representativeness", 0.0) / den["representativeness"]
            if den.get("representativeness")
            else 1.0
        )

        c, f = row.completeness, row.freshness
        r_act = row.f1_labeled
        p_act = max(0.0, 1.0 - row.repr_ks / KS_CAP)

        if drop_shared_components:
            wr = weights["reliability"] / (weights["reliability"] + weights["representativeness"])
            wp = 1.0 - wr
            gate_proxy = min(1.0, r_proxy / tau_r_proxy)
            gate_act = min(1.0, r_act / tau_r_actual)
            s_proxy = gate_proxy * (wr * r_proxy + wp * p_proxy)
            s_actual = gate_act * (wr * r_act + wp * p_act)
        else:
            gate_proxy = min(1.0, c / tau_c) * min(1.0, r_proxy / tau_r_proxy)
            gate_act = min(1.0, c / tau_c) * min(1.0, r_act / tau_r_actual)
            s_proxy = gate_proxy * (
                weights["completeness"] * c
                + weights["freshness"] * f
                + weights["reliability"] * r_proxy
                + weights["representativeness"] * p_proxy
            )
            s_actual = gate_act * (
                weights["completeness"] * c
                + weights["freshness"] * f
                + weights["reliability"] * r_act
                + weights["representativeness"] * p_act
            )

        out[row.scenario].append(
            {
                "window": row.window,
                "start_day": row.start_day,
                "r_proxy": r_proxy,
                "p_proxy": p_proxy,
                "gate_proxy": gate_proxy,
                "s_proxy": s_proxy,
                "s_actual": s_actual,
            }
        )
    return out


def detection_summary(agg: dict[str, list[dict]], delta: float = BASE_DELTA) -> dict:
    """Per-condition detection counts and headline values."""
    base = {r["window"]: r["s_proxy"] for r in agg["baseline"]}
    summary: dict = {"delta": delta, "conditions": {}}
    total_detected = 0
    total_windows = 0
    for cond in ("covariate", "mixed", "concept"):
        rows = agg[cond]
        detected = sum(1 for r in rows if abs(r["s_proxy"] - base[r["window"]]) > delta)
        last = rows[-1]
        base_last = agg["baseline"][-1]
        summary["conditions"][cond] = {
            "windows": len(rows),
            "detected": detected,
            "s_proxy_last": round(last["s_proxy"], 3),
            "s_actual_last": round(last["s_actual"], 3),
            "proxy_actual_gap_last": round(last["s_proxy"] - last["s_actual"], 3),
        }
        total_detected += detected
        total_windows += len(rows)
    summary["baseline_s_proxy_last"] = round(agg["baseline"][-1]["s_proxy"], 3)
    summary["baseline_s_actual_last"] = round(base_last["s_actual"], 3)
    summary["total"] = {"windows": total_windows, "detected": total_detected}
    return summary


# ---------------------------------------------------------------------------
# Variant grid driver
# ---------------------------------------------------------------------------


def run_grid(configs: dict[str, ConfigSignals]) -> dict:
    """Apply the predefined R1-R7 grid over stored signal tables."""
    base = configs["base_30d_logreg_s42"]
    results: dict = {}

    # R1 weights
    r1 = {}
    weight_sets = {
        "W0_base": BASE_WEIGHTS,
        "W1_equal": dict.fromkeys(BASE_WEIGHTS, 0.25),
        "W2_reliability_heavy": {
            "completeness": 0.15,
            "freshness": 0.20,
            "reliability": 0.45,
            "representativeness": 0.20,
        },
        "W3_freshness_light": {
            "completeness": 0.25,
            "freshness": 0.15,
            "reliability": 0.35,
            "representativeness": 0.25,
        },
    }
    for name, w in weight_sets.items():
        r1[name] = detection_summary(aggregate(base, weights=w))
    results["R1_weights"] = r1

    # R2 thresholds
    r2 = {}
    for tau_r in (0.10, 0.15, 0.20):
        r2[f"tau_r_actual_{tau_r}"] = detection_summary(aggregate(base, tau_r_actual=tau_r))
    for tau_rp in (0.45, 0.55, 0.65):
        r2[f"tau_r_proxy_{tau_rp}"] = detection_summary(aggregate(base, tau_r_proxy=tau_rp))
    for tau_c in (0.5, 0.6, 0.7):
        r2[f"tau_c_{tau_c}"] = detection_summary(aggregate(base, tau_c=tau_c))
    for delta in (0.03, 0.05, 0.10):
        r2[f"delta_{delta}"] = detection_summary(aggregate(base), delta=delta)
    results["R2_thresholds"] = r2

    # R3 proxy-group ablation
    r3 = {}
    for cats in (
        ("fea", "unc"),
        ("scr", "unc"),
        ("scr", "fea"),
        ("scr",),
        ("fea",),
        ("unc",),
    ):
        r3["+".join(cats)] = detection_summary(aggregate(base, categories=cats))
    results["R3_proxy_ablation"] = r3

    # R4 coverage-component ablation
    with_cf = aggregate(base)
    without_cf = aggregate(base, drop_shared_components=True)
    corr = {}
    for cond in SCENARIOS:
        a = np.array([r["s_proxy"] for r in with_cf[cond]])
        b = np.array([r["s_actual"] for r in with_cf[cond]])
        a2 = np.array([r["s_proxy"] for r in without_cf[cond]])
        b2 = np.array([r["s_actual"] for r in without_cf[cond]])
        corr[cond] = {
            "with_cf_gap_mean": round(float(np.mean(a - b)), 3),
            "without_cf_gap_mean": round(float(np.mean(a2 - b2)), 3),
            "with_cf_corr": round(float(np.corrcoef(a, b)[0, 1]), 3),
            "without_cf_corr": round(float(np.corrcoef(a2, b2)[0, 1]), 3),
        }
    results["R4_coverage_components"] = {
        "summary_without_cf": detection_summary(without_cf),
        "per_condition": corr,
        "note": "gate reduces to min(1, R/tau_r) when C/F are dropped; "
        "weights renormalized over reliability+representativeness (0.6/0.4)",
    }

    # R5 window schemes / R6 model / R7 seeds: base aggregation on each config
    for rid, key in (
        ("R5_window_15d", "win15_logreg_s42"),
        ("R5_window_45d", "win45_logreg_s42"),
        ("R5_overlap_30d_stride15", "overlap30x15_logreg_s42"),
        ("R6_model_hgb", "base_30d_hgb_s42"),
        ("R7_seed_7", "base_30d_logreg_s7"),
        ("R7_seed_2026", "base_30d_logreg_s2026"),
    ):
        cfg = configs[key]
        results[rid] = {
            "ref_f1": round(cfg.ref_f1, 3),
            "caps": {k: round(v, 3) for k, v in cfg.caps.items()},
            "n_windows_monitored": cfg.n_windows_monitored,
            "summary": detection_summary(aggregate(cfg)),
        }
    results["R7_seed_42"] = {
        "ref_f1": round(base.ref_f1, 3),
        "summary": detection_summary(aggregate(base)),
    }
    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = demo._load_data()

    config_specs = [
        ("base_30d_logreg_s42", "base_30d", 30, 30, "logreg", 42),
        ("win15_logreg_s42", "win15", 15, 15, "logreg", 42),
        ("win45_logreg_s42", "win45", 45, 45, "logreg", 42),
        ("overlap30x15_logreg_s42", "overlap30x15", 30, 15, "logreg", 42),
        ("base_30d_hgb_s42", "base_30d", 30, 30, "hgb", 42),
        ("base_30d_logreg_s7", "base_30d", 30, 30, "logreg", 7),
        ("base_30d_logreg_s2026", "base_30d", 30, 30, "logreg", 2026),
    ]

    configs: dict[str, ConfigSignals] = {}
    for config_id, scheme, wdays, stride, model_class, seed in config_specs:
        print(f"[signals] {config_id} ...", flush=True)
        cfg = compute_config_signals(df, config_id, scheme, wdays, stride, model_class, seed)
        configs[config_id] = cfg
        payload = asdict(cfg)
        (RESULTS_DIR / f"signals_{config_id}.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8"
        )
        print(
            f"  ref_txns={cfg.ref_txns:,} ref_f1={cfg.ref_f1:.3f} "
            f"monitored={cfg.n_windows_monitored} caps={ {k: round(v, 3) for k, v in cfg.caps.items()} }",
            flush=True,
        )

    # Reproduction guard: base config must match frozen v31 headline numbers.
    base_summary = detection_summary(aggregate(configs["base_30d_logreg_s42"]))
    expected = {
        "covariate": (5, 0.105, 0.121),
        "mixed": (5, 0.159, 0.037),
        "concept": (0, 0.294, 0.008),
    }
    for cond, (det, s_prx, s_act) in expected.items():
        got = base_summary["conditions"][cond]
        ok = (
            got["detected"] == det
            and abs(got["s_proxy_last"] - s_prx) < 0.0015
            and abs(got["s_actual_last"] - s_act) < 0.0015
        )
        print(
            f"[repro-guard] {cond}: detected={got['detected']} "
            f"s_proxy_W5={got['s_proxy_last']} s_actual_W5={got['s_actual_last']} "
            f"{'OK' if ok else 'MISMATCH vs v31'}"
        )
        if not ok:
            msg = f"v31 reproduction failed for {cond}: {got} != {det, s_prx, s_act}"
            raise SystemExit(msg)

    grid = run_grid(configs)

    env = {
        "python": platform.python_version(),
        "command": ".venv/bin/python examples/ieee_cis_robustness.py",
        # Provenance only; resolved via PATH by design so the script stays
        # portable across checkouts.
        "git_head": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "seeds": {"model_fit": 42, "injection": [42, 7, 2026]},
        "n_features": len(demo.FEATURE_COLS),
    }
    for mod in ("numpy", "pandas", "sklearn", "scipy", "drift"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - version lookup only
            env[mod] = "unknown"

    out = {
        "environment": env,
        "plan": "predefined bounded variant grid R1-R7; see the module docstring and RELEASE.md",
        "results": grid,
    }
    (RESULTS_DIR / "variants_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nWrote {RESULTS_DIR}/variants_summary.json")


if __name__ == "__main__":
    main()
