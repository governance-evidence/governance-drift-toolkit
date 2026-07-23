# IEEE-CIS robustness study

Sensitivity, ablation, and model-dependence results for the delayed-label
evidence sufficiency evaluation reported in the accompanying manuscript
(preprint arXiv:2604.15740). Release identity and reproduction commands:
[RELEASE.md](RELEASE.md).

Produced by `examples/ieee_cis_robustness.py` (this repository; drift 0.2.2,
python 3.14.5, numpy 2.4.3, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.17.1)
and `examples/ieee_cis_simulator_sensitivity.py` in the
evidence-sufficiency-calc repository. The variant grid was predefined and
fixed before any variant ran. Raw per-window signal tables: `signals_*.json`;
full variant output: `variants_summary.json`; simulator sensitivity:
`ieee_cis_simulator_sensitivity_v32.json` in the calculator repository's
`results/` directory.

Reproduction gate: the base configuration reproduces the frozen v31 numbers
exactly (detection 5/5, 5/5, 0/5; W5 S_proxy 0.105 / 0.159 / 0.294; W5 actual
S 0.121 / 0.037 / 0.008; simulator checkpoints identical to Table 3).

Detection rule: |S_proxy(drift, w) - S_proxy(baseline, w)| > delta, delta =
0.05 unless varied.

## 1. Detection counts by variant (covariate / mixed / concept, total)

| Variant | Covariate | Mixed | Concept | Total |
|---|---|---|---|---|
| Base (W0, tau_r=0.15, tau_rp=0.55, tau_c=0.6, delta=0.05) | 5/5 | 5/5 | 0/5 | 10/15 |
| R1 W1 equal weights | 5/5 | 5/5 | 0/5 | 10/15 |
| R1 W2 reliability-heavy | 5/5 | 5/5 | 0/5 | 10/15 |
| R1 W3 freshness-light | 5/5 | 5/5 | 0/5 | 10/15 |
| R2 tau_r_actual 0.10 / 0.20 | 5/5 | 5/5 | 0/5 | 10/15 |
| R2 tau_r_proxy 0.45 / 0.65 | 5/5 | 5/5 | 0/5 | 10/15 |
| R2 tau_c 0.5 / 0.7 | 5/5 | 5/5 | 0/5 | 10/15 |
| R2 delta 0.03 | 5/5 | 5/5 | 0/5 | 10/15 |
| R2 delta 0.10 | 4/5 | 3/5 | 0/5 | 7/15 |
| R3 fea+unc (drop score) | 5/5 | 5/5 | 0/5 | 10/15 |
| R3 scr+fea (drop uncertainty) | 5/5 | 5/5 | 0/5 | 10/15 |
| R3 scr+unc (drop feature) | 3/5 | 2/5 | 0/5 | 5/15 |
| R3 fea only | 5/5 | 5/5 | 0/5 | 10/15 |
| R3 scr only | 3/5 | 2/5 | 0/5 | 5/15 |
| R3 unc only | 1/5 | 1/5 | 0/5 | 2/15 |
| R4 without shared C/F components | 5/5 | 5/5 | 0/5 | 10/15 |
| R5 15-day windows (11 monitored) | 11/11 | 11/11 | 0/11 | 22/33 |
| R5 45-day windows (3 monitored) | 2/3 | 2/3 | 0/3 | 4/9 |
| R5 30-day windows, 15-day stride (10 monitored) | 10/10 | 10/10 | 0/10 | 20/30 |
| R6 HistGradientBoosting (ref F1 0.228) | 5/5 | 5/5 | 0/5 | 10/15 |
| R7 seed 7 / seed 2026 | 5/5 | 5/5 | 0/5 | 10/15 |

Invariant: the constant-P(X) concept-plus-prior condition is detected in 0
windows in every variant (structural invisibility is configuration-
independent within this grid). Boundary-shifting variants for the
P(X)-changing conditions: delta=0.10 (7/15), 45-day windows (4/9, the
weakest-injection window is missed after cap recalibration widens the
normalization range: PSI cap 0.641, Conf cap 0.758), and feature-category
ablation (5/15 without feature PSI; uncertainty alone 2/15).

## 2. Component attribution (R3)

Feature PSI carries detection: it detects P(X)-changing windows 10/10 alone.
Score-distribution alone detects 5/15 (3/5 covariate, 2/5 mixed);
uncertainty alone 2/15 (1/5, 1/5). Dropping score or uncertainty from the
composite leaves detection unchanged (10/15). Interpretation: in this
calibration the marginal detection contribution of the score-distribution
and uncertainty categories is small; their role is reliability-side coverage
(R_proxy estimation), not detection. Note: solo-category variants remove all
coverage for one estimated dimension, which then defaults to the optimistic
1.0 (e.g. fea-only leaves reliability uncovered and raises absolute S_proxy
levels); divergence-based detection is unaffected by this level shift.

## 3. Shared-component quantification (R4, addresses v31 threat T7)

Per-condition proxy-vs-actual tracking with and without the shared
deterministic completeness/freshness terms (weights renormalized 0.6/0.4
over reliability/representativeness; gate reduces to min(1, R/tau_r)):

| Condition | corr with C/F | corr without C/F | mean gap with | mean gap without |
|---|---|---|---|---|
| Baseline | 0.991 | 0.805 | 0.195 | 0.425 |
| Covariate | 0.988 | 0.930 | 0.085 | 0.174 |
| Mixed | 0.966 | 0.866 | 0.160 | 0.295 |
| Concept+prior | 0.936 | 0.514 | 0.307 | 0.583 |

The shared deterministic components contribute a large part of the apparent
proxy-actual tracking; the residual proxy-estimated signal still tracks the
actual score under P(X)-changing drift (0.87-0.93) and fails under
constant-P(X) drift (0.51). Without C/F, the last-window baseline S_proxy
stays at 0.693 (vs 0.294 with C/F): most of the baseline proxy decline in
the base configuration is the deterministic label-latency decay, not a
drift signal.

## 4. Last-window (W5) values, base scheme

| Variant | S_proxy cov | S_actual cov | S_proxy mix | S_actual mix | S_proxy con | S_actual con | Baseline S_proxy/S_actual |
|---|---|---|---|---|---|---|---|
| Base | 0.105 | 0.121 | 0.159 | 0.037 | 0.294 | 0.008 | 0.294 / 0.167 |
| W1 equal | 0.108 | 0.140 | 0.162 | 0.045 | 0.310 | 0.010 | 0.310 / 0.193 |
| W2 rel-heavy | 0.119 | 0.118 | 0.185 | 0.034 | 0.342 | 0.007 | 0.342 / 0.164 |
| W3 fresh-light | 0.122 | 0.144 | 0.186 | 0.045 | 0.350 | 0.010 | 0.350 / 0.199 |
| HGB | 0.263 | 0.181 | 0.265 | 0.058 | 0.331 | 0.010 | 0.331 / 0.199 |
| seed 7 | 0.103 | 0.117 | 0.156 | 0.039 | 0.294 | 0.008 | 0.294 / 0.167 |
| seed 2026 | 0.104 | 0.114 | 0.157 | 0.033 | 0.294 | 0.009 | 0.294 / 0.167 |

The proxy-actual gap in the concept-plus-prior condition is large and
positive in every variant: 0.286-0.340 across the weight variants and
0.321 for the HGB model class (ablation variants extend the spread, e.g.
0.358 for fea-only): proxies report near-baseline health while actual
sufficiency collapses. The covariate-condition gap is small and
sign-unstable across variants (-0.05 to +0.13), so no claim about proxy
conservatism in detected conditions should rely on its sign. Seed
stability: W5 S_proxy moves at most 0.003 and W5 actual S at most 0.007
across seeds 42/7/2026.

## 5. Simulator checkpoint sensitivity (R8)

S(t) checkpoints (days 30/60/90/180), base drift specs, one-at-a-time:

| Variant | No drift | Covariate | Concept+prior | Mixed |
|---|---|---|---|---|
| base lambda=0.02 tau_r=0.15 | 0.510 / 0.418 / 0.325 / 0.040 | 0.484 / 0.346 / 0.224 / 0.020 | 0.424 / 0.242 / 0.110 / 0.010 | 0.466 / 0.290 / 0.160 / 0.013 |
| lambda=0.01 | 0.562 / 0.484 / 0.384 / 0.046 | 0.534 / 0.408 / 0.277 / 0.026 | 0.467 / 0.282 / 0.131 / 0.012 | 0.514 / 0.341 / 0.196 / 0.016 |
| lambda=0.04 | 0.444 / 0.362 / 0.292 / 0.039 | 0.419 / 0.293 / 0.194 / 0.019 | 0.368 / 0.208 / 0.097 / 0.010 | 0.404 / 0.247 / 0.140 / 0.013 |
| tau_r=0.10 | 0.575 / 0.470 / 0.366 / 0.045 | 0.556 / 0.414 / 0.280 / 0.026 | 0.568 / 0.363 / 0.164 / 0.015 | 0.560 / 0.425 / 0.240 / 0.020 |
| tau_r=0.20 | 0.383 / 0.313 / 0.244 / 0.030 | 0.363 / 0.259 / 0.168 / 0.015 | 0.318 / 0.182 / 0.082 / 0.007 | 0.350 / 0.217 / 0.120 / 0.010 |

Orderings at day 90 and day 180 are stable in every variant (concept+prior
lowest, no-drift highest). Crossing-day statements are threshold-relative:
at tau_r=0.10 the no-drift trajectory is still above 0.5 at day 30 (0.575),
so the base-configuration statement "S falls below 0.5 shortly after day 30"
does not transfer to other gate calibrations. At tau_r=0.10 the day-30
separation between concept+prior (0.568) and no-drift (0.575) nearly
vanishes because the reliability gate barely binds that early.
