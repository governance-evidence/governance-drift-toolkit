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
`ieee_cis_simulator_sensitivity.json` in the calculator repository's
`results/` directory.

Three calibration properties govern how these numbers should be read.
Median imputation statistics are fitted on the reference window of each
window scheme and applied forward, so no monitoring window informs the
values used to fit and calibrate the reference pipeline. Reference F1 is
estimated out-of-sample by five-fold cross-fitting within the reference
window; the in-sample value is reported alongside it as `ref_f1_insample`
and is not used anywhere downstream. Mean gaps are means of absolute
per-window differences, so errors of opposite sign do not cancel.

Reproduction gate: the base configuration reproduces the demo numbers
exactly (separation 5/5, 5/5, 0/5; W5 S_proxy 0.131 / 0.184 / 0.296; W5
actual S 0.120 / 0.035 / 0.007; simulator checkpoints identical to Table 3).

Separation rule: |S_proxy(drift, w) - S_proxy(baseline, w)| > delta, delta =
0.05 unless varied.

## 1. Separation counts by variant (covariate / mixed / concept, total)

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
| R2 delta 0.10 | 4/5 | 2/5 | 0/5 | 6/15 |
| R3 fea+unc (drop score) | 5/5 | 5/5 | 0/5 | 10/15 |
| R3 scr+fea (drop uncertainty) | 5/5 | 5/5 | 0/5 | 10/15 |
| R3 scr+unc (drop feature) | 2/5 | 1/5 | 0/5 | 3/15 |
| R3 fea only | 5/5 | 5/5 | 0/5 | 10/15 |
| R3 scr only | 3/5 | 1/5 | 0/5 | 4/15 |
| R3 unc only | 1/5 | 0/5 | 0/5 | 1/15 |
| R4 without shared C/F components | 5/5 | 5/5 | 0/5 | 10/15 |
| R5 15-day windows (11 monitored) | 11/11 | 11/11 | 0/11 | 22/33 |
| R5 45-day windows (3 monitored) | 2/3 | 2/3 | 0/3 | 4/9 |
| R5 30-day windows, 15-day stride (10 monitored) | 10/10 | 10/10 | 0/10 | 20/30 |
| R6 HistGradientBoosting (ref F1 0.171, in-sample 0.227) | 5/5 | 5/5 | 0/5 | 10/15 |
| R7 seed 7 / seed 2026 | 5/5 | 5/5 | 0/5 | 10/15 |

The concept column is zero in every cell, and that is a property of the
construction rather than a robustness result. The concept-plus-prior
injector edits labels only; the reference model is fitted once on the
uninjected reference window and then held fixed; every quantity entering
S_proxy is a deterministic function of the feature records and that fixed
model. The paired difference therefore vanishes in every window under every
variant, and no grid of calibration variants can either strengthen or weaken
it. The column is retained as an implementation check: it confirms that no
label information reaches the proxy signals through an unintended path. Note
also that `tau_r_actual` enters only S_actual, so varying it cannot change
any separation count; its row is kept for completeness of the executed grid.

Boundary-shifting variants for the conditions that can vary: delta=0.10
(6/15), 45-day windows (4/9, the weakest-injection window is missed after cap
recalibration widens the normalization range: PSI cap 0.693, Conf cap 0.808),
and feature-category ablation (3/15 without feature PSI; uncertainty alone
1/15).

## 2. Component attribution (R3)

Feature PSI carries separation: it separates the feature-reaching windows
10/10 alone. Score-distribution alone separates 4/15 (3/5 covariate, 1/5
mixed); uncertainty alone 1/15 (1/5, 0/5). Dropping score or uncertainty
from the composite leaves separation unchanged (10/15). Interpretation: in
this calibration the marginal separation contribution of the
score-distribution and uncertainty categories is small; their role is
reliability-side coverage (R_proxy estimation), not separation. Note:
solo-category variants remove all coverage for one estimated dimension,
which then defaults to the optimistic 1.0 (e.g. fea-only leaves reliability
uncovered and raises absolute S_proxy levels); divergence-based separation is
unaffected by this level shift.

## 3. Shared-component quantification (R4, addresses v31 threat T7)

Per-condition proxy-vs-actual tracking with and without the shared
deterministic completeness/freshness terms (weights renormalized 0.6/0.4
over reliability/representativeness; gate reduces to min(1, R/tau_r)):

| Condition | corr with C/F | corr without C/F | mean gap with | mean gap without |
|---|---|---|---|---|
| Baseline | 0.995 | 0.981 | 0.195 | 0.422 |
| Covariate | 0.982 | 0.914 | 0.101 | 0.199 |
| Mixed | 0.968 | 0.891 | 0.177 | 0.320 |
| Concept+prior | 0.951 | 0.827 | 0.310 | 0.578 |

The shared deterministic components contribute a large part of the apparent
proxy-actual tracking; the residual proxy-estimated signal still tracks the
actual score under the feature-reaching conditions (0.89-0.91) and tracks it
least well under the label-side condition (0.83), where the mean absolute gap
is also the largest without C/F (0.578). Without C/F, the last-window
baseline S_proxy stays at 0.697 (vs 0.296 with C/F): most of the baseline
proxy decline in the base configuration is the deterministic label-latency
decay, not a drift signal.

## 4. Last-window (W5) values, base scheme

| Variant | S_proxy cov | S_actual cov | S_proxy mix | S_actual mix | S_proxy con | S_actual con | Baseline S_proxy/S_actual |
|---|---|---|---|---|---|---|---|
| Base | 0.131 | 0.120 | 0.184 | 0.035 | 0.296 | 0.007 | 0.296 / 0.152 |
| W1 equal | 0.134 | 0.139 | 0.187 | 0.042 | 0.311 | 0.009 | 0.311 / 0.177 |
| W2 rel-heavy | 0.150 | 0.116 | 0.215 | 0.032 | 0.344 | 0.007 | 0.344 / 0.149 |
| W3 fresh-light | 0.153 | 0.143 | 0.215 | 0.042 | 0.351 | 0.009 | 0.351 / 0.182 |
| HGB | 0.259 | 0.179 | 0.262 | 0.056 | 0.326 | 0.010 | 0.326 / 0.197 |
| seed 7 | 0.129 | 0.116 | 0.184 | 0.037 | 0.296 | 0.007 | 0.296 / 0.152 |
| seed 2026 | 0.127 | 0.115 | 0.182 | 0.033 | 0.296 | 0.008 | 0.296 / 0.152 |

The concept-plus-prior S_proxy equals the baseline S_proxy in every row, as
the construction requires. The proxy-actual gap in that condition is
correspondingly large and positive in every variant: 0.289-0.342 across the
weight variants and 0.316 for the HGB model class (ablation variants extend
the spread, e.g. 0.356 for fea-only): proxies report baseline health while
actual sufficiency collapses. The covariate-condition gap is small and
sign-unstable across variants (-0.05 to +0.12), so no claim about proxy
conservatism in separated conditions should rely on its sign. Seed
stability: W5 S_proxy moves at most 0.004 and W5 actual S at most 0.005
across seeds 42/7/2026.

## 5. Simulator checkpoint sensitivity (R8)

S(t) checkpoints (days 30/60/90/180), base drift specs, one-at-a-time:

| Variant | No drift | Covariate | Concept+prior | Mixed |
|---|---|---|---|---|
| base lambda=0.02 tau_r=0.15 | 0.481 / 0.394 / 0.306 / 0.037 | 0.456 / 0.326 / 0.211 / 0.019 | 0.400 / 0.229 / 0.104 / 0.009 | 0.440 / 0.273 / 0.151 / 0.012 |
| lambda=0.01 | 0.530 / 0.456 / 0.362 / 0.043 | 0.504 / 0.384 / 0.261 / 0.024 | 0.441 / 0.266 / 0.124 / 0.011 | 0.485 / 0.321 / 0.185 / 0.015 |
| lambda=0.04 | 0.419 / 0.341 / 0.274 / 0.036 | 0.395 / 0.276 / 0.182 / 0.018 | 0.348 / 0.197 / 0.092 / 0.009 | 0.381 / 0.233 / 0.131 / 0.012 |
| tau_r=0.10 | 0.572 / 0.468 / 0.364 / 0.044 | 0.554 / 0.412 / 0.278 / 0.026 | 0.566 / 0.343 / 0.155 / 0.014 | 0.558 / 0.410 / 0.226 / 0.019 |
| tau_r=0.20 | 0.361 / 0.295 / 0.230 / 0.028 | 0.342 / 0.244 / 0.158 / 0.014 | 0.300 / 0.171 / 0.078 / 0.007 | 0.330 / 0.205 / 0.113 / 0.009 |

Orderings at day 90 and day 180 are stable in every variant (concept+prior
lowest, no-drift highest). Crossing-day statements are threshold-relative:
in the base configuration the no-drift trajectory falls below 0.5 at day 26,
but at tau_r=0.10 it is still above 0.5 at day 30 (0.572), so the crossing
day does not transfer to other gate calibrations. At tau_r=0.10 the day-30
separation between concept+prior (0.566) and no-drift (0.572) nearly
vanishes because the reliability gate barely binds that early.
