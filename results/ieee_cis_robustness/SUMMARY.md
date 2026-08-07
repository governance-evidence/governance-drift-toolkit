# IEEE-CIS robustness study

Sensitivity, ablation, and model-dependence results for the delayed-label
evidence sufficiency evaluation reported in the accompanying manuscript
(preprint arXiv:2604.15740). Release identity and reproduction commands:
[RELEASE.md](RELEASE.md).

Produced by `examples/ieee_cis_robustness.py` (this repository; drift 0.5.0,
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
actual S 0.120 / 0.035 / 0.007; simulator checkpoints identical to Table 3). Simulator values are those of
Evidence Sufficiency Calculator 0.4.1, reseeded from the cross-fitted reference
F1; earlier releases do not reproduce them.

Separation rule: |S_proxy(drift, w) - S_proxy(baseline, w)| > delta, delta =
0.05 unless varied.

## 1. Separation counts by variant (covariate / mixed / concept, total)

The Total column covers the two conditions whose separation can vary. The
concept-plus-prior column is reported but excluded from it: its zero is forced
by the construction, because the injector edits labels only and every quantity
entering the proxy score is a deterministic function of the untouched feature
records and the fixed reference model. Averaging it together with the empirical
conditions would produce a rate that measures the choice to include it.

| Variant | Covariate | Mixed | Concept | Total |
|---|---|---|---|---|
| Base (W0, tau_r=0.15, tau_rp=0.55, tau_c=0.6, delta=0.05) | 5/5 | 5/5 | 0/5 | 10/10 |
| R1 W1 equal weights | 5/5 | 5/5 | 0/5 | 10/10 |
| R1 W2 reliability-heavy | 5/5 | 5/5 | 0/5 | 10/10 |
| R1 W3 freshness-light | 5/5 | 5/5 | 0/5 | 10/10 |
| R2 tau_r_actual 0.10 | 5/5 | 5/5 | 0/5 | 10/10 |
| R2 tau_r_actual 0.20 | 5/5 | 5/5 | 0/5 | 10/10 |
| R2 tau_r_proxy 0.45 | 5/5 | 5/5 | 0/5 | 10/10 |
| R2 tau_r_proxy 0.65 | 5/5 | 5/5 | 0/5 | 10/10 |
| R2 delta 0.03 | 5/5 | 5/5 | 0/5 | 10/10 |
| R2 delta 0.10 | 4/5 | 2/5 | 0/5 | 6/10 |
| R3 fea+unc (drop score) | 5/5 | 5/5 | 0/5 | 10/10 |
| R3 scr+fea (drop uncertainty) | 5/5 | 5/5 | 0/5 | 10/10 |
| R3 scr+unc (drop feature) | 2/5 | 1/5 | 0/5 | 3/10 |
| R3 fea only | 5/5 | 5/5 | 0/5 | 10/10 |
| R3 scr only | 3/5 | 1/5 | 0/5 | 4/10 |
| R3 unc only | 1/5 | 0/5 | 0/5 | 1/10 |
| R4 without shared C/F | 5/5 | 5/5 | 0/5 | 10/10 |
| R5 15-day windows (11 monitored) | 11/11 | 11/11 | 0/11 | 22/22 |
| R5 45-day windows (3 monitored) | 2/3 | 2/3 | 0/3 | 4/6 |
| R5 30-day windows, 15-day stride (10 monitored) | 10/10 | 10/10 | 0/10 | 20/20 |
| R6 HistGradientBoosting | 5/5 | 5/5 | 0/5 | 10/10 |
| R7 seed 7 | 5/5 | 5/5 | 0/5 | 10/10 |
| R7 seed 2026 | 5/5 | 5/5 | 0/5 | 10/10 |

## 2. Component attribution (R3)

Feature PSI carries separation: it separates the feature-reaching windows
10/10 alone. Score-distribution alone separates 4/10 (3/5 covariate, 1/5
mixed); uncertainty alone 1/10 (1/5, 0/5). Dropping score or uncertainty
from the composite leaves separation unchanged (10/10). Interpretation: in
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

| Variant | No drift | Covariate | Real concept | Mixed |
|---|---|---|---|---|
| base lambda0.02 taur0.15 | 0.480 / 0.392 / 0.305 / 0.037 | 0.455 / 0.324 / 0.210 / 0.019 | 0.399 / 0.228 / 0.103 / 0.009 | 0.438 / 0.272 / 0.150 / 0.012 |
| lambda 0.01 | 0.528 / 0.454 / 0.361 / 0.043 | 0.502 / 0.383 / 0.260 / 0.024 | 0.439 / 0.265 / 0.123 / 0.011 | 0.484 / 0.320 / 0.184 / 0.015 |
| lambda 0.04 | 0.417 / 0.339 / 0.273 / 0.036 | 0.394 / 0.275 / 0.181 / 0.018 | 0.346 / 0.196 / 0.092 / 0.009 | 0.380 / 0.232 / 0.131 / 0.012 |
| tau r 0.1 | 0.572 / 0.468 / 0.364 / 0.044 | 0.554 / 0.412 / 0.278 / 0.026 | 0.566 / 0.342 / 0.155 / 0.014 | 0.558 / 0.409 / 0.225 / 0.019 |
| tau r 0.2 | 0.360 / 0.294 / 0.229 / 0.028 | 0.341 / 0.243 / 0.157 / 0.014 | 0.299 / 0.171 / 0.077 / 0.007 | 0.329 / 0.204 / 0.113 / 0.009 |
