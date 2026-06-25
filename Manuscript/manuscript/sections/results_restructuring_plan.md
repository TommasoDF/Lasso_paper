# Results Section: Restructuring Plan

## Core organizing principle

Three dimensions of the LASSO selection process: **when**, **who**, and **what**.

---

## Proposed structure

### A. Time-series variation — when does perceived predictability arise?

**Theory benchmark.** Under CMCE with constant idiosyncratic volatility, the aggregate
selection rate should be flat. Any systematic co-movement with observable variables is
extra-model.

**Finding 1 — episodic spikes.** Selection is episodic: it spikes during periods of macro
stress (Jan 2012 peak, S&P downgrade, taper tantrum, Brexit, 2016 election) and collapses
during calm periods (Oct 2014 trough).

**Finding 2 — what triggers spikes.** Time-varying fundamental uncertainty (GARCH volatility)
mechanically inflates OLS coefficients, pushing more features over the LASSO threshold by
chance. Evidence:
- Oracle simulation: i.i.d. noise → flat (std ratio 19×); GARCH noise → largely recovers
  spikes (ratio 1.32×, Spearman ρ = 0.02 → 0.38).
- Regression of aggregate selection rate on σ_r and σ_x (R² = 0.12); replacing σ_r with
  GARCH conditional volatility σ_m raises R² to 0.18.

**Finding 3 — what sustains them.** Once selected, features persist. Survival curve, run-length
distribution (median 4 windows, p95 = 78), ACF (first-order AC = 0.89). Consistent with
the self-fulfilling nature of CMCE: a selected belief generates returns that reinforce the
selection signal until the rolling window rolls past the episode.

**⚠ Numbers to verify before finalising:**
- Peak/trough selection rates (Jan 2012 / Oct 2014 values)
- Regression betas and R² (current saved results differ from text: β_σr = 0.155 vs 0.188,
  R² = 0.03 vs 0.12 — needs re-run or re-check)
- GARCH ratio 19× → 1.32×, Spearman 0.02 → 0.38 (needs re-check from notebook)
- Persistence stats: 623,223 episodes, run-length p95 = 78, AC values (from notebook)

---

### B. Cross-sectional variation — which stocks are susceptible?

**Theory benchmark.** Proposition [X]: under a global penalty λ, more volatile stocks
have a higher probability of selection for any given predictor.

**Finding 1 — distribution.** Right-skewed distribution across stocks; ~40% of stocks
near zero selection (selection never occurs). This is consistent with the model: for
low-volatility stocks, returns fluctuate too little for any OLS coefficient to exceed λ.

**Finding 2 — variance-selection relationship.** Strong positive relationship between
return variance and average number of selected features per window (figure).

**Finding 3 — sectoral heterogeneity.** Energy & Mining highest (high-variance,
commodity-driven returns); Financials lowest. Pattern is consistent with the volatility
mechanism rather than with sector-specific information content.

**⚠ Numbers to verify before finalising:**
- 40% near-zero figure (current: 39.7% below 0.3% threshold — consistent)
- Sectoral ranking (from figures — check figures are regenerated with current lambda)

---

### C. What gets selected — statistical accident or prior beliefs?

**Theory benchmark.** With standardized regressors, there is no mechanical reason for
any feature to be selected more frequently than another. Selection should be approximately
uniform across the 190 predictors.

**Finding 1 — mild concentration.** Gini ≈ 0.24; macro series selected ~1.3× more than
topics on average, plausibly because higher autocorrelation → larger rolling-window OLS
coefficients. Top feature: "Announce plan" at ~3.9× the uniform null rate.

**Finding 2 — narrative alignment.** Within-firm, within-topic test (panel regression with
stock and feature FE): features linguistically aligned with a firm's 10-K are marginally
more likely to be selected for that firm. Effect is statistically significant but
economically small; concentrated in Consumer and Financial sectors, absent in Technology
and Healthcare. Suggests agents bring weak prior beliefs but statistical chance remains
the dominant driver.

**⚠ Numbers to verify before finalising:**
- Gini coefficient, macro/topic ratio, top-feature multiplier (from notebook — not in saved CSVs)
- Narrative alignment table (591 firms, 112,290 obs) — check if based on current estimation

---

## Changes vs. current text ordering

| Current order | New order |
|---|---|
| Stage-2 results | Stage-2 results (unchanged — comes before this section) |
| Selection frequency across features | → C, Finding 1 |
| Cross-sectional selection intensity | → B |
| Time-series variation | → A, Finding 1 |
| GARCH / regression decomposition | → A, Findings 2 |
| Persistence | → A, Finding 3 |
| Narrative alignment | → C, Finding 2 |

Persistence moves inside A (it explains how spikes sustain themselves).
The variance-selection figure and distribution move to B as the centrepiece.
Selection frequency and narrative alignment are unified in C under the common
"uniform selection" null.

---

## Pending tasks before writing

1. Re-run or re-check regression evidence (R² discrepancy: 0.12 in text vs 0.03 from CSV).
2. Confirm GARCH simulation stats (19× → 1.32×, Spearman) from notebook.
3. Confirm persistence stats (623,223 episodes, median run 4, AC = 0.89) from notebook.
4. Confirm Gini / macro-topic ratio / top-feature stats from notebook.
5. Update robustness table (tab:comparison) baseline column with current numbers.
6. Check volscaled results CSV exists / re-run volscaled estimation if needed.
7. Decide whether to update regression betas and R² in text, or re-run the analysis.
