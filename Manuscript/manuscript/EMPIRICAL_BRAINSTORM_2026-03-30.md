# Empirical Strategy Brainstorm

**Paper**: Sparse Learning and Endogenous Pockets of Predictability
**Authors**: Tommaso Di Francesco, Stefanie Huber, Jonathan Seim (University of Bonn / University of Cologne)
**Date**: 2026-03-30
**Target journal**: Journal of Finance / Journal of Financial Economics (JEL G12, G14, C55)

---

## Executive Summary

The single strongest empirical strategy that emerges from the four agents is a **clean two-stage design with strict sample separation**: select hyperparameters (s, λ, n) on a held-out period (2005–2009 or a held-out stock subset), freeze them, estimate κ on 2010–2017 with block-bootstrapped standard errors, and validate with a genuine out-of-sample ALM forecast test for 2015–2017. The biggest single risk is the **pre-testing problem embedded in the Bayesian optimisation objective**: because the current objective directly rewards κ significance on the estimation sample, every reported t-statistic is uninformative and the 25%-significant-κ finding is circular. This must be fixed before revision submission or the paper will be desk-rejected. A **permutation placebo test** — randomly reassigning the LASSO belief path β_t across dates and showing κ collapses to zero — is the single cheapest, most powerful fix to pre-empt the hardest referee critique. The analysis that should be started immediately is implementing the held-out hyperparameter selection and the two-stage block bootstrap, since all other results depend on these being credible.

---

## 1. Literature Benchmarks and Referee Expectations

### 1. Closest Empirical Antecedents

#### 1a. Bounded Rationality and Learning in Asset Pricing

**Barberis, Shleifer, and Vishny (1998, JFE)** — "A Model of Investor Sentiment." BSV construct a model where investors update beliefs about a regime-switching earnings process using Bayes' rule but with incorrect priors. Their empirical approach is indirect: they calibrate the model to match post-earnings-announcement drift and long-run reversals in US stock returns. Data: CRSP monthly returns 1926–1995. Methods: simulated moments, calibration. This is the archetype of the "calibrate-to-match-anomalies" approach that referees have internalized but now often find insufficient on its own.

**Daniel, Hirshleifer, and Subrahmanyam (1998, JF)** — "Investor Psychology and Security Markets." DHS introduce overconfidence and self-attribution bias. Empirical content is largely calibration-based, showing the model can reproduce momentum and reversals. Important as a benchmark precisely because later papers have been pushed to do more rigorous structural work.

**Adam, Marcet, and Nicolini (2016, JF)** — "Stock Market Volatility and Learning." The most directly relevant structural predecessor. AMN build a Lucas-tree economy where agents use least-squares learning (constant-gain OLS) to update beliefs. They estimate the constant-gain parameter γ by minimizing the distance between model-implied and empirical autocorrelation patterns of the P/D ratio. Data: annual S&P 500 data 1871–2006. Methods: indirect inference / SMM targeting autocorrelation structure of P/D ratio and return moments. Key finding: γ ≈ 0.0025 fits the data; the learning model substantially outperforms the RE benchmark. This paper establishes the methodological template that Di Francesco et al. most closely follows.

**Adam, Beutel, and Marcet (2017, AER P&P)** — Extends AMN to show that learning-based models can match survey expectation data (Shiller surveys, CFO surveys). Cross-validating structural parameters against survey data is a gold-standard robustness check.

**Nagel and Xu (2022, RFS)** — "Asset Pricing with Fading Memory." Agents forecast using WLS with exponentially decaying weights. Structurally estimates the memory parameter using survey forecasts (Blue Chip, SPF) and asset return predictability. Data: US equity and bond returns, survey forecasts 1968–2018. Methods: MSM/SMM. Arguably the most methodologically sophisticated predecessor: uses WLS (exactly as in the PLM), structurally estimates the gain/decay parameter, and cross-validates against survey forecasts. Referees will draw a direct comparison.

**Eusepi and Preston (2011, AER)** — "Expectations, Learning, and Business Cycle Fluctuations." Structural estimation via Bayesian DSGE comparison of learning model to FIRE benchmark. Established the identification approach of asking whether a learning model's likelihood is higher than FIRE's.

**Collin-Dufresne, Johannes, and Lochstoer (2017, RFS)** — "Asset Pricing When 'This Time Is Different.'" Agents learn about rare disasters using Bayesian updating. Parameters estimated via maximum likelihood. Demonstrates that Bayesian learning with misspecified priors can generate predictability without full irrationality.

---

#### 1b. LASSO and Machine Learning for Return Predictability

**Feng, Giglio, and Xiu (2020, JF)** — "Taming the Factor Zoo." Canonical LASSO/ML paper for asset pricing. Uses double-selection LASSO to select significant risk factors from 150+ candidates, with cross-validation to set the penalty. Key methodological contribution: shows naive LASSO understates standard errors post-selection, requiring the double-selection correction (Belloni, Chernozhukov, and Hansen 2014). Di Francesco et al.'s NLS inference must engage with this — if it does not account for LASSO-induced selection bias, the κ t-statistics are not valid.

**Gu, Kelly, and Xiu (2020, RFS)** — "Empirical Asset Pricing via Machine Learning." Most comprehensive ML return-prediction paper. LASSO achieves OOS R² of approximately 0.3–0.4% monthly. Data: CRSP/Compustat 1957–2016, 30,000+ stock-months. Methods: rolling expanding window, time-series train/validation/test split, portfolio sorts. The negative OOS R² reported by Di Francesco et al. (−0.023) must be contrasted explicitly with GKX's positive but small OOS R².

**Chinco, Clark-Joseph, and Ye (2019, JF)** — "Sparse Signals in the Cross-Section of Returns." Apply LASSO to predict 1-minute-ahead excess returns using lagged returns of all other S&P 500 stocks. Find significant but ephemeral predictability. Data: TAQ, 2001–2014. Methods: LASSO with cross-validated penalty, rolling 1-month estimation window. This paper is the single closest empirical antecedent to the PLM estimation strategy and must be cited prominently.

**Kozak, Nagel, and Santosh (2020, JF)** — "Shrinking the Cross-Section." Uses elastic net / ridge regression in the SDF space to shrink factor exposures. Establishes that regularization is necessary in the cross-section and that the selected factors are sparsely concentrated.

**Kelly and Pruitt (2015, JFE)** — "The Three-Pass Regression Filter." Proposes 3PRF as an alternative to PCA for forecasting with many predictors.

---

#### 1c. Return Predictability Using Text and Topic-Based Predictors

**Bybee, Kelly, Manela, and Xiu (2021, RFS)** — "The Structure of Economic News." Direct source of the 180 WSJ topic attention innovations. Their approach: apply LDA to the full text of WSJ articles 1984–2017, extract 180 topics, construct attention innovations, and show these predict aggregate stock returns and economic activity out-of-sample. Di Francesco et al. inherit both the data source and the challenge that these predictors have weak individual predictability.

**Tetlock (2007, JF)** — "Giving Content to Investor Sentiment." Pioneering paper using WSJ column to construct a daily sentiment measure via the General Inquirer lexicon. Predictive regression of daily DJIA returns on lagged sentiment. Methodological ancestor of the WSJ-text-to-predictor pipeline.

**Tetlock, Saar-Tsechansky, and Macskassy (2008, JF)** — Extends to firm-level return prediction using negative word counts in firm-specific news. Establishes the individual-stock text-based prediction framework.

**Manela and Moreira (2017, JF)** — "News Implied Volatility and Disaster Concerns." Constructs a NVIX measure from WSJ front-page text. Demonstrates that text-derived signals can capture risk-premium variation.

**Calomiris and Mamaysky (2019, JF)** — Uses NLP to classify news tone and context. Data: Reuters news, 51 countries, 2003–2015. Shows text-based signals predict returns and volatility at the individual-asset level.

---

#### 1d. Structural Estimation of Asset Pricing Models via GMM/SMM

**Hansen and Singleton (1982, Econometrica)** — Foundational GMM estimation of Euler-equation-based asset pricing models. Di Francesco et al.'s NLS estimation of κ is spiritually a single-equation analog of this approach.

**Bansal and Yaron (2004, JF)** — "Risks for the Long Run." Estimates a long-run-risks model via SMM targeting 12 moments using annual US data 1929–1998. Establishes the canonical SMM moment-targeting approach. The profession now expects at minimum 8–12 carefully chosen moments, an explicit weighting matrix, and sensitivity analysis.

**Lettau and Ludvigson (2001, JF)** — "Resurrecting the (C)CAPM." GMM estimation using cay as conditioning variable. Establishes that time-varying risk premia models must be tested with formal hypothesis tests on pricing errors, not just R² in cross-sectional regressions.

---

#### 1e. Learning Dynamics and Equilibrium Selection

**Branch and Evans (2006, EJ)** — "A Simple Recursive Forecasting Model." Derives a model where agents choose between forecasting rules based on recent performance. The intellectual predecessor to the CMCE concept: asks whether a non-RE fixed point can be stable under adaptive learning.

**Nagel and Xu (2022, RFS)** — (see above) Most direct methodological comparison for referees.

**Brock and Hommes (1998, Econometrica)** — "Heterogeneous Beliefs and Routes to Chaos." Foundational paper for the endogenous-dynamics-from-learning literature; referees will expect engagement.

**Hommes (2011, JEL)** — Reviews experimental evidence on learning dynamics, providing experimental validation of learning-based models.

---

### 2. Methodological Gold Standard

#### 2a. Estimating Model Hyperparameters (Penalty λ, Window s)

The leading approach, as established by **Gu, Kelly, and Xiu (2020)** and **Feng, Giglio, and Xiu (2020)**, is **time-series cross-validation with a strictly held-out test period**:

1. Split the sample into training, validation (hyperparameter selection), and test (final evaluation, never touched during search) periods.
2. For LASSO penalty: use BIC or time-series walk-forward cross-validation (not k-fold, due to serial dependence).
3. Select window length jointly with the penalty via a grid search on the validation period, targeting out-of-sample Sharpe ratio or OOS R².

**Kozak, Nagel, and Santosh (2020, JF)** set the penalty via economic priors rather than cross-validation — the closest analog to what Di Francesco et al. do. The gold standard fix: use a strictly separated validation set or report performance across a grid of (λ, s) values, showing the main results are not knife-edge.

#### 2b. Identifying and Testing a Feedback Mechanism

Referees will expect:
1. **Direct test of the feedback coefficient κ**: NLS with HAC standard errors is the minimum; the gold standard adds a VAR representation of the (belief, price) system testing Granger causality from beliefs to prices and back.
2. **Instrumental variable strategy**: Find an instrument for belief activation — exogenous news shocks that trigger LASSO selection but are orthogonal to fundamental value.
3. **Panel IV or Bartik-style identification**: Instrument κ_i with stock-level characteristics (volatility, market cap, analyst coverage) that predict learning intensity but are predetermined.

#### 2c. Distinguishing Rational Expectations from Learning-Based Models

Three approaches dominate:
1. **Predictability asymmetry test** (Adam, Marcet, Nicolini 2016): Under FIRE, predictability is zero unconditionally and conditionally. Under learning, predictability is episodic. Formalize as a likelihood ratio test between FIRE null and learning alternative.
2. **Survey forecast consistency test** (Nagel and Xu 2022; Adam and Marcet 2011): Regression of analyst forecast revisions on LASSO-selected predictors tests whether model-implied belief dynamics are consistent with observed expectations. Near-mandatory at top journals since 2018.
3. **Predictive system decomposition** (Cochrane 2008; Lettau and Van Nieuwerburgh 2008): Decompose return variance into cash-flow and discount-rate news using a VAR. Under learning, residual discount-rate variation should be correlated with the model-implied belief process.

#### 2d. Testing Whether Model-Implied Beliefs Are Consistent with Prices

1. **Moment inequality tests**: The model implies E[r_{t+1} | X_t, β_t ≠ 0] > 0. Testable via Andrews and Guggenberger (2009) conditional moment inequality framework.
2. **SMM targeting the joint distribution of (prices, beliefs, predictability)**: Simulate the model at estimated parameters and compare to empirical moments via a χ² overidentification test.
3. **Out-of-sample density forecasting**: Clark and West (2007) test for nested model comparison, comparing ALM predictive density against FIRE benchmark.

---

### 3. Expected Tables and Figures at Top Journals

1. **Table 1 — Summary Statistics of the Stock Sample and Predictor Set** | N stocks, time period, mean/median market cap, return volatility; for 180 WSJ topics: top-10 topics by selection frequency, pairwise correlation structure | Gu, Kelly, and Xiu (2020), Table 1
2. **Table 2 — PLM Estimation Results** | Distribution of in-sample R²_PLM, OOS R²_PLM, number of active predictors per stock-window, fraction of windows with ≥1 active predictor, breakdown by volatility/cap quintile | Chinco, Clark-Joseph, and Ye (2019), Table 2
3. **Table 3 — ALM Estimation Results: κ Distribution and Significance** | Distribution of κ̂_i, fraction significant at 1/5/10%, comparison across volatility quintiles, HAC standard errors, binomial test that fraction significant > multiple-testing threshold | Adam, Marcet, and Nicolini (2016), Table 3
4. **Table 4 — Pocket-of-Predictability Tests** | Conditional R² during active-LASSO periods vs. inactive periods; t-stat for difference; regime-switching regression: r_{t+1} = α + δ·1(β̂_t ≠ 0)·X_t^Tβ̂_t + ε_{t+1} | Rapach, Strauss, and Zhou (2010), Table 5
5. **Table 5 — Cross-Sectional Determinants of Predictability (κ)** | Fama-MacBeth regression of κ̂_i on log(market cap), return volatility, bid-ask spread, analyst coverage, short interest, institutional ownership; double-clustered SEs | Ang, Hodrick, Xing, Zhang (2006), Table 4
6. **Table 6 — Out-of-Sample Portfolio Performance** | Quintile sorts by predicted return f_t; long-short portfolio returns; alphas against FF5 + momentum + short-term reversal; Sharpe ratios | Gu, Kelly, and Xiu (2020), Table 5
7. **Table 7 — Comparison to FIRE Benchmark and Nested Model Tests** | Clark-West test of learning model against (a) FIRE null κ=0, (b) fixed-beliefs model, (c) OLS predictor without LASSO shrinkage | Nagel and Xu (2022), Table 4
8. **Table 8 — Robustness: Alternative Hyperparameters and Subsamples** | Main results for (λ, s) on a 3×3 grid around baseline, for subsamples 2010–2013/2014–2017, for large-cap vs. small-cap, for alternative predictor universes | Kozak, Nagel, Santosh (2020), Table 6
9. **Figure 1 — Time Series of Active Predictor Counts** | Average active coefficients over time overlaid with volatility regimes (VIX) and major market events; illustrates "episodic" nature visually | Chinco, Clark-Joseph, and Ye (2019), Figure 2
10. **Figure 2 — Predicted vs. Actual Returns: ALM Fit** | Scatter/time-series of f_t vs. realized r_{t+1} for a representative stock with NLS-fitted κ curve | Adam, Marcet, and Nicolini (2016), Figure 3
11. **Figure 3 — CMCE Fixed-Point Stability Diagram** | Mapping from β to ALM-implied β* as function of λ; stability of zero fixed point; phase diagram showing convergence/non-convergence | Branch and Evans (2006), Figure 1
12. **Figure 4 — Distribution of κ̂ Across Stocks** | Histogram of κ̂_i; separate distributions for significant and insignificant stocks; simulated distribution under null κ=0 | Nagel and Xu (2022), Figure 2
13. **Figure 5 — Hyperparameter Sensitivity Surface** | OOS R²_ALM (or fraction of significant κ̂) as function of (λ, s) on a heatmap; baseline parameter marked | Gu, Kelly, and Xiu (2020), Figure 3

---

### 4. Cautionary Tales from the Literature

1. **In-Sample Predictability Without OOS Validation** | Goyal and Welch (2008, RFS) showed that virtually all 17 widely-used predictors' in-sample performance disappears out-of-sample, causing a credibility collapse for an entire generation | The paper must either formally argue why OOS R² is not the criterion (in-sample belief-updating is the mechanism), or show portfolio returns are positive OOS
2. **LASSO Inference Without Post-Selection Correction** | Feng, Giglio, and Xiu (2020) document that naive post-LASSO OLS severely understates standard errors due to LASSO-induced selective inference | The Pagan (1984) generated-regressor framework and Escanciano et al. (2021) results on two-step estimation are the relevant corrections for the Stage 2 NLS
3. **Calibration as Substitute for Formal Identification** | Fama (1998, JFE) criticized BSV/DHS for matching anomalies without formally identifying key parameters | Di Francesco et al. must demonstrate κ_i is identified from variation genuinely distinct from what calibrated (λ, s)
4. **Overfitting in High-Dimensional Return Prediction** | Harvey, Liu, and Zhu (2016, RFS) — "and the Cross-Section" — document massive multiple-testing inflation; proposed t > 3.0 threshold | The paper must report Bonferroni-corrected or BH-FDR-adjusted significance for fraction of stocks with significant κ̂, plus a placebo test (permuted topic labels or shuffled returns)
5. **Fixed-Point Arguments Without Stability Analysis** | Marcet and Sargent (1989) and Evans-Honkapohja (2001): empirical relevance of a non-RE fixed point depends on E-stability under the actual recursive algorithm | Must simulate learning dynamics at estimated parameters and demonstrate the system remains bounded between episodes
6. **Time-Varying Parameters Without Structural Break Tests** | Pástor and Stambaugh (2001, JFE): estimated predictive regressions are highly sensitive to structural breaks in the predictor process | Bai-Perron tests on the most frequently selected predictors would be a natural robustness check referees may request

---

## 2. Proposed Empirical Approaches

### Category A — Parameter Estimation

**A1. Hierarchical Bayes across the cross-section with shrinkage on (λ, s)**

Rather than treating each stock's (λ, s) as independent free parameters, embed the NLS/SMM estimation inside a hierarchical model where λ_i and s_i are drawn from cross-sectional distributions with hyperparameters estimated jointly. For each stock, the likelihood comes from the ALM moment conditions. The cross-sectional prior on λ_i is parameterized as a function of observable firm characteristics (market cap, analyst coverage, idiosyncratic volatility), so the hierarchy regularizes identification for thin-data stocks and simultaneously delivers testable cross-sectional predictions. Estimation proceeds by MCMC with a Metropolis-within-Gibbs sampler.

- **Data:** Daily returns and 180-topic attention innovations for all 1,620 NYSE stocks; CRSP characteristics; I/B/E/S analyst coverage.
- **Key output:** Posterior means and credible intervals for hyperparameters of the λ and s distributions; scatter plot of posterior κ_i against firm characteristics with posterior uncertainty bands.
- **Positive result:** Establishes that (λ, s) vary systematically with observables, making the model testable out-of-sample and providing a disciplined way to assign parameters to new stocks.

**A2. SMM with higher-order moments as targets**

Estimate the four parameters (κ, λ, s, σ_d) jointly by SMM targeting: time-series mean and variance of returns, excess kurtosis, first-order autocorrelation, variance of the LASSO-fitted value f_t (pinning down belief volatility), and fraction of trading days with at least one nonzero coefficient (the sparsity rate). The weighting matrix comes from a Newey-West HAC estimator applied to the stacked moment vector. The identification insight: κ shifts skewness without proportionally shifting variance, while σ_d enters both.

- **Data:** Daily returns for the full NYSE panel; Stage 1 rolling-window LASSO estimates provide the sparsity-rate and belief-variance moments without additional estimation.
- **Key output:** A 4×4 parameter covariance table plus a J-test for overidentification; a moment-fit table comparing model-implied and empirical moments for each target.
- **Positive result:** Non-rejection of the overidentifying restrictions confirms that the model's nonlinear structure is not rejected by the data's moment profile.

**A3. Indirect inference via auxiliary GARCH model**

Estimate a GJR-GARCH(1,1) on actual returns and on simulated returns from the ALM, and define the binding function that maps (κ, λ, s, σ_d) to GARCH auxiliary parameters. Minimize the distance between actual and simulated GARCH coefficients. The appeal: GARCH coefficients are well-understood, and their comparison to the ALM binding function makes explicit what the model implies about volatility clustering and asymmetry.

- **Data:** Same daily NYSE return panel. Auxiliary estimation requires standard GARCH; simulation from the ALM requires draws of η and attention innovations X.
- **Key output:** Table comparing auxiliary GARCH parameters estimated on real vs. simulated data; figure showing the binding function α+β = g(κ) with a vertical line at the estimated κ.
- **Positive result:** Small binding-function distance at the estimated κ means the model replicates volatility persistence without explicitly parameterizing a volatility process.

**A4. Rolling-window profile likelihood to map the (λ, s) identification surface**

Map the profile likelihood surface over a grid of (λ, s) values for a representative sample of stocks stratified by market cap and sector. For each grid point, concentrate out κ and σ_d by NLS and record the concentrated log-likelihood. This reveals whether identification is sharp or flat (ridge along a trade-off between λ and s).

- **Data:** Daily returns and attention innovations for a stratified sample of ~100 stocks (20 per size quintile).
- **Key output:** Contour plot of concentrated profile likelihood in (λ, s) space for representative stocks; confidence regions at 90/95% overlaid; histogram across stocks of the likelihood maximum's location.
- **Positive result:** Sharp concentrated likelihood with a unique interior maximum confirms the model is well-identified from time-series data alone.

---

### Category B — Testing Qualitative Predictions

**B1. Portmanteau test for episodic versus persistent predictability**

Construct for each stock a time series of the rolling in-sample R² from the LASSO, then test whether its autocorrelation function decays exponentially (consistent with a persistent AR process) or exhibits rapid die-off (consistent with episodic bursts). Estimate an ARMA model for the R²_t series and test whether the MA component dominates the AR component. Rational time-varying risk premia would produce high AR persistence; LASSO learning produces MA-like bursts.

- **Data:** Rolling LASSO R² estimates already produced in Stage 1.
- **Key output:** Panel of ACF plots; cross-sectional histogram of estimated AR(1) coefficient of the R²_t series; fraction of stocks for which the AR coefficient is not significantly different from zero.
- **Positive result:** AR coefficients concentrated near zero with significant MA terms would confirm episodic structure and distinguish it from persistent-predictability alternatives.

**B2. Fat-tail test against GARCH null using out-of-sample density evaluation**

Compare out-of-sample density forecasts from (i) GARCH(1,1) and (ii) the structural ALM, using the probability integral transform (PIT) and the Berkowitz LR test. The ALM density is obtained by simulation at estimated parameters.

- **Data:** OOS period (e.g., 2018–2022) for stocks in the estimation sample; requires CRSP daily returns and attention innovations for the post-sample period.
- **Key output:** Table of Berkowitz LR statistics and p-values for both models; QQ-plot comparing PIT distributions from GARCH and ALM; focused comparison of left-tail coverage.
- **Positive result:** ALM significantly outperforms GARCH in the tails while both are similar in the center of the distribution, cleanly attributing tail risk to the learning mechanism.

**B3. Return reversal test conditional on LASSO selection events**

The model predicts that when LASSO selects a predictor (a selection event), this inflates the belief index f_t, raises current prices, and lowers expected future returns — a built-in reversal. Test directly: define selection events as days when active LASSO coefficients jump from zero to positive, and run an event study of cumulative abnormal returns in the [0, +20] day window. Under the model, post-selection CARs should be negative. Under rational information-revelation, a selection event should not systematically predict negative future returns.

- **Data:** Stage 1 LASSO coefficient paths for all 1,620 stocks; CRSP daily returns; FF5 factor data from Ken French's website.
- **Key output:** Event-study CAR plot with confidence bands for [−5, +20] days around selection events; CARs shown separately for positive vs. negative belief-index moves; cross-sectional regression of post-event CARs on κ_i.
- **Positive result:** Significantly negative CARs in the [+1, +20] window, larger for high-κ stocks, confirms the reversal prediction and directly tests the ALM pricing equation.

**B4. Variance ratio test at frequencies aligned with the rolling window s**

The model predicts return autocorrelation structure should exhibit a kink at lag s. Construct variance ratios VR(q) = Var(r_{t,q})/(q·Var(r_t)) for q = 1, 2, ..., 2s. Under the model, VR(q) should dip below 1 for q near s (reversal strongest at window frequency) and recover for q >> s. This prediction is specific to the LASSO-learning model and distinct from monotone-VR alternatives.

- **Data:** CRSP daily returns for the full panel; the estimated s from Stage 1.
- **Key output:** Figure showing average VR(q) profile across stocks with theoretical model prediction overlaid and a vertical line at the median estimated s; separate panels by tercile of estimated s.
- **Positive result:** Non-monotone VR profile with a trough near q = s, inconsistent with a monotone-VR alternative, establishes a novel model-specific prediction confirmed in the data.

---

### Category C — Exploiting the Belief Path

**C1. Belief-switch event studies linked to earnings announcements and macro releases**

Define a belief switch as a trading day on which the L1 norm of β_t changes by more than one cross-sectional standard deviation. Test whether belief switches cluster around scheduled macro announcements (FOMC, CPI, NFP) and earnings announcements. If LASSO selection responds to high-salience events, switches should be more frequent on and immediately after announcement days.

- **Data:** Reconstructed β_t paths; economic calendar; earnings announcement dates from Compustat; Bybee et al. attention innovations for topic identification.
- **Key output:** Two-panel figure: (i) average frequency of belief switches by event-relative day (±10 days around earnings, ±5 days around macro releases); (ii) average attention innovation for the "earnings" and "Fed" topics on belief-switch days vs. non-switch days.
- **Positive result:** Belief switches significantly more frequent around earnings and macro announcements, with corresponding news topics more active, establishes that LASSO learning captures genuine information updating.

**C2. Belief index as a signed volatility predictor in a horse race against VIX and realized variance**

Large |f_t| should also predict higher near-term realized variance because the ALM return variance is an increasing function of |f_t|. Test whether |f_t| forecasts next-day and next-week realized variance, controlling for lagged realized variance and VIX. Run panel Fama-MacBeth regressions with stock and time fixed effects.

- **Data:** Reconstructed f_t for all 1,620 stocks; realized variance from 5-minute TAQ data (or WRDS realized library); VIX from CBOE; FF5 controls.
- **Key output:** Fama-MacBeth coefficient table for |f_t| in volatility forecasting regressions at horizons 1, 5, and 20 trading days; R² improvement from adding |f_t|.
- **Positive result:** Statistically significant positive coefficient on |f_t| robust to VIX and lagged realized variance, establishing that model-implied belief state contains incremental information about risk.

**C3. Belief-path correlation across stocks as a novel measure of sentiment comovement**

Construct the cross-sectional dispersion of f_t at each date and ask whether it forecasts market-level returns or cross-sectional return dispersion. Compare to standard sentiment proxies — Baker-Wurgler, AAII sentiment survey, CBOE put-call ratio — in a horse race. Also construct the "aggregate belief activity index" (fraction of stocks with nonzero LASSO coefficients) and test whether it predicts market volatility.

- **Data:** Reconstructed f_{i,t} for all stocks; Baker-Wurgler sentiment index; AAII weekly survey data; CBOE put-call ratio; S&P 500 returns.
- **Key output:** Time-series figure of the aggregate belief activity index against NBER recession dates and major market events; predictive regression table comparing the sentiment horse race at horizons 1w, 4w, 12w.
- **Positive result:** The aggregate belief activity index is a significant predictor of market volatility and return dispersion while traditional sentiment proxies lose significance once it is included.

**C4. Belief-path persistence analysis: do β_t vectors revert to sparsity faster than OLS would predict?**

Construct a parallel OLS belief path (same rolling window, no penalty) and compare the survival function of episode duration. Under LASSO learning, survival function should have a thinner tail than OLS (shorter episodes). The ratio of LASSO to OLS episode duration is a clean empirical summary of the penalty's behavioral bite.

- **Data:** Stage 1 rolling LASSO estimates; parallel rolling OLS estimates on the same data.
- **Key output:** Kaplan-Meier survival curve plot comparing episode duration distributions for LASSO vs. OLS with a log-rank test statistic; cross-sectional regression of median LASSO episode duration on λ_i.
- **Positive result:** Significantly shorter episode duration under LASSO relative to OLS, with duration decreasing in λ, confirms that the penalty is the operative force truncating predictability episodes.

---

### Category D — Cross-Sectional Implications

**D1. The volatility-sparsity nexus: idiosyncratic volatility predicts LASSO selection frequency**

Formalize the cross-sectional model: regress per-stock selection frequency on idiosyncratic volatility (estimated from FF5 residuals), controlling for size, turnover, analyst coverage, institutional ownership, and earnings announcement frequency. Distinguish the statistical-artifact channel from the economic channel using an instrumental variable for volatility.

- **Data:** Stage 1 selection frequencies for 1,620 stocks; CRSP daily returns for FF5 residuals; Compustat; 13F filings.
- **Key output:** Cross-sectional OLS and IV table; scatter plots by idiosyncratic volatility deciles; Robinson-Parzen partial regression plot.
- **Positive result:** Positive, significant, and robust coefficient on idiosyncratic volatility in the IV specification supports the structural interpretation that genuinely uncertain stocks generate more learning activity.

**D2. Sector-level clustering of belief episodes**

Construct sector-level belief-activation dates and ask whether they cluster in calendar time, co-occur with relevant Bybee et al. topic spikes, and whether sector-level return correlations are higher on belief-activation days. Elevated within-sector return correlation on belief-activation days would suggest that LASSO learning amplifies systematic co-movement.

- **Data:** Stage 1 LASSO coefficient paths by stock; GICS sector classifications from Compustat; daily return correlations within sectors; Bybee et al. topic time series.
- **Key output:** Heat map of belief-activation frequency by sector and calendar quarter; bar chart of within-sector pairwise return correlation on belief-activation vs. non-activation days; regression of within-sector correlation on the sector belief-activity index.
- **Positive result:** Significantly higher within-sector return correlations on belief-activation days supports a correlated-learning channel not captured by standard factor models.

**D3. Analyst coverage as a moderator of κ: do information intermediaries dampen feedback?**

Regress estimated κ_i on analyst coverage (I/B/E/S), controlling for size and volatility. Additional test: use broker mergers and exits as plausibly exogenous shocks to coverage and run a difference-in-differences where treatment is a large reduction in analyst coverage.

- **Data:** Estimated κ_i from Stage 2; I/B/E/S analyst forecast files; broker exit events (Irani-Oesch or Kelly-Ljungqvist); CRSP for size controls.
- **Key output:** Cross-sectional regression table of κ_i on analyst coverage; DiD event-study plot of κ_i before and after broker exits; binned scatter of κ_i against log analyst coverage.
- **Positive result:** Significant increase in κ_i following analyst coverage reduction establishes that information intermediaries dampen the LASSO feedback mechanism.

**D4. The liquidity-lambda nexus: bid-ask spread predicts optimal LASSO penalty**

If the LASSO penalty λ is interpreted structurally as a threshold below which investors discard predictors, it should correlate with trading costs. Test cross-sectionally by regressing estimated λ_i on time-averaged effective bid-ask spreads, controlling for volatility and size. Natural experiment: SEC's 2016 tick-size pilot program (exogenous increase in minimum tick sizes for a random subset of small-cap stocks).

- **Data:** Estimated λ_i from Stage 1; TAQ daily effective spreads from WRDS; tick-size pilot program assignment list; CRSP characteristics.
- **Key output:** Cross-sectional regression table; DiD table using tick-size pilot as instrument; scatter plot of λ_i against average spread by spread decile.
- **Positive result:** Positive significant coefficient on bid-ask spread plus a positive DiD effect of the tick-size pilot on λ_i establishes the economic interpretation of λ as a trading-cost threshold.

---

### Category E — Natural Experiments

**E1. MiFID II and Reg FD as plausibly exogenous shifts in predictor dimensionality**

MiFID II (January 3, 2018) and Regulation FD (October 2000) both constituted abrupt changes to the information environment: MiFID II restricted analyst research distribution; Reg FD eliminated selective disclosure. Both plausibly altered the effective number of useful predictors available to investors without directly targeting return volatility. The LASSO-learning model makes sharp predictions: reducing effective predictor dimensionality should reduce selection frequency, reduce average magnitude of f_t, and may reduce κ_i.

For MiFID II, identify affected stocks as those with high pre-treatment analyst report density for EU-domiciled broker-dealers, constructing a continuous treatment intensity variable. Design a staggered DiD exploiting differential exposure to the information shock.

- **Data:** Bybee et al. attention innovations 2015–2020; I/B/E/S analyst reports with broker nationality (EU vs. non-EU); CRSP daily returns; estimated β_t paths reconstructed in pre- and post-MiFID periods; for Reg FD: SEC enforcement actions database and pre-2000 return data.
- **Key output:** Event study plot of LASSO selection frequency and estimated κ_i for high- vs. low-MiFID-exposure stocks; DiD regression table; mechanism table showing OOS R² of the surviving predictor set.
- **Positive result:** Significant reduction in selection frequency and κ_i for high-MiFID-exposure stocks after January 2018 establishes a clean causal link between the information environment and the intensity of the LASSO learning mechanism.

**E2. The COVID-19 information explosion as a positive shock to predictor dimensionality**

COVID-19 generated a massive, sudden expansion in economically relevant news topics with no pre-existing investor priors. The LASSO-learning model predicts that a sudden increase in plausibly relevant predictors, combined with high uncertainty about their coefficients, should generate a spike in selection events, elevated |f_t|, and higher κ-driven volatility. Test by estimating the model separately on pre-COVID (2016–2019) and COVID-onset (March–December 2020) subsamples.

- **Data:** Bybee et al. attention innovations extended through 2020; CRSP daily returns; GICS sector classifications.
- **Key output:** Time series of the aggregate belief-activity index from 2018–2020 with vertical lines at key COVID dates; cross-sectional regression of change in selection frequency on pre-COVID sector-level pandemic exposure; comparison of κ_i distributions pre- vs. post-COVID onset.
- **Positive result:** Documented spike in model-implied learning activity beginning March 2020, concentrated in pandemic-exposed sectors, preceding the peak in realized volatility, would establish that information expansion drives learning activity which amplifies volatility.

---

## 3. Critical Assessment

### A. NLS/SMM Estimation — Critiques

1. **Identification from near-zero belief index: effective sample is tiny** | Severity: **Fatal unless addressed** | Minimal fix: Document per stock the fraction of days with f_t ≠ 0 and distribution of |f_t| on active days. Report how κ estimates and SEs change when conditioning only on the active subsample. If identification collapses to a small number of active periods, the t-stat of 9.5 is misleading.

2. **Generated regressors: β_t is estimated, not observed** | Severity: **Fatal unless addressed** | Minimal fix: Bootstrap the entire two-stage procedure — re-draw daily returns with replacement, re-run Stage 1 LASSO, re-run Stage 2 NLS, collect bootstrap distribution of κ̂. Alternatively, derive analytical standard errors using the delta method applied to the two-stage system (but technically hard given LASSO's non-differentiability; bootstrap is more credible).

3. **Pre-testing / data snooping via hyperparameter selection** | Severity: **Fatal as currently specified** | Minimal fix: Path 1 (clean): split data into hyperparameter-selection period (2005–2009) and held-out evaluation period (2010–2017). Path 2 (partial): remove the κ t-statistic indicator from the optimization objective entirely. Either path requires acknowledging that any result from the current specification is uninformative about κ's true significance.

4. **Overfitting: same data for hyperparameter selection and NLS** | Severity: **Major** | Minimal fix: Report a genuine OOS test — estimate κ on 2010–2014, predict ALM-implied returns for 2015–2017, report OOS R² and Diebold-Mariano statistics against a zero-return benchmark. A subsample stability test (rolling κ estimates) would partially address this.

5. **The 75% non-significant stocks: unresolved interpretation** | Severity: **Major** | Minimal fix: Characterize the 75% systematically — are they smaller, less liquid, less covered? If significant stocks are systematically different in ways correlated with κ, that is evidence for the model's selection mechanism and should be the central cross-sectional test.

6. **Near-zero belief index: flat NLS objective surface** | Severity: **Major** | Minimal fix: Plot the profile likelihood as a function of κ for a representative sample of stocks, documenting whether the objective is genuinely curved or nearly flat.

---

### B. Recovering β_t Paths — Critiques

1. **p/s ratio ≈ 11.8: solution dominated by window-composition noise** | Severity: **Fatal unless addressed** | Minimal fix: Placebo test — randomly permute the dates of the return series and re-run the full two-stage procedure. If placebo β_t paths look qualitatively similar to actual paths, the recovered paths are pure noise.

2. **LASSO overfitting: λ = 0.0025 may be too small for p/s ≈ 11.8** | Severity: **Major** | Minimal fix: Compare λ = 0.0025 to the λ chosen by leave-one-out cross-validation on each window; report the fraction of windows where the calibrated λ is smaller than the CV-optimal λ.

3. **Look-ahead bias in hyperparameter selection** | Severity: **Major** | Minimal fix: Report sensitivity of Stage 1 R² and Stage 2 κ estimates to (s, λ, n) over a grid. If the optimized values sit at a sharp peak surrounded by poor performance, that is a red flag.

4. **OOS R² = −0.023: undermines agents' rationality claim** | Severity: **Major** | Minimal fix: Three responses in increasing credibility: (i) argue agents evaluate in-sample fit, not OOS; (ii) argue LASSO is a signal-extraction device, not a return-forecasting device; (iii) construct a heterogeneous-agent model where each agent privately believes their forecasts are informative even if aggregate is noisy.

---

### C. Cross-Sectional Tests — Critiques

1. **Tautology: volatility–selection test is a near-algebraic property of LASSO** | Severity: **Major** | Minimal fix: The genuine cross-sectional test should involve κ. The model predicts κ should be larger for stocks with more concentrated investor beliefs, more homogeneous investor base, or higher institutional ownership. Test whether κ̂ is systematically related to these firm-level characteristics.

2. **Measurement error in selection frequencies from noisy LASSO** | Severity: **Minor but must address** | Minimal fix: Report bootstrap standard errors for selection frequencies; use EIV correction in cross-sectional regression.

3. **Endogeneity: volatility correlates with everything** | Severity: **Major** | Minimal fix: Run Fama-MacBeth regression of selection frequency on volatility with controls for log market cap, Amihud illiquidity, institutional ownership (13F), analyst coverage (I/B/E/S), and average bid-ask spread.

4. **Multiple testing across 1,620 stocks without adjustment** | Severity: **Major** | Minimal fix: Apply Benjamini-Hochberg FDR correction to κ t-statistics. Report fraction significant after FDR correction at q=0.05 and q=0.10. Additionally, run a pooled test: pool all stock-level NLS residuals and test jointly whether the distribution of κ̂ is shifted above zero, using a block bootstrap preserving cross-sectional dependence.

---

### D. News-Based Predictors — Critiques

1. **Sample length: 2010–2017 is a single post-crisis regime with no recession** | Severity: **Major** | Minimal fix: Split 2010–2017 into two halves and test subsample stability of κ estimates. For a truly credible submission, extend to GDELT or RavenPack data covering a longer history.

2. **WSJ as the news source: is this what investors use?** | Severity: **Major** | Minimal fix: Explicitly scope the claim. Robustness checks using Bloomberg news (via WRDS) or Refinitiv would strengthen external validity. Alternatively, frame WSJ topics as a proxy for the broad information environment and be explicit about the assumption this requires.

3. **AR(1) residuals: measurement error in predictors** | Severity: **Minor but must address** | Minimal fix: Check robustness to alternative detrending methods (HP filter, bandpass, first differences). Report distribution of estimated ρ̂_j and flag topics where AR(1) fit is poor.

4. **LDA topic drift: non-stationarity in predictor set** | Severity: **Minor but must address** | Minimal fix: Test for structural breaks in the time-series properties of each topic (Bai-Perron test). Topics exhibiting significant breaks should be dropped or modeled with regime-change structure.

---

### E. Model Discrimination — Critiques

1. **GARCH generates the same moments: no discriminating test** | Severity: **Fatal unless addressed** | Minimal fix: Identify a moment that is a sharp prediction of LASSO-learning but not of GARCH. Best candidate: the cross-sectional correlation between κ̂ and the fraction of return variance attributable to news-topic predictors. GARCH cannot predict this cross-sectional relationship by construction.

2. **Adaptive-expectations / VAR-learning models generate similar patterns** | Severity: **Major** | Minimal fix: Estimate a competing second-stage model in which f_t is computed using OLS (unrestricted) rather than LASSO, and compare κ̂_OLS to κ̂_LASSO. If sparsity is essential, κ̂_LASSO should be larger, more significant, and produce a better-fitting ALM than κ̂_OLS.

3. **Mean reversion / microstructure as alternative for significant κ stocks** | Severity: **Major** | Minimal fix: Re-run Stage 2 NLS on returns net of Roll (1984) bid-ask spread estimator, or use midquote-to-midquote returns. Show significant-κ stocks are not concentrated in the lowest-liquidity quintile.

---

### F. Data and Feasibility — Critiques

1. **Statistical power: 1,889 days is borderline for within-stock time-series tests** | Severity: **Major** | Minimal fix: Report, for each stock, the number of days with f_t ≠ 0, average duration of active episodes, and distribution of |f_t| on active days. Conduct power calculations for NLS t-test as a function of active-period frequency.

2. **Bayesian optimization: irreproducible black box** | Severity: **Major** | Minimal fix: Report a grid search over a coarse but interpretable grid around (s, λ, n) = (320, 0.0025, 21), showing the objective surface. Release the optimization code and random seed. Report five independent runs with different seeds showing convergence.

3. **Are 180 WSJ topics genuinely high-dimensional?** | Severity: **Minor but must address** | Minimal fix: Report the effective rank of the predictor matrix X_t (e.g., number of PCs explaining 90% of variance) for a representative rolling window. If effective rank is far below 180, reframe the motivation accordingly.

---

### Overall Assessment

**Should be abandoned without fundamental redesign:**
- The hyperparameter optimization objective as currently specified is the single most damaging problem. Any result that flows from an optimization criterion that directly rewards κ significance cannot be presented as evidence for κ ≠ 0. This invalidates the central empirical finding entirely until the data-split design is implemented.
- The cross-sectional volatility test is a near-tautology and should not be presented as a discriminating test. Replace with a cross-sectional test of κ against firm-level characteristics predicted by the feedback mechanism.

**Can survive with major revision:**
- The NLS estimation of κ can survive if the two-stage bootstrap is implemented, the data-split design addresses the pre-testing problem, and active-period diagnostics confirm non-trivial identification variation.
- The β_t recovery can survive if the placebo permutation test shows actual paths are statistically distinguishable from noise paths.
- Model discrimination requires at minimum the LASSO-vs-OLS comparison and the microstructure control.

**Solid foundations needing only incremental fixes:**
- The news predictor design is reasonable for the sample period; AR(1) residual and LDA drift concerns are manageable with robustness tables.
- Sample size and feasibility concerns are surmountable with better diagnostics on active-period counts and clearer power calculations.

**Net verdict:** The paper has a genuinely interesting theoretical mechanism and a sensible empirical design, but as currently specified it cannot be published at JF. The hyperparameter selection criterion alone would cause any careful referee to recommend rejection. A revision implementing the data-split design, the two-stage bootstrap, and the LASSO-vs-OLS discrimination test would address the fatal concerns.

---

## 4. Prioritised Research Agenda (JF Standard)

### 1. Must-Have Results (Non-Negotiable for JF)

**1. Clean Hyperparameter Validation via Hold-Out Sample**
- Table: Hyperparameter Selection Robustness
- Shows: Re-estimate the full model using (s, λ, n) selected exclusively on a hold-out period (e.g., 2005–2009 or a randomly drawn 20% of stocks never seen during Bayesian optimisation). Report κ̂ distribution, mean t-stat, and fraction significant at 5% under these out-of-sample hyperparameters. Also report a grid search showing κ significance is not knife-edge sensitive to small perturbations around the chosen (s, λ, n).
- Sample: NYSE stocks; hyperparameter selection on pre-2010 data or held-out stock subsample; main estimation on 2010–2017.
- Spec detail: Bayesian optimisation frozen on hold-out; primary estimation sample never touched during tuning. Report robustness across ±20% perturbations of each hyperparameter independently. Include a 3-D heatmap of the objective over a grid of (s, λ) with n fixed at 21.
- Why non-negotiable: The current procedure selects hyperparameters by maximising κ significance on the exact same sample used to estimate κ. This is textbook pre-testing and is the single most predictable desk-rejection trigger at JF. No referee will accept "we optimise to find significant κ and then report that κ is significant" as a valid inference procedure.

**2. Generated-Regressors-Corrected Standard Errors**
- Table: Inference Correction for Two-Stage Estimation
- Shows: Re-derive the asymptotic variance of κ̂ accounting for the fact that f_t = x_t^T β_t is itself a first-stage LASSO estimate. Report corrected standard errors alongside naive NLS standard errors and the implied fraction of stocks with significant κ under both.
- Sample: Full 2010–2017 NYSE sample, 1,620 stocks.
- Spec detail: Block bootstrap with block length = rolling window s=320 days; re-run both LASSO stage and NLS stage within each bootstrap draw. Report 95% bootstrap confidence intervals for mean κ̂ and fraction significant.
- Why non-negotiable: Generated-regressors bias is a standard econometric critique that JF referees routinely raise. If the correction substantially widens standard errors, the paper's empirical claim collapses. If it changes little, this becomes a powerful robustness result. Either way, JF will require it.

**3. Genuine Out-of-Sample Test of the ALM**
- Table/Figure: OOS Predictability of the ALM
- Shows: Estimate the model through December 2014; use to generate ALM-implied return forecasts for January 2015–December 2017. Report OOS R² for both the PLM (Stage 1 LASSO alone) and the ALM. Show that the ALM OOS R² exceeds the PLM OOS R², i.e., that incorporating κ adds genuine forecast value beyond the raw LASSO signal.
- Sample: Estimation window 2010–2014; evaluation window 2015–2017. Extended to post-2017 data if available.
- Spec detail: Recursive and rolling-window forecasts both reported. Diebold–Mariano test of forecast equality between ALM and PLM. Clark–West adjustment for nested model comparison. Report by quintile of κ̂.
- Why non-negotiable: The paper currently offers no genuine OOS test of Stage 2. JF editors will ask: if the model cannot forecast returns out of sample, what does it actually explain? The OOS test is the direct empirical counterpart to the structural claim that κ > 0 causes transient predictability.

**4. Extended Sample Period with Structural Stability**
- Table: Full-Sample Estimation and Subsample Stability
- Shows: Replicate full two-stage estimation on (a) pre-crisis 2000–2007, (b) crisis 2008–2009, (c) original post-crisis 2010–2017, and (d) 2017–2022 if data allow. Report κ̂ distributions, fraction significant, mean t-stats by subsample. Include a formal Chow-style test of parameter stability.
- Sample: CRSP daily returns 2000–2022; Bybee et al. news topics or the largest available subset for the full period.
- Spec detail: Re-optimise hyperparameters separately for each subsample using nested hold-out, or apply the frozen hyperparameters from Result 1.
- Why non-negotiable: A seven-year post-crisis window ending in 2017 is the single most common sample-selection critique in asset pricing at JF. Structural stability across longer samples is required for the paper to claim general relevance.

---

### 2. High-Value Additions

**1. Cross-Sectional Asset Pricing Test Using κ̂ as a Characteristic**
- Shows: Sort stocks into quintiles by estimated κ̂; construct long-short portfolio (high-κ minus low-κ); report average returns, CAPM alpha, Fama-French 3- and 5-factor alphas, Stambaugh-Yuan mispricing-factor alpha; Fama-MacBeth regressions of future returns on κ̂ controlling for standard characteristics.
- Why it matters: JF asset-pricing papers are expected to connect to the cross-sectional anomaly literature. If high-κ stocks earn abnormal returns, κ becomes a tradeable signal with economic significance.
- Data feasibility: CRSP/Compustat characteristics; standard factor data from Ken French's website. Feasible immediately given existing κ̂ estimates.

**2. Attention and Investor Heterogeneity: Retail vs. Institutional**
- Shows: Test whether the LASSO feedback (κ̂) is stronger for stocks with higher retail investor attention, using Robinhood popularity data, Google Trends, or SEC EDGAR filing traffic. Regress κ̂ on retail attention controlling for size and volatility.
- Why it matters: Provides micro-foundation evidence for who the LASSO learners are, addressing the implicit behavioural assumption. JF increasingly demands that belief-based models be grounded in evidence about specific investor types.
- Data feasibility: Robinhood popularity data publicly available through mid-2020; Google Trends and EDGAR traffic freely available. Moderately demanding but feasible.

**3. News-Topic Sparsity and the Timing of Predictability Pockets**
- Shows: Map active LASSO coefficient dates to identifiable news events or shifts in Bybee et al. topics. Event-study windows around major earnings, Fed policy, and sector-specific news shocks. The prediction is that predictability pockets should cluster around high-attention episodes for the relevant topic.
- Why it matters: Directly connects the timing of active coefficients to identifiable events, making the mechanism legible to a general finance audience. JF editors respond strongly to results that can be explained in two sentences in a seminar.
- Data feasibility: Bybee et al. data already in the paper. Event dates from the Fed, I/B/E/S, and FRED. Feasible with moderate effort.

**4. Simulation-Based Model Validation (Quantitative Theory Discipline)**
- Shows: Simulate the model at estimated parameters and produce simulated moments for: fraction of active LASSO coefficients over time, autocorrelation structure of returns, distribution of in-sample vs. OOS R², cross-sectional dispersion of κ̂. Compare simulated moments to empirical counterparts using an overidentification test or informal moment-matching table.
- Why it matters: JF expects structural papers to demonstrate that the estimated model can reproduce the key empirical moments that motivated the model. This is the quantitative theory discipline that distinguishes a structural contribution from a reduced-form exercise.
- Data feasibility: Requires only the already-estimated model parameters and simulation code. No additional data needed.

**5. Comparison Against Alternative Learning Rules**
- Shows: Estimate Stage 2 (the ALM feedback mechanism) substituting alternative learning rules for LASSO: Ridge, Elastic Net, OLS with BIC model selection, simple momentum rule. Test whether κ > 0 and ALM OOS R² > 0 only under LASSO, or whether it arises under any regularised learning rule.
- Why it matters: The paper's title and core claim rest specifically on LASSO's sparsity-inducing property. If results are equally strong under Ridge (which does not produce sparse solutions), sparsity is not doing the theoretical work claimed. Conversely, if LASSO uniquely generates the feedback, this is a powerful identification result.
- Data feasibility: Ridge, Elastic Net, and OLS are all available in standard packages. No additional data required. Extremely high-value given the theoretical stakes.

---

### 3. Approaches to Abandon

1. **Current Bayesian Optimisation Procedure as a Main Result** | The procedure must be replaced entirely by the hold-out design. Describe briefly in the appendix as motivation for the held-out hyperparameter values, nothing more.

2. **Volatility as Predictor of Selection Frequency (Without Alternative-Explanation Controls)** | Almost certainly mechanical given the model structure. Higher-volatility assets have larger raw OLS coefficients and survive LASSO thresholding more often — this is a property of LASSO applied to heteroskedastic data. Drop from main text; relegate to appendix.

3. **In-Sample PLM R² = 0.106 as a Primary Performance Metric** | Reporting in-sample R² = 0.106 alongside OOS R² = −0.023 as co-equal results is self-defeating. Either provide a theoretical justification for why agents would continue using a rule with negative OOS R², or replace in-sample R² as motivation with the ALM's positive OOS R² (Must-Have 3). Demote to a diagnostic in an appendix.

---

### 4. Recommended Sequencing

1. **Step 1 (Foundational — must complete before anything else):** Implement the hold-out hyperparameter selection (Must-Have 1). This is the prerequisite for all subsequent inference. Until the tuning-sample / estimation-sample separation is clean, no t-statistic in the paper is interpretable.

2. **Step 2 (Concurrent with Step 1, no dependency):** Extend the data sample to 2000–2022 (Must-Have 4). Confirm availability of Bybee et al. news-topic series outside 2010–2017.

3. **Step 3 (Depends on Steps 1 and 2):** Re-run the full two-stage estimation under clean hyperparameter regime on the extended sample. This produces the paper's new canonical estimates.

4. **Step 4 (Depends on Step 3):** Implement the generated-regressors correction (Must-Have 2). Run the block bootstrap. Computationally intensive; parallelise across stocks.

5. **Step 5 (Depends on Step 3):** Construct the genuine OOS test (Must-Have 3). Split the extended sample at 2014. Run Clark–West and Diebold–Mariano tests.

6. **Step 6 (Depends on Step 3, can overlap with Steps 4–5):** Run the alternative learning rules comparison (High-Value 5). Same estimation infrastructure as Step 3; no additional data requirements.

7. **Step 7 (Depends on Step 3):** Construct κ̂-sorted portfolios and run cross-sectional asset pricing tests (High-Value 1). Merge κ̂ estimates with CRSP/Compustat characteristics.

8. **Step 8 (Depends on Step 3, lower priority):** Run simulation-based model validation (High-Value 4) and attention/investor heterogeneity analysis (High-Value 2).

9. **Step 9 (Depends on Step 3):** Construct news-event timing analysis (High-Value 3). Match LASSO active-coefficient dates to external event calendars.

---

### 5. Key Identification Challenge

**The Challenge:**

The paper's central empirical claim — that κ > 0 measures a structural feedback from LASSO-based beliefs to prices — is identified by the co-movement between the LASSO-implied signal f_t = x_t^T β_t and subsequent returns. The identification problem is that the hyperparameters governing β_t (window length s, penalty λ, lag count n) were chosen specifically to maximise the significance of this co-movement. The parameter κ is not identified independently of the tuning procedure; it is, in the current specification, a direct product of the tuning objective. The paper cannot distinguish between (a) κ > 0 because there is genuine belief-price feedback and (b) κ > 0 because the optimisation found the parameterisation of β_t that maximises the correlation between f_t and r_{t+1} on the estimation sample. These are observationally equivalent under the current design.

A secondary identification challenge: f_t is correlated with lagged returns by construction (it is a function of return-predictive coefficients estimated on lagged returns), so any apparent "feedback" from f_t to r_{t+1} may simply reflect return autocorrelation or momentum.

**The Most Credible Available Solution:**

A two-pronged design:
1. **Hold-out hyperparameter selection** (Must-Have 1) so that the parameters governing β_t are fixed before the estimation sample is touched. This breaks the circularity.
2. **Permutation placebo test**: re-run Stage 2 with f_t constructed from randomly permuted β_t (LASSO coefficients drawn from the empirical distribution but assigned to random dates, destroying any time-series relationship). If κ > 0 survives under the true f_t but collapses under permuted f_t, this provides direct evidence that the signal content of β_t drives the feedback.
3. **Peer-group IV**: instrument f_t with the average LASSO signal across stocks in the same industry-size cell. Peer-group LASSO signals share the common news-topic component of β_t but are orthogonal to the idiosyncratic return autocorrelation of stock i.

The permutation placebo is the single most important of these three: it is cheap to implement, immediately intuitive, and directly falsifies the alternative hypothesis that any smooth function of lagged returns would produce κ > 0 under the optimisation scheme.

---

### 6. Empirical Strategy Memo (to the Authors)

**To:** Tommaso Di Francesco, Stefanie Huber, Jonathan Seim
**From:** JF-Standard Synthesiser
**Re:** Empirical revision priorities for JF submission

---

**Must show:**

The paper currently has a pre-testing problem that will trigger desk rejection at both JF and JFE. The Bayesian optimisation selects hyperparameters (s, λ, n) by maximising the significance of κ on the estimation sample, and then reports κ as a finding. This is circular inference. You must separate the tuning sample from the inference sample completely. The concrete implementation: use data from 2005–2009 (or a randomly held-out 20% of stocks) to select hyperparameters, then freeze them, then estimate the model on 2010–2017. Every table must report results under these frozen hyperparameters. This is not optional.

Second, correct your standard errors for the generated-regressors problem. Run a block bootstrap (block length = s = 320 days) that re-runs both stages inside each draw. If the corrected confidence intervals still yield a mean κ̂ t-stat above 2 and a fraction-significant above 15%, you have a strong result.

Third, produce a genuine out-of-sample test of the ALM. Estimate through 2014, forecast 2015–2017, report OOS R² for the PLM and the ALM separately. The key claim is that κ adds forecast value over the LASSO signal alone.

Fourth, extend your sample. 2010–2017 is a single regime. Replicate core estimates on 2000–2007, 2008–2009, and 2017–2022.

---

**Should show:**

Sort stocks by κ̂ and test whether high-κ portfolios earn abnormal returns controlling for standard factors. This is cheap to produce once you have κ̂ estimates and is the difference between "interesting theory" and "interesting theory with asset pricing consequences."

Compare results to alternative learning rules — Ridge, Elastic Net, momentum. If LASSO uniquely generates κ > 0 and positive ALM OOS R², you have identified the sparsity mechanism. This takes one additional weekend of computation.

Run simulation-based moment matching: simulate the model at estimated parameters and show a table of simulated vs. empirical moments.

Map the timing of active LASSO coefficients to identifiable news events. Show two or three event studies where predictability pockets open and close around high-salience news episodes. This makes the mechanism legible to a general finance audience.

---

**Drop:**

Remove the current Bayesian optimisation procedure from the main text entirely. Describe it briefly in the appendix as motivation for the held-out hyperparameter values, nothing more.

Drop the volatility-predicts-selection-frequency result from the main text. It is almost certainly mechanical — higher-volatility stocks have larger raw OLS coefficients and survive LASSO thresholding more often. Without a credible test ruling out this null, the result invites more criticism than it contributes.

Demote the in-sample PLM R² = 0.106 from the main results. The juxtaposition with OOS R² = −0.023 creates a narrative hole you have not filled. Either fill it with a formal finite-sample decision-theoretic result, or let the in-sample figure appear only as a diagnostic.

---

**The referee's hardest push — and how to pre-empt it:**

The hardest push will be: *"Your κ estimates are a direct product of your optimisation objective. You tuned the model to find κ > 0, and then you report κ > 0. There is no independent variation identifying the feedback mechanism."*

Pre-empt it as follows. In the introduction, state explicitly that you are aware of this concern and address it in three ways:
1. The hold-out hyperparameter design ensures that the parameters governing the LASSO signal are fixed before the estimation sample is touched.
2. A permutation placebo test shows that κ collapses to zero when the LASSO signal is randomly reassigned across dates, confirming that the time-series content of β_t — not its statistical properties — drives the result.
3. A peer-group IV specification instruments f_t with the average LASSO signal from industry-size peers, isolating the common news-topic component from idiosyncratic return autocorrelation.

If all three point to κ > 0 with similar magnitudes, no reasonable referee can sustain the circularity critique. **Run the permutation placebo first.**

---

## Immediate Next Steps

1. **Implement the hold-out hyperparameter selection** (critical — do this before anything else): use 2005–2009 data (or a held-out 20% of stocks) to run the Bayesian optimization; freeze (s, λ, n); re-estimate both stages on 2010–2017 with frozen hyperparameters.
2. **Run the permutation placebo test**: randomly permute return dates, re-run both stages, verify κ collapses to zero; compare sparsity/episode-structure in placebo vs. actual β_t paths.
3. **Implement the two-stage block bootstrap** for corrected standard errors: block length = s = 320 days; parallelize across stocks; report naive vs. corrected standard errors in a new table.
4. **Construct the genuine OOS test of the ALM**: estimate on 2010–2014, forecast 2015–2017, report OOS R² for PLM and ALM separately with Clark-West and Diebold-Mariano statistics.
5. **Extend the data to a longer sample** (2000–2022 if Bybee et al. or an equivalent news dataset covers it): confirm data availability; re-run the model on pre-crisis and post-2017 subsamples to test structural stability of κ.
6. **Run the LASSO-vs-OLS discrimination test**: re-estimate Stage 2 using f_t from rolling OLS (no penalty) and compare κ̂_OLS to κ̂_LASSO in terms of magnitude, significance, and ALM OOS R².
7. **Produce a hyperparameter sensitivity surface**: grid search over (λ, s) ± 20% of the calibrated values; display as a heatmap of fraction-significant-κ; confirm results are not knife-edge.
8. **Characterize the 75% non-significant stocks**: run a cross-sectional regression of κ significance on firm characteristics (size, volatility, analyst coverage, illiquidity); the significant vs. non-significant comparison should be the central cross-sectional test of the feedback mechanism.
9. **Construct κ̂-sorted portfolio returns** and test alphas against FF5 + momentum: merge κ̂ estimates with CRSP/Compustat; construct monthly rebalanced long-short quintile portfolio; report FF5 alpha and t-statistic.
10. **Replace the in-sample R² narrative**: either provide a formal decision-theoretic justification for why agents would rely on a rule with negative OOS R² (citing a specific finite-sample result), or drop in-sample R² from the main text and replace with the ALM OOS R² result as the primary evidence for the model's empirical validity.
