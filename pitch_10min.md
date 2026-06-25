# 10-Minute Colleague Pitch: Sparse Learning and Endogenous Pockets of Predictability

---

**"So, the paper starts from a very simple observation about what modern investors actually do."**

We know that over the last couple of decades, the investment industry has gone heavily into alternative data — news feeds, satellite imagery, textual data — and they process it with machine learning. There are surveys showing that demand for this stuff keeps growing, and that LASSO-type methods are widely used in practice. So the question we ask is: what happens to asset prices when this is the equilibrium behavior? What are the aggregate consequences of everyone running penalized regressions to find predictors?

The surprising answer we get is that the act of *searching* for predictability can itself *generate* predictability — but only transiently and sparsely. That's the core result.

---

**"Let me walk you through the model."**

We use a pretty standard Lucas exchange economy — one risky asset, CRRA preferences, dividend and consumption following a lognormal process. Under rational expectations, the price-dividend ratio is constant and returns are i.i.d. — there's nothing to predict.

Now we follow Adam, Marcet, and Nicolini (2016) and drop the assumption that agents know the return process. They know dividends and consumption perfectly, but they're agnostic about returns. Instead, they *suspect* returns might be predictable from a large set of observable signals — say, news attention series — and they try to learn this relationship from data.

The key departure from the standard learning literature is the learning rule. In standard adaptive learning, agents run OLS on a fixed low-dimensional model and recursively update. Here, the number of candidate predictors is large — potentially larger than the available observations. So OLS is infeasible. Instead, agents run a LASSO: they estimate OLS coefficients and then soft-threshold them, setting anything below a penalty level to zero. This is analytically equivalent to the LASSO in the univariate orthonormal case, and it approximates it well when predictors are approximately uncorrelated, which is the empirically relevant case.

---

**"Now, the mechanism."**

Because agents use finite memory — constant-gain learning, or equivalently a rolling window — their OLS estimates fluctuate randomly around zero. Even when returns are truly unpredictable, the finite-sample correlation between a predictor and returns is noisy. Occasionally, purely by chance, this correlation is large enough that the estimated coefficient survives the soft-threshold.

When that happens, agents start trading on the perceived signal. Their demand shifts prices, and through the asset-pricing equation — where price equals discounted expected future payoffs — this feeds back into realized returns. The perceived predictability becomes *real*. The ALM, which is nonlinear, now generates a genuine correlation between the predictor and returns.

But here's why it's transient. Agents have bounded memory. As new observations arrive and old ones drop out of the rolling window, the noisy realizations that originally triggered the selection get diluted. The estimated coefficient drifts back toward zero, the LASSO zeros it out again, agents stop trading on the signal, and prices revert. The pocket of predictability disappears.

So the life cycle is: **noise spike → selection → belief-price feedback → self-fulfilling predictability → memory fade → reversion**. And this repeats, generating episodic, sparse predictability as an endogenous equilibrium outcome.

---

**"Formally, we show that the economy converges to what we call a Cross-Moment Consistent Equilibrium."**

This is weaker than rational expectations. We don't require the full distributions to match — just that the population OLS projection of actual returns onto the predictor is consistent with what agents believe that projection to be. We prove this equilibrium exists, is unique, and is locally stable under learning. The equilibrium value of the belief coefficient is zero, consistent with returns being unpredictable on average. But the *stochastic fluctuations* around this equilibrium are what generate the interesting dynamics.

We also characterize analytically how the frequency of selection events depends on fundamentals. The probability that the LASSO picks up a predictor is increasing in return volatility and decreasing in the penalty level and predictor variance. So higher-volatility stocks should see more frequent selection — that's a testable prediction.

---

**"Now for the empirics."**

We take daily returns for about 1,600 NYSE-listed stocks from 2010 to 2017. As predictors, we use the 180 daily news attention series from Bybee, Kelly, Manela, and Xiu — these are constructed from the full text of the Wall Street Journal using LDA. We take innovations rather than levels to remove persistence. So we have a 180-dimensional predictor set with no strong prior on which topics matter for which stocks.

The estimation mirrors the model. For each stock, we run a rolling LASSO to get the time-varying belief sequence. Then we plug those beliefs into the nonlinear ALM and estimate the feedback parameter κ by NLS.

The first-stage results are striking. In-sample R² is small but positive — agents find some in-sample fit. Out-of-sample R² is near zero — consistent with the CMCE prediction that the true β is zero. Selection is sparse: the median stock selects fewer than 1% of predictors on average.

Of the stocks with non-zero selection, about 9.5% have a statistically significant κ at the 5% level. The median estimated κ is around 0.28, which is well within the theoretically admissible range and in line with our oracle simulations.

The cross-sectional pattern also matches the theory. More volatile stocks select more predictors on average — the relationship is clearly positive in the data.

---

**"We also document interesting time-series patterns in selection."**

Selection intensity spikes during macro stress events — the European debt crisis, the taper tantrum, Brexit, the 2016 election. We run a regression of the aggregate selection rate on return volatility and predictor volatility, and we get the signs the model predicts: more return volatility means more selection, more predictor volatility means less. But the R² is only 12%, which suggests there's more going on.

We then run a GARCH oracle exercise — we simulate returns with GARCH noise calibrated to each stock's market beta and idiosyncratic volatility, re-run the same rolling LASSO, and compare the simulated selection rate to the actual one. The GARCH simulation reduces the variance gap between actual and simulated selection from 19x down to 1.3x, and the Spearman correlation between the two time series jumps from near zero to 0.38. So GARCH volatility clustering — representing time-varying fundamental uncertainty — explains most of the aggregate episodic variation in perceived predictability.

---

**"So what's the takeaway?"**

We provide a theoretical mechanism for a well-documented empirical fact: return predictability appears in short-lived, shifting, sparse episodes rather than as a stable forecasting relationship. The mechanism doesn't require irrationality in the traditional sense — agents are internally rational given their beliefs, and their learning rule is a sensible response to a genuine high-dimensional problem. The pathology comes from finite memory and the self-referential nature of belief-price feedback.

The paper also speaks to a broader question: as more investors deploy ML techniques on the same data, does that make markets more efficient? Our model says not necessarily — the convergence to CMCE leaves room for transient predictability, and the frequency of these episodes is endogenous to the information environment and return volatility. More data-driven investors don't eliminate the pockets; they might just rotate through them faster.

---

*That's the 10-minute version. Happy to go deeper on any part.*
