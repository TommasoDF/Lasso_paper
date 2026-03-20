# Paper Draft Notes — Market Evolution: Heterogeneous ML Traders, Genetic Algorithms, and Stylized Facts
**Date:** 2026-03-17
**Draft file:** `paper_draft_2026-03-17.tex`

---

## 1. Key Design Choices Adopted

1. **Genetic algorithm as ecological selection mechanism.** Rather than fixed strategy switching (Brock-Hommes logit) or hand-coded rules (SF-ASM classifier systems), agents are architecturally diverse ML models (LSTM, GBM, MLP, LinReg, RF, MA-rule) and the GA operates on their hyperparameter vectors via tournament selection, uniform crossover (within-architecture), and Gaussian mutation. This is the central novelty.

2. **Linear market-maker price formation.** Price is set as `P_t = P_{t-1} exp(λ Σ w_{i,t} a_{i,t} + ε_t)` with proportional wealth-weighted impact. This is a simplification (real impact is concave), explicitly flagged as a vulnerability. Two alternative microstructure specs (CDA, call auction) are deferred to robustness.

3. **Simulated Method of Moments (SMM) for calibration.** The free parameters (λ, σ_ε, transaction cost c, mutation std η) are identified by 7 moment conditions (4 unconditional return moments, Ljung-Box on returns, Ljung-Box on r², Hill tail index, Hurst of |r|, return AC at lags ±1). System has 4 over-identifying restrictions → χ²₄ J-test.

4. **Within-architecture crossover as baseline.** Crossover is restricted to same-architecture parent pairs by default, preserving architectural identity. Cross-architecture ensemble children allowed with probability p_α = 0.05 (robustness variant). This choice makes architectural competition tractable to interpret.

5. **Claim discipline: simulation-based, not causal.** All results are framed as "simulation evidence consistent with…" All tables use placeholders for actual simulation numbers — no fabricated values.

6. **Primary empirical benchmark: SPY daily returns 2000–2023.** Secondary benchmarks: Russell 2000 (IWM), NASDAQ-100 (QQQ), Bitcoin (BTC-USD). Sub-period splits: pre-2008, 2008–2014, 2015–2023.

7. **Four placebo experiments.** (i) Frozen GA (evolution disabled), (ii) Zero-Intelligence (ZI) traders, (iii) Homogeneous-architecture population, (iv) Reversed fitness function. All four must jointly falsify naive alternative explanations before results are credible.

---

## 2. Top 5 Unresolved Risks Before Circulation

1. **SF-ASM differentiation.** Arthur et al. (1997) used a GA on classifier-system strings in a market simulation. Referees at top finance journals will notice. The paper must explicitly benchmark against the SF-ASM and show that (a) modern ML architectures produce materially different dynamics, and (b) the question of *which architecture class survives* is genuinely new. Consider adding a direct SF-ASM replication as a baseline comparison.

2. **Hashimoto et al. (2025) overlap.** arXiv:2511.05207 uses multi-agent RL with heterogeneous preferences to produce stylized facts. If this paper has a pre-print advantage and covers architectural diversity or explicit GA selection, overlap becomes serious. **Action: read the full paper before finalizing the contribution statement.**

3. **SMM non-identification.** With 7 moment conditions and potentially flat regions of the loss surface, θ_hat may not be uniquely identified. Must run 20+ random restarts of the optimizer and report parameter variance. If the surface is flat, reduce to 4 moments and 3 free params.

4. **Computational feasibility of LSTM agents at scale.** 200 agents × 100 refit periods × 100 generations × 50 MC runs requires ~100M gradient steps. Need GPU cluster or must fall back to pre-trained fixed LSTM feature extractors (shared trunk, agent-specific heads). This fallback must be decided before first simulation run.

5. **Genetic drift vs. selection confound.** At N=200, genetic drift may dominate selection signal for rare architectures, making survival rate comparisons unreliable. Must run N ∈ {50, 200, 1000} and show that qualitative survival ordering is stable across population sizes before making architecture-dominance claims.

---

## 3. Immediate Next Empirical Steps

1. **Set up simulation codebase.** Initialize repo structure as per Agent 5's module layout (`src/agents/`, `src/market/`, `src/ga/`, `src/analysis/`). Implement market engine and fitness function first; verify price series is stationary and wealth > 0.

2. **Implement and test ZI-trader placebo.** Build the zero-intelligence baseline before any ML agents. This is the fastest sanity check: if the ZI market already reproduces stylized facts, the evolutionary mechanism is not load-bearing and the paper's contribution collapses.

3. **Download and cache empirical data.** Pull SPY daily OHLCV (2000–2023) via yfinance; compute and save the 7 target moment conditions (`m_data`). These become the fixed SMM targets for all calibration runs.

4. **Profile compute cost with 5 agents × 100 periods × 5 generations.** Measure wall-clock time per generation for each architecture type. Use this to set realistic N, T, G for the full simulation, and decide whether LSTM full-refit or pre-trained trunk is necessary.

5. **Read Hashimoto et al. (2025) in full.** Determine overlap with current design and adjust contribution statement accordingly.

6. **Register target moments pre-calibration.** Write the 7 target moments and the SMM parameter bounds to `config/base_config.yaml` and commit *before* any simulation run to avoid p-hacking on moment selection.

---

## 4. Missing Inputs Required from User / Author

1. **Author list and affiliations.** The draft has `[Author names placeholder]` — fill in before any circulation.

2. **Computation resources.** Confirm available compute (local GPU, HPC cluster, cloud) to determine whether full LSTM refit per generation is feasible or whether the pre-trained trunk fallback must be used.

3. **Architecture scope decision.** Confirm whether reinforcement-learning (RL/DQN) agents should be included in the architecture set alongside LSTM, GBM, MLP, LinReg, RF, MA-rule. Including RL adds conceptual richness but increases implementation complexity substantially.

4. **Target journal.** The draft is written at Journal of Finance / Review of Financial Studies register. If the target is a more computational venue (Journal of Economic Dynamics and Control, Quantitative Finance, Journal of Artificial Societies and Social Simulation), tone and emphasis should shift toward mechanism description and less toward econometric validation.

5. **Companion paper coordination.** Confirm the relationship between this paper and "Sparse Learning and Endogenous Pockets of Predictability" (Di Francesco, Huber, Seim). The draft cites it as a companion; if both papers are under submission simultaneously, the positioning and cross-citation language should be coordinated.

6. **Full citation details for:** Hashimoto et al. (2025), Vie & Farmer Evology arXiv identifiers, Wheeler & Varner (2023), Vyetrenko et al. (2019) published version. These are flagged as `[?]` or placeholder in the assumption log.
