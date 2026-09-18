# Experiment Log — VaR, Expected Shortfall, and Backtesting

**Date:** 2026-09-17
**Project:** stockexp — event-conditioned market-risk platform

Builds on the validated GJR-GARCH-t model from `experiments-2026-09-15.md`. This log covers the question: **if the model says there's a 5% chance of a loss bigger than VaR on any given day, does that actually happen about 5% of the time — and, separately, do breaches land independently or cluster in time?**

---

## Experiment 4 — Historical VaR vs. GJR-GARCH-t Parametric VaR, Backtested

**Purpose:** compute VaR and Expected Shortfall two ways, and backtest both properly rather than trusting either number on its face.

**Methods:**
- **Historical VaR/ES** — empirical quantile (and tail mean) of a trailing 250-day rolling window of daily returns. No distributional assumption.
- **GJR-GARCH-t parametric VaR/ES** — daily one-step-ahead conditional volatility from the validated model (parameters refit on an expanding window every ~21 trading days; the variance recursion updated daily between refits via `arch`'s `.fix()`, so the forecast actually moves day to day rather than holding a stale block-level value), combined with the model's own fitted Student-t tail quantile — not a generic normal assumption.

**Backtests, at α=0.05 (95% VaR):**
- **Kupiec proportion-of-failures test** — is the exception *count* right?
- **Christoffersen independence test** — do exceptions land independently in time, or cluster?
- **Conditional coverage test** — both at once (χ², df=2)

**Validation before use:** the closed-form Student-t Expected Shortfall formula was checked against numerical integration (matched to 4+ decimal places). The Kupiec and Christoffersen likelihood-ratio statistics were both checked against known well-calibrated and miscalibrated reference cases before being trusted on real data — see chat history for the verification runs.

**Bug caught and fixed during this experiment:** the initial Kupiec implementation exponentiated the raw likelihoods before dividing, which underflows to exactly `0.0` once `n` is in the thousands (`0.95**8000` is far below float64's smallest representable positive number), silently producing `nan` for the pooled ~8,600-observation test. Per-symbol counts (n≈375) never triggered it, which is why validation at n=250 didn't catch it. Fixed by computing the likelihood ratio in log space throughout. Confirmed the fix reproduces identical results on the small-n cases already validated, and recovers real numbers on the pooled case (previously `nan`).

### Results

**Forecasted vs. realized volatility (23 symbols, test period):** 0 of 23 symbols over-forecasting volatility by >15%; most ratios sit between 0.86 and 1.12. Average forecast level is not systematically biased.

**Historical VaR (95%):** 438 exceptions / 8,625 observations (5.08%, expected 5.0%). Pooled Kupiec p=0.739 — **not rejected**. 23/23 symbols individually pass the count test. 1/23 symbols (AAPL) shows significant breach clustering (independence p=0.011) despite passing the count test — a real, isolated timing weakness in an otherwise well-calibrated method.

**GJR-GARCH-t parametric VaR (95%):** 322 exceptions / 8,625 observations (3.73%, expected 5.0%). Pooled Kupiec p=1.7×10⁻⁸ — **rejected**. 7/23 symbols individually rejected on count (BA, BAC, COIN, DIS, JPM, NVDA, TSLA), all in the same direction: *fewer* exceptions than expected, i.e. the model is too conservative, not too risky. Of those 7, only BA also shows significant clustering (independence p=0.013); the other 6 have high independence p-values (0.20–0.71) — their breaches are rare but still land independently in time.

**ES/VaR ratio:** historical 1.51x, GJR-GARCH-t 1.46x — broadly similar tail-severity-beyond-VaR between methods.

### Findings

**Historical VaR is cleanly, textbook well-calibrated.** No further work needed there for this universe/period.

**GJR-GARCH-t parametric VaR is measurably too conservative, and three candidate mechanisms were tested and mostly ruled out with direct evidence rather than assumed:**

1. *Average volatility level mismatch* — **refuted.** Forecast/realized ratios for the 7 rejected symbols sit at 0.97–1.12, essentially accurate.
2. *Fat-tail quantile choice* — **refuted, and points the wrong direction.** At α=0.05, a low-ν Student-t quantile is *narrower* than the normal quantile (checked numerically: ν=4.64 gives -1.5464 vs normal's -1.6449), which should produce *more* breaches, not fewer — the opposite of what's observed.
3. *Breach timing/clustering* — **refuted for 6 of the 7 rejected symbols.** Only BA shows genuine clustering; the rest have rare but independently-timed breaches.

With the three obvious explanations eliminated or pointing backwards, the remaining candidate is a subtler one, not yet directly tested: **the fitted tail shape (ν) from the training window may not transfer to the test period's actual realized tail behavior**, independent of volatility level — kurtosis and variance can shift on different schedules. This would require comparing test-period-only empirical quantiles against the fitted ν directly to confirm, which was not done in this pass.

### Limitations

- The remaining miscalibration mechanism (out-of-sample tail-shape/kurtosis mismatch) is a plausible leading hypothesis, not a confirmed finding — flagged honestly as open rather than asserted.
- Christoffersen tests are run per-symbol at n≈375; like the earlier Diebold-Mariano per-symbol tests, this is likely underpowered for subtle clustering effects, the same caveat noted in `experiments-2026-09-15.md`'s Experiment 3c applies here.
- Conditional-coverage and independence tests are not poolable across symbols the way Kupiec's count test is (independence is a property of one ordered series) — no aggregate independence statistic is reported, only the per-symbol count.
- AAPL's historical-VaR clustering finding (the one genuine weakness in an otherwise clean method) was noted but not investigated further.

---

## Recommended Next Step

Two reasonable directions from here: extend to **portfolio-level risk** (correlation matrix across the 23-symbol universe, portfolio VaR vs. sum of individual VaRs, diversification benefit) as the natural continuation of the risk-modeling arc, or treat the modeling arc as complete at this point (daily panel → volatility forecasting → VaR/ES/backtesting, each stage validated with real statistical tests rather than face-value metrics) and shift focus to resume/portfolio presentation of the finished work.

---

## Experiment 5 — Portfolio-Level Risk: Component VaR and Risk Parity

**Purpose:** a single portfolio VaR number ("VaR is 1.98%") is not actionable on its own — it doesn't say where the risk comes from. Decompose portfolio VaR into per-asset contributions via Euler decomposition, and compare equal-weight against a risk-parity allocation to see whether weight and risk contribution actually diverge in practice.

**Method:** parametric (variance-covariance) portfolio VaR — `VaR_p = -z_alpha * sqrt(w'Σw)`, zero-mean — using the sample covariance matrix of daily returns over a trailing 250-day window across all 23 symbols. `ComponentVaR_i = -z_alpha * w_i * (Σw)_i / sigma_p`, which sums to portfolio VaR exactly by construction (Euler's homogeneous-function theorem). `StandaloneVaR_i` uses only that asset's own volatility, ignoring correlation — the risk if it were held alone.

Two allocations compared: equal-weight, and risk parity (weights solved numerically so every asset contributes equally to portfolio risk, not a naive inverse-volatility heuristic).

**Validation before use:** the decomposition was hand-verified on a 2-asset case before implementation (components summed to the exact portfolio VaR to 6 decimal places, standalone sum correctly exceeded it by the expected diversification benefit). The risk-parity optimizer was checked on a synthetic 6-asset portfolio: component VaRs across assets came back within 0.03% of each other (should be identical), and higher-volatility assets correctly showed higher component VaR under equal weighting before parity was applied.

### Results

**Equal-weight portfolio VaR: 1.98%.** Top risk contributors: COIN (11.2% of portfolio risk from a 4.35% weight), AMD (10.3%), INTC (9.8%), SOFI (9.7%), RIVN (8.8%), PLTR (7.6%), TSLA (7.2%) — all higher-beta/higher-volatility names, contributing well above their equal capital weight.

**Three symbols show *negative* component VaR: WMT (-0.18%), CVX (-0.89%), XOM (-1.07%).** Despite carrying real standalone risk (0.10-0.12%), their correlation structure with the rest of the portfolio means a marginal dollar allocated to them *reduces* total portfolio variance — they act as natural hedges in this particular 23-symbol mix, not as risk contributors.

**Diversification benefit: 53%** (sum of standalone VaRs 4.22% vs. actual portfolio VaR 1.98%).

**Risk parity successfully equalizes contributions** — every symbol's component VaR lands within a tight band (4.15%-4.53% of total risk, vs. the equal-weight range of -1.07% to 11.21%), confirming the optimizer converged correctly on real data, not just the synthetic test case. To achieve this, it heavily upweights the negative/low-risk-contribution names (XOM to 11.2% weight, CVX to 11.2%, WMT to 9.5%) and heavily downweights the high-risk-contribution names (COIN down to 1.28%, RIVN to 1.67%, AMD to 1.67%).

**Risk-parity portfolio VaR: 1.18%, a 40.3% reduction vs. equal-weight's 1.98%** — purely from reallocating capital by risk contribution rather than equal dollars, with zero predictive input. Diversification benefit rises to 64.7%.

### Findings

**Weight and risk contribution diverge substantially and predictably by asset type.** Every symbol flagged `risk > weight` under equal weighting is a higher-volatility, higher-beta name (crypto-adjacent, small/mid-cap tech, semis); every symbol flagged `risk < weight` is a lower-volatility, more defensive name (energy, staples, healthcare, financials) — the divergence isn't random, it's exactly what the underlying volatility and correlation structure of the universe predicts.

**Risk-based allocation produces a large, quantified risk reduction with no alpha assumption whatsoever** — a clean, textbook demonstration that where you put capital matters as much as how much you put, holding the asset universe fixed.

### Limitations

- **Parametric-normal, not tied to the validated GJR-GARCH-t per-asset models** from Experiments 3-4. Extending component VaR to a genuinely time-varying, multivariate covariance (DCC-GARCH or a copula) is real additional scope, deliberately not attempted here — this isolates the portfolio-construction question from an unrelated modeling upgrade, but means the covariance matrix itself carries the same normality assumption already shown (in Experiment 3d's residual diagnostics) to be a simplification for several of these symbols.
- Static covariance matrix from a single trailing 250-day window — no consideration of how component VaR or the risk-parity allocation would shift over time, or how often rebalancing would be needed.
- Zero-mean assumption (drops the drift term) — standard at daily horizons but not tested against including it.
- Long-only risk parity (weights bounded to [0.001, 1]) — short positions, which could in principle allow even tighter risk equalization or exploit the negative-component-VaR names more directly, were not considered.
- No transaction costs modeled for the hypothetical equal-weight → risk-parity rebalance.
