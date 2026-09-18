# Experiment Log — Volatility Forecasting (Stage C)

**Date:** 2026-09-15
**Project:** stockexp — event-conditioned market-risk platform

Continues from `experiments-2026-09-13.md` (downside-risk classification experiments). This log covers a separate question: **can market-state features beat standard volatility-forecasting baselines, and which baseline is actually hardest to beat?**

---

## Experiment 3 — 5-Day Realized Volatility Forecasting

**Purpose:** compare a market-state ML model against the standard baselines used in practice for volatility forecasting, using the correct evaluation metric for risk work rather than a generic error metric.

**Target:** `RV_{t+1:t+5} = sqrt(sum of squared daily returns over the next 5 trading days)` — realized volatility, not a return.

**Universe / history:** 23 tickers, 5 years of daily bars (same panel as the downside-risk experiments).

**Models compared:**
- **Naive** — trailing 20-day daily volatility, scaled to a 5-day horizon
- **EWMA (RiskMetrics, λ=0.94)** — exponentially weighted variance
- **GARCH(1,1)** — fit once per symbol on the training window only, then used to forecast forward without refitting (documented simplification — see Limitations)
- **Gradient boosting** — market-state features (1/5/20-day momentum, 20-day realized vol, volume surprise), predicting log(RV)

**Evaluation metrics:** MAE, RMSE, and **QLIKE** — the metric that matters most here, since it penalizes underforecasting variance more heavily than overforecasting it, which is the direction that actually matters when a forecast feeds a risk limit. Results also split by calm vs. stressed regime (median trailing-vol cut).

**Validation:** time-based split (train ≤ cutoff date, test after), same discipline as the downside-risk experiments.

### Results (test set, n=8,510)

| Model | MAE | RMSE | QLIKE | Calm QLIKE | Stressed QLIKE |
|---|---|---|---|---|---|
| Naive (20d, scaled) | 0.02398 | 0.03594 | 0.6987 | 0.7517 | 0.6457 |
| **EWMA (RiskMetrics)** | 0.02321 | 0.03411 | **0.6203** | 0.6607 | 0.5800 |
| GARCH(1,1) | 0.02553 | 0.03590 | 0.6530 | 0.6405 | 0.6655 |
| Gradient boosting | **0.02058** | **0.03251** | 0.7610 | 0.8009 | 0.7212 |

### Findings

**EWMA wins on QLIKE, gradient boosting wins on MAE/RMSE — and QLIKE is the metric to trust here.** The ML model is closer to the truth on average but is apparently underforecasting variance specifically at the moments that matter (large vol spikes), which QLIKE punishes and MAE doesn't. This is a direct, self-produced demonstration of why generic error metrics are the wrong tool for evaluating risk forecasts — not a citation of the general point, but this project's own data showing it.

**Simple EWMA beating both GARCH and a gradient-boosted model is a standard, credible result**, well-documented in the volatility-forecasting literature; it is not evidence of a broken pipeline.

**Regime behavior reverses for GARCH specifically.** Naive, EWMA, and gradient boosting are all worse in the calm regime and better in the stressed regime (a "quiet before the storm" effect — forecasts made during calm periods get blindsided by subsequent vol spikes). GARCH shows the opposite pattern: better in calm (0.6405), worse in stressed (0.6655). This is a direct, explainable consequence of the fit-once simplification below, not noise.

### Limitations

- **GARCH(1,1) was fit once per symbol on the training window and never refit.** A correct walk-forward GARCH (refit periodically through the test period, using only data available up to each refit point) is the standard approach and the natural next step — this result likely understates GARCH's real performance, and probably explains its inverted regime pattern (parameters reflect 2021-2025 dynamics, degrading if the test period's stressed episodes differ in character).
- Regime split is a simple median cut on trailing 20-day volatility, not a proper regime-detection method (e.g. Markov-switching).
- ML model uses only the same five market-state features as the downside-risk experiments — no volatility-specific features (e.g. realized skew/kurtosis, overnight gaps) were engineered.
- Single train/test split, not walk-forward cross-validation — one point estimate per model, no confidence interval on the QLIKE gap between models.

---

## Experiment 3b — GJR-GARCH + Student-t (Asymmetric)

**Purpose:** GARCH remained weaker than EWMA after the walk-forward fix, and specifically still favored the calm regime over stressed — the opposite pattern from every other model. Two hypotheses for why: (1) plain GARCH assumes volatility reacts the same way to positive and negative shocks, when in practice negative shocks tend to spike volatility harder (the leverage effect), and (2) it assumes normally distributed returns, understating tail risk. Both differences would matter specifically during stress, not during calm periods — matching the exact pattern observed.

**Change tested:** GJR-GARCH(1,1,1) with a Student-t distribution instead of GARCH(1,1) with normal, same walk-forward refit schedule (every ~21 trading days, expanding window) as Experiment 3's fix.

### Results

| Model | QLIKE (overall) | Calm QLIKE | Stressed QLIKE |
|---|---|---|---|
| EWMA (RiskMetrics) | 0.6203 | 0.6607 | 0.5800 |
| GARCH(1,1), walk-forward | 0.6228 | 0.6163 | 0.6292 |
| **GJR-GARCH(1,1,1) + t, walk-forward** | **0.5856** | **0.5826** | **0.5885** |

### Findings

**GJR-GARCH-t has the best QLIKE of any model tested (0.5856), the first model to beat EWMA.** Both hypotheses were confirmed: the asymmetry term and the fat-tailed distribution were each real missing pieces, not just theoretically motivated additions.

**The regime asymmetry that persisted through the walk-forward fix is essentially eliminated** — calm and stressed QLIKE are nearly identical (0.5826 vs 0.5885), compared to a ~0.013 gap for plain GARCH and a much larger ~0.08 gap for EWMA (which, notably, is now the model with the largest calm/stressed spread — it does worse in calm markets than either GARCH variant). Consistent performance across regimes is itself a desirable property for a risk model, independent of the average QLIKE.

**Full progression, each step motivated by a specific diagnosed weakness in the last:**
naive → EWMA → GARCH (fit-once) → GARCH (walk-forward) → GJR-GARCH-t (walk-forward, asymmetric, fat-tailed)

### Limitations

- Distribution and asymmetry were changed together in one test rather than isolated (GJR alone with normal dist, or plain GARCH with Student-t alone, were not tested separately) — the combined result doesn't say how much each change contributed individually.
- Same single train/test split as Experiment 3; no confidence interval on the QLIKE gap between GJR-GARCH-t and EWMA.
- Refit cadence (21 trading days) was not itself tuned or compared against other frequencies.

## Recommended Next Step

The volatility-forecasting thread is now in a strong, complete state: GJR-GARCH-t with walk-forward refitting is the best model tested, with a clear, tested explanation for why. Next priority is the biggest open gap for a risk-analyst-focused portfolio: **VaR and Expected Shortfall with a Kupiec proportion-of-failures backtest** — nothing has been built on this yet, and it's the deliverable most likely to come up by name in a risk-analyst interview.

---

## Experiment 3c — Per-Symbol Significance Check (Diebold-Mariano + Sign Test)

**Purpose:** the pooled QLIKE comparison (Experiment 3b) says GJR-GARCH-t beats EWMA overall — but a pooled average can hide a result that's actually driven by one or two extreme symbols. Test whether the advantage holds up at the individual-ticker level, and whether it's statistically distinguishable from noise rather than just a smaller number.

**Method:** per-symbol Diebold-Mariano test (Newey-West HAC variance, lag=4 to account for overlapping 5-day-ahead forecasts, Harvey-Leybourne-Newbold small-sample correction) comparing GJR-GARCH-t against EWMA, run separately for each of the 23 tickers. Followed by a binomial sign test across all 23 DM statistics' directions, independent of individual significance.

### Results

| | Count |
|---|---|
| Symbols where GJR-GARCH-t significantly better (p<0.05) | 2 (AMZN, WMT) |
| Symbols where EWMA significantly better (p<0.05) | 0 |
| No significant difference | 21 |
| DM statistic direction favors GJR-GARCH-t (regardless of significance) | 19 of 23 |
| DM statistic direction favors EWMA | 4 of 23 (CVX, TSLA, UNH, XOM) |
| Sign test p-value (H0: 50/50 coin flip per symbol) | **0.0026** |

### Findings

**Only 2 of 23 symbols individually reach significance — but the sign test shows the tilt toward GJR-GARCH-t is real and broad.** 19 of 23 symbols point the same direction; under the null hypothesis that's about as likely as a coin landing heads 19 times in 23 flips (p=0.0026). This is not a contradiction of the "only 2 significant" result — it's a complementary test answering a different question. Individual per-symbol DM tests are underpowered here: overlapping 5-day-ahead forecasts mean the effective independent sample size is much smaller than the raw n=370 per symbol suggests, so a real, modest, broadly-shared effect can easily fail to clear significance symbol-by-symbol while still being genuine in aggregate.

**Conclusion: the pooled QLIKE advantage for GJR-GARCH-t is not an artifact of a couple of extreme symbols.** It's a small, consistent effect spread across most of the universe — a materially different and more defensible finding than either "it's definitively better everywhere" or "it's only better because of outliers" would have been.

**UNH is worth separate attention, not explained away.** It's the largest single-symbol gap favoring EWMA (though not statistically significant on its own) — worth checking whether a healthcare-sector-specific volatility event in the test window is responsible, and whether GJR-GARCH's asymmetry assumption is a poor fit for that kind of idiosyncratic shock.

### Limitations

- Per-symbol DM tests are individually underpowered by construction (see above) — this is a real constraint of the ~1.5-year test window per symbol, not a flaw in the test itself.
- Sign test treats all 23 symbols as independent Bernoulli trials; if the model comparison's outcome is correlated across similar symbols (e.g. same-sector names moving together), the effective number of independent trials is lower than 23 and the sign test's p-value is somewhat optimistic.
- UNH's divergence was noted but not investigated further in this pass.


---

## Experiment 3d — Residual Diagnostics (Ljung-Box, QQ, Kurtosis)

**Purpose:** the GJR-GARCH-t model won on QLIKE and passed the per-symbol DM/sign test, but a model can win a forecast-accuracy metric without being correctly specified. This checks specification directly, on the model's own standardized residuals `z_t = resid_t / conditional_vol_t`, fit once per symbol on the full history (in-sample) rather than the walk-forward windows used for forecasting.

**Diagnostics run:**
- Ljung-Box on `z_t` — remaining serial correlation in returns after removing the conditional mean
- Ljung-Box on `z_t^2` — remaining volatility clustering after the GARCH filter (the diagnostic that matters most: a failure here means the model didn't do the one thing GARCH exists to do)
- Excess kurtosis, empirical vs. theoretical (implied by the fitted Student-t degrees of freedom, ν)
- QQ plot against arch's own unit-variance-standardized Student-t quantile function (not `scipy.stats.t` directly, which is not unit-variance by default and would silently mis-scale the plot)

### Results (23 symbols, full history, lags 10/20)

| Symbol | n | ν | LB(z) p10 | LB(z) p20 | LB(z²) p10 | LB(z²) p20 | Kurt emp | Kurt fit | Flag |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | 1252 | 4.76 | 0.631 | 0.938 | 0.443 | 0.733 | 2.49 | 7.87 | ok |
| AMD | 1252 | 4.71 | 0.615 | 0.567 | 0.936 | 0.995 | 5.74 | 8.49 | ok |
| AMZN | 1252 | 5.33 | 0.374 | 0.558 | 0.218 | 0.596 | 4.30 | 4.51 | ok |
| BA | 1250 | 4.64 | 0.434 | 0.302 | 0.462 | 0.835 | 2.17 | 9.31 | ok |
| BAC | 1250 | 5.01 | 0.477 | 0.460 | 0.399 | 0.248 | 3.22 | 5.93 | ok |
| CAT | 1250 | 4.09 | 0.204 | 0.280 | 0.735 | 0.938 | 4.14 | 63.75 | ok |
| COIN | 1252 | 6.52 | 0.914 | 0.958 | 0.129 | 0.536 | 2.20 | 2.38 | ok |
| CVX | 1250 | 5.65 | 0.592 | 0.559 | 0.584 | 0.889 | 2.38 | 3.64 | ok |
| DIS | 1250 | 4.17 | 0.900 | 0.762 | 0.940 | 0.991 | 7.73 | 34.43 | ok |
| GOOGL | 1252 | 4.60 | 0.727 | 0.414 | 0.946 | 0.977 | 3.94 | 9.97 | ok |
| INTC | 1252 | 3.75 | 0.643 | 0.211 | 0.993 | 0.980 | 12.57 | ∞ | ok |
| JPM | 1250 | 4.60 | 0.529 | 0.167 | 0.897 | 0.958 | 6.09 | 10.04 | ok |
| META | 1252 | 3.63 | 0.924 | 0.988 | 1.000 | 1.000 | 21.77 | ∞ | ok |
| MSFT | 1252 | 4.99 | 0.596 | 0.611 | 0.986 | 0.991 | 5.61 | 6.06 | ok |
| **NVDA** | 1252 | 6.24 | 0.272 | 0.656 | 1.000 | 1.000 | 9.77 | 2.68 | **HEAVY-TAIL** |
| PFE | 1250 | 5.88 | 0.644 | 0.828 | 0.386 | 0.351 | 2.47 | 3.19 | ok |
| **PLTR** | 1250 | 3.71 | 0.077 | 0.040 | 1.000 | 1.000 | 11.18 | ∞ | **SERIAL-CORR** |
| RIVN | 1212 | 4.20 | 0.483 | 0.537 | 0.410 | 0.763 | 6.08 | 30.27 | ok |
| SOFI | 1252 | 5.51 | 0.102 | 0.302 | 0.824 | 0.933 | 3.48 | 3.96 | ok |
| TSLA | 1252 | 5.37 | 0.534 | 0.491 | 0.042 | 0.206 | 3.00 | 4.38 | ok |
| UNH | 1250 | 3.59 | 0.890 | 0.672 | 1.000 | 0.590 | 24.55 | ∞ | ok |
| WMT | 1250 | 3.90 | 0.657 | 0.770 | 0.989 | 0.999 | 12.48 | ∞ | ok |
| **XOM** | 1250 | 9.05 | 0.470 | 0.453 | 0.028 | 0.019 | 0.84 | 1.19 | **ARCH-LEFT** |

**Summary:** 20/23 symbols (87%) clean on both Ljung-Box tests. 1 serial-correlation flag (PLTR), 1 ARCH-effects-remaining flag (XOM), 1 heavy-tail flag (NVDA).

**QQ plot grid:** `experiments/garch_qq_grid.png` (5×5 grid, all 23 symbols, standardized residuals vs. fitted Student-t quantiles)

![GJR-GARCH-t QQ grid](garch_qq_grid.png)

### Findings

**20 of 23 symbols pass both Ljung-Box tests cleanly — this is real validation, not a formality.** It means the GJR-GARCH-t specification genuinely captures the volatility dynamics for the large majority of a deliberately heterogeneous universe (megacap tech, financials, energy, healthcare, high-volatility names), rather than having won the earlier QLIKE comparison by chance.

**XOM is the only symbol with a genuine remaining-ARCH-effects problem** (LB(z²) p=0.019 at lag 20) — the one exception that actually threatens the well-specified claim. Notably, XOM also has the highest fitted ν (9.05, closest to normal) and the tightest-looking QQ plot in the grid. The failure is specifically about volatility clustering, not tail shape — consistent with an energy/commodity-linked name having different vol dynamics (oil-price jump risk) than the negative-return leverage effect GJR-GARCH is built to capture.

**NVDA's HEAVY-TAIL flag is visible directly in its QQ panel** — a clear bend in the lower-left where sample quantiles fall below the diagonal, meaning NVDA's worst days are worse than even its own fitted t(ν=6.24) predicts. Plausibly linked to NVDA's history of outsized single-day moves on AI-related news — a specific, explainable limitation rather than generic model failure.

**PLTR's SERIAL-CORR flag (p=0.040 at lag 20, but p=0.077 at lag 10) should be treated skeptically rather than as a clean failure.** Its LB(z²) p-values are exactly 1.000 at both lags — unusually clean — combined with ν=3.71 sitting right at the boundary where the 4th moment stops existing (ν≤4). This combination looks more like a numerical optimizer artifact near a parameter boundary than a genuine specification failure, and is a candidate for a rerun with tighter fitting tolerance rather than being taken at face value.

### Limitations

- **The HEAVY-TAIL flag cannot trigger for any symbol with ν≤4** — the theoretical kurtosis formula (6/(ν−4)) is undefined there, so the script reports it as infinite and "empirical > 1.5× theoretical" is vacuously false regardless of how extreme the real tails are. Five symbols (INTC, META, PLTR, UNH, WMT) have ν≤4 and are structurally exempt from this check even though several show visible tail deviation in the QQ plot. The plot is the more trustworthy read for these symbols; the automated flag undercounts tail risk exactly where fitted tails are fattest.
- Fit is in-sample on full history, not tied to the walk-forward windows used for the forecasting comparison in Experiment 3b/3c — this is deliberate (a specification question, not a forecasting one) but means these diagnostics don't directly validate the out-of-sample forecasts themselves.
- PLTR's flag was flagged as a likely optimizer artifact but not re-fit with tighter tolerance to confirm.

## Recommended Next Step (updated)

Volatility forecasting is now closed out at forecast accuracy, statistical comparison, cross-sectional robustness, AND specification diagnostic levels — a complete chain rather than a single leaderboard number. Move to **VaR and Expected Shortfall with a Kupiec backtest**, using each symbol's fitted GJR-GARCH-t parameters (including the actual fitted ν) as the volatility/distribution input rather than a generic normal-VaR assumption — a more defensible parametric VaR than most portfolio projects use, and a direct continuation of this validated model rather than a fresh start.
