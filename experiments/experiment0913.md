# Experiment Log — Event-Conditioned Downside Risk

**Date:** 2026-09-13
**Project:** stockexp — historical price/news explorer, extended into an event-conditioned downside-risk model

Central question tested: **does adding news sentiment to a market-data-only model improve prediction of downside risk?**

Two independent experiments tested this at different time resolutions. Both are documented below with their actual scope, so results aren't read outside the conditions they were produced under.

---

## Experiment 1 — Daily Panel: Market-Only vs. Market+Sentiment

**Purpose:** test whether day-level aggregated news sentiment improves prediction of a stock's 5-day downside risk, on top of market-state features alone.

**Target:** binary — does the 5-day forward cumulative abnormal return (CAR, vs. SPY market model with rolling 60-day beta) fall below -3%.

**Market-only features:** 1/5/20-day momentum, 20-day realized volatility, volume surprise (vs. 20-day average), rolling beta.

**Event features:** daily article count, mean/min FinBERT sentiment score, corporate-event flag (dividends/splits/mergers from Alpaca, 8-K/10-Q/10-K filing dates from SEC EDGAR).

**Relevance filter applied to news:** articles tagged to ≤4 tickers (later tightened to ≤1), excluding known Benzinga roundup/listicle headline patterns.

**Method:** logistic regression, both feature sets scored on the identical train/test rows (an earlier run had a `dropna` bug that silently gave the two models different populations — fixed before these numbers).

**Validation:** time-based split (train on earlier dates, test on later dates) — not random, to avoid training on data from after the test period.

### Results

| Universe | History | Rows | Market-only ROC-AUC | +Sentiment ROC-AUC | Lift |
|---|---|---|---|---|---|
| 8 megacap tech tickers | 2 years | 3,520 | 0.603 | 0.605 | +0.002 |
| 23 tickers, sector-diverse | 5 years | 28,757 | 0.632 | 0.626 | -0.006 |

Market-only PR-AUC in both runs was clearly above the base rate (e.g. 0.315 vs. 0.216 in the 23-ticker run), confirming the market features carry real signal on their own.

**Finding: daily-aggregated news sentiment shows no measurable improvement over market-state features, consistent across two structurally different universes** (a saturated megacap-only set and a sector-diverse set with much sparser news coverage — 83.7% vs. 13.3% of symbol-days had qualifying news).

### Limitations
- Daily aggregation averages every relevant article in a day into one number, discarding whether an article landed before or after a price move — this is the most likely reason sentiment shows no lift here even if a real, faster effect exists.
- Relevance filter is a cheap heuristic (ticker-tag count + regex on known roundup formats), not semantic relevance. A single article that genuinely isn't about the company but is tagged narrowly would still pass.
- Downside threshold (-3% over 5 days) and beta window (60 days) were fixed choices, not tuned or justified against alternatives.
- News-source publish timestamps lag the actual wire time by an unknown, possibly systematic amount (Benzinga-sourced via Alpaca) — this biases any tight time-alignment analysis, though it's a second-order concern for a daily-level target.

---

## Experiment 2 — Article-Level: Minute-Bar Timed Sentiment

**Purpose:** the sharper version of the same question — does an individual article's sentiment predict the price move in the minutes right after its specific publish timestamp, once market context (recent momentum/volatility) is included.

**Target:** binary — did price rise in the N minutes following the article (horizon tested: 30 min).

**Method:** each qualifying article joined to the nearest minute bar at publish time and at publish time + horizon; market context taken from the prior trading day's close. Logistic regression, same time-based split discipline as Experiment 1.

**Constraint:** minute-bar history on the free Alpaca IEX feed is expensive to accumulate (sequential, rate-limited REST pulls), so this experiment is bounded to whatever minute-bar window has been pulled — a fraction of Experiment 1's date range.

### Results

| Minute-bar window | Usable articles (test set) | Market-only ROC-AUC | +Sentiment ROC-AUC | Lift |
|---|---|---|---|---|
| ~30 days | 150 (test n=45) | 0.550 | 0.590 | +0.040 |
| ~60 days | 162 (test n=49) | 0.522 | 0.541 | +0.019 |

**Finding: not conclusive.** Both the market-only and event-aware AUCs, and the lift itself, shifted meaningfully between the two runs despite doubling the data window. That instability is itself the signal — at this sample size (test sets of 45-49 examples), the estimates aren't stable enough to distinguish a real effect from noise.

### Limitations
- **Severely underpowered.** ~150-160 usable articles total is far too few to trust a point estimate; standard error on AUC at this n is roughly ±0.10-0.12.
- Expanding the universe from 8 to 23 tickers barely grew the usable article count (150 → 162) despite doubling the minute-bar window, because most of the newly added tickers (industrials, financials, energy) don't generate frequent single-ticker-tagged news — the constraint is news supply, not minute-bar coverage.
- Getting to a trustworthy sample size (hundreds of test examples) would require months of minute-bar history, which is impractical on the free IEX feed and would likely require a paid data source (e.g. Massive/Polygon flat files, deliberately avoided so far for cost reasons).
- IEX-only minute bars (free tier) see a small fraction of consolidated volume — price/volume figures at the minute level may not reflect the full market's reaction.

---

## Overall Conclusion (as of this date)

The defensible, current statement:

> Market-state features (momentum, volatility, volume, beta) predict 5-day downside risk with ROC-AUC ≈ 0.63 across 23 tickers and 5 years of daily data. Neither daily-aggregated news sentiment nor article-level, minute-timed sentiment showed a stable improvement over that baseline. The daily-level result is well-powered and consistent across two different universes; the article-level result is directionally suggestive but underpowered and not conclusive on its own.

This is a legitimate negative/null result, not a failed experiment — the market-only baseline is real and the methodology (time-based splits, matched populations, relevance filtering) is sound enough to trust the conclusion within its stated scope.

## Recommended Next Step

Given the sentiment signal hasn't stabilized after two honest attempts at two resolutions, and further minute-bar data is expensive to acquire on the current data budget: **prioritize Stage C (volatility forecasting) using the market-only feature set**, which doesn't depend on resolving the sentiment question, over continuing to chase article-level sample size.