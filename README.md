# stockexp

Event-conditioned market risk research, built for the SDSU Financial Markets Lab.

It started as a price and news explorer: click a point on a stock chart, see the news around it. It turned into a small research project. The question is whether news sentiment helps predict downside risk, and how well standard risk models (volatility forecasts, VaR, ES) hold up when tested properly.

Short answer so far: sentiment adds nothing, GJR-GARCH-t is the best volatility model, and historical VaR is better calibrated than the GARCH-based VaR.

## What is in here

```
ingest/       load prices, news, events and sentiment into Postgres
app/          Streamlit explorer (chart, news, event markers)
analysis/     early signal checks
risk/         the research code: features, volatility, VaR/ES, portfolio, stress test
experiments/  dated write-ups of every experiment, with results and limits
migrations/   schema changes
schema.sql    base tables
DATA.md       how the data is laid out and how to query it
```

## Data

- 23 US stocks plus SPY, QQQ and XLK as benchmarks. Daily bars, about 28,700 for the test set period, from Alpaca (free IEX feed, split adjusted).
- News from Alpaca (Benzinga), about 50,800 articles, each scored with FinBERT. Score = P(positive) minus P(negative).
- Corporate events (dividends, splits, earnings) stored in an `events` table.
- Everything is stored in UTC. Daily bars are stamped at New York midnight.

Limits to know about: the IEX feed only covers part of total volume, the 23 stocks are today's survivors, and there are no trading costs anywhere in this project.

More detail in `DATA.md`.

## Setup

```bash
docker compose up -d                  # Postgres on port 5439
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # add your Alpaca paper keys
python ingest/db.py                   # create tables, seed the tickers
python ingest/bars.py --years 5
python ingest/news.py --years 5
python ingest/events.py
python ingest/sentiment.py            # downloads FinBERT, takes a while
streamlit run app/main.py
```

`.env` is in `.gitignore`. Keep it that way. `.env.example` only holds empty variable names.

## How the research went

Each step has its own dated log in `experiments/` with the full output.

**1. Does sentiment predict downside?**

Target: a 5-day cumulative abnormal return below -3%, measured against SPY with a rolling 60-day beta. I trained logistic regression and gradient boosting on market-only features, then added sentiment features and compared on a later test period (time-based split, no shuffling).

- Market-only features give an AUC of about 0.63.
- Adding sentiment changed it by about +0.002. That is noise.
- I also tested article by article on a small set (162 articles). Too few to conclude much.

Early on the two models were scored on different rows because of missing sentiment, which made sentiment look slightly useful. Filling missing sentiment with 0 so both models use the same rows removed that.

**2. Volatility forecasting**

Five models, walk-forward, expanding window, GARCH refit every 21 trading days. Train 2021-09 to 2025-03, test 2025-03 to 2026-09 (8,625 stock-days).

| Model | QLIKE (lower is better) |
|---|---|
| Naive 20-day | 0.6987 |
| EWMA (RiskMetrics) | 0.6203 |
| GARCH(1,1) | 0.6228 |
| GJR-GARCH(1,1,1), Student-t | 0.5856 |
| Gradient boosting on market features | 0.7610 |

GJR-GARCH-t against EWMA, per stock, with the Diebold-Mariano test (Newey-West variance, small-sample correction):

- Significantly better in 2 of 23 stocks (AMZN, WMT). EWMA was never significantly better.
- Better in 19 of 23 by direction. Sign test p = 0.0026.

So the gain is real but small per stock. Only the pooled view shows it clearly.

Residual checks (Ljung-Box on standardized residuals and their squares, kurtosis, QQ plot) were fine for most stocks. XOM, NVDA and PLTR were flagged.

**3. VaR and Expected Shortfall (95%)**

Historical VaR (250-day window) against walk-forward GJR-GARCH-t VaR. Kupiec test for the exception count, Christoffersen for clustering.

| | Exception rate | Pooled Kupiec | Stocks rejected |
|---|---|---|---|
| Historical | 5.08% | p = 0.739 | 0 of 23 |
| GJR-GARCH-t | 3.73% | p < 0.001 | 7 of 23 |

Historical VaR is well calibrated. The GARCH version is too conservative: it predicts too much risk and gets breached less than 5% of the time. Rejected stocks: BA, BAC, COIN, DIS, JPM, NVDA, TSLA.

I checked three explanations and ruled them out:
- Over-forecasting volatility. Forecast vol was within about 15% of realized for every stock (none over-forecast).
- Fat tails not handled. The model uses Student-t.
- Breach clustering. Only one stock (AAPL, historical) showed any.

What is left is probably a mismatch between the fitted tail shape and the out-of-sample tails. I have not tested that yet.

Average ES/VaR ratio: 1.51x historical, 1.46x GARCH.

**4. Portfolio risk**

Euler component VaR on the 23-stock portfolio, using the last 250 days. Components add up exactly to portfolio VaR.

- Equal weight: portfolio VaR 1.98%. Diversification cuts risk by 53% compared with the sum of single-stock VaRs.
- COIN is 4.35% of capital but 11.2% of the risk. WMT, CVX and XOM contribute about zero or slightly negative.
- Risk parity (solved with SLSQP) brings portfolio VaR to 1.18%, which is 40.3% lower than equal weight.

**5. Stress test (written, not run yet)**

`risk/stress_test.py` replays a crisis window, 2022 by default, through the VaR models and tests the portfolio weights fixed before the window. It compares a frozen covariance with one re-estimated every 21 days. It needs more history first:

```bash
python ingest/bars.py --years 9
python risk/stress_test.py
```

Results will go in a new file in `experiments/`.

## Running the research code

```bash
python risk/baseline_vs_event_aware.py   # step 1
python risk/article_level.py
python risk/volatility.py                # step 2 (flags: --skip-gjr --skip-dm --refit-every N)
python risk/garch_diagnostics.py         # writes experiments/garch_qq_grid.png
python risk/var_es.py                    # step 3
python risk/portfolio_var.py             # step 4
python risk/stress_test.py               # step 5
```

The GARCH scripts take a few minutes because of the walk-forward refits.

## Known gaps

- Stress test not run yet.
- The GARCH VaR over-conservatism is unexplained. Tail shape is my next guess.
- XOM showed left-side ARCH effects in the residuals. I have not looked into it.
- PLTR may be an optimizer problem rather than a real feature of the data. Not retested.
- Only daily data is used in the research. Minute bars are set up in the ingest code but unused.
- No transaction costs, no survivorship correction.

## Rules the code follows

- All timestamps are `timestamptz` in UTC. Convert to New York only when displaying.
- Every ingest can be re-run without creating duplicates.
- Nothing downstream of `bars` and `news` knows which vendor the data came from.
