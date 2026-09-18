"""Build the event-observation feature table.

One row per (symbol, trading day) with:
  - market-state features available BEFORE that day closes (momentum,
    volatility, volume) — the "market-only" baseline feature set
  - event/text features from that day's news (sentiment, relevance,
    article count) — the "event-aware" addition
  - forward targets: next-day abnormal return, 5-day cumulative abnormal
    return, and a binary downside flag (CAR_5d < -3%)

Abnormal return needs a market model per symbol: R_i = alpha + beta*R_mkt.
Beta is estimated on a trailing 60-day window and refit at each date, so
it only ever uses information available up to that point — no lookahead.

This is Stage A/B's raw material from the flagship plan. It does NOT yet
do sector returns, VIX/regime features, event-type classification, or
walk-forward CV — those are later phases. Daily bars only for now; this
whole feature set could be rebuilt on minute bars once that history is
wide enough to matter.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest"))
from db import connect, UNIVERSE, BENCHMARKS  # noqa: E402

MARKET_PROXY = "SPY"
BETA_WINDOW = 60       # trading days
DOWNSIDE_THRESHOLD = -0.03
CAR_HORIZON = 5        # trading days

# Same relevance filter as app/main.py's sidebar defaults. Kept as a literal
# copy for now rather than a shared module — worth factoring out once both
# need to change it at the same time.
MAX_RELEVANT_TICKERS = 1  # tightened from 4 — see if single-ticker-only articles
                          # still saturate every trading day for these megacaps
ROUNDUP_PATTERNS = [
    r"^\d+ Stocks? (Moving|To Watch)",
    r"Stocks That Hit (New )?52-Week",
    r"^Benzinga['’]s (Bulls|Bears|Top)",
    r"^Insider (Buying|Selling)",
    r"Mid-Day (Gainers|Losers)",
    r"^Today['’]s Market Wrap",
]
ROUNDUP_SQL = " OR ".join(
    f"n.headline ~* '{p.replace(chr(39), chr(39)+chr(39))}'" for p in ROUNDUP_PATTERNS
)


def load_daily_bars(conn, symbols):
    return pd.read_sql(
        """
        SELECT symbol, ts::date AS date, close, volume
        FROM bars
        WHERE symbol = ANY(%s) AND timeframe = '1Day'
        ORDER BY symbol, ts
        """,
        conn,
        params=(symbols,),
    )


def load_daily_news_features(conn, symbols):
    """Per symbol per day: mean/min sentiment, article count — computed
    only over articles that pass the same relevance filter as the app
    (not tagged to a dozen tickers, not a known roundup format). Without
    this, a day's sentiment average is diluted by broad-market roundups
    and mistagged articles that have nothing to do with the company."""
    return pd.read_sql(
        f"""
        SELECT nt.symbol,
               n.published_at::date AS date,
               COUNT(*)                                   AS n_articles,
               AVG(n.sentiment_score)                       AS avg_sentiment,
               MIN(n.sentiment_score)                       AS min_sentiment,
               AVG(cnt.n_tickers)                           AS avg_tickers_tagged
        FROM news n
        JOIN news_tickers nt ON nt.news_id = n.id
        JOIN (SELECT news_id, COUNT(*) AS n_tickers
              FROM news_tickers GROUP BY news_id) cnt
             ON cnt.news_id = n.id
        WHERE nt.symbol = ANY(%s)
          AND n.sentiment_score IS NOT NULL
          AND cnt.n_tickers <= {MAX_RELEVANT_TICKERS}
          AND NOT ({ROUNDUP_SQL})
        GROUP BY nt.symbol, n.published_at::date
        """,
        conn,
        params=(symbols,),
    )


def load_event_counts(conn, symbols):
    """Per symbol per day: whether a corporate event/filing landed."""
    return pd.read_sql(
        """
        SELECT symbol, event_time::date AS date, COUNT(*) AS n_events,
               string_agg(DISTINCT event_type, ',') AS event_types
        FROM events
        WHERE symbol = ANY(%s)
        GROUP BY symbol, event_time::date
        """,
        conn,
        params=(symbols,),
    )


def rolling_beta(returns_i, returns_mkt, window):
    """Rolling OLS beta/alpha of a single stock against the market proxy.

    Vectorized rolling covariance/variance rather than looping OLS calls —
    same math (beta = cov/var), much faster over a full symbol history.
    """
    cov = returns_i.rolling(window).cov(returns_mkt)
    var = returns_mkt.rolling(window).var()
    beta = cov / var
    alpha = returns_i.rolling(window).mean() - beta * returns_mkt.rolling(window).mean()
    return alpha, beta


def build_features():
    symbols = UNIVERSE  # benchmarks are the market proxy, not modeling targets
    with connect() as conn:
        bars = load_daily_bars(conn, symbols + [MARKET_PROXY])
        news = load_daily_news_features(conn, symbols)
        events = load_event_counts(conn, symbols)

    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["symbol", "date"])
    bars["ret"] = bars.groupby("symbol")["close"].pct_change()

    mkt = bars[bars["symbol"] == MARKET_PROXY][["date", "ret"]].rename(columns={"ret": "mkt_ret"})

    rows = []
    for sym in symbols:
        sdf = bars[bars["symbol"] == sym].merge(mkt, on="date", how="inner").reset_index(drop=True)
        if sdf.empty:
            continue

        alpha, beta = rolling_beta(sdf["ret"], sdf["mkt_ret"], BETA_WINDOW)
        sdf["alpha"] = alpha
        sdf["beta"] = beta
        sdf["expected_ret"] = sdf["alpha"] + sdf["beta"] * sdf["mkt_ret"]
        sdf["abnormal_ret"] = sdf["ret"] - sdf["expected_ret"]

        # Forward CAR over the next CAR_HORIZON trading days, using only
        # abnormal returns that happen AFTER this row's date.
        sdf["fwd_car_5d"] = (
            sdf["abnormal_ret"].shift(-1).rolling(CAR_HORIZON).sum().shift(-(CAR_HORIZON - 1))
        )
        sdf["downside_5d"] = (sdf["fwd_car_5d"] < DOWNSIDE_THRESHOLD).astype("Int64")

        # Market-state features, all backward-looking only.
        sdf["mom_1d"] = sdf["ret"]
        sdf["mom_5d"] = sdf["close"].pct_change(5)
        sdf["mom_20d"] = sdf["close"].pct_change(20)
        sdf["vol_20d"] = sdf["ret"].rolling(20).std()
        sdf["avg_vol_20d"] = sdf["volume"].rolling(20).mean()
        sdf["volume_surprise"] = sdf["volume"] / sdf["avg_vol_20d"]

        sdf["symbol"] = sym
        rows.append(sdf)

    panel = pd.concat(rows, ignore_index=True)

    news["date"] = pd.to_datetime(news["date"])
    events["date"] = pd.to_datetime(events["date"])
    panel = panel.merge(news, on=["symbol", "date"], how="left")
    panel = panel.merge(events, on=["symbol", "date"], how="left")

    panel["n_articles"] = panel["n_articles"].fillna(0)
    panel["n_events"] = panel["n_events"].fillna(0)
    panel["had_news"] = (panel["n_articles"] > 0).astype(int)
    # Neutral, not missing: a day with no relevant article isn't a data gap,
    # it's information (nothing newsworthy happened). Leaving these as NaN
    # meant dropna() silently removed ~16% of rows from the event-aware
    # model only, so it was being scored on a different population than
    # the market-only model — not a fair comparison.
    panel["avg_sentiment"] = panel["avg_sentiment"].fillna(0)
    panel["min_sentiment"] = panel["min_sentiment"].fillna(0)
    panel["avg_tickers_tagged"] = panel["avg_tickers_tagged"].fillna(0)
    panel["had_event"] = (panel["n_events"] > 0).astype(int)

    keep = [
        "symbol", "date", "close",
        # targets
        "abnormal_ret", "fwd_car_5d", "downside_5d",
        # market-only features
        "mom_1d", "mom_5d", "mom_20d", "vol_20d", "volume_surprise", "beta",
        # event/text features
        "n_articles", "avg_sentiment", "min_sentiment", "avg_tickers_tagged",
        "had_news", "had_event", "n_events",
    ]
    panel = panel[keep].dropna(subset=["downside_5d"])
    return panel


if __name__ == "__main__":
    df = build_features()
    out = "features.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} rows, {df['symbol'].nunique()} symbols -> {out}")
    print(f"downside rate: {df['downside_5d'].mean():.1%}")
    print(f"rows with news that day: {(df['n_articles'] > 0).mean():.1%}")