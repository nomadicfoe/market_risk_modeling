"""Article-level test: does an individual article's sentiment predict the
price move right after its specific timestamp, once you also give the
model the market context (recent momentum/volatility) at that moment?

This is a different and sharper instrument than risk/features.py's daily
panel. That version averaged a whole day's news into one number, which
throws away exactly the thing that might matter — whether an article's
sentiment lines up with the move that happens in the minutes after it's
published, not the whole day around it.

Two things this depends on that the daily panel doesn't:
  - minute bars covering the SAME dates as the news you're testing
    (this script only uses news that falls inside your minute-bar window
    — everything else is silently out of scope, same as signal_check.py)
  - enough articles in that overlap to be worth fitting a model on at all

Usage:
    python risk/article_level.py --horizon-min 30
    python risk/article_level.py --horizon-min 60 --max-tickers 1
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest"))
from db import connect, UNIVERSE  # noqa: E402

# Same roundup/relevance filter as risk/features.py — literal copy for now.
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


def load_minute_bars(conn, symbols):
    df = pd.read_sql(
        """
        SELECT symbol, ts, close FROM bars
        WHERE symbol = ANY(%s) AND timeframe = '1Min'
        ORDER BY symbol, ts
        """,
        conn,
        params=(symbols,),
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_articles(conn, symbols, start, end, max_tickers):
    return pd.read_sql(
        f"""
        SELECT nt.symbol, n.published_at, n.sentiment_score, n.sentiment_label,
               cnt.n_tickers,
               (SELECT COUNT(*) FROM news n2
                JOIN news_tickers nt2 ON nt2.news_id = n2.id
                WHERE nt2.symbol = nt.symbol
                  AND n2.published_at BETWEEN n.published_at - INTERVAL '24 hours'
                                           AND n.published_at) AS corroborating_24h
        FROM news n
        JOIN news_tickers nt ON nt.news_id = n.id
        JOIN (SELECT news_id, COUNT(*) AS n_tickers
              FROM news_tickers GROUP BY news_id) cnt
             ON cnt.news_id = n.id
        WHERE nt.symbol = ANY(%s)
          AND n.sentiment_score IS NOT NULL
          AND n.sentiment_label != 'neutral'
          AND cnt.n_tickers <= %s
          AND NOT ({ROUNDUP_SQL})
          AND n.published_at BETWEEN %s AND %s
        ORDER BY n.published_at
        """,
        conn,
        params=(symbols, max_tickers, start, end),
    )


def load_daily_context(conn, symbols):
    """Prior-day market state per symbol/day — the same handful of
    backward-looking features as risk/features.py's market-only set,
    computed lightly here rather than importing that module directly."""
    bars = pd.read_sql(
        """
        SELECT symbol, ts::date AS date, close, volume FROM bars
        WHERE symbol = ANY(%s) AND timeframe = '1Day'
        ORDER BY symbol, ts
        """,
        conn,
        params=(symbols,),
    )
    bars["date"] = pd.to_datetime(bars["date"])
    out = []
    for sym, g in bars.groupby("symbol"):
        g = g.sort_values("date").copy()
        g["mom_5d"] = g["close"].pct_change(5)
        g["vol_20d"] = g["close"].pct_change().rolling(20).std()
        g["volume_surprise"] = g["volume"] / g["volume"].rolling(20).mean()
        out.append(g[["symbol", "date", "mom_5d", "vol_20d", "volume_surprise"]])
    return pd.concat(out, ignore_index=True)


def nearest_price(bars_sym, target_ts, tolerance_minutes=15):
    after = bars_sym[bars_sym["ts"] >= target_ts]
    if after.empty:
        return None
    row = after.iloc[0]
    if (row["ts"] - target_ts).total_seconds() > tolerance_minutes * 60:
        return None
    return row["close"]


def build_dataset(articles, minute_bars, daily_ctx, horizon_min):
    rows = []
    bars_by_symbol = {s: g for s, g in minute_bars.groupby("symbol")}
    ctx_by_symbol = {s: g.sort_values("date") for s, g in daily_ctx.groupby("symbol")}

    for _, a in articles.iterrows():
        sym = a["symbol"]
        bars_sym = bars_by_symbol.get(sym)
        if bars_sym is None:
            continue
        t0 = a["published_at"]
        t1 = t0 + pd.Timedelta(minutes=horizon_min)
        p0 = nearest_price(bars_sym, t0)
        p1 = nearest_price(bars_sym, t1)
        if p0 is None or p1 is None:
            continue

        ctx = ctx_by_symbol.get(sym)
        prior = None
        if ctx is not None:
            eligible = ctx[ctx["date"] < pd.Timestamp(t0.date())]
            if not eligible.empty:
                prior = eligible.iloc[-1]
        if prior is None:
            continue

        ret = (p1 - p0) / p0
        rows.append({
            "published_at": t0,
            "sentiment_score": a["sentiment_score"],
            "n_tickers": a["n_tickers"],
            "corroborating_24h": a["corroborating_24h"],
            "mom_5d": prior["mom_5d"],
            "vol_20d": prior["vol_20d"],
            "volume_surprise": prior["volume_surprise"],
            "fwd_return": ret,
            "up": int(ret > 0),
        })
    return pd.DataFrame(rows)


def time_split(df, train_frac=0.7):
    df = df.sort_values("published_at")
    cutoff = df["published_at"].quantile(train_frac)
    return df[df["published_at"] <= cutoff], df[df["published_at"] > cutoff]


def fit_and_score(train, test, features, label):
    train = train.dropna(subset=features + ["up"])
    test = test.dropna(subset=features + ["up"])
    if train["up"].nunique() < 2 or test["up"].nunique() < 2:
        print(f"{label}: not enough of both classes, skipping")
        return None
    pipe = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(train[features], train["up"])
    proba = pipe.predict_proba(test[features])[:, 1]
    roc = roc_auc_score(test["up"], proba)
    pr = average_precision_score(test["up"], proba)
    print(f"\n{label}  (n_train={len(train)}, n_test={len(test)})")
    print(f"  ROC-AUC: {roc:.3f}   (0.5 = no better than chance)")
    print(f"  PR-AUC:  {pr:.3f}   (base rate = {test['up'].mean():.3f})")
    return {"roc_auc": roc, "pr_auc": pr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--max-tickers", type=int, default=1,
                     help="only articles tagged to this many tickers or fewer")
    ap.add_argument("--symbols", nargs="*")
    args = ap.parse_args()

    symbols = args.symbols or UNIVERSE

    with connect() as conn:
        minute_bars = load_minute_bars(conn, symbols)
        if minute_bars.empty:
            raise SystemExit("No minute bars for any symbol. Run bars.py --timeframe 1Min first.")
        start, end = minute_bars["ts"].min(), minute_bars["ts"].max()

        articles = load_articles(conn, symbols, start, end, args.max_tickers)
        daily_ctx = load_daily_context(conn, symbols)

    print(f"minute-bar coverage: {start} to {end}")
    print(f"{len(articles)} candidate articles in that window")

    df = build_dataset(articles, minute_bars, daily_ctx, args.horizon_min)
    if df.empty:
        raise SystemExit("No articles lined up with both a price and prior-day context. "
                          "Widen minute-bar coverage or lower --max-tickers.")

    print(f"{len(df)} articles usable after joining price + context "
          f"(up rate: {df['up'].mean():.1%})\n")

    if len(df) < 50:
        print(f"⚠ n={len(df)} is too small to trust — treat this as a plumbing check.\n")

    train, test = time_split(df)
    print(f"train: {train['published_at'].min()} to {train['published_at'].max()} ({len(train)})")
    print(f"test:  {test['published_at'].min()} to {test['published_at'].max()} ({len(test)})")

    market_features = ["mom_5d", "vol_20d", "volume_surprise"]
    event_features = market_features + ["sentiment_score", "n_tickers", "corroborating_24h"]

    r1 = fit_and_score(train, test, market_features, "Market-context only")
    r2 = fit_and_score(train, test, event_features, "Market + article sentiment")

    if r1 and r2:
        lift = r2["roc_auc"] - r1["roc_auc"]
        print(f"\nROC-AUC lift from adding sentiment: {lift:+.3f}")


if __name__ == "__main__":
    main()
