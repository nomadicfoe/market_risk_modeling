"""Does sentiment sign actually predict return direction?

For every scored, ticker-specific article, find the nearest bar at or
before publish time and the nearest bar N minutes after, compute the
return, and check whether sign(sentiment) == sign(return).

This is the honest test. If the hit rate isn't meaningfully above 50%,
sentiment isn't a signal on its own here — that's a real, useful finding,
not a failure of the script.

Usage:
    python analysis/signal_check.py --symbol AAPL --horizon-min 30
    python analysis/signal_check.py --horizon-min 60 --min-tickers 1 --max-tickers 3
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest"))
from db import connect  # noqa: E402


def load_scored_news(conn, symbol, min_tickers, max_tickers, start, end):
    return pd.read_sql(
        """
        SELECT n.id, n.published_at, n.headline, n.sentiment_score, n.sentiment_label,
               t.n_tickers
        FROM news n
        JOIN news_tickers nt ON nt.news_id = n.id
        JOIN (SELECT news_id, COUNT(*) AS n_tickers FROM news_tickers GROUP BY news_id) t
             ON t.news_id = n.id
        WHERE nt.symbol = %s
          AND n.sentiment_score IS NOT NULL
          AND n.sentiment_label != 'neutral'
          AND t.n_tickers BETWEEN %s AND %s
          AND n.published_at BETWEEN %s AND %s
        ORDER BY n.published_at
        """,
        conn,
        params=(symbol, min_tickers, max_tickers, start, end),
    )


def load_minute_bars(conn, symbol):
    return pd.read_sql(
        "SELECT ts, close FROM bars WHERE symbol=%s AND timeframe='1Min' ORDER BY ts",
        conn,
        params=(symbol,),
    )


def nearest_price(bars, target_ts, tolerance_minutes=15):
    """Closest bar at or after target_ts, within tolerance. None if too far."""
    after = bars[bars["ts"] >= target_ts]
    if after.empty:
        return None
    row = after.iloc[0]
    if (row["ts"] - target_ts).total_seconds() > tolerance_minutes * 60:
        return None
    return row["close"]


def evaluate(news, bars, horizon_min):
    rows = []
    for _, a in news.iterrows():
        t0 = a["published_at"]
        t1 = t0 + pd.Timedelta(minutes=horizon_min)
        p0 = nearest_price(bars, t0)
        p1 = nearest_price(bars, t1)
        if p0 is None or p1 is None:
            continue
        ret = (p1 - p0) / p0
        rows.append({
            "published_at": t0,
            "headline": a["headline"],
            "sentiment_score": a["sentiment_score"],
            "sentiment_label": a["sentiment_label"],
            "fwd_return": ret,
            "correct_direction": (a["sentiment_score"] > 0) == (ret > 0),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--min-tickers", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=3,
                     help="exclude broad roundups tagged to many tickers")
    ap.add_argument("--start", type=str, default=None, help="ISO date, e.g. 2026-07-01")
    ap.add_argument("--end", type=str, default=None, help="ISO date")
    args = ap.parse_args()

    with connect() as conn:
        bars = load_minute_bars(conn, args.symbol)
        if bars.empty:
            raise SystemExit(f"No minute bars for {args.symbol}. Run bars.py --timeframe 1Min first.")
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)

        # Default to the news window matching whatever minute-bar coverage
        # you actually have, so the join isn't silently starved.
        start = pd.Timestamp(args.start, tz="UTC") if args.start else bars["ts"].min()
        end = pd.Timestamp(args.end, tz="UTC") if args.end else bars["ts"].max()

        news = load_scored_news(conn, args.symbol, args.min_tickers, args.max_tickers, start, end)

    if news.empty:
        raise SystemExit(f"No scored, non-neutral news for {args.symbol} between "
                          f"{start.date()} and {end.date()}. Widen minute-bar coverage "
                          f"(bars.py --timeframe 1Min --days N) or pass --start/--end.")

    result = evaluate(news, bars, args.horizon_min)
    if result.empty:
        raise SystemExit("No articles lined up with a bar within tolerance. "
                          "Try a wider --horizon-min or check minute-bar coverage.")

    hit_rate = result["correct_direction"].mean()
    n = len(result)
    avg_ret_when_pos = result.loc[result["sentiment_score"] > 0, "fwd_return"].mean()
    avg_ret_when_neg = result.loc[result["sentiment_score"] < 0, "fwd_return"].mean()

    print(f"\n{args.symbol}  —  {n} articles, {start.date()} to {end.date()}, "
          f"+{args.horizon_min}min horizon, tagged to {args.min_tickers}-{args.max_tickers} tickers\n")
    if n < 30:
        print(f"⚠ n={n} is too small to draw any conclusion from — treat this as a plumbing "
              f"check, not a result. You want at least a few dozen, ideally 100+.\n")
    print(f"direction hit rate: {hit_rate:.1%}   (50% = sentiment adds nothing)")
    print(f"avg fwd return | positive sentiment: {avg_ret_when_pos:+.3%}")
    print(f"avg fwd return | negative sentiment: {avg_ret_when_neg:+.3%}")

    out_path = f"analysis_{args.symbol}_{args.horizon_min}min.csv"
    result.to_csv(out_path, index=False)
    print(f"\nfull results: {out_path}")


if __name__ == "__main__":
    main()