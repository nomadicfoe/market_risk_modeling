"""Pull OHLCV bars from Alpaca into Postgres.

This is the reference pattern for every ingest script in the project:
paginate, normalize timestamps to UTC, upsert on conflict, record coverage.

Swap this file out for a Massive flat-file loader later. Nothing downstream
should need to change, because everything downstream only reads the bars table.
"""

import argparse
import datetime as dt
import time

import requests

from db import ALPACA_HEADERS, UNIVERSE, BENCHMARKS, connect

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
SOURCE = "alpaca_iex"


def fetch_bars(symbol, start, end, timeframe="1Day"):
    """Yield raw bar dicts, following next_page_token until exhausted."""
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "split",
        "limit": 10000,
        "feed": "iex",
    }
    while True:
        r = requests.get(BARS_URL, params=params, headers=ALPACA_HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        payload = r.json()
        for bar in payload.get("bars", {}).get(symbol, []):
            yield bar
        token = payload.get("next_page_token")
        if not token:
            return
        params["page_token"] = token


def parse_ts(value):
    """Alpaca returns RFC-3339 with a trailing Z. Keep it tz-aware and in UTC."""
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def store(conn, symbol, timeframe, bars):
    rows = [
        (
            symbol,
            parse_ts(b["t"]),
            timeframe,
            b["o"],
            b["h"],
            b["l"],
            b["c"],
            b["v"],
            b.get("n"),
            b.get("vw"),
            SOURCE,
        )
        for b in bars
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO bars (symbol, ts, timeframe, open, high, low, close,
                              volume, trade_count, vwap, source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high,
                low = EXCLUDED.low, close = EXCLUDED.close,
                volume = EXCLUDED.volume, trade_count = EXCLUDED.trade_count,
                vwap = EXCLUDED.vwap, source = EXCLUDED.source
            """,
            rows,
        )
        cur.execute(
            """
            INSERT INTO ingest_log (kind, symbol, timeframe, covered_to)
            VALUES ('bars', %s, %s, %s)
            ON CONFLICT (kind, symbol, timeframe)
            DO UPDATE SET covered_to = GREATEST(ingest_log.covered_to, EXCLUDED.covered_to)
            """,
            (symbol, timeframe, max(r[1] for r in rows)),
        )
    conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2,
                     help="lookback in years, for daily bars")
    ap.add_argument("--days", type=int, default=None,
                     help="lookback in days — overrides --years, use this "
                          "for minute bars (e.g. --timeframe 1Min --days 5)")
    ap.add_argument("--timeframe", default="1Day")
    ap.add_argument("--symbols", nargs="*")
    args = ap.parse_args()

    end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)
    if args.days is not None:
        start = end - dt.timedelta(days=args.days)
    else:
        start = end - dt.timedelta(days=365 * args.years)
    symbols = args.symbols or (UNIVERSE + BENCHMARKS)

    with connect() as conn:
        for sym in symbols:
            bars = fetch_bars(sym, start, end, args.timeframe)
            n = store(conn, sym, args.timeframe, bars)
            print(f"{sym:6s} {args.timeframe}  {n} bars")


if __name__ == "__main__":
    main()