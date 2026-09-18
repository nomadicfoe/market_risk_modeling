"""Pull minute bars from Massive's S3-compatible flat files into Postgres.

Same table as bars.py (timeframe='1Min' instead of '1Day'), same upsert
pattern, but the source is a daily gzipped CSV covering the whole market
rather than a paginated REST call. One file per trading day, filtered down
to your universe after download.

Requires a paid Massive plan (Starter or above) — flat files are gated,
even though *listing* the bucket works on any account. Get S3-style
Access/Secret keys from the Massive dashboard, not your REST API key,
they are a different credential pair.

IMPORTANT — unadjusted prices:
Flat files are NOT split-adjusted. bars.py pulled Alpaca data with
adjustment=split, so if you compare the two sources on a ticker that has
split since your Alpaca backfill, the numbers will not line up. For now
this project only stores minute bars for the recent window (see --days
below), so it is unlikely to cross a split boundary, but do not blindly
merge historical minute data spanning a split without adjusting first.
"""

import argparse
import datetime as dt
import gzip
import io
import time

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError

from db import (
    MASSIVE_ACCESS_KEY,
    MASSIVE_SECRET_KEY,
    MASSIVE_S3_ENDPOINT,
    MASSIVE_BUCKET,
    UNIVERSE,
    BENCHMARKS,
    connect,
)

SOURCE = "massive_sip"


def s3_client():
    session = boto3.Session(
        aws_access_key_id=MASSIVE_ACCESS_KEY,
        aws_secret_access_key=MASSIVE_SECRET_KEY,
    )
    return session.client(
        "s3",
        endpoint_url=MASSIVE_S3_ENDPOINT,
        config=Config(signature_version="s3v4"),
    )


def object_key(date):
    return f"us_stocks_sip/minute_aggs_v1/{date:%Y}/{date:%m}/{date:%Y-%m-%d}.csv.gz"


def trading_days(start, end):
    """Every calendar weekday in range. Market holidays just won't have a
    file, and we skip those with a clean 404 rather than a real error."""
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def fetch_day(client, date, symbols):
    """Download one day's flat file and return rows for our symbols only.

    Returns None if the file doesn't exist (holiday, or too recent —
    flat files land ~11am ET the following day).
    """
    key = object_key(date)
    try:
        obj = client.get_object(Bucket=MASSIVE_BUCKET, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            return None
        raise

    raw = gzip.decompress(obj["Body"].read())
    df = pd.read_csv(io.BytesIO(raw))
    df = df[df["ticker"].isin(symbols)]
    return df


def rows_from_df(symbol_df):
    """window_start is nanoseconds since the Unix epoch, UTC."""
    for _, r in symbol_df.iterrows():
        ts = pd.Timestamp(int(r["window_start"]), unit="ns", tz="UTC").to_pydatetime()
        yield (
            r["ticker"],
            ts,
            "1Min",
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            int(r["volume"]),
            int(r["transactions"]) if not pd.isna(r["transactions"]) else None,
            None,  # flat files don't carry vwap in this schema
            SOURCE,
        )


def store(conn, rows):
    rows = list(rows)
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
                source = EXCLUDED.source
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60,
                     help="how many calendar days back to pull (default 60 — "
                          "minute data is heavy, start small)")
    ap.add_argument("--symbols", nargs="*")
    args = ap.parse_args()

    if not MASSIVE_ACCESS_KEY or not MASSIVE_SECRET_KEY:
        raise SystemExit(
            "MASSIVE_ACCESS_KEY / MASSIVE_SECRET_KEY not set in .env. "
            "These are S3 keys from the Massive dashboard, not your REST API key."
        )

    symbols = set(args.symbols or (UNIVERSE + BENCHMARKS))
    end = dt.date.today() - dt.timedelta(days=1)  # yesterday: today's file may not exist yet
    start = end - dt.timedelta(days=args.days)

    client = s3_client()
    total = 0
    missing = 0

    with connect() as conn:
        for day in trading_days(start, end):
            df = fetch_day(client, day, symbols)
            if df is None:
                missing += 1
                continue
            if df.empty:
                continue
            n = store(conn, rows_from_df(df))
            total += n
            print(f"{day:%Y-%m-%d}  {n:5d} bars")
            time.sleep(0.1)  # be polite, this is a lot of sequential GETs

    print(f"\n{total} bars stored, {missing} days had no file (holidays/weekends/too recent)")


if __name__ == "__main__":
    main()