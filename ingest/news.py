"""Pull news from Alpaca into Postgres.

Same shape as bars.py: paginate, normalize timestamps to UTC, upsert,
record coverage. One difference that matters: an article can tag several
tickers, so it lands in two tables.
"""

import argparse
import datetime as dt
import json
import time

import requests

from db import ALPACA_HEADERS, UNIVERSE, BENCHMARKS, connect

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
PAGE_LIMIT = 50  # hard cap on this endpoint


def fetch_news(symbols, start, end, include_content=False):
    """Yield raw article dicts for all symbols at once, oldest first.

    Requesting every symbol in one call rather than looping per ticker cuts
    the page count by roughly the number of symbols, and articles that tag
    several of them come back once instead of N times.
    """
    params = {
        "symbols": ",".join(symbols),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": PAGE_LIMIT,
        "sort": "asc",
        "include_content": str(include_content).lower(),
    }
    pages = 0
    while True:
        r = requests.get(NEWS_URL, params=params, headers=ALPACA_HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        payload = r.json()

        articles = payload.get("news", [])
        for article in articles:
            yield article

        pages += 1
        if pages % 20 == 0:
            print(f"  ...{pages} pages")

        token = payload.get("next_page_token")
        if not token:
            return
        params["page_token"] = token


def parse_ts(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def store(conn, articles, batch_size=500):
    """Insert articles and their ticker links. Returns (articles, links)."""
    n_articles = 0
    n_links = 0
    buf = []

    def flush(rows):
        nonlocal n_articles, n_links
        if not rows:
            return
        news_rows = []
        link_rows = []
        for a in rows:
            news_rows.append(
                (
                    a["id"],
                    # created_at, NOT updated_at. updated_at moves when the
                    # wire edits a headline, which would silently shift the
                    # event time you are aligning prices against.
                    parse_ts(a["created_at"]),
                    a.get("headline", ""),
                    a.get("summary"),
                    a.get("author"),
                    a.get("source"),
                    a.get("url"),
                    json.dumps(a),
                )
            )
            for sym in a.get("symbols", []):
                link_rows.append((a["id"], sym))

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO news (id, published_at, headline, summary,
                                  author, source, url, raw)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO UPDATE SET
                    headline = EXCLUDED.headline,
                    summary  = EXCLUDED.summary,
                    url      = EXCLUDED.url,
                    raw      = EXCLUDED.raw
                """,
                news_rows,
            )
            cur.executemany(
                """
                INSERT INTO news_tickers (news_id, symbol)
                VALUES (%s,%s)
                ON CONFLICT DO NOTHING
                """,
                link_rows,
            )
        conn.commit()
        n_articles += len(news_rows)
        n_links += len(link_rows)

    for article in articles:
        buf.append(article)
        if len(buf) >= batch_size:
            flush(buf)
            buf = []
    flush(buf)

    return n_articles, n_links


def record_coverage(conn, symbols, covered_to):
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO ingest_log (kind, symbol, timeframe, covered_to)
            VALUES ('news', %s, '', %s)
            ON CONFLICT (kind, symbol, timeframe)
            DO UPDATE SET covered_to = GREATEST(ingest_log.covered_to,
                                                EXCLUDED.covered_to)
            """,
            [(s, covered_to) for s in symbols],
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--content", action="store_true",
                    help="also store the article body (much larger)")
    args = ap.parse_args()

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=365 * args.years)
    symbols = args.symbols or (UNIVERSE + BENCHMARKS)

    print(f"news {start:%Y-%m-%d} to {end:%Y-%m-%d} for {len(symbols)} symbols")

    with connect() as conn:
        articles = fetch_news(symbols, start, end, include_content=args.content)
        n_articles, n_links = store(conn, articles)
        record_coverage(conn, symbols, end)

    print(f"{n_articles} articles, {n_links} ticker links")


if __name__ == "__main__":
    main()