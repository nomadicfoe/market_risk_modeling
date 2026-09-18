"""Corporate events: dividends/splits/mergers/spinoffs from Alpaca, plus
8-K/10-Q/10-K filing dates from SEC EDGAR.

Alpaca corporate actions use your existing REST keys, no new signup.
SEC EDGAR needs no key at all, just a User-Agent header identifying you
(they will rate-limit or block generic/missing ones).
"""

import argparse
import datetime as dt
import time

import requests

from db import ALPACA_HEADERS, UNIVERSE, connect

CORP_ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SEC asks that you identify yourself. Put a real contact here.
SEC_HEADERS = {"User-Agent": "stockexp research project sumanthpaila1@gmail.com"}

FILING_TYPES = {"8-K", "10-Q", "10-K"}


# ---------------------------------------------------------------- Alpaca --

def fetch_corporate_actions(symbols, start, end):
    params = {
        "symbols": ",".join(symbols),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "limit": 1000,
    }
    while True:
        r = requests.get(CORP_ACTIONS_URL, params=params, headers=ALPACA_HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        payload = r.json()
        actions = payload.get("corporate_actions", {})
        # Response is grouped by action type, e.g. {"dividends": [...], "forward_splits": [...]}
        for kind, rows in actions.items():
            for row in rows:
                yield kind, row
        token = payload.get("next_page_token")
        if not token:
            return
        params["page_token"] = token


def parse_alpaca_date(row, kind):
    """Different action types key their date field differently."""
    for key in ("ex_date", "process_date", "payable_date", "declaration_date"):
        if row.get(key):
            return dt.datetime.fromisoformat(row[key]).replace(tzinfo=dt.timezone.utc)
    return None


ACTION_TYPE_MAP = {
    "dividends": "dividend",
    "forward_splits": "forward_split",
    "reverse_splits": "reverse_split",
    "mergers": "merger",
    "spin_offs": "spinoff",
}


def store_corporate_actions(conn, rows):
    out = []
    for kind, row in rows:
        event_type = ACTION_TYPE_MAP.get(kind, kind)
        symbol = row.get("symbol") or row.get("target_symbol")
        ts = parse_alpaca_date(row, kind)
        if not symbol or not ts:
            continue
        title = f"{event_type.replace('_', ' ').title()}: {symbol}"
        out.append((symbol, ts, event_type, title, "alpaca_corp_actions", None, None))
    return _upsert_events(conn, out)


# --------------------------------------------------------------- EDGAR ----

def sec_cik_map():
    """ticker -> zero-padded CIK. Cached in memory for the process lifetime."""
    r = requests.get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {row["ticker"]: row["cik_str"] for row in data.values()}


def fetch_filings(symbol, cik, start, end):
    r = requests.get(SEC_SUBMISSIONS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=30)
    if r.status_code == 404:
        return
    r.raise_for_status()
    recent = r.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accns = recent.get("accessionNumber", [])
    for form, date_str, accn in zip(forms, dates, accns):
        if form not in FILING_TYPES:
            continue
        filed = dt.datetime.fromisoformat(date_str).replace(tzinfo=dt.timezone.utc)
        if not (start <= filed <= end):
            continue
        accn_nodash = accn.replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
               f"{accn_nodash}/{accn}-index.htm")
        yield symbol, filed, form, f"{form} filing: {symbol}", "sec_edgar", url, None


def _upsert_events(conn, rows):
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO events (symbol, event_time, event_type, title, source, url, raw)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, event_time, event_type, source) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--symbols", nargs="*")
    args = ap.parse_args()

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=365 * args.years)
    symbols = args.symbols or UNIVERSE

    with connect() as conn:
        n = store_corporate_actions(conn, fetch_corporate_actions(symbols, start, end))
        print(f"corporate actions: {n} rows")

        print("fetching SEC ticker->CIK map...")
        cik_map = sec_cik_map()

        total_filings = 0
        for sym in symbols:
            cik = cik_map.get(sym)
            if cik is None:
                print(f"{sym:6s}  no CIK found, skipping")
                continue
            rows = list(fetch_filings(sym, cik, start, end))
            n = _upsert_events(conn, rows)
            total_filings += n
            print(f"{sym:6s}  {n} filings")
            time.sleep(0.15)  # SEC asks for <=10 req/sec; stay well under

        print(f"\n{total_filings} filings stored")


if __name__ == "__main__":
    main()
