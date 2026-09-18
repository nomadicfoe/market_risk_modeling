# Stock Event Explorer

Phase 1: pick a ticker, see the price chart, see the news that lines up with it.

## Setup

On Fedora, podman-compose works in place of docker compose.

```bash
podman-compose up -d          # or: docker compose up -d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Alpaca paper keys
python ingest/db.py           # creates tables, seeds the universe
python ingest/bars.py --years 2
streamlit run app/main.py
```

Alpaca paper account keys work for market data. You do not need to fund anything.

## Rules that keep this from rotting

- Every timestamp in Postgres is `timestamptz` in UTC. Convert to New York only
  in the display layer.
- Every ingest is idempotent. Re-running must never duplicate a row.
- Nothing downstream of `bars` and `news` knows which vendor the data came from.
  That is what makes the Massive swap cheap.

## Order of work

1. `ingest/news.py` (stub, write it)
2. Click-a-point-on-the-chart lookup: nearest bar plus news within a window
3. Benchmark overlay, normalized to 0% at window start
4. Unusual move detection: return z-score plus volume ratio versus QQQ
5. Swap price source to Massive minute bars via flat files
