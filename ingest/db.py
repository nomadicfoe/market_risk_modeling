import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("DATABASE_URL", "postgresql://stockexp:stockexp@localhost:5433/stockexp")

ALPACA_KEY = os.getenv("ALPACA_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY or "",
    "APCA-API-SECRET-KEY": ALPACA_SECRET or "",
    "accept": "application/json",
}

MASSIVE_ACCESS_KEY = os.getenv("MASSIVE_ACCESS_KEY")
MASSIVE_SECRET_KEY = os.getenv("MASSIVE_SECRET_KEY")
MASSIVE_S3_ENDPOINT = "https://files.massive.com"
MASSIVE_BUCKET = "flatfiles"

UNIVERSE = [
    # original megacap tech
    "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA",
    # financials
    "JPM", "BAC",
    # energy
    "XOM", "CVX",
    # healthcare
    "UNH", "PFE",
    # industrials
    "CAT", "BA",
    # consumer
    "WMT", "DIS",
    # semis (not already covered)
    "INTC",
    # higher-volatility, less saturated coverage — good for class balance
    "PLTR", "SOFI", "RIVN", "COIN",
]
BENCHMARKS = ["SPY", "QQQ", "XLK"]


def connect():
    return psycopg.connect(DSN)


def init_schema():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "schema.sql")) as f:
        sql = f.read()
    with connect() as conn:
        conn.execute(sql)
        with conn.cursor() as cur:
            for sym in UNIVERSE:
                cur.execute(
                    "INSERT INTO tickers (symbol) VALUES (%s) ON CONFLICT DO NOTHING",
                    (sym,),
                )
            for sym in BENCHMARKS:
                cur.execute(
                    "INSERT INTO tickers (symbol, is_benchmark) VALUES (%s, TRUE) "
                    "ON CONFLICT DO NOTHING",
                    (sym,),
                )
        conn.commit()
    print("schema ready")


if __name__ == "__main__":
    init_schema()