-- Everything is stored in UTC. Convert to America/New_York only at display time.

CREATE TABLE IF NOT EXISTS tickers (
    symbol      TEXT PRIMARY KEY,
    name        TEXT,
    is_benchmark BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS bars (
    symbol      TEXT        NOT NULL REFERENCES tickers(symbol),
    ts          TIMESTAMPTZ NOT NULL,      -- bar OPEN time, UTC
    timeframe   TEXT        NOT NULL,      -- '1Day' or '1Min'
    open        NUMERIC(14,4) NOT NULL,
    high        NUMERIC(14,4) NOT NULL,
    low         NUMERIC(14,4) NOT NULL,
    close       NUMERIC(14,4) NOT NULL,
    volume      BIGINT      NOT NULL,
    trade_count BIGINT,
    vwap        NUMERIC(14,4),
    source      TEXT        NOT NULL,      -- 'alpaca_iex', 'massive_sip', etc.
    PRIMARY KEY (symbol, timeframe, ts)
);

-- BRIN is the right index for append-only time series. Tiny and fast for ranges.
CREATE INDEX IF NOT EXISTS bars_ts_brin ON bars USING BRIN (ts);

CREATE TABLE IF NOT EXISTS news (
    id           BIGINT      PRIMARY KEY,   -- provider id, keeps ingest idempotent
    published_at TIMESTAMPTZ NOT NULL,      -- UTC
    headline     TEXT        NOT NULL,
    summary      TEXT,
    author       TEXT,
    source       TEXT,
    url          TEXT,
    raw          JSONB                      -- keep the original payload
);

CREATE INDEX IF NOT EXISTS news_published_at_idx ON news (published_at);

-- One article can tag several tickers, so the link is its own table.
CREATE TABLE IF NOT EXISTS news_tickers (
    news_id BIGINT NOT NULL REFERENCES news(id) ON DELETE CASCADE,
    symbol  TEXT   NOT NULL,
    PRIMARY KEY (news_id, symbol)
);

CREATE INDEX IF NOT EXISTS news_tickers_symbol_idx ON news_tickers (symbol);

-- Tracks what has already been pulled so re-runs do not refetch everything.
CREATE TABLE IF NOT EXISTS ingest_log (
    kind       TEXT        NOT NULL,   -- 'bars' or 'news'
    symbol     TEXT        NOT NULL,
    timeframe  TEXT        NOT NULL DEFAULT '',
    covered_to TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (kind, symbol, timeframe)
);
