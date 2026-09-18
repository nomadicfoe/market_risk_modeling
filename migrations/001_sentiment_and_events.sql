-- Run once against an existing database to pick up sentiment + events
-- without dropping anything. New installs get this from schema.sql directly.

ALTER TABLE news ADD COLUMN IF NOT EXISTS sentiment_score NUMERIC(5,4);
ALTER TABLE news ADD COLUMN IF NOT EXISTS sentiment_label TEXT;
CREATE INDEX IF NOT EXISTS news_unscored_idx ON news (id) WHERE sentiment_score IS NULL;

CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL   PRIMARY KEY,
    symbol      TEXT        NOT NULL,
    event_time  TIMESTAMPTZ NOT NULL,
    event_type  TEXT        NOT NULL,
    title       TEXT,
    source      TEXT        NOT NULL,
    url         TEXT,
    raw         JSONB,
    UNIQUE (symbol, event_time, event_type, source)
);

CREATE INDEX IF NOT EXISTS events_symbol_time_idx ON events (symbol, event_time);
