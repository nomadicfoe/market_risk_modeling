"""Milestone 1: pick a ticker, see the chart, see the news underneath it.
Milestone 2: click a point, see what happened.
Milestone 3: sentiment, relevance filtering, corporate events.
"""

import datetime as dt
import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest"))
from db import connect, UNIVERSE  # noqa: E402

NY = "America/New_York"

st.set_page_config(page_title="Stock Event Explorer", layout="wide")

# Benzinga listicle/roundup formats that tag a company but aren't really
# about it. Extend this list as you spot more patterns in your own data.
ROUNDUP_PATTERNS = [
    r"^\d+ Stocks? (Moving|To Watch)",
    r"Stocks That Hit (New )?52-Week",
    r"^Benzinga['’]s (Bulls|Bears|Top)",
    r"^Insider (Buying|Selling)",
    r"Mid-Day (Gainers|Losers)",
    r"^Today['’]s Market Wrap",
]
_ROUNDUP_RE = re.compile("|".join(ROUNDUP_PATTERNS), re.IGNORECASE)


@st.cache_data(ttl=300)
def load_bars(symbol, start, end, timeframe="1Day"):
    with connect() as conn:
        df = pd.read_sql(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars
            WHERE symbol = %s AND timeframe = %s AND ts BETWEEN %s AND %s
            ORDER BY ts
            """,
            conn,
            params=(symbol, timeframe, start, end),
        )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        df["volume"] = df["volume"].astype(float)
    return df


@st.cache_data(ttl=300)
def load_news(symbol, start, end):
    """Pulls sentiment + a ticker count per article, so the UI can filter
    out broad multi-ticker roundups without a second round trip."""
    with connect() as conn:
        df = pd.read_sql(
            """
            SELECT n.id, n.published_at, n.headline, n.source, n.url,
                   n.sentiment_score, n.sentiment_label, cnt.n_tickers
            FROM news n
            JOIN news_tickers t ON t.news_id = n.id
            JOIN (SELECT news_id, COUNT(*) AS n_tickers
                  FROM news_tickers GROUP BY news_id) cnt
                 ON cnt.news_id = n.id
            WHERE t.symbol = %s AND n.published_at BETWEEN %s AND %s
            ORDER BY n.published_at DESC
            """,
            conn,
            params=(symbol, start, end),
        )
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_convert(NY)
        df["is_roundup"] = df["headline"].str.contains(_ROUNDUP_RE, regex=True)
    return df


@st.cache_data(ttl=300)
def load_events(symbol, start, end):
    with connect() as conn:
        df = pd.read_sql(
            """
            SELECT event_time, event_type, title, url
            FROM events
            WHERE symbol = %s AND event_time BETWEEN %s AND %s
            ORDER BY event_time
            """,
            conn,
            params=(symbol, start, end),
        )
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True).dt.tz_convert(NY)
    return df


def sentiment_badge(score, label):
    if score is None or pd.isna(score):
        return ""
    if label == "positive":
        return f"🟢 +{score:.2f}"
    if label == "negative":
        return f"🔴 {score:.2f}"
    return f"⚪ {score:.2f}"


def render_news_list(df, limit=100):
    if df.empty:
        st.caption("No articles match the current filters.")
        return
    for _, n in df.head(limit).iterrows():
        badge = sentiment_badge(n.get("sentiment_score"), n.get("sentiment_label"))
        st.markdown(
            f"{badge} **{n['published_at']:%b %d, %I:%M %p}** "
            f"[{n['headline']}]({n['url']}) &nbsp; "
            f"<span style='opacity:0.5'>{n.get('source', '')} · "
            f"{int(n['n_tickers'])} tickers tagged</span>",
            unsafe_allow_html=True,
        )


def nearest_bar_index(bars, target_date):
    day = pd.Timestamp(target_date, tz=NY)
    eligible = bars.index[bars["ts"].dt.date <= day.date()]
    if len(eligible) == 0:
        return bars.index[0]
    return eligible[-1]


def daily_summary(bars, target_date):
    idx = nearest_bar_index(bars, target_date)
    row = bars.loc[idx]
    prev = bars.loc[idx - 1] if idx > bars.index[0] else None

    st.markdown(f"### {row['ts']:%A, %b %d, %Y}")
    cols = st.columns(4)

    if prev is not None:
        chg = row["close"] - prev["close"]
        pct = chg / prev["close"] * 100
        cols[0].metric("Close", f"${row['close']:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
    else:
        cols[0].metric("Close", f"${row['close']:,.2f}")

    intraday_pct = (row["close"] - row["open"]) / row["open"] * 100
    cols[1].metric("Open -> Close", f"{intraday_pct:+.2f}%")

    window = bars[bars["ts"] < row["ts"]].tail(20)
    if not window.empty:
        avg_vol = window["volume"].mean()
        vol_ratio = row["volume"] / avg_vol if avg_vol else float("nan")
        cols[2].metric("Volume vs 20d avg", f"{vol_ratio:.1f}x")
    else:
        cols[2].metric("Volume vs 20d avg", "n/a")

    cols[3].metric("Range", f"${row['low']:,.2f} - ${row['high']:,.2f}")
    return row["ts"].date()


def intraday_window(bars, news, target_date):
    day_bars = bars[bars["ts"].dt.date == target_date].reset_index(drop=True)
    if day_bars.empty:
        st.caption("No minute bars for this day.")
        return

    times = day_bars["ts"].dt.strftime("%H:%M").tolist()
    lo, hi = st.select_slider(
        "Time window (America/New_York)",
        options=times,
        value=(times[0], times[-1]),
    )
    sel = day_bars[(day_bars["ts"].dt.strftime("%H:%M") >= lo)
                   & (day_bars["ts"].dt.strftime("%H:%M") <= hi)]
    if sel.empty:
        return

    start_px, end_px = sel["open"].iloc[0], sel["close"].iloc[-1]
    pct = (end_px - start_px) / start_px * 100
    vol = sel["volume"].sum()

    cols = st.columns(3)
    cols[0].metric(f"{lo} -> {hi}", f"${start_px:,.2f} -> ${end_px:,.2f}", f"{pct:+.2f}%")
    cols[1].metric("Volume in window", f"{vol:,.0f}")
    cols[2].metric("Bars in window", f"{len(sel)}")

    win_start = pd.Timestamp(f"{target_date} {lo}", tz=NY)
    win_end = pd.Timestamp(f"{target_date} {hi}", tz=NY)
    pad = pd.Timedelta(minutes=30)
    win_news = news[(news["published_at"] >= win_start - pad)
                     & (news["published_at"] <= win_end + pad)]

    st.markdown(f"**News within +/-30min of window ({len(win_news)})**")
    render_news_list(win_news, limit=30)


def whats_happening_here(bars, minute_bars, news, target_date, has_minutes):
    day = daily_summary(bars, target_date)

    if has_minutes:
        st.markdown("**Zoom into the day**")
        intraday_window(minute_bars, news, day)
        return

    day_start = pd.Timestamp(day, tz=NY)
    day_end = day_start + pd.Timedelta(days=1)
    day_news = news[(news["published_at"] >= day_start) & (news["published_at"] < day_end)]
    st.markdown(f"**News that day ({len(day_news)})** — run minute bars for a tighter window")
    render_news_list(day_news, limit=30)


symbol = st.sidebar.selectbox("Ticker", UNIVERSE)
timeframe = st.sidebar.radio("Bars", ["1Day", "1Min"], horizontal=True,
                              help="1Min needs: python ingest/bars.py --timeframe 1Min --days 5")
default_lookback = 90 if timeframe == "1Day" else 5
lookback_choices = [30, 90, 180, 365, 730] if timeframe == "1Day" else [1, 3, 5, 10, 30]
lookback = st.sidebar.selectbox("Window (days)", lookback_choices,
                                 index=lookback_choices.index(default_lookback))

st.sidebar.markdown("**News filters**")
max_tickers = st.sidebar.slider(
    "Max tickers tagged", 1, 15, 4,
    help="Articles tagged to many tickers are usually broad roundups, not company-specific news."
)
hide_roundups = st.sidebar.checkbox("Hide known roundup formats", value=True)
sentiment_filter = st.sidebar.radio(
    "Sentiment", ["All", "Positive only", "Negative only", "Non-neutral only"],
    index=0,
)

end = dt.datetime.now(dt.timezone.utc)
start = end - dt.timedelta(days=lookback)

bars = load_bars(symbol, start, end, timeframe=timeframe)
news_raw = load_news(symbol, start, end)
events = load_events(symbol, start, end)

# Apply relevance + sentiment filters once, here, so every panel below
# (chart markers, "what happened here", the full list) sees the same set.
news = news_raw.copy()
if not news.empty:
    news = news[news["n_tickers"] <= max_tickers]
    if hide_roundups:
        news = news[~news["is_roundup"]]
    if sentiment_filter == "Positive only":
        news = news[news["sentiment_label"] == "positive"]
    elif sentiment_filter == "Negative only":
        news = news[news["sentiment_label"] == "negative"]
    elif sentiment_filter == "Non-neutral only":
        news = news[news["sentiment_label"] != "neutral"]

st.title(symbol)

if bars.empty:
    st.warning("No bars. Run ingest/bars.py first.")
    st.stop()

bars = bars.reset_index(drop=True)

if not news_raw.empty and news_raw["sentiment_score"].isna().all():
    st.caption("No articles scored yet — run `python ingest/sentiment.py` to add sentiment.")
if news_raw.empty is False and len(news) < len(news_raw):
    st.caption(f"Showing {len(news)} of {len(news_raw)} articles after filters.")

change = bars["close"].iloc[-1] - bars["close"].iloc[0]
pct = change / bars["close"].iloc[0] * 100
st.metric("Close", f"${bars['close'].iloc[-1]:,.2f}", f"{change:+.2f} ({pct:+.2f}%)")

fig = go.Figure(
    go.Candlestick(
        x=bars["ts"],
        open=bars["open"],
        high=bars["high"],
        low=bars["low"],
        close=bars["close"],
        name=symbol,
    )
)

if not news.empty:
    if timeframe == "1Day":
        daily = bars.set_index("ts")["high"].resample("1D").max()
        counts = news.set_index("published_at").resample("1D").size()
        joined = pd.concat([daily, counts.rename("n")], axis=1).dropna()
        joined = joined[joined["n"] > 0]
        fig.add_trace(
            go.Scatter(
                x=joined.index,
                y=joined["high"] * 1.01,
                mode="markers",
                marker=dict(symbol="triangle-down", size=9, color="#1f77b4"),
                name="news",
                hovertext=[f"{int(n)} articles" for n in joined["n"]],
            )
        )
    else:
        top = bars["high"].max()
        colors = news["sentiment_label"].map(
            {"positive": "#2ca02c", "negative": "#d62728"}
        ).fillna("#1f77b4")
        fig.add_trace(
            go.Scatter(
                x=news["published_at"],
                y=[top * 1.01] * len(news),
                mode="markers",
                marker=dict(symbol="triangle-down", size=8, color=colors),
                name="news",
                hovertext=news["headline"],
            )
        )

if not events.empty:
    top = bars["high"].max()
    fig.add_trace(
        go.Scatter(
            x=events["event_time"],
            y=[top * 1.03] * len(events),
            mode="markers",
            marker=dict(symbol="diamond", size=9, color="#ff7f0e"),
            name="corporate event",
            hovertext=events["title"],
        )
    )

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=520,
    margin=dict(t=20, b=20),
    clickmode="event+select",
)

event_sel = st.plotly_chart(
    fig, use_container_width=True, on_select="rerun", key="price_chart"
)

clicked_date = None
if event_sel and event_sel.get("selection", {}).get("points"):
    point = event_sel["selection"]["points"][0]
    clicked_date = pd.Timestamp(point["x"]).date()

if not events.empty:
    st.caption("🔶 corporate events (dividends, splits, filings) — hover for detail")

st.divider()
st.subheader("What happened here?")

default_date = clicked_date or bars["ts"].iloc[-1].date()
picked = st.date_input(
    "Or pick a day",
    value=default_date,
    min_value=bars["ts"].iloc[0].date(),
    max_value=bars["ts"].iloc[-1].date(),
)

has_minutes = timeframe == "1Min"
whats_happening_here(bars, bars if has_minutes else None, news, picked, has_minutes)

if not events.empty:
    day_events = events[events["event_time"].dt.date == picked]
    if not day_events.empty:
        st.markdown("**Corporate events this day**")
        for _, e in day_events.iterrows():
            st.markdown(f"- **{e['event_type']}**: {e['title']}"
                        + (f" [link]({e['url']})" if e["url"] else ""))

st.divider()
st.subheader(f"All news in window ({len(news)} shown of {len(news_raw)} total)")
if news.empty:
    st.info("No news matches the current filters. Run ingest/news.py and ingest/sentiment.py.")
else:
    render_news_list(news, limit=100)