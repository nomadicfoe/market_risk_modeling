"""Score news headlines with FinBERT (ProsusAI/finbert), a model fine-tuned
on financial text specifically so it doesn't treat ordinary finance words
("liability", "loss", "short") as generically negative the way a general
sentiment model would.

Runs locally, free, CPU is fine for headline-length text. First run
downloads the model (~400MB) from Hugging Face and caches it.

sentiment_score is prob(positive) - prob(negative), so it runs -1..1.
sentiment_label is FinBERT's own top class: positive / negative / neutral.
"""

import argparse

from transformers import pipeline

from db import connect

BATCH_SIZE = 32


def load_model():
    print("loading FinBERT (first run downloads ~400MB)...")
    return pipeline("text-classification", model="ProsusAI/finbert",
                     top_k=None, truncation=True)


def score_batch(model, texts):
    """Returns (score, label) per text. score = P(positive) - P(negative)."""
    results = model(texts, batch_size=BATCH_SIZE)
    out = []
    for scores in results:
        by_label = {r["label"]: r["score"] for r in scores}
        pos = by_label.get("positive", 0.0)
        neg = by_label.get("negative", 0.0)
        top_label = max(scores, key=lambda r: r["score"])["label"]
        out.append((pos - neg, top_label))
    return out


def fetch_unscored(conn, limit):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, headline FROM news WHERE sentiment_score IS NULL LIMIT %s",
            (limit,),
        )
        return cur.fetchall()


def store_scores(conn, ids, scores):
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE news SET sentiment_score=%s, sentiment_label=%s WHERE id=%s",
            [(s, lbl, i) for i, (s, lbl) in zip(ids, scores)],
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="max articles to score this run (default: all unscored)")
    args = ap.parse_args()

    model = load_model()
    total = 0

    with connect() as conn:
        while True:
            chunk_limit = min(args.limit - total, 1000) if args.limit else 1000
            if chunk_limit <= 0:
                break
            rows = fetch_unscored(conn, chunk_limit)
            if not rows:
                break
            ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]
            scores = score_batch(model, texts)
            store_scores(conn, ids, scores)
            total += len(rows)
            print(f"scored {total}...")

    print(f"\ndone — {total} articles scored")


if __name__ == "__main__":
    main()
