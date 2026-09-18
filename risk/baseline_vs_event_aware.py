"""The central experiment: does adding news/sentiment features improve
downside-risk prediction over a market-data-only baseline?

Target: will this symbol's 5-day cumulative abnormal return fall below -3%?
Two models, same algorithm, different feature sets, same time-based split
so neither sees the future.

This is Stage A/B's comparison row from the flagship plan, done as one
clean experiment rather than the full multi-model table. Extend the
MODELS dict below once this baseline is trustworthy.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MARKET_FEATURES = ["mom_1d", "mom_5d", "mom_20d", "vol_20d", "volume_surprise", "beta"]
EVENT_FEATURES = ["n_articles", "avg_sentiment", "min_sentiment", "had_event"]


def time_split(df, train_frac=0.7):
    """Split by date, not randomly — a random split would let the model
    train on days after the ones it's tested on, which is lookahead bias
    dressed up as cross-validation."""
    df = df.sort_values("date")
    cutoff = df["date"].quantile(train_frac)
    return df[df["date"] <= cutoff], df[df["date"] > cutoff]


def fit_and_score(train, test, features, label):
    train = train.dropna(subset=features + ["downside_5d"])
    test = test.dropna(subset=features + ["downside_5d"])

    if train["downside_5d"].nunique() < 2 or test["downside_5d"].nunique() < 2:
        print(f"{label}: not enough of both classes in this split, skipping")
        return None

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    pipe.fit(train[features], train["downside_5d"])
    proba = pipe.predict_proba(test[features])[:, 1]

    roc = roc_auc_score(test["downside_5d"], proba)
    pr = average_precision_score(test["downside_5d"], proba)
    brier = brier_score_loss(test["downside_5d"], proba)

    print(f"\n{label}  (n_train={len(train)}, n_test={len(test)}, "
          f"features={features})")
    print(f"  ROC-AUC: {roc:.3f}   (0.5 = no better than chance)")
    print(f"  PR-AUC:  {pr:.3f}   (baseline = {test['downside_5d'].mean():.3f}, "
          f"the positive rate)")
    print(f"  Brier:   {brier:.3f}   (lower is better, 0 = perfect)")
    return {"roc_auc": roc, "pr_auc": pr, "brier": brier}


def main():
    df = pd.read_parquet("features.parquet")
    train, test = time_split(df)

    print(f"train: {train['date'].min().date()} to {train['date'].max().date()} "
          f"({len(train)} rows)")
    print(f"test:  {test['date'].min().date()} to {test['date'].max().date()} "
          f"({len(test)} rows)")

    r1 = fit_and_score(train, test, MARKET_FEATURES, "Market-only baseline")
    r2 = fit_and_score(train, test, MARKET_FEATURES + EVENT_FEATURES, "Market + event-aware")

    if r1 and r2:
        lift = r2["roc_auc"] - r1["roc_auc"]
        print(f"\nROC-AUC lift from adding event features: {lift:+.3f}")
        if abs(lift) < 0.02:
            print("That's within noise for this data size — not a real effect either way yet.")


if __name__ == "__main__":
    main()
