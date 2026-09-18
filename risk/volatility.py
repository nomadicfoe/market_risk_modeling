"""Stage C: 5-day realized volatility forecasting.

Central question: do market-state features (momentum, volume, beta)
improve volatility forecasts over standard historical/EWMA/GARCH
baselines?

Target: RV_{t+1:t+5} = sqrt(sum of squared daily returns over the NEXT
5 trading days) — realized volatility, not a return.

Baselines:
  - naive: trailing 20-day daily vol, scaled to a 5-day horizon
  - EWMA (RiskMetrics lambda=0.94): exponentially weighted vol
  - GARCH(1,1): walk-forward — refit per symbol on an expanding window
    every ~21 trading days (configurable via --refit-every) through the
    test period, rather than fit once and forecast blind. This is the
    fix for the fit-once simplification documented in
    experiments/experiments-2026-09-15.md, which likely caused GARCH's
    backwards regime pattern (better in calm, worse in stressed) in
    that run.

ML model: gradient boosting on the same market-state features used in
risk/features.py, predicting log(RV) since realized vol is strictly
positive and right-skewed.

Evaluation: MAE/RMSE on RV directly, plus QLIKE on the implied variance.
QLIKE matters more than MAE for vol forecasts specifically because it
penalizes UNDERforecasting risk more heavily than overforecasting it —
the asymmetry that actually matters when the forecast feeds into a risk
limit or a margin calculation.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, binomtest
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest"))
from db import connect, UNIVERSE  # noqa: E402

RV_HORIZON = 5
BETA_WINDOW = 60
EWMA_LAMBDA = 0.94  # RiskMetrics standard

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


def load_returns(conn, symbols):
    bars = pd.read_sql(
        "SELECT symbol, ts::date AS date, close, volume FROM bars "
        "WHERE symbol = ANY(%s) AND timeframe = '1Day' ORDER BY symbol, ts",
        conn, params=(symbols,),
    )
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["symbol", "date"])
    bars["ret"] = bars.groupby("symbol")["close"].pct_change()
    return bars


def build_panel(bars):
    """One row per symbol/date: market-state features + forward RV target."""
    rows = []
    for sym, g in bars.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True).copy()

        fwd_sq_sum = g["ret"].shift(-1).pow(2).rolling(RV_HORIZON).sum().shift(-(RV_HORIZON - 1))
        g["fwd_rv_5d"] = np.sqrt(fwd_sq_sum)

        g["hist_vol_20d"] = g["ret"].rolling(20).std()
        g["naive_fwd_vol"] = g["hist_vol_20d"] * np.sqrt(RV_HORIZON)

        ewma_var = np.full(len(g), np.nan)
        r2 = g["ret"].pow(2).values
        seed = np.nanmean(r2[:20]) if len(r2) >= 20 else np.nan
        prev = seed
        for i in range(len(g)):
            if np.isnan(r2[i]):
                ewma_var[i] = prev
                continue
            prev = EWMA_LAMBDA * (prev if not np.isnan(prev) else r2[i]) + (1 - EWMA_LAMBDA) * r2[i]
            ewma_var[i] = prev
        g["ewma_fwd_vol"] = np.sqrt(ewma_var * RV_HORIZON)

        g["mom_1d"] = g["ret"]
        g["mom_5d"] = g["close"].pct_change(5)
        g["mom_20d"] = g["close"].pct_change(20)
        g["vol_20d"] = g["hist_vol_20d"]
        g["volume_surprise"] = g["volume"] / g["volume"].rolling(20).mean()

        g["symbol"] = sym
        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def garch_forecast(bars, train_cutoff, refit_every=21, asymmetric=False):
    """Walk-forward GARCH per symbol: refit on an EXPANDING window
    (train start -> current point) every `refit_every` trading days
    through the test period, rather than fitting once and forecasting
    forward blind. Between refits, the same forecast is held constant —
    refitting at every single day (~8,600 fits total) would cost a lot
    of runtime for negligible extra accuracy over a ~monthly cadence,
    which is the standard practitioner compromise.

    asymmetric=True fits GJR-GARCH(1,1,1) with a Student-t distribution
    instead of plain GARCH(1,1) with normal. Two changes bundled together
    because both target the same weakness plain GARCH showed even after
    the walk-forward fix (worse in stressed regimes than calm — the
    opposite pattern from every other model): GJR's asymmetry term lets
    volatility react harder to negative shocks than positive ones (the
    leverage effect), and Student-t accounts for fatter tails than a
    normal distribution assumes — both differences matter specifically
    when markets are stressed, not when they're calm.

    Returns a Series aligned to the input index, NaN wherever a forecast
    couldn't be produced (too little training history, fit failure)."""
    out = pd.Series(np.nan, index=bars.index)
    if not HAS_ARCH:
        return out

    for sym, g in bars.groupby("symbol"):
        g = g.sort_values("date")
        test_idx = list(g.index[g["date"] > train_cutoff])
        if not test_idx:
            continue

        for start in range(0, len(test_idx), refit_every):
            block = test_idx[start:start + refit_every]
            first_date = g.loc[block[0], "date"]
            train_slice = g[g["date"] < first_date].dropna(subset=["ret"])
            if len(train_slice) < 100:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if asymmetric:
                        am = arch_model(train_slice["ret"] * 100, vol="Garch",
                                         p=1, o=1, q=1, dist="t")
                    else:
                        am = arch_model(train_slice["ret"] * 100, vol="Garch",
                                         p=1, q=1, dist="normal")
                    res = am.fit(disp="off")
                    fc = res.forecast(horizon=RV_HORIZON, reindex=False)
                    total_var = fc.variance.values[-1].sum() / (100 ** 2)
                    garch_rv = np.sqrt(total_var)
            except Exception:
                continue
            out.loc[block] = garch_rv

    return out


def qlike_loss_series(actual_rv, forecast_rv):
    """Per-observation QLIKE loss (not averaged) — the raw series a
    Diebold-Mariano test needs, as opposed to qlike() below which
    returns the mean."""
    av, fv = actual_rv ** 2, forecast_rv ** 2
    ratio = av / fv
    return (ratio - np.log(ratio) - 1).to_numpy()


def dm_test(loss_a, loss_b, horizon):
    """Diebold-Mariano test: is model A's average loss significantly
    different from model B's, given the two loss series come from the
    SAME ordered time series (same symbol, sorted by date)?

    Negative DM stat + small p-value => A has significantly LOWER loss
    (A wins). Positive => B wins.

    Uses a Newey-West-style HAC variance for the mean loss differential,
    with lag = horizon-1, because forecasts for a 5-day-ahead target
    made on consecutive days overlap and are therefore autocorrelated —
    treating them as independent would understate the variance and
    overstate significance. Includes the Harvey-Leybourne-Newbold
    small-sample correction, standard practice for DM at the sample
    sizes typical of a few years of daily data.

    Returns (dm_stat, p_value), or None if there isn't enough data.
    """
    d = loss_a - loss_b
    T = len(d)
    if T < 20:
        return None

    L = max(horizon - 1, 0)
    mean_d = d.mean()
    gamma0 = np.var(d, ddof=0)
    nw_var = gamma0
    for k in range(1, L + 1):
        if T <= k:
            break
        cov_k = np.cov(d[:-k], d[k:])[0, 1]
        nw_var += 2 * (1 - k / (L + 1)) * cov_k

    var_mean = nw_var / T
    if var_mean <= 0:
        return None

    dm_stat = mean_d / np.sqrt(var_mean)
    hln = np.sqrt(max((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T, 0))
    dm_adj = dm_stat * hln
    p_value = 2 * (1 - t_dist.cdf(abs(dm_adj), df=max(T - 1, 1)))
    return dm_adj, p_value


def per_symbol_report(test, symbols, model_a_col, model_b_col, model_a_name, model_b_name):
    """QLIKE per symbol for both models, plus a per-symbol DM test —
    answers 'does the winning model actually win broadly, or is a couple
    of volatile names dragging the pooled average?'"""
    print(f"\n--- Per-symbol: {model_a_name} vs {model_b_name} ---\n")
    print(f"{'symbol':8s} {'n':>5s} {model_a_name[:12]:>14s} {model_b_name[:12]:>14s}  "
          f"{'DM stat':>9s} {'p-value':>9s}  winner")

    a_wins = b_wins = no_sig = skipped = 0
    dm_favors_a = dm_favors_b = 0  # direction only, regardless of significance
    for sym in sorted(symbols):
        d = test[test["symbol"] == sym].dropna(subset=["fwd_rv_5d", model_a_col, model_b_col])
        d = d.sort_values("date")
        if len(d) < 20:
            skipped += 1
            continue

        ql_a = qlike(d["fwd_rv_5d"], d[model_a_col])
        ql_b = qlike(d["fwd_rv_5d"], d[model_b_col])

        loss_a = qlike_loss_series(d["fwd_rv_5d"], d[model_a_col])
        loss_b = qlike_loss_series(d["fwd_rv_5d"], d[model_b_col])
        result = dm_test(loss_a, loss_b, RV_HORIZON)

        if result is None:
            print(f"{sym:8s} {len(d):5d} {ql_a:14.4f} {ql_b:14.4f}  {'n/a':>9s} {'n/a':>9s}  —")
            continue

        dm_stat, p_value = result
        if dm_stat < 0:
            dm_favors_a += 1
        elif dm_stat > 0:
            dm_favors_b += 1

        if p_value < 0.05:
            winner = model_a_name if dm_stat < 0 else model_b_name
            if dm_stat < 0:
                a_wins += 1
            else:
                b_wins += 1
        else:
            winner = "no sig. diff."
            no_sig += 1

        print(f"{sym:8s} {len(d):5d} {ql_a:14.4f} {ql_b:14.4f}  {dm_stat:9.3f} {p_value:9.3f}  {winner}")

    print(f"\nSummary: {model_a_name} sig. better in {a_wins} symbols, "
          f"{model_b_name} sig. better in {b_wins}, no significant difference in {no_sig} "
          f"({skipped} skipped — too few observations)")

    n_valid = dm_favors_a + dm_favors_b
    if n_valid >= 5:
        sign_result = binomtest(dm_favors_a, n=n_valid, p=0.5, alternative="two-sided")
        print(f"\nSign test across {n_valid} symbols: {model_a_name} favored in "
              f"{dm_favors_a}, {model_b_name} favored in {dm_favors_b} "
              f"(by DM stat direction, regardless of individual significance)")
        print(f"  p-value = {sign_result.pvalue:.4f}  "
              f"(H0: each symbol is a coin flip between the two models)")
        if sign_result.pvalue < 0.05:
            leader = model_a_name if dm_favors_a > dm_favors_b else model_b_name
            print(f"  -> the tilt toward {leader} is broader than chance would produce, "
                  f"even though most individual symbols aren't significant on their own.")
        else:
            print("  -> no significant directional tilt across the universe.")


def qlike(actual_rv, forecast_rv):
    av, fv = actual_rv ** 2, forecast_rv ** 2
    ratio = av / fv
    return float(np.mean(ratio - np.log(ratio) - 1))


def evaluate(df, col, label, split_by_regime=False):
    d = df.dropna(subset=["fwd_rv_5d", col])
    if d.empty:
        print(f"{label}: no rows to evaluate")
        return None
    mae = float((d["fwd_rv_5d"] - d[col]).abs().mean())
    rmse = float(np.sqrt(((d["fwd_rv_5d"] - d[col]) ** 2).mean()))
    ql = qlike(d["fwd_rv_5d"], d[col])
    print(f"{label:22s}  MAE={mae:.5f}  RMSE={rmse:.5f}  QLIKE={ql:.4f}  (n={len(d)})")

    if split_by_regime:
        med = d["vol_20d"].median()
        calm = d[d["vol_20d"] <= med]
        stressed = d[d["vol_20d"] > med]
        for name, sub in [("  calm regime", calm), ("  stressed regime", stressed)]:
            if sub.empty:
                continue
            ql_sub = qlike(sub["fwd_rv_5d"], sub[col])
            print(f"{name:22s}  QLIKE={ql_sub:.4f}  (n={len(sub)})")

    return {"mae": mae, "rmse": rmse, "qlike": ql}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--refit-every", type=int, default=21,
                     help="trading days between GARCH refits (default: ~monthly)")
    ap.add_argument("--skip-gjr", action="store_true",
                     help="skip the asymmetric GJR-GARCH-t model (roughly doubles GARCH runtime)")
    ap.add_argument("--skip-dm", action="store_true",
                     help="skip the per-symbol Diebold-Mariano significance test")
    args = ap.parse_args()

    symbols = args.symbols or UNIVERSE
    print(f"connecting, loading daily bars for {len(symbols)} symbols...", flush=True)
    with connect() as conn:
        bars = load_returns(conn, symbols)
    print(f"loaded {len(bars)} bars — building features...", flush=True)

    panel = build_panel(bars)
    panel = panel.sort_values("date")
    cutoff = panel["date"].quantile(args.train_frac)
    train, test = panel[panel["date"] <= cutoff], panel[panel["date"] > cutoff]

    print(f"train: {train['date'].min().date()} to {train['date'].max().date()} ({len(train)})")
    print(f"test:  {test['date'].min().date()} to {test['date'].max().date()} ({len(test)})\n")

    if not HAS_ARCH:
        print("`arch` package not installed — skipping GARCH baselines. "
              "pip install arch to include them.\n")
    else:
        n_blocks_approx = max(1, len(test) // len(symbols) // args.refit_every)
        print(f"walk-forward GARCH(1,1): refitting every {args.refit_every} trading days "
              f"(~{n_blocks_approx} refits/symbol, {len(symbols)} symbols)...", flush=True)
        test = test.copy()
        test["garch_fwd_vol"] = garch_forecast(panel, cutoff, args.refit_every).loc[test.index]
        print("GARCH done.", flush=True)

        if not args.skip_gjr:
            print(f"walk-forward GJR-GARCH(1,1,1) + Student-t: same refit schedule "
                  f"(~{n_blocks_approx} refits/symbol)...", flush=True)
            test["gjr_fwd_vol"] = garch_forecast(
                panel, cutoff, args.refit_every, asymmetric=True
            ).loc[test.index]
            print("GJR-GARCH done.", flush=True)
        print()

    features = ["mom_1d", "mom_5d", "mom_20d", "vol_20d", "volume_surprise"]
    train_ml = train.dropna(subset=features + ["fwd_rv_5d"])
    test_ml = test.dropna(subset=features + ["fwd_rv_5d"])

    model = GradientBoostingRegressor(random_state=0)
    model.fit(train_ml[features], np.log(train_ml["fwd_rv_5d"].clip(lower=1e-6)))
    test = test.copy()
    test.loc[test_ml.index, "ml_fwd_vol"] = np.exp(model.predict(test_ml[features]))

    print("--- Results on test set ---\n")
    evaluate(test, "naive_fwd_vol", "Naive (20d, scaled)", split_by_regime=True)
    evaluate(test, "ewma_fwd_vol", "EWMA (RiskMetrics)", split_by_regime=True)
    if HAS_ARCH:
        evaluate(test, "garch_fwd_vol", "GARCH(1,1)", split_by_regime=True)
        if not args.skip_gjr:
            evaluate(test, "gjr_fwd_vol", "GJR-GARCH(1,1,1) + t", split_by_regime=True)
    evaluate(test, "ml_fwd_vol", "Gradient boosting (market feats)", split_by_regime=True)

    if HAS_ARCH and not args.skip_gjr and not args.skip_dm:
        per_symbol_report(test, symbols, "gjr_fwd_vol", "ewma_fwd_vol",
                           "GJR-GARCH-t", "EWMA")


if __name__ == "__main__":
    main()
