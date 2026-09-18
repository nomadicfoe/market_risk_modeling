"""Value at Risk, Expected Shortfall, and the Kupiec backtest.

The question this answers: if you say there's a 5% chance of a loss
bigger than VaR on any given day, does that actually happen about 5%
of the time? A VaR number nobody has backtested is just a guess with
decimal places.

Two VaR methods, both walk-forward and both backtested the same way:

  Historical VaR   -- empirical quantile of trailing daily returns
                      (250-day rolling window). No distributional
                      assumption, the standard baseline every risk
                      desk computes regardless of what else they use.

  Parametric VaR    -- uses the validated GJR-GARCH-t model from
  (GJR-GARCH-t)         volatility.py: daily one-step-ahead conditional
                      volatility (refit every ~21 trading days,
                      expanding window, params FIXED and the variance
                      recursion updated daily with realized returns in
                      between refits) combined with the model's own
                      fitted Student-t tail — not a generic normal
                      assumption, which would understate tail risk for
                      exactly the fat-tailed symbols the diagnostics
                      already flagged.

Expected Shortfall (ES) is reported alongside VaR for both methods —
the average loss GIVEN that VaR was breached, which is what actually
matters for tail risk (two VaRs can be identical while one has a much
worse ES behind it).

Kupiec proportion-of-failures test: for each VaR series, count how
many days the actual loss exceeded VaR, compare that count against
the expected count under the stated confidence level via a likelihood
ratio test. Closed-form ES formula and the Kupiec LR statistic were
both verified against numerical integration / known reference cases
before being used here — see chat history for the verification.

Usage:
    python risk/var_es.py
    python risk/var_es.py --alpha 0.01 --symbols AAPL NVDA
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as t_dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest"))
from db import connect, UNIVERSE  # noqa: E402

HIST_WINDOW = 250
REFIT_EVERY = 21

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


def load_returns(conn, symbols):
    bars = pd.read_sql(
        "SELECT symbol, ts::date AS date, close FROM bars "
        "WHERE symbol = ANY(%s) AND timeframe = '1Day' ORDER BY symbol, ts",
        conn, params=(symbols,),
    )
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["symbol", "date"])
    bars["ret"] = bars.groupby("symbol")["close"].pct_change()
    return bars


def historical_var(returns, window, alpha):
    """Empirical quantile of trailing returns, expressed as a positive
    loss number. NaN until `window` observations are available."""
    q = returns.rolling(window).quantile(alpha)
    return -q  # a loss, so sign-flipped to be positive


def historical_es(returns, window, alpha):
    """Mean of the trailing window's returns that fall at/below that
    window's own VaR quantile — the standard historical-simulation ES."""
    def es_of_window(w):
        thresh = np.quantile(w, alpha)
        tail = w[w <= thresh]
        return -tail.mean() if len(tail) else np.nan
    return returns.rolling(window).apply(es_of_window, raw=True)


def parametric_es_factor(alpha, nu):
    """Closed-form ES of a UNIT-VARIANCE Student-t at tail probability
    alpha, matching arch's internal standardization convention. Verified
    against numerical integration (see module docstring)."""
    t_alpha = t_dist.ppf(alpha, nu)
    pdf_at = t_dist.pdf(t_alpha, nu)
    es_raw = -(pdf_at / alpha) * (nu + t_alpha ** 2) / (nu - 1)
    scale = np.sqrt((nu - 2) / nu)
    return es_raw * scale


def walk_forward_garch_var_es(bars, train_cutoff, alpha, refit_every=REFIT_EVERY):
    """Daily one-step-ahead VaR and ES per symbol using GJR-GARCH-t.

    Parameters are refit on an EXPANDING window every `refit_every`
    trading days (same cadence as volatility.py). Between refits, the
    conditional-variance RECURSION is still updated daily using
    realized returns via arch's .fix() (parameters held constant, state
    updated) + forecast(horizon=1) — this is what makes daily VaR values
    meaningful for a Kupiec/Christoffersen-style backtest, rather than
    holding one stale forecast constant across a whole 21-day block the
    way volatility.py's 5-day RV forecast does (fine for that use case,
    wrong for this one).
    """
    var_out = pd.Series(np.nan, index=bars.index)
    es_out = pd.Series(np.nan, index=bars.index)
    sigma_out = pd.Series(np.nan, index=bars.index)
    if not HAS_ARCH:
        return var_out, es_out, sigma_out

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

            full_slice = g[g["date"] <= g.loc[block[-1], "date"]].dropna(subset=["ret"])
            n_train = len(train_slice)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    am_fit = arch_model(train_slice["ret"] * 100, vol="Garch",
                                         p=1, o=1, q=1, dist="t")
                    res = am_fit.fit(disp="off")

                    am_full = arch_model(full_slice["ret"] * 100, vol="Garch",
                                          p=1, o=1, q=1, dist="t")
                    fixed = am_full.fix(res.params)
                    fc = fixed.forecast(horizon=1, start=n_train, reindex=False)
            except Exception:
                continue

            sigma_path = np.sqrt(fc.variance.values[:, 0]) / 100
            mu = res.params.get("mu", 0.0) / 100
            nu = res.params["nu"]

            t_q = res.model.distribution.ppf([alpha], [nu])[0]
            es_factor = parametric_es_factor(alpha, nu)

            var_path = -(mu + sigma_path * t_q)
            es_path = -(mu + sigma_path * es_factor)

            var_out.loc[block] = var_path[:len(block)]
            es_out.loc[block] = es_path[:len(block)]
            sigma_out.loc[block] = sigma_path[:len(block)]

    return var_out, es_out, sigma_out


def kupiec_pof(n, x, p):
    """Kupiec proportion-of-failures likelihood ratio test.
    n=observations, x=exceptions (VaR breaches), p=expected exception
    rate (e.g. 0.05 for 95% VaR). Returns (LR statistic, p-value).
    Verified against known well/mis-calibrated reference cases.

    Computed in LOG SPACE throughout, not by exponentiating the raw
    likelihoods and dividing — (1-p)**(n-x) underflows to exactly 0.0
    once n is in the thousands (0.95**8000 is far below float64's
    smallest representable positive number), which produces a silent
    0/0 = nan for any large pooled sample. This bit a pooled ~8600-
    observation backtest on first use; per-symbol counts (n~375) never
    triggered it, which is why it wasn't caught by validation at n=250."""
    if n == 0:
        return np.nan, np.nan
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        pi_hat = x / n
        log_L_null = (n - x) * np.log(1 - p) + x * np.log(p)
        log_L_alt = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
        lr = -2 * (log_L_null - log_L_alt)
    return lr, chi2.sf(lr, df=1)


def christoffersen_independence(exceptions):
    """Christoffersen (1998) independence test: do VaR breaches cluster
    in time, rather than falling independently? Tests whether the
    breach sequence follows a first-order Markov chain with different
    transition probabilities (pi01 != pi11) against the null that
    breaches are i.i.d. (pi01 == pi11 == overall breach rate).

    This is the test that actually distinguishes "VaR is the wrong
    width on average" (what Kupiec catches) from "VaR is roughly the
    right width but badly timed, breaches come in runs" (what Kupiec
    CANNOT catch — a model that breaches 5 days in a row then stays
    quiet for a year can still pass Kupiec's count-only test).

    exceptions: 0/1 array in time order (sorted by date).
    Returns (LR stat, p-value), or None if there's nothing to test
    (fewer than 2 observations, or all exceptions/no exceptions so one
    of the transition probabilities can't be estimated).

    Verified: identical exception COUNTS, one scattered independently
    and one crammed into a contiguous block, correctly give p~0.20 vs
    p~1e-27 — see chat history for the verification.
    """
    exc = np.asarray(exceptions).astype(int)
    n = len(exc)
    if n < 2:
        return None
    trans = list(zip(exc[:-1], exc[1:]))
    n00 = sum(1 for a, b in trans if a == 0 and b == 0)
    n01 = sum(1 for a, b in trans if a == 0 and b == 1)
    n10 = sum(1 for a, b in trans if a == 1 and b == 0)
    n11 = sum(1 for a, b in trans if a == 1 and b == 1)
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return None
    pi01 = n01 / n0
    pi11 = n11 / n1
    pi_ = (n01 + n11) / (n0 + n1)

    def term(count, prob):
        return 0.0 if count == 0 else count * np.log(prob)

    logL0 = term(n00 + n10, 1 - pi_) + term(n01 + n11, pi_)
    logL1 = term(n00, 1 - pi01) + term(n01, pi01) + term(n10, 1 - pi11) + term(n11, pi11)
    lr = -2 * (logL0 - logL1)
    return lr, chi2.sf(lr, df=1)


def backtest_report(bars, ret_col, var_col, es_col, alpha, label):
    """Per-symbol exception counts, Kupiec test (is the COUNT right?),
    Christoffersen test (are breaches INDEPENDENT over time, or do they
    cluster?), and the combined conditional-coverage test (both at
    once, chi-square df=2) — plus a pooled Kupiec summary. Pooling
    across symbols only makes sense for the count-based Kupiec test;
    independence is inherently a property of one ordered time series,
    so Christoffersen/conditional-coverage stay per-symbol only."""
    print(f"\n--- {label} (alpha={alpha}, i.e. {int((1-alpha)*100)}% VaR) ---\n")
    print(f"{'symbol':8s} {'n':>5s} {'exc':>4s} {'rate':>7s} "
          f"{'Kupiec p':>9s} {'indep. p':>9s} {'cond.cov p':>10s}  verdict")

    total_n = total_x = 0
    n_reject = n_ok = 0
    n_clustered = 0
    for sym in sorted(bars["symbol"].unique()):
        d = bars[bars["symbol"] == sym].dropna(subset=[ret_col, var_col]).sort_values("date")
        if len(d) < 30:
            continue
        n = len(d)
        exc_series = (d[ret_col] < -d[var_col]).astype(int).to_numpy()
        exceptions = int(exc_series.sum())

        lr_pof, p_pof = kupiec_pof(n, exceptions, alpha)

        ind_result = christoffersen_independence(exc_series)
        if ind_result is None:
            p_ind_str, p_cc_str = "n/a", "n/a"
            clustered = False
        else:
            lr_ind, p_ind = ind_result
            lr_cc = lr_pof + lr_ind
            p_cc = chi2.sf(lr_cc, df=2)
            p_ind_str, p_cc_str = f"{p_ind:.3f}", f"{p_cc:.3f}"
            clustered = p_ind < 0.05

        if p_pof < 0.05:
            n_reject += 1
        else:
            n_ok += 1
        if clustered:
            n_clustered += 1

        verdict_bits = []
        if p_pof < 0.05:
            verdict_bits.append("wrong count")
        if clustered:
            verdict_bits.append("CLUSTERED")
        verdict = ", ".join(verdict_bits) if verdict_bits else "ok"

        total_n += n
        total_x += exceptions
        print(f"{sym:8s} {n:5d} {exceptions:4d} {exceptions/n:7.2%} "
              f"{p_pof:9.3f} {p_ind_str:>9s} {p_cc_str:>10s}  {verdict}")

    if total_n:
        lr, pval = kupiec_pof(total_n, total_x, alpha)
        print(f"\nPooled (Kupiec count only — independence isn't poolable across symbols): "
              f"{total_x} exceptions / {total_n} obs "
              f"({total_x/total_n:.3%}, expected {alpha:.1%})  "
              f"LR={lr:.3f} p={pval:.3f}  "
              f"{'REJECT' if pval < 0.05 else 'not rejected'}")
        print(f"Per-symbol: {n_ok} count-calibrated, {n_reject} wrong count, "
              f"{n_clustered} show significant breach clustering (p<0.05)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--alpha", type=float, default=0.05,
                     help="tail probability, e.g. 0.05 for 95%% VaR, 0.01 for 99%%")
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    if not HAS_ARCH:
        raise SystemExit("`arch` package not installed. pip install arch")

    symbols = args.symbols or UNIVERSE
    print(f"connecting, loading daily bars for {len(symbols)} symbols...", flush=True)
    with connect() as conn:
        bars = load_returns(conn, symbols)
    print(f"loaded {len(bars)} bars\n", flush=True)

    bars = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    cutoff = bars["date"].quantile(args.train_frac)

    print("computing historical VaR/ES (250-day rolling window)...", flush=True)
    bars["hist_var"] = bars.groupby("symbol")["ret"].transform(
        lambda r: historical_var(r, HIST_WINDOW, args.alpha))
    bars["hist_es"] = bars.groupby("symbol")["ret"].transform(
        lambda r: historical_es(r, HIST_WINDOW, args.alpha))

    print(f"computing walk-forward GJR-GARCH-t VaR/ES "
          f"(refit every {REFIT_EVERY} days, {len(symbols)} symbols)...", flush=True)
    garch_var, garch_es, garch_sigma = walk_forward_garch_var_es(bars, cutoff, args.alpha)
    bars["garch_var"] = garch_var
    bars["garch_es"] = garch_es
    bars["garch_sigma"] = garch_sigma
    print("done.\n", flush=True)

    test = bars[bars["date"] > cutoff]

    print("--- Forecasted vol vs. realized vol (is the model over-forecasting risk?) ---\n")
    print(f"{'symbol':8s} {'mean fcst sigma':>16s} {'realized std':>14s} {'ratio':>7s}  note")
    over_forecasting = 0
    for sym in sorted(test["symbol"].unique()):
        d = test[test["symbol"] == sym].dropna(subset=["garch_sigma", "ret"])
        if len(d) < 30:
            continue
        mean_fcst = d["garch_sigma"].mean()
        realized = d["ret"].std()
        ratio = mean_fcst / realized if realized else float("nan")
        note = ""
        if ratio > 1.15:
            note = "over-forecasting vol"
            over_forecasting += 1
        elif ratio < 0.87:
            note = "under-forecasting vol"
        print(f"{sym:8s} {mean_fcst:16.5f} {realized:14.5f} {ratio:7.2f}  {note}")
    print(f"\n{over_forecasting} of {test['symbol'].nunique()} symbols over-forecasting vol by >15%")

    backtest_report(test, "ret", "hist_var", "hist_es", args.alpha, "Historical VaR")
    backtest_report(test, "ret", "garch_var", "garch_es", args.alpha, "GJR-GARCH-t parametric VaR")

    print(f"\nAvg ES/VaR ratio (how much worse the tail is than VaR alone) "
          f"— historical: {(test['hist_es']/test['hist_var']).mean():.2f}x, "
          f"GJR-GARCH-t: {(test['garch_es']/test['garch_var']).mean():.2f}x")


if __name__ == "__main__":
    main()
