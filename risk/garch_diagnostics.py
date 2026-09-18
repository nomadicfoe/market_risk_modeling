"""Specification diagnostics for the GJR-GARCH(1,1,1) + Student-t model.

Forecast accuracy (QLIKE) -> statistical comparison (Diebold-Mariano,
sign test) -> THIS: specification diagnostics -> then VaR.

A model can win on QLIKE by luck or by genuinely capturing the return
process. These diagnostics check the latter directly, on the model's
own standardized residuals z_t = resid_t / conditional_vol_t:

  Ljung-Box on z_t     -- any serial correlation LEFT in returns after
                          removing the conditional mean? Should be
                          absent (p > 0.05) for most symbols if the
                          mean specification is adequate.
  Ljung-Box on z_t^2   -- any volatility clustering LEFT after the
                          GARCH filter? This is the one that actually
                          matters: a significant result here means the
                          model did NOT fully capture the ARCH effects
                          it exists to capture.
  Excess kurtosis       -- empirical vs. what the FITTED Student-t
                          degrees of freedom (nu) implies (6/(nu-4)
                          for nu>4). If empirical >> theoretical, real
                          tails are heavier than even the fitted t
                          distribution accounts for.
  QQ plot               -- visual version of the kurtosis check, against
                          arch's own (correctly unit-variance-scaled)
                          Student-t quantile function, not scipy.stats.t
                          directly, which is NOT unit-variance by
                          default and would silently distort the plot.

Fit is done ONCE per symbol on the FULL available history (not the
walk-forward windows from volatility.py) — this is a specification
question about the functional form, not an out-of-sample forecasting
exercise, so an in-sample fit on all available data is the standard and
correct choice here.

Usage:
    python risk/garch_diagnostics.py
    python risk/garch_diagnostics.py --symbols AAPL NVDA TSLA
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # headless — this is a terminal script, not a notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest"))
from db import connect, UNIVERSE  # noqa: E402

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

LB_LAGS = [10, 20]
OUTPUT_PNG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "garch_qq_grid.png",
)


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


def fit_gjr_t(returns):
    """Fit GJR-GARCH(1,1,1) + Student-t on a full return series.
    Returns the fitted result, or None on failure."""
    r = returns.dropna()
    if len(r) < 200:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(r * 100, vol="Garch", p=1, o=1, q=1, dist="t")
            res = am.fit(disp="off")
        return res
    except Exception:
        return None


def diagnose_symbol(res):
    """Run all four diagnostics on one fitted model's residuals."""
    z = res.std_resid.dropna()
    nu = float(res.params.get("nu", np.nan))

    lb_z = acorr_ljungbox(z, lags=LB_LAGS, return_df=True)
    lb_z2 = acorr_ljungbox(z ** 2, lags=LB_LAGS, return_df=True)

    ex_kurt_emp = float(kurtosis(z, fisher=True))
    ex_kurt_theory = 6.0 / (nu - 4) if nu > 4 else float("inf")

    return {
        "n": len(z),
        "nu": nu,
        "lb_z_p10": float(lb_z.loc[10, "lb_pvalue"]),
        "lb_z_p20": float(lb_z.loc[20, "lb_pvalue"]),
        "lb_z2_p10": float(lb_z2.loc[10, "lb_pvalue"]),
        "lb_z2_p20": float(lb_z2.loc[20, "lb_pvalue"]),
        "kurt_emp": ex_kurt_emp,
        "kurt_theory": ex_kurt_theory,
        "z": z,
    }


def qq_quantiles(res, z):
    """Theoretical quantiles from arch's OWN Student-t distribution
    object, which is unit-variance-standardized to match std_resid.
    scipy.stats.t is NOT unit-variance by default (var = nu/(nu-2)) and
    would silently mis-scale the plot if used directly."""
    n = len(z)
    probs = (np.arange(1, n + 1) - 0.5) / n
    nu = res.params["nu"]
    theo_q = res.model.distribution.ppf(probs, [nu])
    samp_q = np.sort(z)
    return theo_q, samp_q


def plot_qq_grid(results_by_symbol, out_path):
    symbols = sorted(results_by_symbol.keys())
    n = len(symbols)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, sym in zip(axes, symbols):
        res, diag = results_by_symbol[sym]
        theo_q, samp_q = qq_quantiles(res, diag["z"])
        ax.scatter(theo_q, samp_q, s=6, alpha=0.5)
        lims = [min(theo_q.min(), samp_q.min()), max(theo_q.max(), samp_q.max())]
        ax.plot(lims, lims, color="red", linewidth=1)
        ax.set_title(f"{sym} (nu={diag['nu']:.1f})", fontsize=10)
        ax.set_xlabel("theoretical (fitted t)")
        ax.set_ylabel("sample")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("GJR-GARCH-t standardized residuals vs. fitted Student-t quantiles", y=1.00)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    args = ap.parse_args()

    if not HAS_ARCH:
        raise SystemExit("`arch` package not installed. pip install arch")

    symbols = args.symbols or UNIVERSE
    print(f"connecting, loading daily bars for {len(symbols)} symbols...", flush=True)
    with connect() as conn:
        bars = load_returns(conn, symbols)
    print(f"loaded {len(bars)} bars — fitting GJR-GARCH-t per symbol (in-sample, full history)...\n", flush=True)

    header = (f"{'symbol':8s} {'n':>5s} {'nu':>6s}  "
              f"{'LB(z) p10':>10s} {'LB(z) p20':>10s}  "
              f"{'LB(z2) p10':>11s} {'LB(z2) p20':>11s}  "
              f"{'kurt emp':>9s} {'kurt fit':>9s}  flags")
    print(header)

    results_by_symbol = {}
    n_serial_corr = n_arch_left = n_heavy_tail = n_fit_fail = 0

    for sym in sorted(symbols):
        sym_returns = bars.loc[bars["symbol"] == sym, "ret"]
        res = fit_gjr_t(sym_returns)
        if res is None:
            print(f"{sym:8s}  fit failed or insufficient data")
            n_fit_fail += 1
            continue

        diag = diagnose_symbol(res)
        results_by_symbol[sym] = (res, diag)

        flags = []
        if diag["lb_z_p20"] < 0.05:
            flags.append("SERIAL-CORR")
            n_serial_corr += 1
        if diag["lb_z2_p20"] < 0.05:
            flags.append("ARCH-LEFT")
            n_arch_left += 1
        if np.isfinite(diag["kurt_theory"]) and diag["kurt_emp"] > 1.5 * diag["kurt_theory"]:
            flags.append("HEAVY-TAIL")
            n_heavy_tail += 1
        flag_str = ",".join(flags) if flags else "ok"

        print(f"{sym:8s} {diag['n']:5d} {diag['nu']:6.2f}  "
              f"{diag['lb_z_p10']:10.3f} {diag['lb_z_p20']:10.3f}  "
              f"{diag['lb_z2_p10']:11.3f} {diag['lb_z2_p20']:11.3f}  "
              f"{diag['kurt_emp']:9.2f} {diag['kurt_theory']:9.2f}  {flag_str}")

    n_ok = len(results_by_symbol) - n_serial_corr - n_arch_left
    print(f"\nSummary ({len(results_by_symbol)} symbols fit, {n_fit_fail} failed):")
    print(f"  Remaining serial correlation in returns (LB(z) p20 < 0.05): {n_serial_corr}")
    print(f"  Remaining ARCH effects / volatility clustering (LB(z2) p20 < 0.05): {n_arch_left}")
    print(f"  Empirical tails notably heavier than fitted t implies (>1.5x): {n_heavy_tail}")
    print(f"  (LB-z2 is the diagnostic that matters most — a significant result there "
          f"means the GARCH filter did not fully remove volatility clustering, which is "
          f"the one thing a GARCH model exists to do.)")

    if results_by_symbol:
        plot_qq_grid(results_by_symbol, OUTPUT_PNG)
        print(f"\nQQ plot grid saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
