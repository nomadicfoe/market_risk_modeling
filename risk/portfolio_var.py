"""Portfolio-level VaR: risk contribution, not just a portfolio number.

"Portfolio VaR is $100k" tells you nothing actionable. "NVDA contributes
$18k of it despite being a 10% position" tells you where the risk
actually lives. Component VaR via Euler decomposition answers that,
exactly — the components sum to the total portfolio VaR by
construction (this is Euler's homogeneous-function theorem applied to
VaR, which is degree-1 homogeneous in weights under the
variance-covariance/parametric-normal assumption used here), not an
approximation.

Method: parametric (variance-covariance) portfolio VaR, using the
sample covariance matrix of daily returns over a trailing window.
NOT the per-asset GJR-GARCH-t models from volatility.py/var_es.py —
extending component VaR to a genuinely multivariate, time-varying
covariance (DCC-GARCH or a copula) is real additional scope, flagged
here rather than done. The parametric-normal version is the standard
textbook context this decomposition is taught in, and it isolates the
actual point of this exercise (where does portfolio risk concentrate)
from an unrelated modeling upgrade.

For portfolio return R_p = w'r with r ~ (mu, Sigma):
    sigma_p = sqrt(w' Sigma w)
    VaR_p = -z_alpha * sigma_p                    (zero-mean version)
    ComponentVaR_i = -z_alpha * w_i * (Sigma w)_i / sigma_p
    sum_i ComponentVaR_i = VaR_p                   (exact, verified by
                                                     hand calculation —
                                                     see chat history)
    StandaloneVaR_i = -z_alpha * w_i * sigma_i     (that asset's own
                                                     vol only, ignoring
                                                     correlation — the
                                                     risk if it were
                                                     evaluated alone)

Two weight schemes compared: equal-weight, and risk parity (weights
chosen so every asset contributes equally to portfolio risk, solved
by numerical optimization — not a naive inverse-volatility heuristic).

Usage:
    python risk/portfolio_var.py
    python risk/portfolio_var.py --window 250 --alpha 0.05
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest"))
from db import connect, UNIVERSE  # noqa: E402


def load_return_matrix(conn, symbols, window):
    bars = pd.read_sql(
        "SELECT symbol, ts::date AS date, close FROM bars "
        "WHERE symbol = ANY(%s) AND timeframe = '1Day' ORDER BY symbol, ts",
        conn, params=(symbols,),
    )
    bars["date"] = pd.to_datetime(bars["date"])
    bars["ret"] = bars.groupby("symbol")["close"].pct_change()
    wide = bars.pivot(index="date", columns="symbol", values="ret").dropna()
    return wide.tail(window)


def component_var(w, Sigma, alpha):
    """Euler component VaR. Returns (portfolio_VaR, component_VaR array,
    standalone_VaR array). All zero-mean (drops the drift term — the
    standard simplification at daily horizons, where estimated mean
    return is usually noise relative to volatility)."""
    z = norm.ppf(alpha)  # negative
    sigma_p = np.sqrt(w @ Sigma @ w)
    var_p = -z * sigma_p
    Sigma_w = Sigma @ w
    comp = -z * w * Sigma_w / sigma_p
    asset_sigma = np.sqrt(np.diag(Sigma))
    standalone = -z * w * asset_sigma
    return var_p, comp, standalone


def risk_parity_weights(Sigma, n):
    """Solve for weights where every asset contributes equally to
    portfolio risk (not the naive 1/sigma_i heuristic — a proper
    numerical solve of the equal-risk-contribution condition)."""
    def objective(w):
        w = np.abs(w)
        w = w / w.sum()
        sigma_p = np.sqrt(w @ Sigma @ w)
        rc = w * (Sigma @ w) / sigma_p  # risk contribution (vol units)
        target = sigma_p / n
        return np.sum((rc - target) ** 2)

    w0 = np.full(n, 1.0 / n)
    bounds = [(0.001, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        print(f"  warning: risk-parity optimizer did not fully converge "
              f"({res.message}) — result may be approximate")
    w = np.abs(res.x)
    return w / w.sum()


def print_table(symbols, w, var_p, comp, standalone, label):
    print(f"\n--- {label} ---\n")
    print(f"Portfolio VaR: {var_p:.4%} of portfolio value\n")
    print(f"{'symbol':8s} {'weight':>8s} {'standalone VaR':>15s} "
          f"{'component VaR':>14s} {'% portfolio risk':>17s}  note")

    order = np.argsort(-comp)
    div_benefit = 1 - var_p / standalone.sum()
    for i in order:
        pct_risk = comp[i] / var_p
        weight_vs_risk = pct_risk / w[i] if w[i] > 0 else float("nan")
        note = ""
        if weight_vs_risk > 1.3:
            note = "risk > weight"
        elif weight_vs_risk < 0.7:
            note = "risk < weight"
        print(f"{symbols[i]:8s} {w[i]:8.2%} {standalone[i]:15.4%} "
              f"{comp[i]:14.4%} {pct_risk:17.2%}  {note}")

    print(f"\nSum of standalone VaRs: {standalone.sum():.4%}  "
          f"(vs. portfolio VaR {var_p:.4%} — {div_benefit:.1%} diversification benefit)")
    print(f"Sum of component VaRs:  {comp.sum():.4%}  (must equal portfolio VaR — Euler property)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    symbols = args.symbols or UNIVERSE
    print(f"connecting, loading {args.window}-day return matrix for {len(symbols)} symbols...",
          flush=True)
    with connect() as conn:
        returns = load_return_matrix(conn, symbols, args.window)

    symbols = list(returns.columns)
    n = len(symbols)
    print(f"loaded {len(returns)} days x {n} symbols\n", flush=True)

    Sigma = returns.cov().values

    # Equal weight
    w_eq = np.full(n, 1.0 / n)
    var_eq, comp_eq, stand_eq = component_var(w_eq, Sigma, args.alpha)
    print_table(symbols, w_eq, var_eq, comp_eq, stand_eq, "Equal-weight portfolio")

    # Risk parity
    print("\nsolving risk-parity weights...", flush=True)
    w_rp = risk_parity_weights(Sigma, n)
    var_rp, comp_rp, stand_rp = component_var(w_rp, Sigma, args.alpha)
    print_table(symbols, w_rp, var_rp, comp_rp, stand_rp, "Risk-parity portfolio")

    print(f"\nEqual-weight portfolio VaR: {var_eq:.4%}")
    print(f"Risk-parity portfolio VaR:  {var_rp:.4%}")
    print(f"Risk-parity reduces portfolio VaR by {(1 - var_rp/var_eq)*100:.1f}% "
          f"vs. equal weight, by construction spreading risk rather than capital evenly.")


if __name__ == "__main__":
    main()
