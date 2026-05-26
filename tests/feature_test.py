"""Offline tests for the added features: return-scenario blend, historical
bootstrap, FHSS, Division 296, and Age Pension. No network required.

Run: PYTHONPATH=. .venv/bin/python tests/feature_test.py
"""

import numpy as np
import pandas as pd

from finplan import age_pension, config
from finplan.market_data import ReturnStats, apply_return_scenario
from finplan.simulation import Plan, run_montecarlo
from finplan.superann import SuperAccount


def synth(tickers, years=20, seed=0):
    """Synthetic ReturnStats with a populated annual history (for bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    mean_log = np.full(n, 0.08)
    vol = np.full(n, 0.15)
    corr = np.full((n, n), 0.6)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vol, vol) * corr
    L = np.linalg.cholesky(cov)
    ann = np.expm1(mean_log + rng.standard_normal((years, n)) @ L.T)
    idx = pd.date_range("2005-12-31", periods=years, freq="YE")
    annual = pd.DataFrame(ann, columns=list(tickers), index=idx)
    return ReturnStats(list(tickers), mean_log, cov, {t: years for t in tickers}, [], annual)


def main():
    plan = Plan()
    tk = plan.all_tickers()
    stats = synth(tk)

    # 1. Return-scenario blend pulls the mean toward the (lower) anchor.
    base = float(stats.mean_simple.mean())
    adj = apply_return_scenario(stats, blend=1.0, equity_anchor=0.07, bond_anchor=0.04)
    blended = float(adj.mean_simple.mean())
    print(f"1) scenario blend: mean return {base:.3%} -> anchored {blended:.3%}")
    assert blended < base, "blend toward lower anchor should reduce expected return"
    # volatility (cov) must be untouched
    assert np.allclose(adj.cov_log, stats.cov_log)

    # 2. Bootstrap runs with sufficient history…
    r = run_montecarlo(plan, config.SimConfig(n_paths=300, seed=2, sampling_method="bootstrap"), "buy", stats=stats)
    print(f"2) bootstrap: method_used={r.method_used} ok={r.bootstrap_ok}")
    assert r.method_used == "bootstrap" and r.bootstrap_ok
    # …and gracefully falls back to MVN with too little history.
    short = synth(tk, years=3)
    r2 = run_montecarlo(plan, config.SimConfig(n_paths=100, sampling_method="bootstrap"), "buy", stats=short)
    print(f"   short-history bootstrap -> method_used={r2.method_used} ok={r2.bootstrap_ok}")
    assert r2.method_used == "mvn" and not r2.bootstrap_ok

    # 3. FHSS (pre-tax saving released for the deposit) should not hurt the buyer.
    sim = config.SimConfig(n_paths=500, seed=3)
    p_no = Plan(starting_cash=120_000, fhss_enabled=False)
    p_yes = Plan(starting_cash=120_000, fhss_enabled=True, fhss_annual=15_000)
    nw_no = np.median(run_montecarlo(p_no, sim, "buy", stats=stats).terminal("net_worth"))
    nw_yes = np.median(run_montecarlo(p_yes, sim, "buy", stats=stats).terminal("net_worth"))
    print(f"3) FHSS: median terminal NW  no-FHSS ${nw_no:,.0f}  vs  FHSS ${nw_yes:,.0f}")
    assert nw_yes >= nw_no * 0.999, "FHSS tax concession should help (or be ~neutral)"

    # 4. Division 296 taxes earnings on the balance above $3M.
    s0 = SuperAccount(5_000_000, div296=False); f0 = s0.step(gross_return=0.07, salary=0, pension_phase=True)
    s1 = SuperAccount(5_000_000, div296=True);  f1 = s1.step(gross_return=0.07, salary=0, pension_phase=True)
    print(f"4) Div296 tax on $5M @ 7%: ${f1['div296_tax']:,.0f}  (balance {s1.balance:,.0f} < {s0.balance:,.0f})")
    assert f1["div296_tax"] > 0 and s1.balance < s0.balance
    assert f0["div296_tax"] == 0

    # 5. Age Pension means test: more assets -> less pension.
    p_low = age_pension.age_pension(200_000, 200_000, homeowner=True)
    p_high = age_pension.age_pension(2_000_000, 2_000_000, homeowner=True)
    print(f"5) Age Pension: $200k assets -> ${p_low:,.0f}/yr ; $2M assets -> ${p_high:,.0f}/yr")
    assert p_low > p_high and p_high == 0

    print("\nFEATURE TEST PASSED")


if __name__ == "__main__":
    main()
