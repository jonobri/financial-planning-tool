"""Offline smoke test: exercises the engine with synthetic return stats so it
runs without any network access. Run: .venv/bin/python tests/smoke_test.py
"""

import numpy as np

from finplan import config
from finplan.market_data import ReturnStats
from finplan.simulation import Plan, run_montecarlo, housing_purchasing_power
import pandas as pd


def synthetic_stats(tickers):
    n = len(tickers)
    # plausible annual log-return means / vols
    mean_log = np.full(n, 0.07)
    vol = np.full(n, 0.15)
    corr = np.full((n, n), 0.6)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vol, vol) * corr
    return ReturnStats(
        tickers=list(tickers),
        mean_log=mean_log,
        cov_log=cov,
        history_years={t: 15 for t in tickers},
        used_fallback=[],
        annual_simple=pd.DataFrame(columns=list(tickers)),
    )


def main():
    plan = Plan()
    stats = synthetic_stats(plan.all_tickers())
    sim = config.SimConfig(n_paths=500, seed=1)

    buy = run_montecarlo(plan, sim, scenario="buy", stats=stats)
    rent = run_montecarlo(plan, sim, scenario="rent", stats=stats)

    assert buy.components["net_worth"].shape == (500, plan.years())
    assert np.all(np.isfinite(buy.components["net_worth"]))

    nw = buy.percentiles("net_worth")
    print("=== BUY scenario: net worth percentiles (selected ages) ===")
    print(nw.loc[[plan.current_age, plan.buy_age, plan.retirement_age, plan.end_age]].round(0))

    print("\n=== Terminal net worth (age %d) ===" % plan.end_age)
    for label, res in [("buy", buy), ("rent", rent)]:
        tv = res.terminal("net_worth")
        print(f"{label:>5}: median ${np.median(tv):,.0f} | "
              f"10th ${np.percentile(tv,10):,.0f} | 90th ${np.percentile(tv,90):,.0f}")

    print("\n=== Super at retirement (age %d) ===" % plan.retirement_age)
    si = plan.retirement_age - plan.current_age
    sup = buy.components["super"][:, si]
    print(f"median ${np.median(sup):,.0f} | 10th ${np.percentile(sup,10):,.0f} | 90th ${np.percentile(sup,90):,.0f}")

    gap = buy.components["spending_gap"]
    print(f"\nPaths with any retirement spending gap: {np.mean(gap.sum(axis=1) > 0)*100:.1f}%")

    print("\n=== Housing purchasing power (first/last few ages) ===")
    pp = housing_purchasing_power(plan, sim, stats=stats)
    print(pp[["salary", "deposit_capacity", "borrowing_capacity", "max_affordable_price"]].round(0).head())

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
