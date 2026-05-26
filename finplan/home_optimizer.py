"""Home-buying optimiser: deterministic analysis of *when* to buy and *how big*
a deposit to put down, framed around "dead money" (rent vs mortgage interest —
neither builds equity) and the opportunity cost of tying cash up in a deposit.

This is a transparent, expected-return model (no Monte Carlo) so it can sweep
many deposit sizes and purchase timings quickly. It reuses ``finplan.housing``
for stamp duty, LMI and mortgage amortisation.

Fair rent-vs-buy comparison: both paths spend the same annual budget
(``annual_surplus`` = take-home pay minus living costs) on housing + investing.
Whatever isn't spent on rent or mortgage+ownership is invested at
``investment_return``; the buyer also builds home equity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import housing


@dataclass
class HomeParams:
    annual_surplus: float            # take-home pay − living costs (for housing + investing)
    rent_now: float                  # current annual rent for an equivalent home
    rent_growth: float = 0.035
    property_growth: float = 0.055
    mortgage_rate: float = 0.062
    mortgage_term: int = 30
    investment_return: float = 0.075  # opportunity cost / portfolio return (nominal)
    ownership_cost_rate: float = 0.012
    inflation: float = 0.025
    state: str = "NSW"
    first_home_buyer: bool = True
    horizon_years: int = 30          # comparison horizon
    selling_cost: float = 0.025      # agent + costs, applied to equity at the horizon

    def real(self, nominal: float, year: int) -> float:
        return nominal / (1 + self.inflation) ** year


# ---------------------------------------------------------------------------
def project(price_now: float, deposit_fraction: float, years_to_buy: int,
            savings_now: float, p: HomeParams) -> pd.DataFrame:
    """Year-by-year projection of the buy path and the rent-forever counterfactual.

    Returns a DataFrame indexed by year (1..horizon) with net worth for both
    paths plus cumulative rent / interest / ownership ("dead money") components.
    """
    # Buy-path state
    liquid_b = savings_now
    owned = False
    home_value = 0.0
    mortgage: housing.Mortgage | None = None
    cum_interest = cum_own = cum_rent_b = 0.0
    stamp = lmi = 0.0
    feasible = True
    # Rent-path state
    liquid_r = savings_now
    cum_rent_r = 0.0

    rows = []
    for y in range(1, p.horizon_years + 1):
        # --- purchase event (at the start of the chosen year) ---
        if not owned and (y - 1) == years_to_buy:
            buy_price = price_now * (1 + p.property_growth) ** years_to_buy
            deposit = deposit_fraction * buy_price
            pc = housing.purchase_costs(buy_price, deposit, p.state, p.first_home_buyer)
            if liquid_b < pc.cash_required - 1:
                feasible = False          # couldn't actually fund the deposit + costs
            liquid_b -= pc.cash_required
            stamp, lmi = pc.stamp_duty, pc.lmi
            mortgage = housing.Mortgage(pc.loan, p.mortgage_rate, p.mortgage_term)
            owned = True
            home_value = buy_price

        # --- buy path cashflow ---
        if owned:
            home_value *= (1 + p.property_growth)
            m = mortgage.step_year()
            cum_interest += m["interest"]
            oc = home_value * p.ownership_cost_rate
            cum_own += oc
            housing_cost_b = m["interest"] + m["principal"] + oc
        else:
            rent_b = p.rent_now * (1 + p.rent_growth) ** (y - 1)
            cum_rent_b += rent_b
            housing_cost_b = rent_b
        liquid_b = liquid_b * (1 + p.investment_return) + (p.annual_surplus - housing_cost_b)
        loan_bal = mortgage.balance if mortgage else 0.0
        equity = max(home_value - loan_bal, 0.0)
        nw_b = liquid_b + (equity * (1 - p.selling_cost) if owned else 0.0)

        # --- rent-forever path cashflow ---
        rent_r = p.rent_now * (1 + p.rent_growth) ** (y - 1)
        cum_rent_r += rent_r
        liquid_r = liquid_r * (1 + p.investment_return) + (p.annual_surplus - rent_r)

        dead_buy = cum_interest + cum_own + stamp + lmi
        rows.append({
            "year": y,
            "nw_buy": nw_b, "nw_rent": liquid_r,
            "nw_buy_real": p.real(nw_b, y), "nw_rent_real": p.real(liquid_r, y),
            "cum_interest": cum_interest, "cum_ownership": cum_own,
            "cum_rent_buy": cum_rent_b, "cum_rent_rent": cum_rent_r,
            "dead_money_buy": dead_buy, "dead_money_rent": cum_rent_r,
            "loan_balance": loan_bal, "home_value": home_value if owned else 0.0,
            "equity": equity, "liquid_buy": liquid_b, "feasible": feasible,
        })
    return pd.DataFrame(rows).set_index("year")


def _purchase_snapshot(price_now, deposit_fraction, years_to_buy, p):
    buy_price = price_now * (1 + p.property_growth) ** years_to_buy
    deposit = deposit_fraction * buy_price
    pc = housing.purchase_costs(buy_price, deposit, p.state, p.first_home_buyer)
    repay = housing.monthly_payment(pc.loan, p.mortgage_rate, p.mortgage_term)
    return buy_price, deposit, pc, repay


# ---------------------------------------------------------------------------
def deposit_sweep(price_now, deposit_fractions, years_to_buy, savings_now, p) -> pd.DataFrame:
    """Vary the deposit size for a fixed purchase time. Shows the dead-money and
    net-worth trade-off (more deposit → less interest, but more cash tied up)."""
    out = []
    for f in deposit_fractions:
        df = project(price_now, f, years_to_buy, savings_now, p)
        last = df.iloc[-1]
        buy_price, deposit, pc, repay = _purchase_snapshot(price_now, f, years_to_buy, p)
        out.append({
            "deposit_pct": f, "deposit": deposit, "loan": pc.loan, "lmi": pc.lmi,
            "stamp_duty": pc.stamp_duty, "monthly_repayment": repay,
            "total_interest": last["cum_interest"],
            "year1_interest": df.iloc[0]["cum_interest"],
            "annual_rent_now": p.rent_now,
            "terminal_nw_buy": last["nw_buy_real"], "terminal_nw_rent": last["nw_rent_real"],
            "feasible": bool(last["feasible"]),
        })
    return pd.DataFrame(out)


def timing_sweep(price_now, years_list, deposit_fraction, savings_now, p) -> pd.DataFrame:
    """Vary the purchase year for a fixed deposit fraction. Buying later means a
    bigger saved deposit but more rent paid and a higher price."""
    out = []
    for yb in years_list:
        df = project(price_now, deposit_fraction, yb, savings_now, p)
        last = df.iloc[-1]
        buy_price, deposit, pc, repay = _purchase_snapshot(price_now, deposit_fraction, yb, p)
        out.append({
            "buy_in_years": yb, "buy_price": buy_price, "deposit_needed": deposit + pc.stamp_duty,
            "loan": pc.loan, "lmi": pc.lmi,
            "rent_paid_before_buy": df["cum_rent_buy"].iloc[-1],
            "terminal_nw_buy": last["nw_buy_real"], "terminal_nw_rent": last["nw_rent_real"],
            "feasible": bool(last["feasible"]),
        })
    return pd.DataFrame(out)


def breakeven_year(df: pd.DataFrame) -> int | None:
    """First year the buy path's net worth overtakes renting (None if never)."""
    ahead = df.index[df["nw_buy"] >= df["nw_rent"]]
    return int(ahead[0]) if len(ahead) else None
