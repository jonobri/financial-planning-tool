"""Home-buying optimiser page: when to buy, how big a deposit, and the
"dead money" comparison (rent vs mortgage interest). Prefills from your saved
plan (profiles/autosave.json) and reuses the finplan engine.

This is a deterministic, expected-return analysis — a complement to the main
planner's Monte Carlo outcomes.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from finplan import storage, tax
from finplan.home_optimizer import (HomeParams, breakeven_year, deposit_sweep,
                                    project, timing_sweep)

st.set_page_config(page_title="Home Buying Optimiser", layout="wide", page_icon="🏠")
S = storage.load()


def g(key, default):
    return S.get(key, default)


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x/1e6:,.2f}M"
    if x >= 1e3:
        return f"{sign}${x/1e3:,.0f}k"
    return f"{sign}${x:,.0f}"


st.title("🏠 Home Buying Optimiser")
st.caption("When to buy, how much deposit, and the **dead-money** trade-off — rent and mortgage interest "
           "are both money you never get back, while principal builds equity. Deterministic (expected-return) "
           "analysis; prefilled from your saved plan. **Not financial advice.**")

# --- Inputs (prefilled from the saved plan) --------------------------------
saved_salary = g("current_salary", 110_000)
saved_living = g("living_expenses", 45_000)
take_home = saved_salary - tax.total_income_tax(saved_salary)
default_surplus = int(max(take_home - saved_living, 0))
default_rent = int(g("home_price", 750_000) * g("rent_yield", 3.8) / 100)
default_savings = int(g("starting_cash", 30_000) + g("starting_portfolio", 20_000))

with st.expander("⚙️ Inputs (prefilled from your saved plan — adjust as needed)", expanded=True):
    c = st.columns(4)
    home_price = c[0].number_input("Home price (today's $)", 0, 10_000_000, int(g("home_price", 750_000)), step=25_000, key="ho_price")
    savings_now = c[1].number_input("Savings available now", 0, 10_000_000, default_savings, step=10_000, key="ho_savings")
    annual_surplus = c[2].number_input("Annual surplus (take-home − living)", 0, 1_000_000, default_surplus, step=2_000, key="ho_surplus",
                                       help="Your yearly budget for housing + investing. Prefilled from salary − income tax − living costs.")
    rent_now = c[3].number_input("Current annual rent (equivalent home)", 0, 500_000, default_rent, step=1_000, key="ho_rent")

    c = st.columns(4)
    mortgage_rate = c[0].slider("Mortgage rate %", 1.0, 12.0, float(g("mortgage_rate", 6.2)), 0.1, key="ho_mrate") / 100
    investment_return = c[1].slider("Investment return % (opportunity cost)", 0.0, 12.0, 7.5, 0.1, key="ho_invest",
                                    help="Return your deposit/savings would earn if invested instead. If this beats property growth, smaller deposits look better.") / 100
    property_growth = c[2].slider("Property growth %", 0.0, 10.0, float(g("property_growth", 5.5)), 0.1, key="ho_pgrow") / 100
    ownership_cost = c[3].slider("Ownership costs % of value/yr", 0.0, 4.0, float(g("property_costs", 1.2)), 0.1, key="ho_own") / 100

    c = st.columns(4)
    mortgage_term = c[0].slider("Mortgage term (yrs)", 10, 30, int(g("mortgage_term", 30)), key="ho_term")
    horizon = c[1].slider("Comparison horizon (yrs)", 10, 40, 30, key="ho_horizon")
    rent_growth = c[2].slider("Rent growth %", 0.0, 8.0, float(g("rent_growth", 3.5)), 0.1, key="ho_rgrow") / 100
    inflation = c[3].slider("Inflation %", 0.0, 6.0, float(g("inflation", 2.5)), 0.1, key="ho_infl") / 100

    c = st.columns(4)
    state = c[0].selectbox("State (stamp duty)", ["NSW", "VIC", "OTHER"],
                           index=["NSW", "VIC", "OTHER"].index(g("state", "NSW")) if g("state", "NSW") in ["NSW", "VIC", "OTHER"] else 0, key="ho_state")
    first_home_buyer = c[1].checkbox("First home buyer concession", value=bool(g("first_home_buyer", True)), key="ho_fhb")

p = HomeParams(
    annual_surplus=annual_surplus, rent_now=rent_now, rent_growth=rent_growth,
    property_growth=property_growth, mortgage_rate=mortgage_rate, mortgage_term=mortgage_term,
    investment_return=investment_return, ownership_cost_rate=ownership_cost, inflation=inflation,
    state=state, first_home_buyer=first_home_buyer, horizon_years=horizon,
)

tabs = st.tabs(["💰 Deposit size", "⏱️ When to buy", "📉 Dead money over time"])

# ===========================================================================
# Deposit size
# ===========================================================================
with tabs[0]:
    st.subheader("How big a deposit? (buying now)")
    st.caption("Bigger deposit → smaller loan → less interest (dead money), and no LMI above 20%. But it also "
               "ties up cash that could be invested. With investment return above your mortgage rate, smaller "
               "deposits can win; below it, bigger deposits win.")
    fracs = [round(x, 2) for x in np.arange(0.05, 0.55, 0.05)]
    ds = deposit_sweep(home_price, fracs, 0, savings_now, p)
    feas = ds[ds["feasible"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds["deposit_pct"] * 100, y=ds["total_interest"], name="Total interest paid (dead money)",
                             line=dict(color="#dc2626", width=3), yaxis="y1"))
    fig.add_trace(go.Scatter(x=ds["deposit_pct"] * 100, y=ds["terminal_nw_buy"], name="Net worth at horizon (real)",
                             line=dict(color="#2563eb", width=3), yaxis="y2"))
    fig.add_vline(x=20, line_dash="dot", line_color="gray", annotation_text="20% (LMI avoided)")
    if len(feas):
        best = feas.loc[feas["terminal_nw_buy"].idxmax()]
        fig.add_vline(x=best["deposit_pct"] * 100, line_dash="dash", line_color="green",
                      annotation_text=f"best feasible: {best['deposit_pct']*100:.0f}%")
    fig.update_layout(height=440, xaxis_title="Deposit (% of price)",
                      yaxis=dict(title="Total interest ($)", tickformat="$,.0f"),
                      yaxis2=dict(title="Net worth (real $)", overlaying="y", side="right", tickformat="$,.0f"),
                      hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    if len(feas):
        best = feas.loc[feas["terminal_nw_buy"].idxmax()]
        st.success(f"**Best feasible deposit ≈ {best['deposit_pct']*100:.0f}%** "
                   f"({money(best['deposit'])}): loan {money(best['loan'])}, "
                   f"{money(best['monthly_repayment'])}/mo, total interest {money(best['total_interest'])} over {horizon}y. "
                   f"Year-1 interest {money(best['year1_interest'])} vs current rent {money(rent_now)}.")
    infeasible = ds[~ds["feasible"]]
    if len(infeasible):
        st.caption(f"⚠️ Deposits ≥ {infeasible['deposit_pct'].min()*100:.0f}% aren't affordable with {money(savings_now)} saved.")

    show = ds.copy()
    show["deposit_pct"] = (show["deposit_pct"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(show[["deposit_pct", "deposit", "loan", "lmi", "monthly_repayment", "total_interest",
                       "terminal_nw_buy", "feasible"]].style.format(
        {"deposit": "${:,.0f}", "loan": "${:,.0f}", "lmi": "${:,.0f}", "monthly_repayment": "${:,.0f}",
         "total_interest": "${:,.0f}", "terminal_nw_buy": "${:,.0f}"}), width="stretch")

# ===========================================================================
# When to buy
# ===========================================================================
with tabs[1]:
    st.subheader("When to buy?")
    st.caption("Buying sooner stops rent and captures property growth, but means a smaller deposit (more interest, "
               "maybe LMI). Waiting builds a bigger deposit but you pay more rent and face a higher price. "
               "Net worth is at your horizon, in today's dollars.")
    deposit_frac_t = st.slider("Deposit assumed (% of price at purchase)", 5, 50, 20, 5, key="ho_tdep") / 100
    years_list = list(range(0, min(16, horizon - 1)))
    ts = timing_sweep(home_price, years_list, deposit_frac_t, savings_now, p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["buy_in_years"], y=ts["terminal_nw_buy"], name="Buy (net worth, real)",
                             line=dict(color="#2563eb", width=3)))
    fig.add_trace(go.Scatter(x=ts["buy_in_years"], y=ts["terminal_nw_rent"], name="Rent forever (net worth, real)",
                             line=dict(color="#d97706", width=2, dash="dash")))
    fig.add_trace(go.Bar(x=ts["buy_in_years"], y=ts["rent_paid_before_buy"], name="Rent paid before buying",
                         marker_color="rgba(220,38,38,0.25)", yaxis="y2"))
    feas_t = ts[ts["feasible"]]
    if len(feas_t):
        bb = feas_t.loc[feas_t["terminal_nw_buy"].idxmax()]
        fig.add_vline(x=bb["buy_in_years"], line_dash="dash", line_color="green",
                      annotation_text=f"best: in {int(bb['buy_in_years'])}y")
    fig.update_layout(height=440, xaxis_title="Buy in N years from now",
                      yaxis=dict(title="Net worth at horizon (real $)", tickformat="$,.0f"),
                      yaxis2=dict(title="Rent paid before buying ($)", overlaying="y", side="right", tickformat="$,.0f"),
                      hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    first_feasible = ts[ts["feasible"]]["buy_in_years"].min() if ts["feasible"].any() else None
    if len(feas_t):
        bb = feas_t.loc[feas_t["terminal_nw_buy"].idxmax()]
        msg = (f"**Best time to buy ≈ {int(bb['buy_in_years'])} years from now** "
               f"(price {money(bb['buy_price'])}, deposit+duty {money(bb['deposit_needed'])}).")
        if first_feasible is not None and first_feasible > 0:
            msg += f" Earliest you can afford a {deposit_frac_t*100:.0f}% deposit is in {int(first_feasible)} years."
        st.success(msg)
    if len(feas_t) and ts["terminal_nw_rent"].iloc[0] > feas_t["terminal_nw_buy"].max():
        st.info("On these assumptions, **renting and investing beats buying** at every timing — because your "
                "investment return is above property growth. Lower it (or raise property growth/rent) to flip this.")

# ===========================================================================
# Dead money over time
# ===========================================================================
with tabs[2]:
    st.subheader("Dead money & break-even over time")
    st.caption("For one specific choice: cumulative rent (if you rented) vs cumulative mortgage interest + "
               "ownership costs (if you bought) — the money neither path gets back — plus where buying's net "
               "worth overtakes renting.")
    c = st.columns(2)
    dep_c = c[0].slider("Deposit (% of price)", 5, 50, 20, 5, key="ho_cdep") / 100
    buy_in_c = c[1].slider("Buy in N years", 0, min(15, horizon - 1), 0, key="ho_cwhen")
    df = project(home_price, dep_c, buy_in_c, savings_now, p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["dead_money_rent"], name="Dead money — rent (cumulative)",
                             line=dict(color="#d97706", width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df["dead_money_buy"], name="Dead money — interest + ownership + upfront",
                             line=dict(color="#dc2626", width=3)))
    fig.update_layout(height=380, xaxis_title="Year", yaxis_title="Cumulative dead money ($, nominal)",
                      yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["nw_buy_real"], name="Buy — net worth (real)", line=dict(color="#2563eb", width=3)))
    fig2.add_trace(go.Scatter(x=df.index, y=df["nw_rent_real"], name="Rent — net worth (real)", line=dict(color="#d97706", width=3)))
    be = breakeven_year(df)
    if be:
        fig2.add_vline(x=be, line_dash="dash", line_color="green", annotation_text=f"buy overtakes: yr {be}")
    fig2.update_layout(height=380, xaxis_title="Year", yaxis_title="Net worth (real $)",
                       yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig2, width="stretch")

    last = df.iloc[-1]
    m = st.columns(4)
    m[0].metric("Net worth — buy", money(last["nw_buy_real"]))
    m[1].metric("Net worth — rent", money(last["nw_rent_real"]))
    m[2].metric("Buy advantage", money(last["nw_buy_real"] - last["nw_rent_real"]))
    m[3].metric("Break-even", f"year {be}" if be else "never (in horizon)")
    if not bool(last["feasible"]):
        st.warning(f"⚠️ A {dep_c*100:.0f}% deposit isn't affordable {('now' if buy_in_c==0 else f'in {buy_in_c} years')} "
                   f"with {money(savings_now)} saved plus accumulated savings — results assume you fund the gap.")
