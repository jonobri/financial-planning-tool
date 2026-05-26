"""Streamlit front end for the Australian financial planning tool.

Run:  streamlit run app.py   (inside the project venv)

Sidebar = your plan (life stages, portfolio, super, housing, assumptions).
Main area = projections: net-worth fan charts, retirement outcomes, housing
purchasing power, and the rent-vs-buy counterfactual — all with Monte Carlo
error bands from real ETF return history.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from finplan import config
from finplan.etf_universe import ETF_UNIVERSE
from finplan.market_data import compute_return_stats
from finplan.simulation import Plan, housing_purchasing_power, run_montecarlo

st.set_page_config(page_title="FinPlan — AU Financial Planner", layout="wide", page_icon="📈")

PALETTE = {"buy": "#2563eb", "rent": "#d97706", "band": "rgba(37,99,235,0.15)", "band2": "rgba(37,99,235,0.28)"}


# ---------------------------------------------------------------------------
# Cached heavy work
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading ETF history…")
def load_stats(tickers: tuple[str, ...]):
    return compute_return_stats(list(tickers))


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x/1e6:,.2f}M"
    if x >= 1e3:
        return f"{sign}${x/1e3:,.0f}k"
    return f"{sign}${x:,.0f}"


def weight_inputs(label: str, defaults: dict[str, float], key: str) -> dict[str, float]:
    """Multiselect tickers + per-ticker weight inputs, returned normalised."""
    options = list(ETF_UNIVERSE.keys())
    fmt = {t: ETF_UNIVERSE[t].label for t in options}
    chosen = st.multiselect(
        label, options, default=list(defaults), format_func=lambda t: fmt[t], key=f"{key}_sel"
    )
    weights: dict[str, float] = {}
    for t in chosen:
        weights[t] = st.number_input(
            f"  {t.replace('.AX','')} weight %", 0.0, 100.0,
            value=float(defaults.get(t, 100.0 / max(len(chosen), 1))), step=5.0, key=f"{key}_{t}",
        )
    total = sum(weights.values()) or 1.0
    return {t: w / total for t, w in weights.items()}


def fan_chart(result, component, real, color, name, fig=None, pcts=(10, 25, 50, 75, 90)):
    """Median line with shaded 10–90 and 25–75 percentile bands vs age."""
    df = result.percentiles(component, pcts=pcts, real=real)
    ages = df.index
    fig = fig or go.Figure()
    rgba = lambda hexc, a: f"rgba({int(hexc[1:3],16)},{int(hexc[3:5],16)},{int(hexc[5:7],16)},{a})"
    fig.add_trace(go.Scatter(x=ages, y=df["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=df["p10"], fill="tonexty", fillcolor=rgba(color, 0.12),
                             line=dict(width=0), name=f"{name} 10–90%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=df["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=df["p25"], fill="tonexty", fillcolor=rgba(color, 0.25),
                             line=dict(width=0), name=f"{name} 25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=df["p50"], line=dict(color=color, width=3), name=f"{name} median",
                             hovertemplate="age %{x}: %{y:$,.0f}<extra></extra>"))
    return fig


# ---------------------------------------------------------------------------
# Sidebar — the plan (inside a form so heavy recompute only runs on submit)
# ---------------------------------------------------------------------------
st.sidebar.title("📋 Your plan")
with st.sidebar.form("plan_form"):
    st.subheader("Life stages")
    current_age = st.slider("Current age", 18, 70, 32)
    retirement_age = st.slider("Retirement age", current_age + 1, 75, max(60, current_age + 1))
    end_age = st.slider("Plan to age", retirement_age + 1, 105, 95)

    st.subheader("Income & expenses (today's $)")
    current_salary = st.number_input("Gross salary (excl. super)", 0, 2_000_000, 110_000, step=5_000)
    living_expenses = st.number_input("Living expenses / yr (excl. housing)", 0, 500_000, 45_000, step=1_000)
    retirement_spending = st.number_input("Desired retirement spend / yr", 0, 500_000, 60_000, step=1_000)

    st.subheader("Starting balances")
    starting_cash = st.number_input("Cash / savings", 0, 5_000_000, 30_000, step=5_000)
    starting_portfolio = st.number_input("ETF portfolio (outside super)", 0, 10_000_000, 20_000, step=5_000)
    starting_super = st.number_input("Superannuation", 0, 10_000_000, 90_000, step=5_000)

    st.subheader("📈 Investing — your CMC portfolio")
    portfolio_weights = weight_inputs("Outside-super ETFs", {"VAS.AX": 40, "VGS.AX": 60}, "pf")
    monthly_dca = st.number_input("Monthly DCA contribution", 0, 100_000, 1_500, step=100)

    st.subheader("🏦 Superannuation")
    salary_sacrifice = st.number_input("Extra salary sacrifice / yr", 0, 30_000, 0, step=500)
    with st.expander("Super investment mix"):
        super_weights = weight_inputs("Super ETFs / proxy", {"VGS.AX": 55, "VAS.AX": 30, "VAF.AX": 15}, "su")

    st.subheader("🏠 Housing")
    buy_home = st.checkbox("Model buying a home", value=True)
    buy_age = st.slider("Purchase age", current_age, end_age, min(35, end_age))
    home_price = st.number_input("Home price (today's $)", 0, 10_000_000, 850_000, step=25_000)
    deposit_target = st.number_input("Cash deposit", 0, 5_000_000, 170_000, step=10_000)
    mortgage_term = st.slider("Mortgage term (yrs)", 10, 30, 30)
    state = st.selectbox("State (stamp duty)", ["NSW", "VIC", "OTHER"])
    first_home_buyer = st.checkbox("First home buyer concession", value=True)
    fhss_enabled = st.checkbox("Use First Home Super Saver (FHSS)", value=False,
                               help="Make pre-tax super contributions and release them for the deposit.")
    fhss_annual = st.number_input("  FHSS contribution / yr", 0, 15_000, 15_000, step=1_000,
                                  disabled=not fhss_enabled)

    st.subheader("📉 Return scenario")
    return_scenario = st.selectbox(
        "Return assumptions",
        ["Historical (full window)", "Long-run blend (50%)", "Conservative (long-run)", "Custom"],
        index=1,
        help="Recent ETF history can overstate the future. Blend it toward sustainable long-run anchors.",
    )
    if return_scenario == "Historical (full window)":
        return_blend, equity_anchor, bond_anchor = 0.0, 0.08, 0.04
    elif return_scenario == "Long-run blend (50%)":
        return_blend, equity_anchor, bond_anchor = 0.5, 0.08, 0.04
    elif return_scenario == "Conservative (long-run)":
        return_blend, equity_anchor, bond_anchor = 1.0, 0.07, 0.035
    else:
        return_blend = st.slider("Blend toward long-run %", 0, 100, 50) / 100
        equity_anchor = st.slider("Long-run equity return %", 4.0, 12.0, 8.0, 0.1) / 100
        bond_anchor = st.slider("Long-run bond return %", 1.0, 6.0, 4.0, 0.1) / 100

    st.subheader("🏛️ Policy")
    include_age_pension = st.checkbox("Include Age Pension (means-tested)", value=True)
    div296 = st.checkbox("Apply Division 296 ($3M super tax)", value=False)

    with st.expander("⚙️ Economic assumptions"):
        inflation = st.slider("Inflation (CPI) %", 0.0, 6.0, 2.5, 0.1) / 100
        wage_growth = st.slider("Wage growth %", 0.0, 8.0, 3.5, 0.1) / 100
        cash_rate = st.slider("Cash / savings rate %", 0.0, 8.0, 4.0, 0.1) / 100
        mortgage_rate = st.slider("Mortgage rate %", 1.0, 12.0, 6.2, 0.1) / 100
        property_growth = st.slider("Property growth %", 0.0, 10.0, 5.5, 0.1) / 100
        property_growth_vol = st.slider("Property volatility %", 0.0, 20.0, 9.0, 0.5) / 100
        rent_yield = st.slider("Rent yield %", 1.0, 8.0, 3.8, 0.1) / 100
        rent_growth = st.slider("Rent growth %", 0.0, 8.0, 3.5, 0.1) / 100
        property_costs = st.slider("Ownership costs % of value", 0.0, 4.0, 1.2, 0.1) / 100

    with st.expander("🎲 Simulation"):
        sampling_label = st.radio("Sampling method", ["Monte Carlo (normal)", "Historical bootstrap"],
                                  help="Bootstrap resamples real past return sequences (preserving actual "
                                       "crashes); needs ≥8 years of common history or it falls back to Monte Carlo.")
        sampling_method = "bootstrap" if "bootstrap" in sampling_label.lower() else "mvn"
        block_years = st.slider("Bootstrap block (yrs)", 1, 10, 4, disabled=sampling_method != "bootstrap")
        n_paths = st.select_slider("Monte Carlo paths", [500, 1000, 2000, 5000], value=2000)
        seed = st.number_input("Random seed", 0, 99999, 42)

    submitted = st.form_submit_button("▶ Run projection", width="stretch", type="primary")


# ---------------------------------------------------------------------------
# Build objects + run (on submit or first load)
# ---------------------------------------------------------------------------
def build_and_run():
    assumptions = config.Assumptions(
        inflation=inflation, wage_growth=wage_growth, cash_rate=cash_rate,
        mortgage_rate=mortgage_rate, property_growth=property_growth,
        property_growth_vol=property_growth_vol, rent_yield=rent_yield,
        rent_growth=rent_growth, property_costs=property_costs,
    )
    sim = config.SimConfig(
        n_paths=int(n_paths), seed=int(seed), assumptions=assumptions,
        return_blend=return_blend, equity_anchor=equity_anchor, bond_anchor=bond_anchor,
        sampling_method=sampling_method, block_years=int(block_years),
        include_age_pension=include_age_pension, div296=div296,
    )
    plan = Plan(
        current_age=current_age, retirement_age=retirement_age, end_age=end_age,
        current_salary=current_salary, living_expenses=living_expenses,
        retirement_spending=retirement_spending, starting_cash=starting_cash,
        starting_portfolio=starting_portfolio, starting_super=starting_super,
        portfolio_weights=portfolio_weights or {"VAS.AX": 1.0}, monthly_dca=monthly_dca,
        salary_sacrifice=salary_sacrifice, super_weights=super_weights or {"VGS.AX": 1.0},
        fhss_enabled=fhss_enabled, fhss_annual=fhss_annual,
        buy_home=buy_home, buy_age=buy_age, home_price=home_price, deposit_target=deposit_target,
        mortgage_term=mortgage_term, state=state, first_home_buyer=first_home_buyer,
    )
    stats = load_stats(tuple(plan.all_tickers()))
    with st.spinner("Running Monte Carlo…"):
        buy = run_montecarlo(plan, sim, scenario="buy", stats=stats)
        rent = run_montecarlo(plan, sim, scenario="rent", stats=stats)
        power = housing_purchasing_power(plan, sim, stats=stats)
    return {"plan": plan, "sim": sim, "stats": stats, "buy": buy, "rent": rent, "power": power}


if submitted or "results" not in st.session_state:
    st.session_state["results"] = build_and_run()
R = st.session_state["results"]
buy, rent, power, stats, plan = R["buy"], R["rent"], R["power"], R["stats"], R["plan"]
primary = buy if plan.buy_home else rent

if R["sim"].sampling_method == "bootstrap" and not primary.bootstrap_ok:
    st.warning("⚠️ Historical bootstrap needs ≥8 years of common history across your ETFs — "
               "not enough for this selection, so Monte Carlo (normal) was used instead.")


# ---------------------------------------------------------------------------
# Header + view controls
# ---------------------------------------------------------------------------
st.title("📈 Australian Financial Planning Tool")
c1, c2 = st.columns([3, 1])
with c2:
    real = st.radio("Dollars shown in", ["Today's $ (real)", "Future $ (nominal)"], index=0) == "Today's $ (real)"
with c1:
    st.caption(
        "Monte Carlo projection from real ASX ETF return history. "
        "**Not financial advice** — assumptions are simplified; verify against the ATO and a licensed adviser."
    )

tabs = st.tabs(["🧭 Net worth", "🏖️ Retirement", "🏠 Housing power", "⚖️ Rent vs buy", "📊 Data & assumptions"])

# ---------------------------------------------------------------------------
# Tab 1 — Net worth
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Projected net worth")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Median net worth at retirement", money(np.median(primary.at_age("net_worth", retirement_age, real))))
    k2.metric("Median terminal net worth", money(np.median(primary.terminal("net_worth", real))))
    k3.metric("10th percentile (terminal)", money(np.percentile(primary.terminal("net_worth", real), 10)))
    k4.metric("90th percentile (terminal)", money(np.percentile(primary.terminal("net_worth", real), 90)))

    fig = fan_chart(primary, "net_worth", real, PALETTE["buy"], "Net worth")
    fig.add_vline(x=retirement_age, line_dash="dot", line_color="gray", annotation_text="retire")
    if plan.buy_home:
        fig.add_vline(x=plan.buy_age, line_dash="dot", line_color="green", annotation_text="buy")
    fig.update_layout(height=460, xaxis_title="Age", yaxis_title="Net worth (AUD)",
                      yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("What's it made of? (median path)")
    comp_fig = go.Figure()
    for comp, col, label in [
        ("super", "#7c3aed", "Super"), ("portfolio", "#2563eb", "ETF portfolio"),
        ("home_equity", "#16a34a", "Home equity"), ("cash", "#64748b", "Cash"),
    ]:
        med = primary.percentiles(comp, pcts=(50,), real=real)["p50"]
        comp_fig.add_trace(go.Scatter(x=primary.ages, y=med, stackgroup="one", name=label, line=dict(color=col, width=0.5)))
    comp_fig.update_layout(height=380, xaxis_title="Age", yaxis_title="Median value (AUD)",
                           yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(comp_fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2 — Retirement
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader(f"Outcomes at retirement (age {retirement_age})")
    sup_ret = primary.at_age("super", retirement_age, real)
    liquid = (primary.at_age("super", retirement_age, real)
              + primary.at_age("portfolio", retirement_age, real)
              + primary.at_age("cash", retirement_age, real))
    gap_prob = float(np.mean(primary.components["spending_gap"].sum(axis=1) > 1))
    sustainable = np.median(liquid) * 0.04  # 4% rule on liquid assets

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median super balance", money(np.median(sup_ret)))
    m2.metric("Median liquid assets", money(np.median(liquid)))
    m3.metric("Sustainable income (4% rule)", money(sustainable) + " /yr")
    m4.metric("Chance of a shortfall", f"{gap_prob*100:.0f}%",
              help="Probability that desired spending can't be funded from cash, super, portfolio and home equity in some year.")

    if R["sim"].include_age_pension:
        pens = primary.matrix("age_pension", real=real)
        mask = primary.ages >= config.AGE_PENSION_AGE
        if mask.any():
            med_pension = float(np.median(pens[:, mask].mean(axis=1)))
            st.caption(f"💸 Median Age Pension received: ~{money(med_pension)}/yr "
                       f"(averaged over age {config.AGE_PENSION_AGE}+, means-tested). "
                       "This supplements drawdowns and reduces shortfall risk.")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Super balance at retirement — distribution**")
        hist = go.Figure(go.Histogram(x=sup_ret, nbinsx=40, marker_color="#7c3aed"))
        hist.add_vline(x=float(np.median(sup_ret)), line_dash="dash", annotation_text="median")
        hist.update_layout(height=320, xaxis_tickformat="$,.0f", xaxis_title="Super (AUD)", yaxis_title="Paths")
        st.plotly_chart(hist, width="stretch")
    with cc2:
        st.markdown("**Super trajectory (with error bands)**")
        sfig = fan_chart(primary, "super", real, "#7c3aed", "Super")
        sfig.add_vline(x=retirement_age, line_dash="dot", annotation_text="retire")
        sfig.update_layout(height=320, xaxis_title="Age", yaxis_tickformat="$,.0f", hovermode="x unified",
                           showlegend=False)
        st.plotly_chart(sfig, width="stretch")

    if gap_prob > 0.05:
        st.warning(
            f"In {gap_prob*100:.0f}% of simulated paths, desired retirement spending of "
            f"{money(retirement_spending)}/yr can't be fully funded across the whole plan. "
            "Consider higher contributions, later retirement, lower spend, or releasing home equity."
        )
    else:
        st.success(f"Desired spending of {money(retirement_spending)}/yr is funded in "
                   f"{(1-gap_prob)*100:.0f}% of simulated paths.")

# ---------------------------------------------------------------------------
# Tab 3 — Housing power
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("How much home can you afford, by age?")
    st.caption("Expected-return projection. Affordable price = the lower of your deposit capacity "
               "(at 20% deposit) and serviceable loan (repayments ≤ 35% of gross income), at 80% LVR.")
    pf = power.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pf.index, y=pf["max_price_deposit"], name="Limited by deposit",
                             line=dict(color="#16a34a", dash="dot")))
    fig.add_trace(go.Scatter(x=pf.index, y=pf["max_price_serviceability"], name="Limited by borrowing capacity",
                             line=dict(color="#d97706", dash="dot")))
    fig.add_trace(go.Scatter(x=pf.index, y=pf["max_affordable_price"], name="Max affordable", line=dict(color="#2563eb", width=3)))
    fig.add_trace(go.Scatter(x=pf.index, y=np.full(len(pf), plan.home_price), name="Target home price",
                             line=dict(color="black", dash="dash")))
    fig.update_layout(height=440, xaxis_title="Age", yaxis_title="Property price (nominal AUD)",
                      yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    affordable = pf[pf["max_affordable_price"] >= plan.home_price]
    if len(affordable):
        first_age = int(affordable.index[0])
        st.success(f"You can afford a {money(plan.home_price)} home (today's $) from around **age {first_age}** "
                   f"({money(pf.loc[first_age,'borrowing_capacity'])} borrowing + {money(pf.loc[first_age,'deposit_capacity'])} deposit).")
    else:
        st.info(f"On these settings a {money(plan.home_price)} home isn't reached before retirement. "
                "Lower the price, raise savings, or extend the timeline.")
    st.dataframe(pf[["salary", "deposit_capacity", "borrowing_capacity", "max_affordable_price"]]
                 .style.format("${:,.0f}"), width="stretch")

# ---------------------------------------------------------------------------
# Tab 4 — Rent vs buy
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Rent vs buy — lifetime net worth")
    st.caption("Both scenarios invest spare cashflow. The renter invests the deposit and any "
               "saving from lower housing costs; the buyer builds home equity. Returns use the same paths.")
    fig = fan_chart(buy, "net_worth", real, PALETTE["buy"], "Buy")
    fig = fan_chart(rent, "net_worth", real, PALETTE["rent"], "Rent", fig=fig)
    fig.update_layout(height=460, xaxis_title="Age", yaxis_title="Net worth (AUD)",
                      yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    bt, rt = buy.terminal("net_worth", real), rent.terminal("net_worth", real)
    d1, d2, d3 = st.columns(3)
    d1.metric("Median terminal — Buy", money(np.median(bt)))
    d2.metric("Median terminal — Rent", money(np.median(rt)))
    diff = np.median(bt) - np.median(rt)
    d3.metric("Buy advantage (median)", money(diff), delta=("Buy wins" if diff > 0 else "Rent wins"))

    win = float(np.mean(bt > rt))
    dollar_kind = "today's (real)" if real else "future (nominal)"
    st.markdown(f"**Buying ends ahead of renting in {win*100:.0f}% of simulated paths** "
                f"(terminal net worth, {dollar_kind} dollars).")
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=bt, name="Buy", opacity=0.6, marker_color=PALETTE["buy"], nbinsx=40))
    hist.add_trace(go.Histogram(x=rt, name="Rent", opacity=0.6, marker_color=PALETTE["rent"], nbinsx=40))
    hist.update_layout(barmode="overlay", height=340, xaxis_title="Terminal net worth (AUD)",
                       xaxis_tickformat="$,.0f", yaxis_title="Paths", legend=dict(orientation="h"))
    st.plotly_chart(hist, width="stretch")

# ---------------------------------------------------------------------------
# Tab 5 — Data & assumptions
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("ETF return statistics (from real history)")
    st.caption("Annualised expected return & volatility per ETF, derived from downloaded Yahoo Finance history. "
               "ETFs with little history fall back to asset-class assumptions. "
               "⚠️ Recent windows (e.g. global/Nasdaq) reflect a strong bull market and may overstate the future.")
    tbl = stats.as_table()
    st.dataframe(
        tbl.style.format({"exp_return": "{:.1%}", "volatility": "{:.1%}", "history_yrs": "{:.0f}"}),
        width="stretch",
    )
    if stats.used_fallback:
        st.info("Using asset-class fallbacks (short history): " + ", ".join(stats.used_fallback))

    st.subheader("Portfolio summary")
    pfw = R["plan"].portfolio_weights
    exp = sum(w * stats.mean_simple[stats.tickers.index(t)] for t, w in pfw.items() if t in stats.tickers)
    st.write(f"Blended expected return of your CMC portfolio: **{exp:.1%}** nominal / "
             f"**{(1+exp)/(1+inflation)-1:.1%}** real.")
    st.json({t: f"{w:.0%}" for t, w in pfw.items()})

    st.subheader("Key assumptions in this run")
    a = R["sim"].assumptions
    sim_cfg = R["sim"]
    st.json({
        "inflation": f"{a.inflation:.1%}", "wage_growth": f"{a.wage_growth:.1%}",
        "cash_rate": f"{a.cash_rate:.1%}", "mortgage_rate": f"{a.mortgage_rate:.1%}",
        "property_growth": f"{a.property_growth:.1%}", "rent_yield": f"{a.rent_yield:.1%}",
        "rent_growth": f"{a.rent_growth:.1%}", "monte_carlo_paths": sim_cfg.n_paths,
        "sampling_method": primary.method_used,
        "return_blend": f"{sim_cfg.return_blend:.0%} toward {sim_cfg.equity_anchor:.1%} equity / {sim_cfg.bond_anchor:.1%} bond",
        "age_pension": sim_cfg.include_age_pension, "div296_applied": sim_cfg.div296,
        "fhss": plan.fhss_enabled,
        "super_guarantee": f"{config.SUPER_GUARANTEE_RATE:.0%}", "concessional_cap": config.CONCESSIONAL_CAP,
    })
