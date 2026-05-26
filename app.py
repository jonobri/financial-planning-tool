"""Streamlit front end for the Australian financial planning tool.

Run:  streamlit run app.py   (inside the project venv)

Inputs live in tabs at the top of the page (they update live); the projection
charts sit below. Press "Run projection" to recompute the Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finplan import config, storage
from finplan.etf_universe import ETF_UNIVERSE
from finplan.market_data import compute_return_stats
from finplan.simulation import Plan, housing_purchasing_power, run_montecarlo

st.set_page_config(page_title="FinPlan — AU Financial Planner", layout="wide", page_icon="📈")
PALETTE = {"buy": "#2563eb", "rent": "#d97706"}

# Auto-saved plan inputs (restored on startup; re-saved on every run).
STORE = storage.load()


def g(key, default):
    return STORE.get(key, default)


def opt_index(options, key, fallback):
    val = STORE.get(key, fallback)
    return options.index(val) if val in options else options.index(fallback)


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


def weight_inputs(label: str, defaults: dict[str, float], key: str):
    """Multiselect tickers + per-ticker weight inputs (live, restored, normalised).
    Returns (normalised_weights, raw_dict_to_persist)."""
    options = list(ETF_UNIVERSE.keys())
    fmt = {t: ETF_UNIVERSE[t].label for t in options}
    saved_sel = [t for t in g(f"{key}_sel", list(defaults)) if t in options]
    chosen = st.multiselect(label, options, default=saved_sel or list(defaults),
                            format_func=lambda t: fmt[t], key=f"{key}_sel")
    weights: dict[str, float] = {}
    cols = st.columns(max(len(chosen), 1)) if chosen else []
    for i, t in enumerate(chosen):
        with cols[i]:
            weights[t] = st.number_input(
                f"{t.replace('.AX','')} %", 0.0, 100.0,
                value=float(g(f"{key}_w_{t}", defaults.get(t, 100.0 / max(len(chosen), 1)))),
                step=5.0, key=f"{key}_{t}")
    total = sum(weights.values()) or 1.0
    raw = {f"{key}_sel": chosen, **{f"{key}_w_{t}": weights[t] for t in chosen}}
    return {t: w / total for t, w in weights.items()}, raw


def fan_chart(result, component, real, color, name, fig=None, pcts=(25, 50, 75)):
    """Median line with a shaded 25–75% band (the 10–90% band is intentionally
    omitted — geared funds' wide tails made it unreadable)."""
    df = result.percentiles(component, pcts=pcts, real=real)
    ages = df.index
    fig = fig or go.Figure()
    rgba = lambda h, a: f"rgba({int(h[1:3],16)},{int(h[3:5],16)},{int(h[5:7],16)},{a})"
    fig.add_trace(go.Scatter(x=ages, y=df["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=df["p25"], fill="tonexty", fillcolor=rgba(color, 0.22),
                             line=dict(width=0), name=f"{name} 25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=df["p50"], line=dict(color=color, width=3), name=f"{name} median",
                             hovertemplate="age %{x}: %{y:$,.0f}<extra></extra>"))
    return fig


# ===========================================================================
# Header
# ===========================================================================
st.title("📈 Australian Financial Planning Tool")
st.caption("Monte Carlo projection from real ASX ETF return history. "
           "**Not financial advice** — assumptions are simplified; verify against the ATO and a licensed adviser.")

_PATHS = [500, 1000, 2000, 5000]
_SCENARIOS = ["Historical (full window)", "Long-run blend (50%)", "Conservative (long-run)", "Custom"]
_SAMPLING = ["Monte Carlo (normal)", "Historical bootstrap"]
_STATES = ["NSW", "VIC", "OTHER"]

st.markdown("#### ⚙️ Your plan")
itabs = st.tabs(["1 · You & income", "2 · Portfolio & saving", "3 · Housing", "4 · Assumptions & scenario"])

# --- Tab 1: You & income ---------------------------------------------------
with itabs[0]:
    c = st.columns(3)
    current_age = c[0].slider("Current age", 18, 75, int(g("current_age", 32)))
    retirement_age = c[1].slider("Retirement age", 30, 80, int(g("retirement_age", 60)))
    end_age = c[2].slider("Plan to age", 60, 105, int(g("end_age", 95)))
    c = st.columns(3)
    current_salary = c[0].number_input("Gross salary (excl. super)", 0, 2_000_000, int(g("current_salary", 110_000)), step=5_000)
    living_expenses = c[1].number_input("Living expenses / yr (excl. housing)", 0, 500_000, int(g("living_expenses", 45_000)), step=1_000)
    retirement_spending = c[2].number_input("Desired retirement spend / yr", 0, 500_000, int(g("retirement_spending", 60_000)), step=1_000)
    c = st.columns(3)
    starting_cash = c[0].number_input("Starting cash / savings", 0, 5_000_000, int(g("starting_cash", 30_000)), step=5_000)
    starting_portfolio = c[1].number_input("Starting ETF portfolio", 0, 10_000_000, int(g("starting_portfolio", 20_000)), step=5_000)
    starting_super = c[2].number_input("Starting super", 0, 10_000_000, int(g("starting_super", 90_000)), step=5_000)

    st.markdown("**Income step-ups (optional)** — salary jumps to these amounts at the given ages "
                "(e.g. promotions); between them it grows at the wage-growth rate.")
    ms_default = g("salary_milestones", [])
    ms_df = (pd.DataFrame(ms_default, columns=["age", "salary"]) if ms_default
             else pd.DataFrame({"age": pd.Series([], dtype="Int64"), "salary": pd.Series([], dtype="float")}))
    edited_ms = st.data_editor(
        ms_df, num_rows="dynamic", width="stretch", key="ms_editor",
        column_config={
            "age": st.column_config.NumberColumn("Age", min_value=18, max_value=80, step=1),
            "salary": st.column_config.NumberColumn("Salary $", min_value=0, step=5_000, format="$%d"),
        })
    salary_milestones = [(int(r["age"]), float(r["salary"])) for _, r in edited_ms.iterrows()
                         if pd.notna(r["age"]) and pd.notna(r["salary"])]

# --- Tab 2: Portfolio & saving ---------------------------------------------
with itabs[1]:
    st.markdown("**Your CMC (outside-super) ETF portfolio**")
    portfolio_weights, pf_raw = weight_inputs("ETFs", {"VAS.AX": 40, "VGS.AX": 60}, "pf")
    st.markdown("**Super investment mix**")
    super_weights, su_raw = weight_inputs("Super ETFs / proxy", {"VGS.AX": 55, "VAS.AX": 30, "VAF.AX": 15}, "su")

    st.divider()
    st.markdown("**Saving** — each year, whatever's left after tax, living costs and housing is your savings. "
                "Choose how to split it (the mortgage portion only applies once you own a home):")
    c = st.columns([2, 1])
    split_pct = c[0].slider("→ Extra mortgage repayments %  (the rest goes to ETF investing)", 0, 100, int(g("savings_split_pct", 40)))
    savings_split_to_mortgage = split_pct / 100
    c[1].metric("Split", f"{split_pct}% mortgage", f"{100-split_pct}% ETFs", delta_color="off")
    salary_sacrifice = st.number_input("Extra super salary sacrifice / yr", 0, 30_000, int(g("salary_sacrifice", 0)), step=500)

# --- Tab 3: Housing --------------------------------------------------------
with itabs[2]:
    c = st.columns(3)
    buy_home = c[0].checkbox("Model buying a home", value=bool(g("buy_home", True)))
    first_home_buyer = c[1].checkbox("First home buyer concession", value=bool(g("first_home_buyer", True)))
    state = c[2].selectbox("State (stamp duty)", _STATES, index=opt_index(_STATES, "state", "NSW"))
    c = st.columns(3)
    buy_age = c[0].slider("Purchase age", 18, 80, int(g("buy_age", 35)))
    home_price = c[1].number_input("Home price (today's $)", 0, 10_000_000, int(g("home_price", 850_000)), step=25_000)
    deposit_target = c[2].number_input("Cash deposit", 0, 5_000_000, int(g("deposit_target", 170_000)), step=10_000)
    c = st.columns(3)
    mortgage_term = c[0].slider("Mortgage term (yrs)", 10, 30, int(g("mortgage_term", 30)))
    fhss_enabled = c[1].checkbox("Use First Home Super Saver (FHSS)", value=bool(g("fhss_enabled", False)),
                                 help="Make pre-tax super contributions and release them for the deposit.")
    fhss_annual = c[2].number_input("FHSS contribution / yr", 0, 15_000, int(g("fhss_annual", 15_000)),
                                    step=1_000, disabled=not fhss_enabled)
    fhss_already = st.number_input("FHSS already contributed to date ($)", 0, 50_000, int(g("fhss_already", 0)),
                                   step=1_000, disabled=not fhss_enabled,
                                   help="Voluntary FHSS contributions you've already made. Counts toward the $50k cap; "
                                        "assumed concessional (~85% releasable, plus earnings until purchase).")

# --- Tab 4: Assumptions & scenario -----------------------------------------
with itabs[3]:
    c = st.columns([2, 1, 1])
    return_scenario = c[0].selectbox("Return scenario", _SCENARIOS, index=opt_index(_SCENARIOS, "return_scenario", _SCENARIOS[1]),
                                     help="Recent ETF history can overstate the future. Blend it toward sustainable long-run anchors.")
    if return_scenario == "Historical (full window)":
        return_blend, equity_anchor, bond_anchor = 0.0, 0.08, 0.04
    elif return_scenario == "Long-run blend (50%)":
        return_blend, equity_anchor, bond_anchor = 0.5, 0.08, 0.04
    elif return_scenario == "Conservative (long-run)":
        return_blend, equity_anchor, bond_anchor = 1.0, 0.07, 0.035
    else:
        return_blend = c[1].slider("Blend → long-run %", 0, 100, int(g("return_blend_pct", 50))) / 100
        equity_anchor = c[2].slider("Equity anchor %", 4.0, 12.0, float(g("equity_anchor_pct", 8.0)), 0.1) / 100
        bond_anchor = c[2].slider("Bond anchor %", 1.0, 6.0, float(g("bond_anchor_pct", 4.0)), 0.1) / 100

    c = st.columns(3)
    include_age_pension = c[0].checkbox("Include Age Pension (means-tested)", value=bool(g("include_age_pension", True)))
    div296 = c[1].checkbox("Apply Division 296 ($3M super tax)", value=bool(g("div296", False)))
    sampling_label = c[2].radio("Sampling method", _SAMPLING, index=opt_index(_SAMPLING, "sampling_label", _SAMPLING[0]),
                                help="Bootstrap resamples real past sequences; needs ≥8 yrs common history or it falls back.")
    sampling_method = "bootstrap" if "bootstrap" in sampling_label.lower() else "mvn"

    with st.expander("Economic assumptions"):
        c = st.columns(3)
        inflation = c[0].slider("Inflation (CPI) %", 0.0, 6.0, float(g("inflation", 2.5)), 0.1) / 100
        wage_growth = c[1].slider("Wage growth %", 0.0, 8.0, float(g("wage_growth", 3.5)), 0.1) / 100
        cash_rate = c[2].slider("Cash rate %", 0.0, 8.0, float(g("cash_rate", 4.0)), 0.1) / 100
        mortgage_rate = c[0].slider("Mortgage rate %", 1.0, 12.0, float(g("mortgage_rate", 6.2)), 0.1) / 100
        property_growth = c[1].slider("Property growth %", 0.0, 10.0, float(g("property_growth", 5.5)), 0.1) / 100
        property_growth_vol = c[2].slider("Property volatility %", 0.0, 20.0, float(g("property_growth_vol", 9.0)), 0.5) / 100
        rent_yield = c[0].slider("Rent yield %", 1.0, 8.0, float(g("rent_yield", 3.8)), 0.1) / 100
        rent_growth = c[1].slider("Rent growth %", 0.0, 8.0, float(g("rent_growth", 3.5)), 0.1) / 100
        property_costs = c[2].slider("Ownership costs %", 0.0, 4.0, float(g("property_costs", 1.2)), 0.1) / 100
    with st.expander("Simulation"):
        c = st.columns(3)
        block_years = c[0].slider("Bootstrap block (yrs)", 1, 10, int(g("block_years", 4)), disabled=sampling_method != "bootstrap")
        _np = int(g("n_paths", 2000))
        n_paths = c[1].select_slider("Monte Carlo paths", _PATHS, value=_np if _np in _PATHS else 2000)
        seed = c[2].number_input("Random seed", 0, 99999, int(g("seed", 42)))

# --- Run / reset row -------------------------------------------------------
if not (current_age < retirement_age < end_age and current_age <= buy_age <= end_age):
    st.warning("Check the ages: need current < retirement < plan-to age, and purchase age within range. "
               "I'll clamp them to run.")
rc = st.columns([2, 1, 4])
submitted = rc[0].button("▶ Run projection", type="primary", width="stretch")
if rc[1].button("↺ Reset", width="stretch"):
    storage.clear()
    st.session_state.clear()
    st.rerun()
rc[2].caption("💾 Auto-saved locally as you edit — restored next time." if STORE
              else "💾 Your plan auto-saves locally as you edit it.")


# ===========================================================================
# Build + run
# ===========================================================================
def plan_inputs() -> dict:
    d = {
        "current_age": current_age, "retirement_age": retirement_age, "end_age": end_age,
        "current_salary": current_salary, "living_expenses": living_expenses,
        "retirement_spending": retirement_spending, "starting_cash": starting_cash,
        "starting_portfolio": starting_portfolio, "starting_super": starting_super,
        "salary_milestones": [[int(a), float(s)] for a, s in salary_milestones],
        "savings_split_pct": int(savings_split_to_mortgage * 100), "salary_sacrifice": salary_sacrifice,
        "buy_home": buy_home, "buy_age": buy_age, "home_price": home_price,
        "deposit_target": deposit_target, "mortgage_term": mortgage_term,
        "state": state, "first_home_buyer": first_home_buyer,
        "fhss_enabled": fhss_enabled, "fhss_annual": fhss_annual, "fhss_already": fhss_already,
        "return_scenario": return_scenario, "return_blend_pct": round(return_blend * 100, 1),
        "equity_anchor_pct": round(equity_anchor * 100, 1), "bond_anchor_pct": round(bond_anchor * 100, 1),
        "include_age_pension": include_age_pension, "div296": div296,
        "inflation": round(inflation * 100, 2), "wage_growth": round(wage_growth * 100, 2),
        "cash_rate": round(cash_rate * 100, 2), "mortgage_rate": round(mortgage_rate * 100, 2),
        "property_growth": round(property_growth * 100, 2), "property_growth_vol": round(property_growth_vol * 100, 2),
        "rent_yield": round(rent_yield * 100, 2), "rent_growth": round(rent_growth * 100, 2),
        "property_costs": round(property_costs * 100, 2),
        "sampling_label": sampling_label, "block_years": block_years, "n_paths": n_paths, "seed": seed,
    }
    d.update(pf_raw)
    d.update(su_raw)
    return d


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
    # Clamp ages to a consistent ordering.
    ca = current_age
    ra = max(retirement_age, ca + 1)
    ea = max(end_age, ra + 1)
    ba = min(max(buy_age, ca), ea)
    plan = Plan(
        current_age=ca, retirement_age=ra, end_age=ea,
        current_salary=current_salary, living_expenses=living_expenses,
        retirement_spending=retirement_spending, starting_cash=starting_cash,
        starting_portfolio=starting_portfolio, starting_super=starting_super,
        portfolio_weights=portfolio_weights or {"VAS.AX": 1.0},
        savings_split_to_mortgage=savings_split_to_mortgage, salary_milestones=salary_milestones,
        salary_sacrifice=salary_sacrifice, super_weights=super_weights or {"VGS.AX": 1.0},
        fhss_enabled=fhss_enabled, fhss_annual=fhss_annual, fhss_already_contributed=fhss_already,
        buy_home=buy_home, buy_age=ba, home_price=home_price, deposit_target=deposit_target,
        mortgage_term=mortgage_term, state=state, first_home_buyer=first_home_buyer,
    )
    stats = load_stats(tuple(plan.all_tickers()))
    with st.spinner("Running Monte Carlo…"):
        buy = run_montecarlo(plan, sim, scenario="buy", stats=stats)
        rent = run_montecarlo(plan, sim, scenario="rent", stats=stats)
        power = housing_purchasing_power(plan, sim, stats=stats)
    return {"plan": plan, "sim": sim, "stats": stats, "buy": buy, "rent": rent, "power": power}


storage.save(plan_inputs())              # auto-save on every edit (not just on Run)
if submitted or "results" not in st.session_state:
    st.session_state["results"] = build_and_run()
R = st.session_state["results"]
buy, rent, power, stats, plan = R["buy"], R["rent"], R["power"], R["stats"], R["plan"]
primary = buy if plan.buy_home else rent
ret_age = plan.retirement_age

# --- Alerts ----------------------------------------------------------------
if R["sim"].sampling_method == "bootstrap" and not primary.bootstrap_ok:
    st.warning("⚠️ Historical bootstrap needs ≥8 years of common history across your ETFs — "
               "not enough for this selection, so Monte Carlo (normal) was used instead.")

cfg = primary.components["cashflow_gap"]
cf_prob = float(np.mean(cfg.sum(axis=1) > 1))
if cf_prob > 0.02:
    worst = float(np.median(cfg.max(axis=1)))
    st.error(f"🚩 **Affordability:** in {cf_prob*100:.0f}% of paths your commitments exceed income + savings in some "
             f"working year (median worst-year shortfall ~{money(worst)}/yr). The plan may be unaffordable as set — "
             "try a cheaper home, smaller deposit, income step-ups, lower expenses, or directing less to the mortgage.")

# ===========================================================================
# Results
# ===========================================================================
st.markdown("#### 📊 Projection")
hc = st.columns([3, 1])
real = hc[1].radio("Dollars shown in", ["Today's $ (real)", "Future $ (nominal)"], index=0) == "Today's $ (real)"
tabs = st.tabs(["🧭 Net worth", "🏖️ Retirement", "🏠 Housing power", "⚖️ Rent vs buy", "📊 Data & assumptions"])

with tabs[0]:
    st.subheader("Projected net worth")
    k = st.columns(4)
    k[0].metric("Median net worth at retirement", money(np.median(primary.at_age("net_worth", ret_age, real))))
    k[1].metric("Median terminal net worth", money(np.median(primary.terminal("net_worth", real))))
    k[2].metric("10th percentile (terminal)", money(np.percentile(primary.terminal("net_worth", real), 10)))
    k[3].metric("90th percentile (terminal)", money(np.percentile(primary.terminal("net_worth", real), 90)))

    if plan.fhss_enabled and plan.buy_home:
        rel = primary.components["fhss_released"].sum(axis=1)
        rel = rel[rel > 0]
        if len(rel):
            st.caption(f"🏦 **FHSS released at purchase (median ~{money(float(np.median(rel)))}, nominal):** "
                       "85% of concessional contributions + 100% of non-concessional + earnings, "
                       "net of the withdrawal tax (marginal rate − 30%). Added to your deposit.")

    fig = fan_chart(primary, "net_worth", real, PALETTE["buy"], "Net worth")
    fig.add_vline(x=ret_age, line_dash="dot", line_color="gray", annotation_text="retire")
    if plan.buy_home:
        fig.add_vline(x=plan.buy_age, line_dash="dot", line_color="green", annotation_text="buy")
    fig.update_layout(height=460, xaxis_title="Age", yaxis_title="Net worth (AUD)",
                      yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("What's it made of? (median path)")
    comp_fig = go.Figure()
    for comp, col, label in [("super", "#7c3aed", "Super"), ("portfolio", "#2563eb", "ETF portfolio"),
                             ("home_equity", "#16a34a", "Home equity"), ("cash", "#64748b", "Cash")]:
        med = primary.percentiles(comp, pcts=(50,), real=real)["p50"]
        comp_fig.add_trace(go.Scatter(x=primary.ages, y=med, stackgroup="one", name=label, line=dict(color=col, width=0.5)))
    comp_fig.update_layout(height=380, xaxis_title="Age", yaxis_title="Median value (AUD)",
                           yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(comp_fig, width="stretch")

with tabs[1]:
    st.subheader(f"Outcomes at retirement (age {ret_age})")
    sup_ret = primary.at_age("super", ret_age, real)
    liquid = (primary.at_age("super", ret_age, real) + primary.at_age("portfolio", ret_age, real)
              + primary.at_age("cash", ret_age, real))
    gap_prob = float(np.mean(primary.components["spending_gap"].sum(axis=1) > 1))
    sustainable = np.median(liquid) * 0.04

    m = st.columns(4)
    m[0].metric("Median super balance", money(np.median(sup_ret)))
    m[1].metric("Median liquid assets", money(np.median(liquid)))
    m[2].metric("Sustainable income (4% rule)", money(sustainable) + " /yr")
    m[3].metric("Chance of a shortfall", f"{gap_prob*100:.0f}%",
                help="Probability desired spending can't be funded from cash, super, portfolio and home equity in some year.")

    if R["sim"].include_age_pension:
        pens = primary.matrix("age_pension", real=real)
        mask = primary.ages >= config.AGE_PENSION_AGE
        if mask.any():
            med_pension = float(np.median(pens[:, mask].mean(axis=1)))
            st.caption(f"💸 Median Age Pension received: ~{money(med_pension)}/yr (averaged over age "
                       f"{config.AGE_PENSION_AGE}+, means-tested). Supplements drawdowns and lowers shortfall risk.")

    cc = st.columns(2)
    with cc[0]:
        st.markdown("**Super balance at retirement — distribution**")
        h = go.Figure(go.Histogram(x=sup_ret, nbinsx=40, marker_color="#7c3aed"))
        h.add_vline(x=float(np.median(sup_ret)), line_dash="dash", annotation_text="median")
        h.update_layout(height=320, xaxis_tickformat="$,.0f", xaxis_title="Super (AUD)", yaxis_title="Paths")
        st.plotly_chart(h, width="stretch")
    with cc[1]:
        st.markdown("**Super trajectory (with error bands)**")
        sfig = fan_chart(primary, "super", real, "#7c3aed", "Super")
        sfig.add_vline(x=ret_age, line_dash="dot", annotation_text="retire")
        sfig.update_layout(height=320, xaxis_title="Age", yaxis_tickformat="$,.0f", hovermode="x unified", showlegend=False)
        st.plotly_chart(sfig, width="stretch")

    if gap_prob > 0.05:
        st.warning(f"In {gap_prob*100:.0f}% of paths, desired retirement spending of {money(retirement_spending)}/yr "
                   "can't be fully funded. Consider higher contributions, later retirement, lower spend, or home equity.")
    else:
        st.success(f"Desired spending of {money(retirement_spending)}/yr is funded in {(1-gap_prob)*100:.0f}% of paths.")

with tabs[2]:
    st.subheader("How much home can you afford, by age?")
    st.caption("Expected-return projection. Affordable price = the lower of deposit capacity (20% deposit) and "
               "serviceable loan (repayments ≤ 35% of gross income), at 80% LVR.")
    pf = power.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pf.index, y=pf["max_price_deposit"], name="Limited by deposit", line=dict(color="#16a34a", dash="dot")))
    fig.add_trace(go.Scatter(x=pf.index, y=pf["max_price_serviceability"], name="Limited by borrowing", line=dict(color="#d97706", dash="dot")))
    fig.add_trace(go.Scatter(x=pf.index, y=pf["max_affordable_price"], name="Max affordable", line=dict(color="#2563eb", width=3)))
    fig.add_trace(go.Scatter(x=pf.index, y=np.full(len(pf), plan.home_price), name="Target price", line=dict(color="black", dash="dash")))
    fig.update_layout(height=440, xaxis_title="Age", yaxis_title="Property price (nominal AUD)",
                      yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    affordable = pf[pf["max_affordable_price"] >= plan.home_price]
    if len(affordable):
        fa = int(affordable.index[0])
        st.success(f"You can afford a {money(plan.home_price)} home (today's $) from around **age {fa}** "
                   f"({money(pf.loc[fa,'borrowing_capacity'])} borrowing + {money(pf.loc[fa,'deposit_capacity'])} deposit).")
    else:
        st.info(f"On these settings a {money(plan.home_price)} home isn't reached before retirement. "
                "Lower the price, raise savings, or extend the timeline.")
    st.dataframe(pf[["salary", "deposit_capacity", "borrowing_capacity", "max_affordable_price"]].style.format("${:,.0f}"), width="stretch")

with tabs[3]:
    st.subheader("Rent vs buy — lifetime net worth")
    st.caption("Both scenarios invest spare cashflow. The renter invests the deposit and any saving from lower "
               "housing costs; the buyer builds home equity. Same return paths.")
    fig = fan_chart(buy, "net_worth", real, PALETTE["buy"], "Buy")
    fig = fan_chart(rent, "net_worth", real, PALETTE["rent"], "Rent", fig=fig)
    fig.update_layout(height=460, xaxis_title="Age", yaxis_title="Net worth (AUD)",
                      yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    bt, rt = buy.terminal("net_worth", real), rent.terminal("net_worth", real)
    d = st.columns(3)
    d[0].metric("Median terminal — Buy", money(np.median(bt)))
    d[1].metric("Median terminal — Rent", money(np.median(rt)))
    diff = np.median(bt) - np.median(rt)
    d[2].metric("Buy advantage (median)", money(diff), delta=("Buy wins" if diff > 0 else "Rent wins"))

    win = float(np.mean(bt > rt))
    dollar_kind = "today's (real)" if real else "future (nominal)"
    st.markdown(f"**Buying ends ahead of renting in {win*100:.0f}% of paths** (terminal net worth, {dollar_kind} dollars).")
    h = go.Figure()
    h.add_trace(go.Histogram(x=bt, name="Buy", opacity=0.6, marker_color=PALETTE["buy"], nbinsx=40))
    h.add_trace(go.Histogram(x=rt, name="Rent", opacity=0.6, marker_color=PALETTE["rent"], nbinsx=40))
    h.update_layout(barmode="overlay", height=340, xaxis_title="Terminal net worth (AUD)",
                    xaxis_tickformat="$,.0f", yaxis_title="Paths", legend=dict(orientation="h"))
    st.plotly_chart(h, width="stretch")

with tabs[4]:
    st.subheader("ETF return statistics (from real history)")
    st.caption("Annualised expected return & volatility per ETF from Yahoo Finance history. Short-history ETFs "
               "fall back to asset-class assumptions. ⚠️ Recent windows (global/Nasdaq/geared) reflect a strong bull "
               "market and may overstate the future — use the return scenario to temper them.")
    st.dataframe(stats.as_table().style.format({"exp_return": "{:.1%}", "volatility": "{:.1%}", "history_yrs": "{:.0f}"}), width="stretch")
    if stats.used_fallback:
        st.info("Using asset-class fallbacks (short history): " + ", ".join(stats.used_fallback))

    st.subheader("Portfolio summary")
    pfw = plan.portfolio_weights
    exp = sum(w * stats.mean_simple[stats.tickers.index(t)] for t, w in pfw.items() if t in stats.tickers)
    st.write(f"Blended expected return of your CMC portfolio: **{exp:.1%}** nominal / **{(1+exp)/(1+inflation)-1:.1%}** real.")
    st.json({ETF_UNIVERSE[t].label if t in ETF_UNIVERSE else t: f"{w:.0%}" for t, w in pfw.items()})

    st.subheader("Key assumptions in this run")
    a, sc = R["sim"].assumptions, R["sim"]
    st.json({
        "inflation": f"{a.inflation:.1%}", "wage_growth": f"{a.wage_growth:.1%}", "cash_rate": f"{a.cash_rate:.1%}",
        "mortgage_rate": f"{a.mortgage_rate:.1%}", "property_growth": f"{a.property_growth:.1%}",
        "rent_yield": f"{a.rent_yield:.1%}", "monte_carlo_paths": sc.n_paths, "sampling_method": primary.method_used,
        "return_blend": f"{sc.return_blend:.0%} toward {sc.equity_anchor:.1%} equity / {sc.bond_anchor:.1%} bond",
        "savings_split": f"{plan.savings_split_to_mortgage:.0%} to mortgage", "income_milestones": plan.salary_milestones or "none",
        "age_pension": sc.include_age_pension, "div296_applied": sc.div296, "fhss": plan.fhss_enabled,
        "super_guarantee": f"{config.SUPER_GUARANTEE_RATE:.0%}", "concessional_cap": config.CONCESSIONAL_CAP,
    })
