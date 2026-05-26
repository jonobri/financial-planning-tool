"""The life simulation + Monte Carlo engine.

A ``Plan`` describes the user's life stages and choices. ``run_montecarlo``
samples thousands of correlated return paths for the chosen ETFs (and the super
mix and property), runs the full year-by-year cashflow for each, and aggregates
the results into percentile bands.

Pots tracked each year:
    cash        Savings / offset (earns the cash rate; interest is taxable).
    portfolio   Outside-super ETF investments (DCA in; total return; dividends
                taxed yearly with franking; CGT on sale via average cost base).
    super       Superannuation (SG + salary sacrifice; 15% in; concessional
                earnings tax in accumulation; tax-free in pension phase).
    home_equity Property value minus mortgage balance.

Cashflow waterfall:
    Working years  — salary funds tax, living costs, housing, and DCA; leftover
                     cash above a buffer is swept into the portfolio.
    Retirement     — spending is drawn cash -> super (if accessible) -> portfolio.

Two scenarios can be compared: ``buy`` (purchase at a chosen year) and ``rent``
(rent forever, invest what would have been spent on the home) — the rent-vs-buy
counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import age_pension, config, housing, tax
from .etf_universe import ETF_UNIVERSE
from .market_data import ReturnStats, apply_return_scenario, compute_return_stats
from .superann import SuperAccount


# ---------------------------------------------------------------------------
@dataclass
class Plan:
    # Life stages
    current_age: int = 32
    retirement_age: int = 60
    end_age: int = 95

    # Income & expenses (nominal, today's dollars at start)
    current_salary: float = 110_000
    salary_growth: float | None = None          # None -> assumptions.wage_growth
    living_expenses: float = 45_000             # annual, excl. housing
    retirement_spending: float = 60_000         # annual desired spend in retirement

    # Starting balances
    starting_cash: float = 30_000
    starting_portfolio: float = 20_000
    starting_super: float = 90_000

    # Investing (outside super, "CMC" account)
    portfolio_weights: dict[str, float] = field(default_factory=lambda: {"VAS.AX": 0.40, "VGS.AX": 0.60})
    # Saving: surplus (take-home − living − housing) is split between extra
    # mortgage repayments and ETF investing. 0 = all to ETFs, 1 = all to mortgage.
    savings_split_to_mortgage: float = 0.40

    # Optional career income step-ups: list of (age, new_salary). Between
    # milestones, salary grows at `salary_growth` / wage growth.
    salary_milestones: list[tuple[int, float]] = field(default_factory=list)

    # Super
    salary_sacrifice: float = 0.0
    super_weights: dict[str, float] = field(
        default_factory=lambda: {"VGS.AX": 0.55, "VAS.AX": 0.30, "VAF.AX": 0.15}
    )

    # First Home Super Saver (FHSS) — voluntary contributions released at purchase
    fhss_enabled: bool = False
    fhss_annual: float = 15_000
    fhss_already_contributed: float = 0.0   # gross FHSS already contributed to date

    # Housing
    buy_home: bool = True
    buy_age: int = 35
    home_price: float = 850_000
    deposit_target: float = 170_000             # cash deposit contributed
    mortgage_term: int = 30
    state: str = "NSW"
    first_home_buyer: bool = True

    def years(self) -> int:
        return self.end_age - self.current_age + 1

    def all_tickers(self) -> list[str]:
        return sorted(set(self.portfolio_weights) | set(self.super_weights))


@dataclass
class SimResult:
    years: np.ndarray                  # calendar years
    ages: np.ndarray
    components: dict[str, np.ndarray]  # name -> (n_paths, n_years), nominal AUD
    stats: ReturnStats
    plan: Plan
    scenario: str
    inflation: float = 0.025
    method_used: str = "mvn"
    bootstrap_ok: bool = True

    def _deflator(self) -> np.ndarray:
        """(1+inflation)^t per year, to convert nominal -> today's dollars."""
        t = self.ages - self.ages[0]
        return (1 + self.inflation) ** t

    def matrix(self, component: str, real: bool = True) -> np.ndarray:
        data = self.components[component]
        return data / self._deflator() if real else data

    def percentiles(self, component: str, pcts=(10, 25, 50, 75, 90), real: bool = True) -> pd.DataFrame:
        data = self.matrix(component, real=real)
        rows = {f"p{p}": np.percentile(data, p, axis=0) for p in pcts}
        return pd.DataFrame(rows, index=self.ages)

    def terminal(self, component: str, real: bool = True) -> np.ndarray:
        return self.matrix(component, real=real)[:, -1]

    def at_age(self, component: str, age: int, real: bool = True) -> np.ndarray:
        idx = int(age - self.ages[0])
        return self.matrix(component, real=real)[:, idx]


# ---------------------------------------------------------------------------
def _weighted(weights: dict[str, float], attr: str) -> float:
    """Distribution-yield / franking weighted average over a portfolio."""
    total = sum(weights.values()) or 1.0
    return sum(
        w / total * getattr(ETF_UNIVERSE[t], attr)
        for t, w in weights.items()
        if t in ETF_UNIVERSE
    )


def _weight_vector(weights: dict[str, float], tickers: list[str]) -> np.ndarray:
    v = np.array([weights.get(t, 0.0) for t in tickers])
    s = v.sum()
    return v / s if s else v


def _sample_asset_returns(stats: ReturnStats, n_paths: int, n_years: int, rng) -> np.ndarray:
    """Correlated simple annual returns, shape (n_paths, n_years, n_assets)."""
    n = len(stats.tickers)
    L = np.linalg.cholesky(stats.cov_log + 1e-12 * np.eye(n))
    z = rng.standard_normal((n_paths, n_years, n))
    log_r = stats.mean_log + z @ L.T
    return np.expm1(log_r)


def _bootstrap_asset_returns(stats: ReturnStats, n_paths, n_years, rng, block: int):
    """Block bootstrap of real historical annual returns.

    Resamples wrap-around blocks of actual past years (preserving real crashes
    and cross-asset co-movement), then recentres each asset to the (possibly
    scenario-adjusted) target mean. Returns ``None`` if there isn't a long enough
    common history across the selected assets — the caller then falls back to MVN.
    """
    df = stats.annual_simple.reindex(columns=stats.tickers).dropna(axis=0, how="any")
    hist = df.values
    T = hist.shape[0]
    if T < 8:
        return None
    shift = stats.mean_simple - hist.mean(axis=0)   # recentre to target mean
    out = np.empty((n_paths, n_years, hist.shape[1]))
    for p in range(n_paths):
        y = 0
        while y < n_years:
            start = int(rng.integers(0, T))
            length = min(block, n_years - y)
            for k in range(length):
                out[p, y + k] = hist[(start + k) % T]
            y += length
    return out + shift


def _sample_property_returns(a: config.Assumptions, n_paths, n_years, rng) -> np.ndarray:
    m, v = a.property_growth, a.property_growth_vol
    sigma = np.sqrt(np.log1p((v / (1 + m)) ** 2))
    mu = np.log1p(m) - 0.5 * sigma**2
    return np.expm1(rng.normal(mu, sigma, (n_paths, n_years)))


def _salary_schedule(plan: Plan, wage_growth: float) -> dict[int, float]:
    """Salary for each age: grows at wage_growth, jumps to milestone values, and
    is zero from retirement age onward."""
    milestones = {int(a): float(s) for a, s in plan.salary_milestones}
    schedule: dict[int, float] = {}
    sal = plan.current_salary
    for age in range(plan.current_age, plan.end_age + 1):
        if age in milestones:
            sal = milestones[age]
        schedule[age] = sal if age < plan.retirement_age else 0.0
        sal *= (1 + wage_growth)
    return schedule


# ---------------------------------------------------------------------------
def run_montecarlo(
    plan: Plan,
    sim: config.SimConfig | None = None,
    scenario: str = "buy",
    stats: ReturnStats | None = None,
) -> SimResult:
    """Run the full Monte Carlo for one scenario ('buy' or 'rent')."""
    sim = sim or config.SimConfig()
    a = sim.assumptions
    rng = np.random.default_rng(sim.seed)

    tickers = plan.all_tickers()
    if stats is None:
        stats = compute_return_stats(tickers)
    # Shrink returns toward long-run anchors if requested (scenario blend).
    stats = apply_return_scenario(stats, sim.return_blend, sim.equity_anchor, sim.bond_anchor)

    pf_w = _weight_vector(plan.portfolio_weights, stats.tickers)
    su_w = _weight_vector(plan.super_weights, stats.tickers)

    n_years = plan.years()
    n_paths = sim.n_paths

    # Sample correlated asset returns by the chosen method.
    method_used, bootstrap_ok = sim.sampling_method, True
    asset_r = None
    if sim.sampling_method == "bootstrap":
        asset_r = _bootstrap_asset_returns(stats, n_paths, n_years, rng, sim.block_years)
        if asset_r is None:                # insufficient common history
            bootstrap_ok, method_used = False, "mvn"
    if asset_r is None:
        asset_r = _sample_asset_returns(stats, n_paths, n_years, rng)

    port_r = asset_r @ pf_w               # (n_paths, n_years)
    super_r = asset_r @ su_w
    prop_r = _sample_property_returns(a, n_paths, n_years, rng)

    wage_growth = plan.salary_growth if plan.salary_growth is not None else a.wage_growth
    div_yield = _weighted(plan.portfolio_weights, "dist_yield")
    franking = _weighted(plan.portfolio_weights, "franking")
    super_income_yield = _weighted(plan.super_weights, "dist_yield")
    # Deterministic salary trajectory: grow at wage_growth, jump at milestones,
    # zero once retired. Same across all paths.
    salary_by_age = _salary_schedule(plan, wage_growth)

    # Output buffers
    comp_names = [
        "net_worth", "super", "portfolio", "cash", "home_equity", "property_value",
        "mortgage", "spending_gap", "cashflow_gap", "age_pension", "fhss_released",
    ]
    out = {k: np.zeros((n_paths, n_years)) for k in comp_names}

    cash_rate = a.cash_rate
    buy_year_index = plan.buy_age - plan.current_age
    # FHSS contributions are only made before purchase, in the buy scenario.
    fhss_active = plan.fhss_enabled and scenario == "buy" and plan.buy_home

    for p in range(n_paths):
        cash = plan.starting_cash
        portfolio = plan.starting_portfolio
        cost_base = plan.starting_portfolio
        sup = SuperAccount(plan.starting_super, income_yield=super_income_yield, div296=sim.div296)
        owns_home = False
        property_value = 0.0
        mortgage: housing.Mortgage | None = None
        equity_released = 0.0          # cumulative downsizing / reverse-mortgage draws
        # Seed from any FHSS already contributed (assumed concessional: ~85%
        # releasable, then grows with returns until purchase).
        fhss_contributed = plan.fhss_already_contributed
        fhss_balance = (0.85 * plan.fhss_already_contributed) if fhss_active else 0.0
        fhss_nonconc = 0.0             # non-concessional principal (tax-free on release)

        for t in range(n_years):
            age = plan.current_age + t
            retired = age >= plan.retirement_age
            pension_phase = retired and age >= config.PRESERVATION_AGE
            salary = salary_by_age[age]
            # Index living costs to inflation so they stay constant in real terms.
            infl_factor = (1 + a.inflation) ** t
            living = plan.living_expenses * infl_factor
            retirement_spend = plan.retirement_spending * infl_factor

            # ---- super step (uses this year's super return) ----
            # Earnings are tax-free once in pension phase (retired and age >= 60);
            # drawdowns themselves are handled in the retirement waterfall below.
            sup.step(
                gross_return=super_r[p, t],
                salary=salary if not retired else 0.0,
                salary_sacrifice=plan.salary_sacrifice if not retired else 0.0,
                total_income=salary,
                pension_phase=pension_phase,
            )

            # ---- FHSS: grow earmarked balance and add a contribution ----
            # Contribute up to $15k/yr and $50k total (the FHSS caps). Amounts
            # within the remaining concessional cap are pre-tax (15% in, releasable
            # at 85%); any excess goes in as non-concessional (released in full).
            fhss_contrib = 0.0
            if fhss_active:
                fhss_balance *= (1 + super_r[p, t] * (1 - config.SUPER_EARNINGS_TAX_RATE))
                if not retired and t < buy_year_index and fhss_contributed < config.FHSS_TOTAL_CAP:
                    fhss_contrib = min(
                        plan.fhss_annual, config.FHSS_ANNUAL_CAP,
                        config.FHSS_TOTAL_CAP - fhss_contributed,
                    )
                    sg = salary * config.SUPER_GUARANTEE_RATE
                    cap_room = max(config.CONCESSIONAL_CAP - sg - plan.salary_sacrifice, 0.0)
                    conc = min(fhss_contrib, cap_room)          # pre-tax portion
                    nonconc = fhss_contrib - conc               # after-tax portion
                    fhss_contributed += fhss_contrib
                    # Releasable principal: 85% of concessional + 100% of non-concessional.
                    fhss_balance += conc * (1 - config.CONTRIBUTIONS_TAX_RATE) + nonconc
                    fhss_nonconc += nonconc                     # tax-free-on-release principal

            # ---- interest & dividends (taxable) ----
            interest_income = cash * cash_rate if cash > 0 else 0.0
            taxable_interest = interest_income
            dividends = max(portfolio, 0.0) * div_yield

            # ---- housing purchase event ----
            if (
                scenario == "buy"
                and plan.buy_home
                and not owns_home
                and t == buy_year_index
            ):
                pc = housing.purchase_costs(
                    plan.home_price, plan.deposit_target, plan.state, plan.first_home_buyer
                )
                # Release FHSS savings to cash. The assessable part (released
                # concessional principal + all earnings) is taxed at marginal rate
                # less 30%; non-concessional principal comes out tax-free.
                if fhss_balance > 0:
                    assessable = max(fhss_balance - fhss_nonconc, 0.0)
                    fhss_tax = tax.fhss_withdrawal_tax(assessable, salary)
                    net_release = fhss_balance - fhss_tax
                    out["fhss_released"][p, t] = net_release
                    cash += net_release
                    fhss_balance = 0.0
                need = pc.cash_required
                # Fund from cash, then by selling portfolio (CGT applies).
                from_cash = min(max(cash, 0.0), need)
                cash -= from_cash
                remaining = need - from_cash
                if remaining > 0 and portfolio > 0:
                    sell = min(portfolio, remaining)
                    gain = sell * (1 - cost_base / portfolio)
                    cgt = tax.capital_gains_tax(gain, salary)
                    cost_base *= (1 - sell / portfolio)
                    portfolio -= sell
                    remaining -= sell
                    cash -= cgt           # CGT paid from cash (may go negative)
                cash -= remaining         # any remaining gap -> negative cash (funding gap)
                mortgage = housing.Mortgage(pc.loan, a.mortgage_rate, plan.mortgage_term)
                owns_home = True
                property_value = plan.home_price

            # ---- housing ongoing costs ----
            if owns_home:
                property_value *= (1 + prop_r[p, t])
                m = mortgage.step_year() if mortgage else {"interest": 0, "principal": 0, "balance": 0}
                housing_cash = m["interest"] + m["principal"] + housing.annual_ownership_cost(property_value, a)
                mortgage_balance = mortgage.balance if mortgage else 0.0
            else:
                # Renting (always in 'rent' scenario, or pre-purchase in 'buy').
                rent = plan.home_price * a.rent_yield * (1 + a.rent_growth) ** t
                housing_cash = rent
                mortgage_balance = 0.0

            # ---- taxes ---- (FHSS contributions are pre-tax, like salary sacrifice)
            taxable_emp = (salary - plan.salary_sacrifice - fhss_contrib if not retired else 0.0) + taxable_interest
            inc_tax = tax.total_income_tax(max(taxable_emp, 0.0))
            div_tax = tax.tax_on_dividends(dividends, franking, max(taxable_emp, 0.0))

            # ---- portfolio growth (total return incl. reinvested dividends) ----
            portfolio = max(portfolio, 0.0) * (1 + port_r[p, t])
            cost_base += dividends            # reinvested dividends raise cost base

            # ---- cashflow ----
            pension = 0.0
            cashflow_gap = 0.0
            if not retired:
                cash_salary = salary - plan.salary_sacrifice - fhss_contrib
                cash += cash_salary + interest_income - inc_tax - div_tax - living - housing_cash
                # Total savings = cash above a 6-month buffer. Split between extra
                # mortgage repayments and ETF investing.
                buffer = 0.5 * living
                savings = max(cash - buffer, 0.0)
                if savings > 0:
                    has_loan = owns_home and mortgage is not None and mortgage.balance > 0
                    to_mortgage = savings * plan.savings_split_to_mortgage if has_loan else 0.0
                    paid = min(to_mortgage, mortgage.balance) if mortgage else 0.0
                    if mortgage:
                        mortgage.balance -= paid
                    invest = savings - paid                 # remainder (incl. any unused) to ETFs
                    cash -= savings
                    portfolio += invest
                    cost_base += invest
                # If income didn't cover outgoings (e.g. a stretched deposit),
                # sell investments rather than carrying a negative cash balance.
                if cash < 0 and portfolio > 0:
                    sell = min(portfolio, -cash)
                    gain = sell * (1 - cost_base / portfolio)
                    cost_base *= (1 - sell / portfolio)
                    portfolio -= sell
                    cash += sell
                    cgt = tax.capital_gains_tax(gain, salary)
                    pay = min(portfolio, cgt)
                    portfolio -= pay
                    cash -= (cgt - pay)
                # If commitments still exceed income + assets, the plan is
                # unaffordable this year: record the shortfall and floor cash at 0
                # (rather than accumulating phantom debt).
                if cash < 0:
                    cashflow_gap = -cash
                    cash = 0.0
                spending_gap = 0.0
            else:
                # Retirement: fund spending + housing from cash -> super -> portfolio.
                cash += interest_income
                # Age Pension (means-tested) supplements income from age 67.
                if sim.include_age_pension and age >= config.AGE_PENSION_AGE:
                    financial = max(cash, 0.0) + max(portfolio, 0.0) + sup.balance
                    pension = age_pension.age_pension(
                        assessable_assets=financial, financial_assets=financial,
                        other_income=0.0, homeowner=owns_home,
                    )
                need = max(retirement_spend + housing_cash + inc_tax + div_tax - interest_income - pension, 0.0)

                take_cash = min(max(cash, 0.0), need)
                cash -= take_cash
                need -= take_cash

                if need > 0 and pension_phase and sup.balance > 0:
                    draw = min(sup.balance, need)
                    sup.balance -= draw           # tax-free in pension phase
                    need -= draw

                if need > 0 and portfolio > 0:
                    sell = min(portfolio, need)
                    gain = sell * (1 - cost_base / portfolio)
                    cost_base *= (1 - sell / portfolio)
                    portfolio -= sell
                    need -= sell
                    # Settle CGT from remaining portfolio; fund any remainder from
                    # the next steps (super already drawn, then equity) rather than
                    # overdrawing cash.
                    cgt = tax.capital_gains_tax(gain, 0.0)
                    pay = min(portfolio, cgt)
                    portfolio -= pay
                    need += (cgt - pay)

                # Last resort: release home equity (downsize / reverse mortgage),
                # up to 60% of the property value, leaving a buffer.
                if need > 0 and owns_home:
                    available_equity = max(0.6 * property_value - mortgage_balance - equity_released, 0.0)
                    release = min(available_equity, need)
                    equity_released += release
                    need -= release

                spending_gap = max(need, 0.0)

            # ---- record ---- (FHSS savings sit in super until released)
            # Re-read the mortgage balance so any extra repayment this year shows.
            mortgage_balance = mortgage.balance if (owns_home and mortgage) else 0.0
            home_equity = max(property_value - mortgage_balance - equity_released, 0.0) if owns_home else 0.0
            net_worth = cash + portfolio + sup.balance + fhss_balance + home_equity
            out["net_worth"][p, t] = net_worth
            out["super"][p, t] = sup.balance + fhss_balance
            out["portfolio"][p, t] = max(portfolio, 0.0)
            out["cash"][p, t] = cash
            out["home_equity"][p, t] = home_equity
            out["property_value"][p, t] = property_value if owns_home else 0.0
            out["mortgage"][p, t] = mortgage_balance
            out["spending_gap"][p, t] = spending_gap
            out["cashflow_gap"][p, t] = cashflow_gap
            out["age_pension"][p, t] = pension

    ages = np.arange(plan.current_age, plan.end_age + 1)
    years = sim.start_year + (ages - plan.current_age)
    return SimResult(
        years=years, ages=ages, components=out, stats=stats,
        plan=plan, scenario=scenario, inflation=a.inflation,
        method_used=method_used, bootstrap_ok=bootstrap_ok,
    )


# ---------------------------------------------------------------------------
def housing_purchasing_power(
    plan: Plan,
    sim: config.SimConfig | None = None,
    serviceability_ratio: float = 0.35,
    deposit_fraction: float = 0.20,
    stats: ReturnStats | None = None,
) -> pd.DataFrame:
    """Deterministic (expected-return) projection of max affordable home price by age.

    Combines two constraints at an 80% LVR (no LMI):
      * Deposit capacity = accumulated cash + portfolio.
      * Borrowing capacity = max loan whose P&I repayment is within
        ``serviceability_ratio`` of gross income.
    """
    sim = sim or config.SimConfig()
    a = sim.assumptions
    tickers = plan.all_tickers()
    if stats is None:
        stats = compute_return_stats(tickers)
    stats = apply_return_scenario(stats, sim.return_blend, sim.equity_anchor, sim.bond_anchor)
    pf_w = _weight_vector(plan.portfolio_weights, stats.tickers)
    exp_port = float(stats.mean_simple @ pf_w)
    wage_growth = plan.salary_growth if plan.salary_growth is not None else a.wage_growth
    salary_by_age = _salary_schedule(plan, wage_growth)

    cash = plan.starting_cash
    portfolio = plan.starting_portfolio
    rows = []
    r = a.mortgage_rate / 12
    n = plan.mortgage_term * 12
    annuity = (1 - (1 + r) ** (-n)) / r if r else n

    for age in range(plan.current_age, plan.retirement_age + 1):
        salary = salary_by_age.get(age, plan.current_salary)
        deposit_capacity = max(cash + portfolio, 0.0)
        monthly_capacity = serviceability_ratio * salary / 12
        max_loan = monthly_capacity * annuity
        price_by_deposit = deposit_capacity / deposit_fraction
        price_by_serviceability = max_loan / (1 - deposit_fraction)
        max_price = min(price_by_deposit, price_by_serviceability)
        rows.append(
            {
                "age": age,
                "year": sim.start_year + (age - plan.current_age),
                "salary": salary,
                "deposit_capacity": deposit_capacity,
                "borrowing_capacity": max_loan,
                "max_price_deposit": price_by_deposit,
                "max_price_serviceability": price_by_serviceability,
                "max_affordable_price": max_price,
            }
        )
        # Advance one year (expected returns). Pre-purchase, all savings are
        # liquid/invested toward the deposit.
        inc_tax = tax.total_income_tax(salary)
        rent = plan.home_price * a.rent_yield * (1 + a.rent_growth) ** (age - plan.current_age)
        savings = max(salary - inc_tax - plan.living_expenses - rent, 0.0)
        portfolio = portfolio * (1 + exp_port) + savings
        cash = max(cash, 0.0)

    return pd.DataFrame(rows).set_index("age")
