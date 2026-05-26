# 📈 Australian Financial Planning Tool

An interactive Monte Carlo planner for Australian personal finance. It projects
**retirement outcomes**, **home purchasing power**, and the **rent‑vs‑buy**
counterfactual — using *real* historical returns of individual ASX‑listed ETFs,
with error bands across thousands of simulated futures.

Built for someone dollar‑cost‑averaging into a CMC Markets brokerage account,
contributing to **Superannuation**, and weighing up buying a home.

> ⚠️ **Not financial advice.** Tax/super rules are simplified (see
> [Methodology](#methodology--assumptions)). Verify against the ATO and a
> licensed adviser before making decisions.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>). The first
run downloads ETF price history from Yahoo Finance and caches it under
`data_cache/` (refreshed daily).

Your sidebar inputs **auto-save** to `profiles/autosave.json` on every run and
are restored next time you open the app — use **↺ Reset to defaults** to clear
them. Both `data_cache/` and `profiles/` are git-ignored and stay local.

Run the offline tests (no network needed):

```bash
PYTHONPATH=. .venv/bin/python tests/smoke_test.py     # end-to-end engine
PYTHONPATH=. .venv/bin/python tests/feature_test.py   # scenario blend, bootstrap, FHSS, Div296, pension
```

---

## What it does

| Tab | Answers |
|-----|---------|
| 🧭 **Net worth** | Fan chart of total net worth over your life, with 10–90% / 25–75% Monte Carlo bands, plus a stacked breakdown into super / ETFs / home equity / cash. |
| 🏖️ **Retirement** | Super balance and liquid assets at retirement, sustainable income (4% rule), and the probability of a spending shortfall. |
| 🏠 **Housing power** | Maximum affordable home price by age — the lower of your deposit capacity and serviceable borrowing capacity — vs your target price. |
| ⚖️ **Rent vs buy** | Two scenarios on identical return paths: buy a home, or rent forever and invest the difference. Shows who ends ahead and how often. |
| 📊 **Data & assumptions** | The real per‑ETF expected return / volatility used, history length, fallbacks, and every assumption in the run. |

All dollar figures toggle between **today's dollars (real)** and **future
dollars (nominal)**.

### 🏠 Home Buying Optimiser (second page)

A second page (in the sidebar nav, prefilled from your saved plan) focuses on the
home-purchase decision with a **deterministic** "dead money" lens:

| Tab | Answers |
|-----|---------|
| 💰 **Deposit size** | How big a deposit to put down — trading off less interest (and avoiding LMI above 20%) against the opportunity cost of tying up cash. Flags the best feasible deposit. |
| ⏱️ **When to buy** | Net worth at your horizon for buying in N years vs renting forever, plus rent paid while you wait. |
| 📉 **Dead money over time** | Cumulative rent vs cumulative mortgage interest + ownership costs for a chosen deposit/timing, and the year buying's net worth overtakes renting. |

Rent and mortgage interest are both "dead money" (no equity); principal isn't.
When your investment return beats property growth, smaller deposits / renting
look better; below it, bigger deposits / buying win.

---

## How it works

1. **Real returns** — `market_data.py` downloads adjusted‑close history (total
   return, net of fees) for your chosen ETFs via `yfinance`, then estimates the
   annualised mean and **covariance** of log returns. Implausible data points
   are clipped; short‑history ETFs fall back to asset‑class assumptions.
2. **Scenarios** — recent ETF windows can be over‑optimistic, so a *return
   scenario* control shrinks each asset's mean toward a long‑run anchor (e.g.
   pull the portfolio back toward ~7%) while keeping real volatility and
   correlations. Choose **Monte Carlo** (multivariate‑normal) or a **historical
   bootstrap** that resamples real past return sequences (preserving actual
   crashes); the bootstrap falls back to Monte Carlo if histories are too short.
3. **Simulation** — thousands of correlated annual return paths, plus an
   independent path for property growth, run through a full year‑by‑year
   cashflow each.
3. **The cashflow** routes salary through Australian income tax, super
   contributions, saving and housing, then aggregates the paths into percentile
   bands. Each year's savings (take‑home − living − housing) are split between
   extra mortgage repayments and ETF investing by a single slider.

### Project structure

```
finplan/
  config.py        FY2025‑26 tax/super constants + macro assumptions + stamp duty
  etf_universe.py  Curated ASX ETFs (fees, franking, distribution yield)
  market_data.py   yfinance download + cache + return statistics
  tax.py           Income tax, Medicare, CGT (50% discount), franking, Div 293
  superann.py      SG + salary sacrifice, contributions/earnings tax, drawdown
  housing.py       Stamp duty (+FHB), LMI, mortgage amortisation, ownership costs
  simulation.py    Life simulation, Monte Carlo engine, housing purchasing power
  home_optimizer.py  Deterministic deposit/timing sweeps + dead-money analysis
app.py                       Streamlit UI (main planner)
pages/1_🏠_Home_Buying_Optimiser.py   Second page: home-buying optimiser
tests/smoke_test.py  Offline end‑to‑end engine test
```

---

## Methodology & assumptions

**Tax & super (FY2025‑26, simplified):**
- Resident income tax brackets (post‑"Stage 3"), 2% Medicare levy above a single
  threshold.
- Super Guarantee **12%**, concessional cap **$30,000**, 15% contributions tax,
  Div 293 extra 15% above $250k income.
- Super earnings taxed 15% (≈10% on the capital‑growth share) in accumulation;
  **tax‑free** in pension phase (retired and age ≥ 60).
- Optional **Division 296**: extra 15% on earnings attributable to the super
  balance above $3M (threshold not indexed).
- CGT outside super uses the **50% discount** (held > 12 months); gains tracked
  via an average cost base.
- Franked dividends grossed up at the 30% company rate with refundable credits.
- Optional **FHSS**: pre‑tax voluntary contributions ($15k/yr, $50k cap)
  released at purchase, taxed at marginal rate less 30%.
- Optional **Age Pension** (single, ~2024‑25): the lower of the assets test and
  the deeming‑based income test, supplementing retirement drawdowns from age 67.

**Housing:**
- NSW / VIC / generic stamp‑duty schedules with first‑home‑buyer concessions
  (simplified). LMI applied (and capitalised) above 80% LVR. Standard amortising
  mortgage. In retirement, home equity can be released (downsizing / reverse
  mortgage) up to 60% of value as a last resort.

**Cashflow & saving:**
- Living and retirement spending are held constant in **real** terms (indexed to
  inflation). Each year's surplus above a 6‑month buffer is your *total savings*,
  split between **extra mortgage repayments** and **ETF investing** by a slider —
  so you can compare paying down the loan vs investing.
- **Income step‑ups**: optional age→salary milestones model promotions; salary
  grows at the wage‑growth rate between them.
- If commitments ever exceed income + assets, the year is flagged as a
  **cashflow gap** (an affordability warning) rather than accruing phantom debt.

**Portfolio:**
- Includes **Betashares geared** ETFs (GEAR, GGUS, GHHF, G200). Return stats use
  a robust clip that preserves geared funds' real high volatility (e.g. GEAR's
  ~‑60% month in 2020) while still removing corrupt data points.

### Known limitations
- Long‑horizon (60+ year) **nominal** figures look huge; read results in *real*
  dollars. Recent ETF windows (global / Nasdaq) reflect a strong bull market and
  may **overstate** future returns — use the *return scenario* control to blend
  toward a conservative long‑run anchor.
- Age Pension and FHSS model a **single** person; no couples, no detailed
  Medicare/offset phase‑ins, no land tax, no negative gearing on investment
  property.
- Property returns are modelled independently of equities; mortgage and cash
  rates are deterministic.

---

## Customising

- **ETF universe**: add tickers in `finplan/etf_universe.py` (use the `.AX`
  suffix). Fallback return/vol live in `ASSET_CLASS_FALLBACK`.
- **Tax/super constants**: edit `finplan/config.py` to roll forward to a new
  financial year.
- **Scenarios**: the engine is plain Python — `run_montecarlo(plan, sim,
  scenario=...)` returns numpy arrays you can analyse directly.
