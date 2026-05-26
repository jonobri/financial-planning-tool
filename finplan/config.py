"""Australian tax & superannuation constants and global modelling assumptions.

Figures are for the **FY2025-26** financial year (1 July 2025 – 30 June 2026)
unless noted. All amounts are in nominal AUD.

These are deliberately kept as plain dataclasses/dicts so the Streamlit UI can
override any assumption at runtime without touching the engine.

Sources (as legislated / published at time of writing):
    - Resident income tax rates 2024-25 onward ("Stage 3" cuts), ATO.
    - Super Guarantee rate 12% from 1 July 2025.
    - Concessional contributions cap $30,000; Div 293 threshold $250,000.
    - CGT 50% discount for assets held > 12 months.
    - Superannuation preservation age 60 (born on/after 1 July 1964).

NOTE: This tool gives projections, not financial advice. Rules are simplified
(e.g. low-income offsets, Medicare levy thresholds, and indexation of caps over
time are approximated). Verify specifics against the ATO before acting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Income tax (resident individuals, FY2025-26)
# ---------------------------------------------------------------------------
# Each tuple is (lower_bound_inclusive, marginal_rate). Tax is the sum of the
# rate applied to income within each band above the lower bound.
INCOME_TAX_BRACKETS_2025_26: list[tuple[float, float]] = [
    (0, 0.00),
    (18_200, 0.16),
    (45_000, 0.30),
    (135_000, 0.37),
    (190_000, 0.45),
]

MEDICARE_LEVY_RATE = 0.02
# Simplified: full 2% levy applies above this taxable income. Real rules phase
# the levy in between a lower and upper threshold; we approximate with a single
# threshold for a single adult.
MEDICARE_LEVY_THRESHOLD = 27_222


# ---------------------------------------------------------------------------
# Superannuation
# ---------------------------------------------------------------------------
SUPER_GUARANTEE_RATE = 0.12          # 12% from 1 July 2025
CONCESSIONAL_CAP = 30_000            # annual concessional (pre-tax) cap
NON_CONCESSIONAL_CAP = 120_000       # annual non-concessional (after-tax) cap
CONTRIBUTIONS_TAX_RATE = 0.15        # 15% tax on concessional contributions
SUPER_EARNINGS_TAX_RATE = 0.15       # 15% on earnings in accumulation phase
# Effective tax on the capital-growth portion of super returns is lower because
# of the 1/3 CGT discount inside super (15% * 2/3 = 10%).
SUPER_CG_EARNINGS_TAX_RATE = 0.10
DIV293_THRESHOLD = 250_000           # extra 15% contributions tax above this
DIV293_EXTRA_RATE = 0.15
PRESERVATION_AGE = 60                # can access super (born on/after 1/7/1964)
# Maximum super balance assumed taxed at concessional rate. Earnings above the
# transfer balance cap in retirement are taxed; ignored in this simplified model.


# ---------------------------------------------------------------------------
# Capital gains tax (assets held outside super)
# ---------------------------------------------------------------------------
CGT_DISCOUNT = 0.50                  # 50% discount on gains held > 12 months
CGT_DISCOUNT_MIN_HOLD_DAYS = 365

# ---------------------------------------------------------------------------
# Franking (dividend imputation)
# ---------------------------------------------------------------------------
COMPANY_TAX_RATE = 0.30              # used to gross up franked dividends


# ---------------------------------------------------------------------------
# Division 296 — extra 15% tax on earnings attributable to the super balance
# above $3,000,000 (legislated to apply from 1 July 2025; threshold NOT indexed).
# ---------------------------------------------------------------------------
DIV296_BALANCE_THRESHOLD = 3_000_000
DIV296_RATE = 0.15


# ---------------------------------------------------------------------------
# First Home Super Saver (FHSS) scheme — simplified.
# Voluntary contributions (concessional here) up to $15k/yr and $50k total can be
# released for a first-home deposit. On release, concessional amounts are taxed at
# marginal rate less a 30% offset.
# ---------------------------------------------------------------------------
FHSS_ANNUAL_CAP = 15_000
FHSS_TOTAL_CAP = 50_000
FHSS_WITHDRAWAL_OFFSET = 0.30


# ---------------------------------------------------------------------------
# Age Pension (single, ~2024-25 rates). Tax-free at these levels. We take the
# lower of the assets test and the income (deeming) test. Couples not modelled.
# ---------------------------------------------------------------------------
AGE_PENSION_AGE = 67
AGE_PENSION_MAX_SINGLE = 29_754              # base + max supplements, annual
PENSION_ASSETS_FREE_HOMEOWNER = 314_000
PENSION_ASSETS_FREE_NONHOMEOWNER = 566_000
PENSION_ASSETS_TAPER = 78.0                  # $/yr reduction per $1,000 over free area
PENSION_INCOME_FREE_SINGLE = 5_512           # annual income-test free area
PENSION_INCOME_TAPER = 0.50                  # 50c per $1 of income over free area
DEEMING_THRESHOLD_SINGLE = 62_600
DEEMING_RATE_LOW = 0.0025
DEEMING_RATE_HIGH = 0.0225


# ---------------------------------------------------------------------------
# Long-run return anchors (nominal total return) for the "return scenario" blend.
# Used to shrink optimistic recent history toward sustainable long-run figures.
# ---------------------------------------------------------------------------
DEFAULT_EQUITY_ANCHOR = 0.080
DEFAULT_BOND_ANCHOR = 0.040


# ---------------------------------------------------------------------------
# Global economic assumptions (user-overridable defaults)
# ---------------------------------------------------------------------------
@dataclass
class Assumptions:
    """Macro assumptions used across the projection. All rates are annual."""

    inflation: float = 0.025          # CPI, used to convert nominal -> real
    wage_growth: float = 0.035        # nominal salary growth
    cash_rate: float = 0.040          # return on cash / offset / savings
    mortgage_rate: float = 0.062      # nominal home loan interest rate
    property_growth: float = 0.055    # nominal residential property growth
    property_growth_vol: float = 0.09 # std dev of annual property growth
    rent_yield: float = 0.038         # annual rent as % of property value
    rent_growth: float = 0.035        # nominal growth in rent
    property_costs: float = 0.012     # ongoing ownership costs (rates, strata,
                                      # maintenance, insurance) as % of value/yr
    transaction_cost_buy: float = 0.0 # buyer agent/legal (stamp duty handled
                                      # separately); kept 0 by default
    selling_cost: float = 0.025       # agent + marketing on property sale

    def real(self, nominal_rate: float) -> float:
        """Convert a nominal annual rate to a real (post-inflation) rate."""
        return (1 + nominal_rate) / (1 + self.inflation) - 1


# ---------------------------------------------------------------------------
# Stamp duty schedules (transfer duty) — simplified standard schedules.
# Keyed by state. Each is a list of (threshold, base_duty, rate_above_threshold).
# Duty = base_duty + rate * (value - threshold) for the highest threshold <= value.
# These approximate FY2024-25 schedules and ignore many concessions.
# ---------------------------------------------------------------------------
STAMP_DUTY_SCHEDULES: dict[str, list[tuple[float, float, float]]] = {
    # NSW transfer duty (approx; rounded per-$100 rules simplified to marginal).
    "NSW": [
        (0, 0, 0.0125),
        (17_000, 212, 0.015),
        (37_000, 512, 0.0175),
        (99_000, 1_597, 0.035),
        (372_000, 11_152, 0.045),
        (1_240_000, 50_212, 0.055),
        (4_670_000, 238_862, 0.07),
    ],
    # VIC general (non-PPR) duty, simplified marginal schedule.
    "VIC": [
        (0, 0, 0.014),
        (25_000, 350, 0.024),
        (130_000, 2_870, 0.05),
        (960_000, 44_370, 0.055),
        (2_000_000, 110_000, 0.065),
    ],
    # Rough national fallback (~ QLD-ish) if state unknown.
    "OTHER": [
        (0, 0, 0.015),
        (75_000, 1_125, 0.035),
        (540_000, 17_325, 0.045),
        (1_000_000, 38_025, 0.0575),
    ],
}

# First-home-buyer full-exemption / concession thresholds (simplified).
# Below `exempt_below` => no duty; between => linear concession to `concession_above`.
FHB_STAMP_DUTY: dict[str, dict[str, float]] = {
    "NSW": {"exempt_below": 800_000, "concession_above": 1_000_000},
    "VIC": {"exempt_below": 600_000, "concession_above": 750_000},
    "OTHER": {"exempt_below": 0, "concession_above": 0},
}


@dataclass
class SimConfig:
    """Top-level simulation configuration."""

    start_year: int = 2026
    n_paths: int = 2_000             # Monte Carlo paths
    seed: int | None = 42
    percentiles: tuple[int, ...] = (10, 25, 50, 75, 90)
    assumptions: Assumptions = field(default_factory=Assumptions)

    # --- return scenario: blend historical means toward long-run anchors ---
    return_blend: float = 0.0        # 0 = pure historical, 1 = pure anchor
    equity_anchor: float = DEFAULT_EQUITY_ANCHOR
    bond_anchor: float = DEFAULT_BOND_ANCHOR

    # --- sampling method ---
    sampling_method: str = "mvn"     # "mvn" (multivariate normal) or "bootstrap"
    block_years: int = 4             # block length for the historical bootstrap

    # --- policy toggles ---
    include_age_pension: bool = True
    div296: bool = False             # apply the $3M Division 296 super tax
