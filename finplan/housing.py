"""Home purchase mechanics: stamp duty, lenders mortgage insurance (LMI),
mortgage amortisation, and the per-year cashflows needed for the rent-vs-buy
counterfactual.

Stamp duty and FHB concessions use the simplified schedules in ``config``.
LMI is an approximation of typical lender premiums by loan-to-value ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


# ---------------------------------------------------------------------------
def stamp_duty(value: float, state: str = "NSW", first_home_buyer: bool = False) -> float:
    """Transfer (stamp) duty on a purchase, with optional FHB concession."""
    state = state if state in config.STAMP_DUTY_SCHEDULES else "OTHER"
    schedule = config.STAMP_DUTY_SCHEDULES[state]

    base_duty = 0.0
    for threshold, base, rate in schedule:
        if value >= threshold:
            base_duty = base + rate * (value - threshold)
    full_duty = base_duty

    if first_home_buyer:
        fhb = config.FHB_STAMP_DUTY.get(state, config.FHB_STAMP_DUTY["OTHER"])
        below, above = fhb["exempt_below"], fhb["concession_above"]
        if value <= below:
            return 0.0
        if below < value < above and above > below:
            fraction = (value - below) / (above - below)
            return full_duty * fraction
    return full_duty


def lmi_premium(loan_amount: float, property_value: float) -> float:
    """Approximate lenders mortgage insurance premium when LVR > 80%."""
    if property_value <= 0:
        return 0.0
    lvr = loan_amount / property_value
    if lvr <= 0.80:
        return 0.0
    if lvr <= 0.85:
        rate = 0.010
    elif lvr <= 0.90:
        rate = 0.020
    elif lvr <= 0.95:
        rate = 0.037
    else:
        rate = 0.045
    return loan_amount * rate


@dataclass
class PurchaseCosts:
    price: float
    deposit: float
    loan: float
    lvr: float
    stamp_duty: float
    lmi: float
    other_costs: float          # conveyancing, inspection, etc.
    cash_required: float        # deposit + stamp duty + lmi + other

    @property
    def upfront_fees(self) -> float:
        return self.stamp_duty + self.lmi + self.other_costs


def purchase_costs(
    price: float,
    deposit: float,
    state: str = "NSW",
    first_home_buyer: bool = False,
    other_costs: float = 3_000.0,
) -> PurchaseCosts:
    """Compute loan size and total cash required to complete a purchase.

    LMI is capitalised into the loan if the deposit is below 20%, matching
    common practice; the deposit shown is the borrower's cash contribution.
    """
    loan_before_lmi = max(price - deposit, 0.0)
    lmi = lmi_premium(loan_before_lmi, price)
    loan = loan_before_lmi + lmi            # LMI capitalised onto the loan
    duty = stamp_duty(price, state, first_home_buyer)
    cash_required = deposit + duty + other_costs
    lvr = loan / price if price else 0.0
    return PurchaseCosts(
        price=price,
        deposit=deposit,
        loan=loan,
        lvr=lvr,
        stamp_duty=duty,
        lmi=lmi,
        other_costs=other_costs,
        cash_required=cash_required,
    )


# ---------------------------------------------------------------------------
def monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    """Standard amortising monthly repayment."""
    if principal <= 0:
        return 0.0
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r) ** (-n))


@dataclass
class Mortgage:
    balance: float
    annual_rate: float
    term_years: int
    _payment: float = 0.0

    def __post_init__(self):
        self._payment = monthly_payment(self.balance, self.annual_rate, self.term_years)

    @property
    def annual_repayment(self) -> float:
        return self._payment * 12

    def step_year(self, extra_repayment: float = 0.0) -> dict[str, float]:
        """Advance 12 monthly payments. Returns interest/principal split."""
        r = self.annual_rate / 12
        interest_paid = 0.0
        principal_paid = 0.0
        for _ in range(12):
            if self.balance <= 0:
                break
            interest = self.balance * r
            principal = min(self._payment - interest, self.balance)
            self.balance -= principal
            interest_paid += interest
            principal_paid += principal
        if extra_repayment > 0 and self.balance > 0:
            extra = min(extra_repayment, self.balance)
            self.balance -= extra
            principal_paid += extra
        return {
            "interest": interest_paid,
            "principal": principal_paid,
            "balance": self.balance,
        }


def annual_ownership_cost(property_value: float, assumptions: config.Assumptions) -> float:
    """Rates, strata, insurance, maintenance for the year (excludes mortgage)."""
    return property_value * assumptions.property_costs


def annual_rent(property_value: float, assumptions: config.Assumptions) -> float:
    """Annual rent for an equivalent home, from the rental-yield assumption."""
    return property_value * assumptions.rent_yield
