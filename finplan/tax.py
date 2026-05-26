"""Australian personal tax calculations: income tax, Medicare levy, CGT with the
50% discount, dividend franking credits, and the Div 293 super surcharge.

All functions are pure and vectorisable over scalars; the simulation calls them
per-year per-path. Rules are simplified (see config.py) but capture the levers
that matter for retirement/housing decisions.
"""

from __future__ import annotations

from . import config


def income_tax(taxable_income: float, brackets=None) -> float:
    """Progressive resident income tax (excludes Medicare levy)."""
    if brackets is None:
        brackets = config.INCOME_TAX_BRACKETS_2025_26
    if taxable_income <= 0:
        return 0.0
    tax = 0.0
    for i, (lower, rate) in enumerate(brackets):
        upper = brackets[i + 1][0] if i + 1 < len(brackets) else float("inf")
        if taxable_income > lower:
            taxed_in_band = min(taxable_income, upper) - lower
            tax += taxed_in_band * rate
        else:
            break
    return tax


def medicare_levy(taxable_income: float) -> float:
    if taxable_income <= config.MEDICARE_LEVY_THRESHOLD:
        return 0.0
    return taxable_income * config.MEDICARE_LEVY_RATE


def marginal_rate(taxable_income: float, brackets=None) -> float:
    """Top marginal rate (incl. Medicare) applying to the next dollar earned."""
    if brackets is None:
        brackets = config.INCOME_TAX_BRACKETS_2025_26
    rate = 0.0
    for lower, r in brackets:
        if taxable_income >= lower:
            rate = r
    if taxable_income > config.MEDICARE_LEVY_THRESHOLD:
        rate += config.MEDICARE_LEVY_RATE
    return rate


def total_income_tax(taxable_income: float) -> float:
    """Income tax + Medicare levy."""
    return income_tax(taxable_income) + medicare_levy(taxable_income)


# ---------------------------------------------------------------------------
# Dividends & franking
# ---------------------------------------------------------------------------
def franking_credit(cash_dividend: float, franked_pct: float) -> float:
    """Imputation credit attached to a (partially) franked cash dividend.

    A fully franked dividend carries a credit of div * (company_rate / (1-rate)).
    """
    cr = config.COMPANY_TAX_RATE
    return cash_dividend * franked_pct * (cr / (1 - cr))


def tax_on_dividends(
    cash_dividend: float,
    franked_pct: float,
    other_taxable_income: float,
) -> float:
    """Net tax on a dividend given other income, after franking credit offset.

    Returns the *additional* tax (can be negative if surplus credits refund
    other tax). The grossed-up dividend is taxed at the marginal rate; the
    franking credit is a refundable offset.
    """
    credit = franking_credit(cash_dividend, franked_pct)
    grossed_up = cash_dividend + credit
    rate = marginal_rate(other_taxable_income + grossed_up)
    return grossed_up * rate - credit


# ---------------------------------------------------------------------------
# Capital gains tax (outside super)
# ---------------------------------------------------------------------------
def capital_gains_tax(
    capital_gain: float,
    other_taxable_income: float,
    held_over_12_months: bool = True,
) -> float:
    """Tax on a realised capital gain, applying the 50% discount if eligible.

    The (discounted) gain is added to assessable income and taxed at the
    resulting marginal rate (incl. Medicare).
    """
    if capital_gain <= 0:
        return 0.0
    discounted = capital_gain * (1 - config.CGT_DISCOUNT) if held_over_12_months else capital_gain
    rate = marginal_rate(other_taxable_income + discounted)
    return discounted * rate


# ---------------------------------------------------------------------------
# Superannuation contributions tax
# ---------------------------------------------------------------------------
def fhss_withdrawal_tax(released_amount: float, other_income: float) -> float:
    """Tax on a First Home Super Saver release of concessional amounts.

    The released amount is taxed at the member's marginal rate less a 30% offset
    (floored at zero) — the headline FHSS tax concession.
    """
    if released_amount <= 0:
        return 0.0
    rate = max(marginal_rate(other_income + released_amount) - config.FHSS_WITHDRAWAL_OFFSET, 0.0)
    return released_amount * rate


def contributions_tax(concessional_contribution: float, total_income: float) -> float:
    """15% contributions tax, plus Div 293 extra 15% for high earners.

    Div 293 applies the extra 15% to concessional contributions to the extent
    that (income + contributions) exceeds the $250k threshold.
    """
    tax = concessional_contribution * config.CONTRIBUTIONS_TAX_RATE
    combined = total_income + concessional_contribution
    if combined > config.DIV293_THRESHOLD:
        excess = min(concessional_contribution, combined - config.DIV293_THRESHOLD)
        tax += excess * config.DIV293_EXTRA_RATE
    return tax
