"""Superannuation accumulation & drawdown.

Models a single super account through accumulation (working years) and pension
(retirement) phase:

* Contributions: employer SG + optional salary sacrifice, capped at the
  concessional cap, taxed at 15% on the way in (plus Div 293 for high earners).
* Earnings: taxed at 15% on the income component and ~10% on the capital-growth
  component during accumulation; **tax-free** once in pension phase (post age 60
  and retired).
* Drawdown: withdrawals in pension phase are tax-free (age 60+).

The investment return each year is supplied by the simulation (it comes from the
same Monte Carlo engine, using the member's chosen super investment mix).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, tax


@dataclass
class SuperAccount:
    balance: float = 0.0
    income_yield: float = 0.025      # income (taxable) share of total return
    div296: bool = False             # apply the $3M Division 296 extra tax

    def concessional_contribution(
        self,
        salary: float,
        sg_rate: float = config.SUPER_GUARANTEE_RATE,
        salary_sacrifice: float = 0.0,
    ) -> float:
        """Gross concessional contribution for the year, capped at the cap."""
        gross = salary * sg_rate + salary_sacrifice
        return min(gross, config.CONCESSIONAL_CAP)

    def step(
        self,
        gross_return: float,
        salary: float,
        sg_rate: float = config.SUPER_GUARANTEE_RATE,
        salary_sacrifice: float = 0.0,
        total_income: float = 0.0,
        pension_phase: bool = False,
        drawdown: float = 0.0,
    ) -> dict[str, float]:
        """Advance the account one year. Returns a dict of flows.

        Order: apply this year's earnings, then add net contributions, then take
        any pension drawdown. ``total_income`` is the member's other income, used
        only to decide Div 293 liability on contributions.
        """
        opening = self.balance

        # --- earnings ---
        if pension_phase:
            net_return = gross_return  # tax-free in pension phase
        else:
            income_part = min(self.income_yield, gross_return) if gross_return > 0 else 0.0
            growth_part = gross_return - income_part
            net_return = (
                income_part * (1 - config.SUPER_EARNINGS_TAX_RATE)
                + growth_part * (1 - config.SUPER_CG_EARNINGS_TAX_RATE)
            )
        earnings = opening * net_return
        self.balance = opening + earnings

        # --- Division 296: extra 15% on earnings attributable to balance > $3M ---
        div296_tax = 0.0
        if self.div296 and earnings > 0 and opening > config.DIV296_BALANCE_THRESHOLD:
            excess_fraction = (opening - config.DIV296_BALANCE_THRESHOLD) / opening
            div296_tax = earnings * excess_fraction * config.DIV296_RATE
            self.balance -= div296_tax

        # --- contributions (accumulation only) ---
        gross_contrib = 0.0
        contrib_tax = 0.0
        if not pension_phase:
            gross_contrib = self.concessional_contribution(salary, sg_rate, salary_sacrifice)
            contrib_tax = tax.contributions_tax(gross_contrib, total_income)
            self.balance += gross_contrib - contrib_tax

        # --- drawdown (pension only) ---
        withdrawn = 0.0
        if pension_phase and drawdown > 0:
            withdrawn = min(drawdown, self.balance)
            self.balance -= withdrawn

        return {
            "opening": opening,
            "earnings": earnings,
            "div296_tax": div296_tax,
            "gross_contribution": gross_contrib,
            "contributions_tax": contrib_tax,
            "net_contribution": gross_contrib - contrib_tax,
            "withdrawn": withdrawn,
            "closing": self.balance,
        }


def can_access(age: float) -> bool:
    """Whether super is accessible (preservation age met)."""
    return age >= config.PRESERVATION_AGE
