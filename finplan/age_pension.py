"""Age Pension means testing (single person, ~2024-25 rates).

Centrelink pays the **lower** of two tests:
  * Assets test  — pension reduces as assessable assets exceed a free area
                   (the family home is exempt; the free area is higher for
                   non-homeowners).
  * Income test  — pension reduces as assessable income (including *deemed*
                   income on financial assets) exceeds a free area.

Couples are not modelled. Pension payments at these levels are tax-free.
"""

from __future__ import annotations

from . import config


def deemed_income(financial_assets: float) -> float:
    """Centrelink deeming on financial assets (single rates)."""
    fa = max(financial_assets, 0.0)
    low = min(fa, config.DEEMING_THRESHOLD_SINGLE) * config.DEEMING_RATE_LOW
    high = max(fa - config.DEEMING_THRESHOLD_SINGLE, 0.0) * config.DEEMING_RATE_HIGH
    return low + high


def assets_test(assessable_assets: float, homeowner: bool) -> float:
    free = (
        config.PENSION_ASSETS_FREE_HOMEOWNER if homeowner
        else config.PENSION_ASSETS_FREE_NONHOMEOWNER
    )
    excess = max(assessable_assets - free, 0.0)
    reduction = (excess / 1_000) * config.PENSION_ASSETS_TAPER
    return max(config.AGE_PENSION_MAX_SINGLE - reduction, 0.0)


def income_test(assessable_income: float) -> float:
    excess = max(assessable_income - config.PENSION_INCOME_FREE_SINGLE, 0.0)
    reduction = excess * config.PENSION_INCOME_TAPER
    return max(config.AGE_PENSION_MAX_SINGLE - reduction, 0.0)


def age_pension(
    assessable_assets: float,
    financial_assets: float,
    other_income: float = 0.0,
    homeowner: bool = True,
) -> float:
    """Annual Age Pension entitlement = min(assets test, income test).

    ``assessable_assets`` excludes the family home. ``financial_assets`` (cash,
    shares, super in pension phase) drive the deemed income for the income test.
    """
    a = assets_test(assessable_assets, homeowner)
    i = income_test(other_income + deemed_income(financial_assets))
    return max(min(a, i), 0.0)
