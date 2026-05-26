"""Curated universe of ASX-listed ETFs commonly used for DCA portfolios.

Each ETF carries:
    ticker        Yahoo Finance ticker (ASX tickers use the .AX suffix).
    name          Human-readable name.
    asset_class   Broad category, used for grouping & default correlations.
    mer           Management expense ratio (annual %, as a decimal).
    franking      Approx. proportion of distributions that are franked (0-1).
                  Drives franking-credit calculation for Australian equity.
    dist_yield    Typical annual distribution (income) yield as a decimal.
                  The remainder of total return is treated as capital growth.
    inception     Rough ASX inception year (data may be shorter than history).

These are reasonable defaults; the engine prefers *actual* downloaded history
for return/volatility and only falls back to assumptions when data is missing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ETF:
    ticker: str
    name: str
    asset_class: str
    mer: float
    franking: float
    dist_yield: float
    inception: int

    @property
    def label(self) -> str:
        return f"{self.ticker.replace('.AX', '')} — {self.name}"


ETF_UNIVERSE: dict[str, ETF] = {
    etf.ticker: etf
    for etf in [
        # --- Australian equity ---
        ETF("VAS.AX", "Vanguard Australian Shares (ASX300)", "AU Equity", 0.0007, 0.75, 0.040, 2009),
        ETF("A200.AX", "Betashares Australia 200", "AU Equity", 0.0004, 0.80, 0.040, 2018),
        ETF("STW.AX", "SPDR S&P/ASX 200", "AU Equity", 0.0005, 0.80, 0.042, 2001),
        ETF("IOZ.AX", "iShares Core S&P/ASX 200", "AU Equity", 0.0005, 0.80, 0.041, 2010),
        # --- Global / developed equity ---
        ETF("VGS.AX", "Vanguard MSCI Intl (ex-Aus)", "Global Equity", 0.0018, 0.0, 0.018, 2014),
        ETF("IVV.AX", "iShares S&P 500", "US Equity", 0.0004, 0.0, 0.015, 2007),
        ETF("NDQ.AX", "Betashares Nasdaq 100", "US Equity", 0.0048, 0.0, 0.005, 2015),
        ETF("IWLD.AX", "iShares Core MSCI World (ex-Aus)", "Global Equity", 0.0009, 0.0, 0.017, 2016),
        ETF("BGBL.AX", "Betashares Global Shares (ex-Aus)", "Global Equity", 0.0008, 0.0, 0.016, 2023),
        # --- Diversified / all-in-one ---
        ETF("DHHF.AX", "Betashares Diversified All Growth", "Diversified Growth", 0.0019, 0.30, 0.030, 2019),
        ETF("VDHG.AX", "Vanguard Diversified High Growth", "Diversified Growth", 0.0027, 0.35, 0.045, 2017),
        # --- Emerging markets / small cap ---
        ETF("VGE.AX", "Vanguard FTSE Emerging Markets", "EM Equity", 0.0048, 0.0, 0.030, 2013),
        # --- Bonds / defensive ---
        ETF("VAF.AX", "Vanguard Australian Fixed Interest", "AU Bonds", 0.0010, 0.0, 0.030, 2012),
        ETF("VGB.AX", "Vanguard Australian Govt Bond", "AU Bonds", 0.0016, 0.0, 0.025, 2012),
        # --- Property ---
        ETF("VAP.AX", "Vanguard Australian Property (REIT)", "AU Property", 0.0023, 0.20, 0.045, 2011),
        # --- Gold ---
        ETF("GOLD.AX", "Global X Physical Gold", "Gold", 0.0040, 0.0, 0.0, 2003),
        # --- Betashares geared (internally leveraged) ---
        # GEAR/GGUS are the ~50-65% LVR "hedge fund" geared products; GHHF/G200 are
        # the more moderately geared (~30-40% LVR) Wealth Builder range.
        ETF("GEAR.AX", "Betashares Geared Australian Equity (G)", "Geared Equity", 0.0080, 0.70, 0.020, 2014),
        ETF("GGUS.AX", "Betashares Geared US Equity – Currency Hedged (G)", "Geared Equity", 0.0080, 0.0, 0.010, 2015),
        ETF("GHHF.AX", "Betashares Wealth Builder Diversified All Growth Geared", "Geared Diversified", 0.0035, 0.25, 0.020, 2024),
        ETF("G200.AX", "Betashares Wealth Builder Australia 200 Geared", "Geared Diversified", 0.0035, 0.65, 0.025, 2024),
    ]
}

# Sensible long-run nominal total-return / volatility fallbacks per asset class,
# used only when downloaded history is too short. Real, well-documented guesses.
ASSET_CLASS_FALLBACK: dict[str, tuple[float, float]] = {
    # asset_class: (mean_annual_total_return, annual_volatility)
    "AU Equity": (0.090, 0.16),
    "Global Equity": (0.095, 0.15),
    "US Equity": (0.105, 0.17),
    "Diversified Growth": (0.080, 0.12),
    "EM Equity": (0.085, 0.20),
    "AU Bonds": (0.035, 0.05),
    "AU Property": (0.080, 0.18),
    "Gold": (0.060, 0.16),
    "Geared Equity": (0.120, 0.32),       # ~2.3x exposure; high return, high vol
    "Geared Diversified": (0.105, 0.22),  # ~1.5x exposure (Wealth Builder range)
}


def build_anchors(equity_anchor: float, bond_anchor: float) -> dict[str, float]:
    """Long-run nominal total-return anchor per asset class, from two knobs.

    Used by the 'return scenario' blend to pull optimistic recent history back
    toward sustainable long-run expectations.
    """
    return {
        "AU Equity": equity_anchor,
        "Global Equity": equity_anchor,
        "US Equity": equity_anchor + 0.005,
        "EM Equity": equity_anchor + 0.005,
        "Diversified Growth": 0.85 * equity_anchor + 0.15 * bond_anchor,
        "AU Bonds": bond_anchor,
        "AU Property": equity_anchor - 0.010,
        "Gold": 0.050,
        # Geared long-run anchor ≈ exposure*equity − (exposure−1)*borrow_cost.
        "Geared Equity": 2.3 * equity_anchor - 0.075,
        "Geared Diversified": 1.5 * equity_anchor - 0.030,
    }


def default_portfolio() -> dict[str, float]:
    """A common 'two-fund' Australian DCA split: home bias + global."""
    return {"VAS.AX": 0.40, "VGS.AX": 0.60}


def tickers() -> list[str]:
    return list(ETF_UNIVERSE.keys())
