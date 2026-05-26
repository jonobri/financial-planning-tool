"""Market data: download real ASX ETF history via yfinance, cache it locally,
and derive the return statistics the Monte Carlo engine needs.

Design notes
------------
* yfinance ``auto_adjust=True`` 'Close' prices already reflect reinvested
  distributions and are net of the fund's management fee, so annual changes are
  a good proxy for **total return**.
* We work in **log returns** so that compounding and the multivariate-normal
  Monte Carlo are internally consistent.
* For ETFs with too little history (e.g. BGBL launched 2023) we substitute the
  asset-class fallback mean/volatility from ``etf_universe`` while preserving
  whatever correlation structure the overlapping data provides.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .etf_universe import ASSET_CLASS_FALLBACK, ETF_UNIVERSE, build_anchors

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_TTL_SECONDS = 24 * 3600          # re-download at most once per day
MIN_HISTORY_YEARS = 5                  # below this we lean on fallbacks
TRADING_MONTHS = 12


# ---------------------------------------------------------------------------
@dataclass
class ReturnStats:
    """Annualised statistics for a set of tickers, in **log-return** space."""

    tickers: list[str]
    mean_log: np.ndarray                # shape (n,)
    cov_log: np.ndarray                 # shape (n, n)
    history_years: dict[str, float]
    used_fallback: list[str]
    annual_simple: pd.DataFrame         # historical annual simple returns

    @property
    def mean_simple(self) -> np.ndarray:
        """Approx. expected simple annual return per ticker."""
        var = np.diag(self.cov_log)
        return np.exp(self.mean_log + 0.5 * var) - 1

    @property
    def vol_simple(self) -> np.ndarray:
        """Approx. annual volatility of simple returns per ticker."""
        var = np.diag(self.cov_log)
        m = np.exp(self.mean_log + 0.5 * var)
        return m * np.sqrt(np.expm1(var))

    def as_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": self.tickers,
                "exp_return": self.mean_simple,
                "volatility": self.vol_simple,
                "history_yrs": [round(self.history_years.get(t, 0), 1) for t in self.tickers],
                "fallback": [t in self.used_fallback for t in self.tickers],
            }
        ).set_index("ticker")


# ---------------------------------------------------------------------------
def _cache_path(ticker: str, interval: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_')}_{interval}.parquet"


def _fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def get_price_history(
    tickers: list[str],
    interval: str = "1mo",
    period: str = "max",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, columns = tickers.

    Downloads missing/stale tickers from Yahoo Finance and caches each one to
    ``data_cache/``. Network failures degrade gracefully: a ticker that cannot
    be fetched is simply omitted (the caller falls back to assumptions).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    series: dict[str, pd.Series] = {}
    to_download: list[str] = []

    for t in tickers:
        cp = _cache_path(t, interval)
        if use_cache and _fresh(cp):
            try:
                series[t] = pd.read_parquet(cp)["close"]
                continue
            except Exception:
                pass
        to_download.append(t)

    if to_download:
        downloaded = _download(to_download, interval, period)
        for t, s in downloaded.items():
            series[t] = s
            try:
                s.rename("close").to_frame().to_parquet(_cache_path(t, interval))
            except Exception:
                pass  # caching is best-effort

    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1)
    df.columns = list(series.keys())
    return df.sort_index()


def _download(tickers: list[str], interval: str, period: str) -> dict[str, pd.Series]:
    """Fetch adjusted close from yfinance, normalising its column layout."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required to download market data") from exc

    raw = yf.download(
        tickers,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,  # avoid yfinance's sqlite tz-cache "database is locked" races
    )
    out: dict[str, pd.Series] = {}
    if raw is None or len(raw) == 0:
        return out

    # Multi-ticker: columns are a MultiIndex (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=-1)
        for t in tickers:
            if t in close.columns:
                s = close[t].dropna()
                if len(s) > 1:
                    out[t] = s
    else:
        # Single ticker: flat columns.
        s = raw["Close"].dropna() if "Close" in raw.columns else raw.iloc[:, 0].dropna()
        if len(s) > 1:
            out[tickers[0]] = s
    return out


# ---------------------------------------------------------------------------
def _fallback_log_moments(ticker: str) -> tuple[float, float]:
    """(log mean, log std) for a ticker from its asset-class fallback, net MER."""
    etf = ETF_UNIVERSE.get(ticker)
    asset_class = etf.asset_class if etf else "Global Equity"
    m, v = ASSET_CLASS_FALLBACK.get(asset_class, (0.08, 0.15))
    if etf:
        m -= etf.mer  # fallback figures are index-like (gross); net the fee
    sigma_log = np.sqrt(np.log1p((v / (1 + m)) ** 2))
    mu_log = np.log1p(m) - 0.5 * sigma_log**2
    return mu_log, sigma_log


def compute_return_stats(
    tickers: list[str],
    interval: str = "1mo",
    use_cache: bool = True,
) -> ReturnStats:
    """Build annualised log-return mean vector and covariance matrix.

    Correlations come from overlapping history. Means and volatilities come
    from history for tickers with >= ``MIN_HISTORY_YEARS`` of data, otherwise
    from asset-class fallbacks (correlations are still kept from any overlap).
    """
    tickers = list(tickers)
    n = len(tickers)
    prices = get_price_history(tickers, interval=interval, use_cache=use_cache)

    history_years: dict[str, float] = {}
    used_fallback: list[str] = []

    # Monthly log returns per ticker (may have differing lengths / NaNs).
    if prices.empty:
        log_rets = pd.DataFrame(columns=tickers)
    else:
        log_rets = np.log(prices / prices.shift(1))
        # Clip implausible monthly moves (|log| > 0.5 ~ -39%/+65%). For broad
        # ETFs these are almost always corrupt Yahoo data points, not real moves.
        log_rets = log_rets.clip(lower=-0.5, upper=0.5)
        for t in tickers:
            if t in log_rets.columns:
                history_years[t] = log_rets[t].dropna().shape[0] / TRADING_MONTHS

    # Data-derived monthly mean/std/correlation from pairwise-complete data.
    data_mean = {}
    data_std = {}
    for t in tickers:
        if t in log_rets.columns:
            r = log_rets[t].dropna()
            if len(r) >= 2:
                data_mean[t] = r.mean()
                data_std[t] = r.std(ddof=1)

    corr = log_rets.corr() if not log_rets.empty else pd.DataFrame()

    # Final annualised per-ticker mean (log) and std (log).
    mean_log = np.zeros(n)
    std_log = np.zeros(n)
    for i, t in enumerate(tickers):
        yrs = history_years.get(t, 0.0)
        if yrs >= MIN_HISTORY_YEARS and t in data_mean:
            mean_log[i] = data_mean[t] * TRADING_MONTHS
            std_log[i] = data_std[t] * np.sqrt(TRADING_MONTHS)
        else:
            mu, sig = _fallback_log_moments(t)
            mean_log[i] = mu
            std_log[i] = sig
            used_fallback.append(t)

    # Build covariance: cov = D R D, with R from data (identity-ish if absent).
    R = np.eye(n)
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i == j:
                continue
            if not corr.empty and ti in corr.columns and tj in corr.columns:
                c = corr.loc[ti, tj]
                if np.isfinite(c):
                    R[i, j] = c
    R = _nearest_psd(R)
    D = np.diag(std_log)
    cov_log = D @ R @ D

    # Historical annual simple returns (for display / bootstrap option).
    if prices.empty:
        annual_simple = pd.DataFrame(columns=tickers)
    else:
        annual_prices = prices.resample("YE").last()
        annual_simple = annual_prices.pct_change().dropna(how="all").reindex(columns=tickers)

    return ReturnStats(
        tickers=tickers,
        mean_log=mean_log,
        cov_log=cov_log,
        history_years=history_years,
        used_fallback=used_fallback,
        annual_simple=annual_simple,
    )


def apply_return_scenario(
    stats: ReturnStats,
    blend: float,
    equity_anchor: float,
    bond_anchor: float,
) -> ReturnStats:
    """Shrink each asset's expected return toward its long-run anchor.

    ``blend`` of 0 leaves history untouched; 1 replaces the mean with the anchor.
    Volatilities and correlations (and the raw annual history used for bootstrap)
    are kept from real data — only the central tendency is adjusted.
    """
    if blend <= 0:
        return stats
    blend = min(blend, 1.0)
    anchors = build_anchors(equity_anchor, bond_anchor)
    hist_simple = stats.mean_simple
    var = np.diag(stats.cov_log)
    new_mean_log = stats.mean_log.copy()
    for i, t in enumerate(stats.tickers):
        etf = ETF_UNIVERSE.get(t)
        cls = etf.asset_class if etf else "Global Equity"
        target = anchors.get(cls, equity_anchor)
        blended_simple = (1 - blend) * hist_simple[i] + blend * target
        new_mean_log[i] = np.log1p(blended_simple) - 0.5 * var[i]
    return replace(stats, mean_log=new_mean_log)


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so the correlation matrix is valid for sampling."""
    vals, vecs = np.linalg.eigh((matrix + matrix.T) / 2)
    vals = np.clip(vals, 1e-8, None)
    psd = vecs @ np.diag(vals) @ vecs.T
    # Re-normalise to unit diagonal (it's a correlation matrix).
    d = np.sqrt(np.diag(psd))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd
