"""Exploratory data analysis: data-quality checks and descriptive analytics.

Every check returns a :class:`CheckResult` so that the CLI, the notebook and the tests
consume the same numbers. ``ERROR`` checks guard invariants the pipeline relies on;
``INFO`` checks quantify known imperfections that later phases must handle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy import stats

from hedgelab.black76 import implied_vol

logger = logging.getLogger(__name__)

TICK = 0.01
"""Minimum price increment of ICE Brent futures and options (USD/bbl)."""

STRIKE_STEP = 5.0
"""Listed strike spacing in this dataset (USD/bbl)."""

# ICE Futures Europe full-day closures for Brent in the sample years: Good Friday,
# Christmas Day and New Year's Day. Easter Monday and UK bank holidays are trading days.
KNOWN_ICE_CLOSURES: frozenset[pd.Timestamp] = frozenset(
    pd.Timestamp(d)
    for d in (
        "2020-01-01", "2020-04-10", "2020-12-25",
        "2021-01-01", "2021-04-02",
        "2022-04-15", "2022-12-26",
        "2023-01-02", "2023-04-07",
    )
)  # fmt: skip

MONEYNESS_BINS: tuple[float, ...] = (-2.0, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 2.0)


class Severity(StrEnum):
    """How a failed check should be treated."""

    ERROR = "error"
    INFO = "info"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one data-quality check.

    Attributes
    ----------
    name : str
        Short identifier.
    description : str
        What is being tested, in words.
    n_tested : int
        Number of items (rows, slices, dates) examined.
    n_failed : int
        Number of items violating the condition.
    severity : Severity
        ``ERROR`` if a failure would invalidate downstream results.
    comment : str
        Interpretation / how downstream phases handle it.
    """

    name: str
    description: str
    n_tested: int
    n_failed: int
    severity: Severity
    comment: str = ""

    @property
    def passed(self) -> bool:
        """Whether no item failed."""
        return self.n_failed == 0


def checks_to_frame(checks: Iterable[CheckResult]) -> pd.DataFrame:
    """Tabulate check results.

    Parameters
    ----------
    checks : Iterable[CheckResult]
        Results of :func:`run_quality_checks`.

    Returns
    -------
    pd.DataFrame
        One row per check, with a ``passed`` column.
    """
    rows = [
        {
            "check": c.name,
            "description": c.description,
            "tested": c.n_tested,
            "failed": c.n_failed,
            "severity": c.severity.value,
            "passed": c.passed,
            "comment": c.comment,
        }
        for c in checks
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- enrichment


def enrich_options(options: pd.DataFrame) -> pd.DataFrame:
    """Add analytics columns (time to expiry, moneyness, spreads, implied vols).

    Parameters
    ----------
    options : pd.DataFrame
        Output of :func:`hedgelab.loaders.load_option_quotes`.

    Returns
    -------
    pd.DataFrame
        Copy with columns ``tau, is_call, log_moneyness, is_otm, intrinsic,
        extrinsic, mid, rel_spread, iv_settle, iv_bid, iv_ask, iv_width``.

    Notes
    -----
    ``tau`` is ACT/365 from the provided calendar-day count. Implied vols use the
    undiscounted Black-76 formula (r = 0), appropriate for futures-style margined
    options, whose American early-exercise premium is zero. ``iv_width`` is the
    bid-ask width in vol points (percent).
    """
    df = options.copy()
    df["tau"] = df["time_to_expiry_days"].astype(float) / 365.0
    df["is_call"] = df["option_type"].eq("Call")
    F = df["underlying_settlement"]
    K = df["strike"]
    df["log_moneyness"] = np.log(K / F)
    df["is_otm"] = np.where(df["is_call"], K >= F, K <= F)
    df["intrinsic"] = np.where(df["is_call"], (F - K).clip(lower=0), (K - F).clip(lower=0))
    df["extrinsic"] = df["settlement_price"] - df["intrinsic"]
    df["mid"] = 0.5 * (df["option_bid"] + df["option_ask"])
    df["rel_spread"] = (df["option_ask"] - df["option_bid"]) / df["mid"]
    for src, col in (("settlement_price", "iv_settle"), ("option_bid", "iv_bid")):
        df[col] = implied_vol(df[src], F, K, df["tau"], df["is_call"])
    df["iv_ask"] = implied_vol(df["option_ask"], F, K, df["tau"], df["is_call"])
    df["iv_width"] = 100.0 * (df["iv_ask"] - df["iv_bid"])
    return df


# --------------------------------------------------------------------------- checks


def _check(
    name: str,
    description: str,
    mask_failed: pd.Series[bool] | np.ndarray,
    severity: Severity = Severity.ERROR,
    comment: str = "",
) -> CheckResult:
    failed = np.asarray(mask_failed, dtype=bool)
    return CheckResult(name, description, int(failed.size), int(failed.sum()), severity, comment)


def calendar_checks(options: pd.DataFrame, panel: pd.DataFrame) -> list[CheckResult]:
    """Trading-calendar sanity checks.

    Parameters
    ----------
    options : pd.DataFrame
        Raw option quotes.
    panel : pd.DataFrame
        Futures panel from :func:`hedgelab.loaders.build_futures_panel`.

    Returns
    -------
    list[CheckResult]
        Weekend dates, unexplained missing weekdays, row-count consistency.
    """
    dates = pd.DatetimeIndex(np.sort(options["date"].unique()))
    all_dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    weekdays = pd.bdate_range(all_dates.min(), all_dates.max())
    missing = weekdays.difference(all_dates)
    unexplained = [d for d in missing if d not in KNOWN_ICE_CLOSURES]
    per_slice = options.groupby(["date", "contract_code"]).size()
    return [
        _check(
            "no_weekend_dates",
            "No observation falls on a Saturday/Sunday",
            all_dates.day_of_week >= 5,
        ),
        CheckResult(
            "missing_weekdays_are_ice_holidays",
            "Every missing weekday is a known ICE Brent closure",
            len(missing),
            len(unexplained),
            Severity.ERROR,
            f"missing: {', '.join(f'{d:%Y-%m-%d}' for d in missing)}",
        ),
        _check(
            "unique_date_ticker",
            "One row per (date, option_ticker)",
            options.duplicated(["date", "option_ticker"]).to_numpy(),
        ),
        _check(
            "even_chain_rows",
            "Each (date, chain) has a call and a put on every strike",
            (per_slice % 2 != 0).to_numpy(),
        ),
        _check(
            "ttm_matches_expiry",
            "time_to_expiry_days == expiration - date",
            (
                (options["expiration"] - options["date"]).dt.days
                != options["time_to_expiry_days"].astype(int)
            ).to_numpy(),
        ),
        _check(
            "no_quote_after_expiry",
            "No option row is dated after its expiry",
            (options["date"] > options["expiration"]).to_numpy(),
        ),
        CheckResult(
            "dates_in_training_window",
            "Option dates lie in the training window 2020-01-06 .. 2021-04-06",
            len(dates),
            int(
                ((dates < pd.Timestamp("2020-01-06")) | (dates > pd.Timestamp("2021-04-06"))).sum()
            ),
            Severity.ERROR,
        ),
    ]


def futures_checks(options: pd.DataFrame, panel: pd.DataFrame) -> list[CheckResult]:
    """Consistency checks on futures quotes.

    Parameters
    ----------
    options : pd.DataFrame
        Raw option quotes.
    panel : pd.DataFrame
        Futures panel.

    Returns
    -------
    list[CheckResult]
        Uniqueness, settle-within-quotes, positivity, spread-column consistency.
    """
    per_key = options.groupby(["date", "future_contract_code"])["underlying_settlement"]
    spread_diff = (
        options["underlying_spread"] - (options["underlying_ask"] - options["underlying_bid"])
    ).abs()
    return [
        _check(
            "unique_underlying_per_date",
            "One underlying settle per (date, future) across option rows",
            (per_key.nunique() > 1).to_numpy(),
        ),
        _check(
            "future_settle_within_quotes",
            "bid <= settlement <= ask for every futures quote",
            (
                (panel["settlement"] < panel["bid"] - 1e-9)
                | (panel["settlement"] > panel["ask"] + 1e-9)
            ).to_numpy(),
        ),
        _check(
            "future_prices_positive",
            "Futures prices strictly positive (no negative-WTI-style prints)",
            (panel[["settlement", "bid", "ask"]] <= 0).any(axis=1).to_numpy(),
        ),
        _check(
            "zero_width_futures_quotes",
            "Futures bid < ask (zero width understates transaction costs)",
            (panel["bid"] >= panel["ask"]).to_numpy(),
            Severity.INFO,
            "bid == ask on ~5% of prints: floor the hedge half-spread at half a tick",
        ),
        _check(
            "underlying_spread_column",
            "underlying_spread == underlying_ask - underlying_bid",
            (spread_diff > 1e-6).to_numpy(),
            Severity.INFO,
            "column is off by one tick on many rows: recompute ask - bid, never use it",
        ),
    ]


def option_checks(enriched: pd.DataFrame) -> list[CheckResult]:
    """No-arbitrage and quote-quality checks on option prices.

    Parameters
    ----------
    enriched : pd.DataFrame
        Output of :func:`enrich_options`.

    Returns
    -------
    list[CheckResult]
        Quote consistency, intrinsic bound, parity, monotonicity, convexity, liquidity.

    Notes
    -----
    Put-call parity for futures-style margined options is undiscounted,
    ``C - P = F - K``; a residual within one tick confirms the r = 0 (Asay) convention.
    Convexity is tested on the 5-dollar butterfly with a one-tick tolerance, since
    settlement prices are rounded to the tick.
    """
    q = enriched.dropna(subset=["option_bid", "option_ask"])
    has_settle = enriched["settlement_price"].notna()
    has_quotes = enriched[["option_bid", "option_ask"]].notna().all(axis=1)

    s = enriched.dropna(subset=["settlement_price"])
    wide = s.pivot_table(
        index=["date", "contract_code", "strike"], columns="option_type", values="settlement_price"
    ).dropna()
    fwd = s.groupby(["date", "contract_code", "strike"])["underlying_settlement"].first()
    parity = wide["Call"] - wide["Put"] - (fwd.reindex(wide.index) - wide.index.get_level_values(2))

    listed = enriched.groupby(["date", "contract_code"])["strike"]
    off_grid = (enriched["strike"] % STRIKE_STEP).abs() > 1e-9
    holes = listed.apply(lambda k: bool(np.any(np.diff(np.sort(k.unique())) > STRIKE_STEP + 1e-9)))

    mono_fail: list[bool] = []
    conv_fail: list[bool] = []
    for (_, _, typ), sl in s.groupby(["date", "contract_code", "option_type"]):
        sl = sl.sort_values("strike")
        px = sl["settlement_price"].to_numpy()
        k = sl["strike"].to_numpy()
        dp = np.diff(px)
        mono_fail.append(bool(np.any(dp > 1e-9) if typ == "Call" else np.any(dp < -1e-9)))
        # Slope differences scaled to a 5-USD butterfly, so grid holes are handled.
        fly = np.diff(dp / np.diff(k)) * STRIKE_STEP if len(px) > 2 else np.array([])
        conv_fail.append(bool(np.any(fly < -TICK - 1e-9)))

    otm_q = q[q["is_otm"]]
    stale = enriched.sort_values(["option_ticker", "date"]).groupby("option_ticker")
    unchanged = (stale["option_bid"].diff() == 0) & (stale["option_ask"].diff() == 0)
    wide_iv = s.pivot_table(
        index=["date", "contract_code", "strike"], columns="option_type", values="iv_settle"
    ).dropna()
    return [
        _check(
            "no_crossed_option_quotes",
            "option bid <= ask",
            (q["option_bid"] > q["option_ask"]).to_numpy(),
        ),
        _check(
            "option_settle_within_quotes",
            "option bid <= settlement <= ask",
            (
                (q["settlement_price"] < q["option_bid"] - 1e-9)
                | (q["settlement_price"] > q["option_ask"] + 1e-9)
            ).to_numpy(),
        ),
        _check(
            "settle_iff_quotes",
            "settlement present exactly when bid/ask present",
            (has_settle != has_quotes).to_numpy(),
        ),
        _check(
            "settle_above_intrinsic",
            "settlement >= intrinsic value (extrinsic >= 0)",
            (s["extrinsic"] < -1e-9).to_numpy(),
        ),
        _check(
            "ask_above_intrinsic",
            "ask >= intrinsic (no free exercise arbitrage)",
            (q["option_ask"] < q["intrinsic"] - 1e-9).to_numpy(),
        ),
        _check(
            "put_call_parity_r0",
            "|C - P - (F - K)| <= 1 tick (undiscounted parity)",
            (parity.abs() > TICK + 1e-9).to_numpy(),
            comment="parity holds without discounting -> Black-76 with r = 0 is the right model",
        ),
        _check(
            "call_put_same_iv",
            "Call and put settlement IVs coincide at each strike (to 1e-4)",
            ((wide_iv["Call"] - wide_iv["Put"]).abs() > 1e-4).to_numpy(),
            Severity.INFO,
            "one exchange vol surface; mismatches only at tick-floor prices (IV undefined)",
        ),
        _check(
            "strike_grid_5usd",
            "Every listed strike is a multiple of 5 USD",
            off_grid.to_numpy(),
        ),
        _check(
            "no_strike_holes",
            "Listed strikes per (date, chain) are contiguous (no missing interior strike)",
            holes.to_numpy(),
            Severity.INFO,
            "a few chains skip an interior strike: interpolate the smile, never assume the grid",
        ),
        _check(
            "monotone_in_strike",
            "Calls non-increasing, puts non-decreasing in strike",
            np.array(mono_fail),
        ),
        _check(
            "butterfly_convexity",
            "Settlement prices convex in strike (5-USD butterfly >= -1 tick)",
            np.array(conv_fail),
            comment="tolerance of one tick because settlements are rounded to 0.01",
        ),
        _check(
            "iv_inverts_on_otm_quotes",
            "OTM bid and ask both invert to a Black-76 vol",
            (otm_q["iv_bid"].isna() | otm_q["iv_ask"].isna()).to_numpy(),
            Severity.INFO,
            "failures are tick-floor bids (0.01) below the model floor: filter in calibration",
        ),
        _check(
            "tick_floor_bids",
            "OTM bids above the 0.01 floor",
            (otm_q["option_bid"] <= TICK + 1e-9).to_numpy(),
            Severity.INFO,
            "bid at the price floor = no real bid; exclude from smile fits",
        ),
        _check(
            "fresh_quotes",
            "bid/ask changed vs the previous day for the same option",
            unchanged.to_numpy(),
            Severity.INFO,
            "unchanged bid and ask day-on-day flags a possibly stale quote",
        ),
    ]


def fixing_window_checks(
    panel: pd.DataFrame, legs: tuple[str, str], month: date
) -> list[CheckResult]:
    """Verify both legs have a settlement on every business day of the fixing month.

    Parameters
    ----------
    panel : pd.DataFrame
        Futures panel.
    legs : tuple[str, str]
        Futures codes of the first- and second-nearby legs, e.g. ``("COJ21", "COK21")``.
    month : date
        Any date in the averaging month.

    Returns
    -------
    list[CheckResult]
        One result per leg.
    """
    start = pd.Timestamp(month.year, month.month, 1)
    days = pd.bdate_range(start, start + pd.offsets.MonthEnd(0))
    days = days[~days.isin(list(KNOWN_ICE_CLOSURES))]
    out = []
    for leg in legs:
        have = set(panel.loc[panel["contract"] == leg, "date"])
        out.append(
            _check(
                f"fixing_window_{leg}",
                f"{leg} settles on every business day of {start:%b %Y}",
                np.array([d not in have for d in days]),
            )
        )
    return out


def run_quality_checks(
    options: pd.DataFrame,
    enriched: pd.DataFrame,
    panel: pd.DataFrame,
    legs: tuple[str, str],
    fixing_month: date,
) -> list[CheckResult]:
    """Run every data-quality check.

    Parameters
    ----------
    options : pd.DataFrame
        Raw option quotes.
    enriched : pd.DataFrame
        Output of :func:`enrich_options`.
    panel : pd.DataFrame
        Futures panel.
    legs : tuple[str, str]
        Futures codes of the two structure legs.
    fixing_month : date
        Averaging month of the structure.

    Returns
    -------
    list[CheckResult]
        All results, calendar first.
    """
    results = [
        *calendar_checks(options, panel),
        *futures_checks(options, panel),
        *option_checks(enriched),
        *fixing_window_checks(panel, legs, fixing_month),
    ]
    n_err = sum(not c.passed and c.severity is Severity.ERROR for c in results)
    logger.info("%d checks run, %d blocking failures", len(results), n_err)
    return results


# --------------------------------------------------------------------------- analytics


def expiry_table(options: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Option expiry vs futures last observed date per contract.

    Parameters
    ----------
    options : pd.DataFrame
        Raw option quotes.
    panel : pd.DataFrame
        Futures panel.

    Returns
    -------
    pd.DataFrame
        Index = future code; columns ``option_code, option_expiry, first_date,
        last_option_date, last_future_date``.
    """
    tab = options.groupby("future_contract_code").agg(
        option_code=("contract_code", "first"),
        option_expiry=("expiration", "first"),
        first_date=("date", "min"),
        last_option_date=("date", "max"),
    )
    tab["last_future_date"] = panel.groupby("contract")["date"].max().reindex(tab.index)
    return tab.sort_values("option_expiry")


def settlement_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot the futures panel to a date x contract settlement matrix.

    Parameters
    ----------
    panel : pd.DataFrame
        Futures panel.

    Returns
    -------
    pd.DataFrame
        Settlement prices; columns ordered by contract maturity.
    """
    mat = panel.pivot(index="date", columns="contract", values="settlement")
    month_codes = "FGHJKMNQUVXZ"
    order = sorted(mat.columns, key=lambda c: (int(c[3:]), month_codes.index(c[2])))
    return mat[order]


def log_returns(panel: pd.DataFrame, contracts: Iterable[str]) -> pd.DataFrame:
    """Daily settlement log-returns for the given contracts (common dates only).

    Parameters
    ----------
    panel : pd.DataFrame
        Futures panel.
    contracts : Iterable[str]
        Futures codes.

    Returns
    -------
    pd.DataFrame
        Log-returns, NaN rows dropped.
    """
    mat = settlement_matrix(panel)[list(contracts)]
    logs = pd.DataFrame(np.log(mat.to_numpy()), index=mat.index, columns=mat.columns)
    return logs.diff().dropna()


def realised_vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Compute rolling annualised close-to-close realised volatility.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-returns.
    window : int, default 21
        Window length in trading days.

    Returns
    -------
    pd.DataFrame
        sqrt(252) * rolling std.
    """
    out: pd.DataFrame = returns.rolling(window).std() * float(np.sqrt(252.0))
    return out


def rolling_correlation(returns: pd.DataFrame, a: str, b: str, window: int) -> pd.Series[float]:
    """Compute the rolling Pearson correlation of two return series.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-returns.
    a, b : str
        Column names.
    window : int
        Window in trading days.

    Returns
    -------
    pd.Series[float]
        Correlation per date.
    """
    return returns[a].rolling(window).corr(returns[b])


def ewma_correlation(returns: pd.DataFrame, a: str, b: str, lam: float = 0.94) -> pd.Series[float]:
    """RiskMetrics-style EWMA correlation.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-returns.
    a, b : str
        Column names.
    lam : float, default 0.94
        Decay factor.

    Returns
    -------
    pd.Series[float]
        Correlation per date (first 20 observations dropped as burn-in).
    """
    r = returns[[a, b]]
    alpha = 1.0 - lam
    cov = (r[a] * r[b]).ewm(alpha=alpha).mean()
    var_a = (r[a] ** 2).ewm(alpha=alpha).mean()
    var_b = (r[b] ** 2).ewm(alpha=alpha).mean()
    return (cov / np.sqrt(var_a * var_b)).iloc[20:]


def atm_vol(enriched: pd.DataFrame, option_code: str) -> pd.Series[float]:
    """At-the-money settlement IV per date, linearly interpolated in log-moneyness.

    Parameters
    ----------
    enriched : pd.DataFrame
        Output of :func:`enrich_options`.
    option_code : str
        Option chain code, e.g. ``"COJ1"``.

    Returns
    -------
    pd.Series[float]
        ATM implied vol indexed by date.
    """
    sl = enriched[
        (enriched["contract_code"] == option_code)
        & enriched["is_otm"]
        & enriched["iv_settle"].notna()
    ]

    def _interp(g: pd.DataFrame) -> float:
        g = g.sort_values("log_moneyness")
        return float(np.interp(0.0, g["log_moneyness"], g["iv_settle"]))

    out: pd.Series[float] = sl.groupby("date")[["log_moneyness", "iv_settle"]].apply(_interp)
    return out.rename(option_code)


def strike_coverage(enriched: pd.DataFrame, option_code: str) -> pd.DataFrame:
    """Compute the listed and liquid strike range per date for one chain.

    Parameters
    ----------
    enriched : pd.DataFrame
        Output of :func:`enrich_options`.
    option_code : str
        Option chain code.

    Returns
    -------
    pd.DataFrame
        Columns ``forward, k_min, k_max, k_min_quoted, k_max_quoted, n_quoted``.
    """
    sl = enriched[enriched["contract_code"] == option_code]
    quoted = sl[sl["option_bid"] > TICK + 1e-9]
    out = sl.groupby("date").agg(
        forward=("underlying_settlement", "first"),
        k_min=("strike", "min"),
        k_max=("strike", "max"),
    )
    q = quoted.groupby("date").agg(
        k_min_quoted=("strike", "min"),
        k_max_quoted=("strike", "max"),
        n_quoted=("strike", "size"),
    )
    return out.join(q)


def _bucket(values: pd.Series[float]) -> pd.Series[str]:
    cats = pd.cut(values, list(MONEYNESS_BINS))
    return cats.astype(str)


def liquidity_by_moneyness(enriched: pd.DataFrame) -> pd.DataFrame:
    """Quote availability and width by log-moneyness bucket (OTM side).

    Parameters
    ----------
    enriched : pd.DataFrame
        Output of :func:`enrich_options`.

    Returns
    -------
    pd.DataFrame
        Index = bucket; columns ``n, missing_share, floor_bid_share,
        median_rel_spread, median_iv_width``.
    """
    otm = enriched[enriched["is_otm"]].copy()
    otm["bucket"] = _bucket(otm["log_moneyness"])
    order = [str(i) for i in pd.cut(pd.Series([0.0]), list(MONEYNESS_BINS)).cat.categories]
    agg = otm.groupby("bucket").agg(
        n=("strike", "size"),
        missing_share=("option_bid", lambda v: float(v.isna().mean())),
        floor_bid_share=("option_bid", lambda v: float((v <= TICK + 1e-9).mean())),
        median_rel_spread=("rel_spread", "median"),
        median_iv_width=("iv_width", "median"),
    )
    return agg.reindex([o for o in order if o in agg.index])


def missing_share_grid(enriched: pd.DataFrame) -> pd.DataFrame:
    """Share of missing quotes by (time-to-expiry bucket, log-moneyness bucket).

    Parameters
    ----------
    enriched : pd.DataFrame
        Output of :func:`enrich_options`.

    Returns
    -------
    pd.DataFrame
        Rows = tau bucket (months), columns = log-moneyness bucket.
    """
    df = enriched.copy()
    df["m_bucket"] = pd.cut(df["log_moneyness"], list(MONEYNESS_BINS))
    df["t_bucket"] = pd.cut(df["tau"] * 12, [0, 1, 3, 6, 12, 18, 25], include_lowest=True)
    grid = df.pivot_table(
        index="t_bucket",
        columns="m_bucket",
        values="option_bid",
        aggfunc=lambda v: float(v.isna().mean()),
        observed=False,
    )
    grid.index = [f"{iv.left:.0f}-{iv.right:.0f}m" for iv in grid.index]
    grid.columns = [str(c) for c in grid.columns]
    return grid


def smile_snapshot(enriched: pd.DataFrame, option_code: str, on: date) -> pd.DataFrame:
    """OTM smile (settle, bid, ask IV) of one chain on one date.

    Parameters
    ----------
    enriched : pd.DataFrame
        Output of :func:`enrich_options`.
    option_code : str
        Option chain code.
    on : date
        Observation date.

    Returns
    -------
    pd.DataFrame
        Sorted by strike; columns ``strike, option_type, forward, iv_bid,
        iv_settle, iv_ask, option_bid, option_ask``.
    """
    sl = enriched[
        (enriched["contract_code"] == option_code)
        & (enriched["date"] == pd.Timestamp(on))
        & enriched["is_otm"]
    ]
    cols = ["strike", "option_type", "underlying_settlement", "iv_bid", "iv_settle", "iv_ask"]
    out = sl[[*cols, "option_bid", "option_ask"]].rename(
        columns={"underlying_settlement": "forward"}
    )
    return out.sort_values("strike").reset_index(drop=True)


def atm_term_structure(enriched: pd.DataFrame, on: date) -> pd.DataFrame:
    """ATM implied vol of every listed chain on one date.

    Parameters
    ----------
    enriched : pd.DataFrame
        Output of :func:`enrich_options`.
    on : date
        Observation date.

    Returns
    -------
    pd.DataFrame
        Columns ``contract_code, expiration, tau, atm_iv`` sorted by expiry.
    """
    day = enriched[(enriched["date"] == pd.Timestamp(on)) & enriched["is_otm"]]
    day = day.dropna(subset=["iv_settle"])
    rows = []
    for code, g in day.groupby("contract_code"):
        g = g.sort_values("log_moneyness")
        rows.append(
            {
                "contract_code": code,
                "expiration": g["expiration"].iloc[0],
                "tau": g["tau"].iloc[0],
                "atm_iv": float(np.interp(0.0, g["log_moneyness"], g["iv_settle"])),
            }
        )
    return pd.DataFrame(rows).sort_values("expiration").reset_index(drop=True)


def summarise_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of daily log-returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-returns.

    Returns
    -------
    pd.DataFrame
        Annualised vol, skew, excess kurtosis, worst/best day per column.
    """
    metrics: dict[str, Callable[[pd.Series[float]], float]] = {
        "ann_vol": lambda s: float(s.std() * np.sqrt(252.0)),
        "skew": lambda s: float(stats.skew(s.to_numpy(), bias=False)),
        "excess_kurtosis": lambda s: float(stats.kurtosis(s.to_numpy(), bias=False)),
        "worst_day": lambda s: float(s.to_numpy().min()),
        "best_day": lambda s: float(s.to_numpy().max()),
    }
    return pd.DataFrame({k: returns.apply(f) for k, f in metrics.items()})
