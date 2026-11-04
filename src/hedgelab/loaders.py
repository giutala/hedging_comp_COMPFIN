"""Loading, schema validation and as-of filtering of the raw course dataset."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from hedgelab.exceptions import DataLoadError, DataValidationError, LookAheadError

logger = logging.getLogger(__name__)

OPTION_COLUMNS: tuple[str, ...] = (
    "date",
    "option_ticker",
    "contract_code",
    "future_contract_code",
    "strike",
    "option_type",
    "expiration",
    "settlement_price",
    "underlying_settlement",
    "underlying_bid",
    "underlying_ask",
    "sofr_rate",
    "time_to_expiry_days",
    "option_bid",
    "option_ask",
)
FUTURES_COLUMNS: tuple[str, ...] = ("date", "settlement", "bid", "ask")


def load_option_quotes(path: Path) -> pd.DataFrame:
    """Load and validate the unpivoted option bid/ask parquet.

    Parameters
    ----------
    path : Path
        Path to ``unpivoted_bid_offer_train_clean.parquet``.

    Returns
    -------
    pd.DataFrame
        One row per (date, option_ticker), with ``date`` and ``expiration`` as
        ``datetime64`` and ``sofr_rate`` converted from percent to a decimal rate.

    Raises
    ------
    DataLoadError
        If the file does not exist or cannot be read.
    DataValidationError
        If required columns are missing, keys are duplicated, or quotes are crossed.

    Notes
    -----
    ``sofr_rate`` is stored in percent in the raw file (e.g. ``1.55``); it is divided by
    100 once here so that every downstream function works in decimal units.
    Rows with missing bid/ask are kept: the liquidity filter is a calibration decision,
    not a loading one.
    """
    if not path.exists():
        raise DataLoadError(f"option quotes file not found: {path}")
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise DataLoadError(f"cannot read {path}: {exc}") from exc

    missing = set(OPTION_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"option quotes missing columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["sofr_rate"] = df["sofr_rate"] / 100.0

    if df.duplicated(["date", "option_ticker"]).any():
        raise DataValidationError("duplicate (date, option_ticker) rows in option quotes")
    quoted = df["option_bid"].notna() & df["option_ask"].notna()
    if (df.loc[quoted, "option_bid"] > df.loc[quoted, "option_ask"]).any():
        raise DataValidationError("crossed option quotes (bid > ask)")
    if (df["underlying_bid"] > df["underlying_ask"]).any():
        raise DataValidationError("crossed futures quotes (bid > ask)")

    logger.info("Loaded %d option quotes over %d dates", len(df), df["date"].nunique())
    return df


def load_futures_series(path: Path) -> dict[str, pd.DataFrame]:
    """Load the per-contract futures settlement/bid/ask workbook.

    Parameters
    ----------
    path : Path
        Path to the xlsx file; each sheet is one futures contract (e.g. ``COJ21``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping contract code -> DataFrame indexed by ``date`` with columns
        ``settlement``, ``bid``, ``ask``.

    Raises
    ------
    DataLoadError
        If the file does not exist or cannot be read.
    DataValidationError
        If a sheet lacks the expected columns or has duplicated dates.
    """
    if not path.exists():
        raise DataLoadError(f"futures workbook not found: {path}")
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except (OSError, ValueError) as exc:
        raise DataLoadError(f"cannot read {path}: {exc}") from exc

    out: dict[str, pd.DataFrame] = {}
    for name, sheet in sheets.items():
        missing = set(FUTURES_COLUMNS) - set(sheet.columns)
        if missing:
            raise DataValidationError(f"sheet {name!r} missing columns: {sorted(missing)}")
        frame = sheet.loc[:, list(FUTURES_COLUMNS)].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if frame["date"].duplicated().any():
            raise DataValidationError(f"sheet {name!r} has duplicated dates")
        out[str(name)] = frame.set_index("date").sort_index().astype(float)
    logger.info("Loaded futures series for %s", ", ".join(out))
    return out


def filter_as_of(df: pd.DataFrame, as_of: date, date_col: str = "date") -> pd.DataFrame:
    """Keep only rows observable on or before ``as_of`` (D-1 temporal discipline).

    Parameters
    ----------
    df : pd.DataFrame
        Any time-stamped market-data frame.
    as_of : date
        Valuation date; no information dated after it may be used.
    date_col : str
        Name of the date column; if absent, the index is used.

    Returns
    -------
    pd.DataFrame
        Filtered copy.

    Raises
    ------
    LookAheadError
        If the post-filter invariant ``max(date) <= as_of`` does not hold.

    Notes
    -----
    This must be the first operation of every pricing, calibration and hedging
    function, so that an implicit "today" can never leak future data into a backtest.
    """
    stamps = df[date_col] if date_col in df.columns else df.index.to_series()
    cutoff = pd.Timestamp(as_of)
    out = df.loc[(stamps <= cutoff).to_numpy()].copy()
    kept = out[date_col] if date_col in out.columns else out.index.to_series()
    if len(kept) and kept.max() > cutoff:
        raise LookAheadError(f"data after as_of={as_of} survived filtering")
    return out
