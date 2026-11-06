"""Shared data types exchanged between the calibration and hedging domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class CollarSpec:
    """Terms of the zero-cost Asian collar: long call, short put, averaged price.

    Parameters
    ----------
    call_strike : float
        Strike of the long call.
    put_strike : float
        Strike of the short put; solved so the structure costs zero at inception.
    avg_start : date
        First fixing date of the averaging window.
    avg_end : date
        Last fixing date of the averaging window.
    legs : tuple of str
        Futures contract codes the averaging window is split across (e.g. ``COJ21``).
    """

    call_strike: float
    put_strike: float
    avg_start: date
    avg_end: date
    legs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the terms."""
        if not 0 < self.put_strike <= self.call_strike:
            raise ValueError("strikes must satisfy 0 < put_strike <= call_strike")
        if self.avg_start > self.avg_end:
            raise ValueError("avg_start must not be after avg_end")
        if not self.legs:
            raise ValueError("legs must not be empty")


@dataclass(frozen=True)
class CalibratedModel:
    """Output of a calibration: model parameters valid on one date.

    Parameters
    ----------
    model : str
        Model name (e.g. ``"NIG"``, ``"VG"``).
    as_of : date
        Date of the market data used; the model must not be used before it.
    params : dict of str to float
        Calibrated parameters keyed by name.
    iv_rmse : float
        Root-mean-square implied-volatility error of the fit.
    """

    model: str
    as_of: date
    params: dict[str, float] = field(default_factory=dict)
    iv_rmse: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fit summary."""
        if not self.params:
            raise ValueError("params must not be empty")
        if self.iv_rmse < 0:
            raise ValueError("iv_rmse must be >= 0")


@dataclass(frozen=True)
class HedgePosition:
    """Target hedge portfolio on one date.

    Parameters
    ----------
    as_of : date
        Date the position is set.
    lots : dict of str to float
        Signed quantity per instrument ticker (futures or options);
        positive is long, negative is short.
    """

    as_of: date
    lots: dict[str, float] = field(default_factory=dict)
