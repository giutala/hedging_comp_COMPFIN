"""Interfaces between the calibration and hedging domains.

Each domain codes against these protocols and never imports the other's modules.
Every method takes an explicit ``as_of`` date; implementations must call
``loaders.filter_as_of`` first so no data after ``as_of`` is ever used.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from hedgelab.contracts.models import CalibratedModel, CollarSpec, HedgePosition


@runtime_checkable
class Calibrator(Protocol):
    """Fits a model to the option chain observed on ``as_of``."""

    def calibrate(self, chain: pd.DataFrame, as_of: date) -> CalibratedModel:
        """Return the model calibrated on quotes dated up to ``as_of``.

        Raises
        ------
        CalibrationError
            If the fit fails or does not converge.
        """
        ...


@runtime_checkable
class Pricer(Protocol):
    """Prices the collar under a calibrated model."""

    def price(self, model: CalibratedModel, spec: CollarSpec, as_of: date) -> float:
        """Return the collar value on ``as_of`` (long call minus short put).

        Raises
        ------
        PricingError
            If no valid price can be produced.
        """
        ...


@runtime_checkable
class Hedger(Protocol):
    """Computes the hedge portfolio for the short collar."""

    def hedge(self, model: CalibratedModel, spec: CollarSpec, as_of: date) -> HedgePosition:
        """Return the target hedge on ``as_of`` given the calibrated model."""
        ...
