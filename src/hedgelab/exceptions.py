"""Domain-specific exceptions.

Every expected failure state in the pipeline raises one of these, so callers never
need to catch the broad ``Exception`` base class.
"""

from __future__ import annotations


class HedgeLabError(Exception):
    """Base class for all hedgelab errors."""


class ConfigError(HedgeLabError):
    """Configuration is missing or invalid (e.g. a bad path in ``.env``)."""


class DataLoadError(HedgeLabError):
    """A raw input file is missing or cannot be parsed."""


class DataValidationError(HedgeLabError):
    """Loaded data violates a schema or a market-data invariant."""


class StaleDataError(HedgeLabError):
    """No market data is available on or close enough before the requested ``as_of``."""


class LookAheadError(HedgeLabError):
    """Data dated after ``as_of`` reached a pricing, calibration or hedging function."""


class CalibrationError(HedgeLabError):
    """Implied-vol inversion, smile fit or correlation estimation failed."""


class PricingError(HedgeLabError):
    """A pricer failed to produce a valid price (e.g. root-finding did not converge)."""
