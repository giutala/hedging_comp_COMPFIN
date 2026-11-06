"""Contracts shared by the calibration and hedging domains.

Importing rules
---------------
- ``calibration`` and ``hedging`` code may import from ``hedgelab.contracts``.
- ``hedgelab.contracts`` must not import from either domain.

Every file here is jointly owned: see ``OWNERS.md``.
"""

from __future__ import annotations

from hedgelab.contracts.models import CalibratedModel, CollarSpec, HedgePosition
from hedgelab.contracts.protocols import Calibrator, Hedger, Pricer

__all__ = [
    "CalibratedModel",
    "Calibrator",
    "CollarSpec",
    "HedgePosition",
    "Hedger",
    "Pricer",
]
