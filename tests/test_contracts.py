"""Tests for the shared contracts."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedgelab.contracts import (
    CalibratedModel,
    Calibrator,
    CollarSpec,
    HedgePosition,
    Hedger,
    Pricer,
)

D = date(2020, 1, 6)


def _spec(**kw: object) -> CollarSpec:
    args: dict[str, object] = {
        "call_strike": 78.0,
        "put_strike": 60.0,
        "avg_start": date(2021, 2, 1),
        "avg_end": date(2021, 2, 26),
        "legs": ("COJ21", "COK21"),
    }
    args.update(kw)
    return CollarSpec(**args)  # type: ignore[arg-type]


class TestCollarSpec:
    def test_valid(self) -> None:
        assert _spec().legs == ("COJ21", "COK21")

    @pytest.mark.parametrize(
        "kw",
        [
            {"put_strike": 80.0},
            {"put_strike": 0.0},
            {"avg_start": date(2021, 3, 1)},
            {"legs": ()},
        ],
    )
    def test_invalid(self, kw: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _spec(**kw)


class TestCalibratedModel:
    def test_valid(self) -> None:
        assert CalibratedModel("NIG", D, {"alpha": 1.0}, 0.01).model == "NIG"

    def test_empty_params(self) -> None:
        with pytest.raises(ValueError):
            CalibratedModel("NIG", D)

    def test_negative_error(self) -> None:
        with pytest.raises(ValueError):
            CalibratedModel("NIG", D, {"alpha": 1.0}, -0.1)


class _Calib:
    def calibrate(self, chain: pd.DataFrame, as_of: date) -> CalibratedModel:
        return CalibratedModel("NIG", as_of, {"alpha": 1.0})


class _Pricer:
    def price(self, model: CalibratedModel, spec: CollarSpec, as_of: date) -> float:
        return 0.0


class _Hedger:
    def hedge(self, model: CalibratedModel, spec: CollarSpec, as_of: date) -> HedgePosition:
        return HedgePosition(as_of, {"COJ21": -1.0})


def test_protocols_runtime_checkable() -> None:
    assert isinstance(_Calib(), Calibrator)
    assert isinstance(_Pricer(), Pricer)
    assert isinstance(_Hedger(), Hedger)
    assert not isinstance(object(), Calibrator)


def test_domains_chain_through_contracts() -> None:
    model = _Calib().calibrate(pd.DataFrame(), D)
    assert _Hedger().hedge(model, _spec(), D).lots == {"COJ21": -1.0}
