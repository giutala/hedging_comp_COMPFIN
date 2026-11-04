from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hedgelab import eda
from hedgelab.black76 import black76_price
from hedgelab.exceptions import DataValidationError
from hedgelab.loaders import build_futures_panel


def _synthetic_options() -> pd.DataFrame:
    """Two days of one clean chain priced by Black-76 and rounded to the tick."""
    rows = []
    for d, F in ((pd.Timestamp("2020-01-06"), 61.4), (pd.Timestamp("2020-01-07"), 61.0)):
        expiry = pd.Timestamp("2021-02-23")
        days = (expiry - d).days
        for K in np.arange(45.0, 86.0, 5.0):
            for typ in ("Call", "Put"):
                px = round(float(black76_price(F, K, days / 365, 0.25, typ == "Call")), 2)
                rows.append(
                    {
                        "date": d,
                        "option_ticker": f"COJ1{typ[0]} {K:.2f} Comdty",
                        "contract_code": "COJ1",
                        "future_contract_code": "COJ21",
                        "strike": K,
                        "option_type": typ,
                        "expiration": expiry,
                        "settlement_price": px,
                        "underlying_settlement": F,
                        "underlying_bid": F - 0.02,
                        "underlying_ask": F + 0.02,
                        "underlying_spread": 0.04,
                        "sofr_rate": 0.0155,
                        "time_to_expiry_days": days,
                        "option_bid": max(px - 0.05, 0.01),
                        "option_ask": px + 0.05,
                    }
                )
    return pd.DataFrame(rows)


def _errors(checks: list[eda.CheckResult]) -> dict[str, int]:
    return {c.name: c.n_failed for c in checks if c.severity is eda.Severity.ERROR}


def test_clean_chain_passes_all_blocking_checks() -> None:
    opts = _synthetic_options()
    panel = build_futures_panel(opts, {})
    enriched = eda.enrich_options(opts)
    checks = [*eda.calendar_checks(opts, panel), *eda.option_checks(enriched)]
    assert all(v == 0 for v in _errors(checks).values()), _errors(checks)


def test_enrich_recovers_flat_vol() -> None:
    enriched = eda.enrich_options(_synthetic_options())
    near = enriched[enriched["is_otm"] & (enriched["log_moneyness"].abs() < 0.15)]
    np.testing.assert_allclose(near["iv_settle"], 0.25, atol=2e-3)


def test_crossed_quote_and_parity_break_are_detected() -> None:
    opts = _synthetic_options()
    opts.loc[0, ["option_bid", "option_ask"]] = [5.0, 4.0]
    opts.loc[1, "settlement_price"] += 0.5  # put leg of first strike -> parity break
    errs = _errors(eda.option_checks(eda.enrich_options(opts)))
    assert errs["no_crossed_option_quotes"] == 1
    assert errs["put_call_parity_r0"] == 1


def test_convexity_violation_is_detected() -> None:
    opts = _synthetic_options()
    mid = (
        (opts["date"] == "2020-01-06") & (opts["strike"] == 65.0) & (opts["option_type"] == "Call")
    )
    opts.loc[mid, "settlement_price"] += 1.0
    errs = _errors(eda.option_checks(eda.enrich_options(opts)))
    assert errs["butterfly_convexity"] >= 1


def test_fixing_window_check_flags_missing_day() -> None:
    days = pd.bdate_range("2021-02-01", "2021-02-26")
    panel = pd.DataFrame({"date": days[:-1], "contract": "COJ21", "settlement": 60.0})
    (res,) = eda.fixing_window_checks(panel, ("COJ21",), date(2021, 2, 1))  # type: ignore[arg-type]
    assert res.n_tested == 20
    assert res.n_failed == 1


def test_futures_panel_rejects_disagreeing_sources() -> None:
    opts = _synthetic_options()
    wb = pd.DataFrame(
        {"settlement": [99.0], "bid": [98.9], "ask": [99.1]},
        index=pd.DatetimeIndex([pd.Timestamp("2020-01-06")], name="date"),
    )
    with pytest.raises(DataValidationError):
        build_futures_panel(opts, {"COJ21": wb})


def test_futures_panel_extends_with_workbook() -> None:
    opts = _synthetic_options()
    wb = pd.DataFrame(
        {"settlement": [61.4, 62.0], "bid": [61.38, 61.98], "ask": [61.42, 62.02]},
        index=pd.DatetimeIndex(pd.to_datetime(["2020-01-06", "2020-01-08"]), name="date"),
    )
    panel = build_futures_panel(opts, {"COJ21": wb})
    assert list(panel["date"]) == list(pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]))
    assert panel.set_index("date").loc["2020-01-08", "source"] == "workbook"


def test_ewma_correlation_of_identical_series_is_one() -> None:
    r = pd.DataFrame({"a": np.random.default_rng(0).normal(size=100)})
    r["b"] = r["a"]
    np.testing.assert_allclose(eda.ewma_correlation(r, "a", "b"), 1.0)
