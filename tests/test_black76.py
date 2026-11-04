from __future__ import annotations

import numpy as np
import pytest

from hedgelab.black76 import black76_delta, black76_price, implied_vol


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("tau", [0.05, 0.5, 2.0])
def test_implied_vol_roundtrip(is_call: bool, tau: float) -> None:
    K = np.array([40.0, 55.0, 60.0, 65.0, 85.0])
    sigma = np.array([0.45, 0.30, 0.25, 0.22, 0.20])
    px = black76_price(60.0, K, tau, sigma, is_call)
    intrinsic = np.maximum(60.0 - K, 0.0) if is_call else np.maximum(K - 60.0, 0.0)
    has_time_value = px - intrinsic > 1e-6  # otherwise IV is numerically undetermined
    iv = implied_vol(px, 60.0, K, tau, is_call)
    np.testing.assert_allclose(iv[has_time_value], sigma[has_time_value], atol=1e-8)
    assert has_time_value.sum() >= 4


def test_no_time_value_gives_nan() -> None:
    # deep ITM 3-week put: price equals intrinsic to machine precision
    px = black76_price(60.0, 85.0, 0.05, 0.2, False)
    assert np.isnan(implied_vol(px, 60.0, 85.0, 0.05, False))


def test_put_call_parity_undiscounted() -> None:
    K = np.linspace(30, 90, 13)
    c = black76_price(61.4, K, 1.1, 0.25, True)
    p = black76_price(61.4, K, 1.1, 0.25, False)
    np.testing.assert_allclose(c - p, 61.4 - K, atol=1e-12)


def test_discount_scales_price() -> None:
    assert black76_price(60, 60, 1, 0.3, True, discount=0.9) == pytest.approx(
        0.9 * black76_price(60, 60, 1, 0.3, True)
    )


def test_atm_approximation() -> None:
    # ATM Black price ~ 0.4 F sigma sqrt(T) for small vol.
    assert black76_price(100, 100, 1, 0.1, True) == pytest.approx(0.4 * 100 * 0.1, rel=1e-2)


def test_implied_vol_out_of_bounds_is_nan() -> None:
    # below intrinsic, above upper bound, zero time
    iv = implied_vol(
        [4.0, 70.0, 1.0], [60.0, 60.0, 60.0], [55.0, 55.0, 60.0], [1.0, 1.0, 0.0], True
    )
    assert np.isnan(iv).all()


@pytest.mark.parametrize("is_call", [True, False])
def test_delta_matches_finite_difference(is_call: bool) -> None:
    K = np.array([45.0, 60.0, 80.0])
    h = 1e-4
    up = black76_price(60.0 + h, K, 0.8, 0.3, is_call)
    dn = black76_price(60.0 - h, K, 0.8, 0.3, is_call)
    np.testing.assert_allclose(
        black76_delta(60.0, K, 0.8, 0.3, is_call), (up - dn) / (2 * h), atol=1e-7
    )
