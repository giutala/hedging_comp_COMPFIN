"""Undiscounted Black-76 pricing and implied-volatility inversion (Asay, 1982)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

VOL_LOWER = 1e-4
VOL_UPPER = 5.0


def black76_price(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    tau: npt.ArrayLike,
    sigma: npt.ArrayLike,
    is_call: npt.ArrayLike,
    discount: npt.ArrayLike = 1.0,
) -> FloatArray:
    """Black-76 option price on a futures contract (vectorised).

    Parameters
    ----------
    forward : array_like
        Futures price ``F``.
    strike : array_like
        Strike ``K``.
    tau : array_like
        Time to expiry in years (must be > 0).
    sigma : array_like
        Black volatility (annualised).
    is_call : array_like of bool
        ``True`` for calls, ``False`` for puts.
    discount : array_like, default 1.0
        Discount factor; ``1.0`` gives the undiscounted (futures-style margined,
        Asay 1982) price used for listed ICE Brent options.

    Returns
    -------
    FloatArray
        Option prices, broadcast to the common shape of the inputs.

    Notes
    -----
    Futures-style margining means the premium is never paid up front: it is marked to
    market like the future itself. The cost of carrying the premium therefore vanishes
    and the arbitrage-free price is the *undiscounted* Black formula, i.e. ``r = 0``.
    """
    F = np.asarray(forward, dtype=float)
    K = np.asarray(strike, dtype=float)
    vol_sqrt_t = np.asarray(sigma, dtype=float) * np.sqrt(np.asarray(tau, dtype=float))
    call = np.asarray(is_call, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    call_px = F * ndtr(d1) - K * ndtr(d2)
    put_px = K * ndtr(-d2) - F * ndtr(-d1)
    out: FloatArray = np.asarray(discount, dtype=float) * np.where(call, call_px, put_px)
    return out


def implied_vol(
    price: npt.ArrayLike,
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    tau: npt.ArrayLike,
    is_call: npt.ArrayLike,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> FloatArray:
    """Invert the undiscounted Black-76 formula by vectorised bisection.

    Parameters
    ----------
    price : array_like
        Observed option prices (undiscounted).
    forward, strike, tau, is_call : array_like
        As in :func:`black76_price`.
    tol : float, default 1e-10
        Absolute tolerance on volatility.
    max_iter : int, default 200
        Maximum bisection iterations.

    Returns
    -------
    FloatArray
        Implied volatilities; ``NaN`` where the price is outside the no-arbitrage
        band ``(intrinsic, upper bound)`` or the inversion bracket ``[1e-4, 5]``.

    Notes
    -----
    Bisection is chosen over Newton because deep-OTM and tick-floor quotes have
    near-zero vega, where Newton steps explode; bisection is guaranteed to converge
    on the monotone price-vol map and is fully vectorised, so ~10^5 quotes invert in
    well under a second.
    """
    P, F, K, T, call = np.broadcast_arrays(
        np.asarray(price, dtype=float),
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(is_call, dtype=bool),
    )
    intrinsic = np.where(call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = np.where(call, F, K)
    valid = np.isfinite(P) & (T > 0) & (intrinsic < P) & (upper > P)

    lo = np.full(P.shape, VOL_LOWER)
    hi = np.full(P.shape, VOL_UPPER)
    valid &= black76_price(F, K, T, hi, call) >= P
    valid &= black76_price(F, K, T, lo, call) <= P
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        too_high = black76_price(F, K, T, mid, call) > P
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
        if np.nanmax(np.where(valid, hi - lo, 0.0), initial=0.0) < tol:
            break
    out: FloatArray = np.where(valid, 0.5 * (lo + hi), np.nan)
    return out
