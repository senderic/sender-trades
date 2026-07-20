"""Options pricing and strategy evaluation utilities."""

from __future__ import annotations

import math

from src.models.recommendation import Direction


def _days_to_expiry() -> int:
    return 0


def compute_otm_strike(
    underlying_price: float,
    direction: Direction,
    delta_target: float = 0.30,
) -> float:
    """Compute an out-of-the-money strike price for a given delta target.

    For 0DTE options, delta changes rapidly with small moneyness moves.
    This approximates the OTM distance as roughly 2 % per 1.0 delta
    (e.g. 30-delta → ~0.6 % OTM), which produces strikes that actually
    exist in the chain.

    Args:
        underlying_price: Current price of the underlying asset.
        direction: CALL or PUT direction.
        delta_target: Target delta for the option (default 0.30).

    Returns:
        The computed OTM strike price rounded to the nearest $1.00 increment.
    """
    if underlying_price <= 0:
        return 0.0
    otm_distance = delta_target * 0.02
    multiplier = 1 + otm_distance if direction == Direction.CALL else 1 - otm_distance
    raw = underlying_price * multiplier
    return _round_to_strike(raw, increment=1.0)


def _round_to_strike(price: float, increment: float = 1.0) -> float:
    """Round a price to the nearest valid strike increment.

    Args:
        price: The price to round.
        increment: Strike increment (default 1.0).

    Returns:
        The rounded strike price.
    """
    return round(price / increment) * increment


def estimate_delta(
    underlying_price: float,
    strike: float,
    days_to_expiry: int,
    iv: float = 0.20,
    direction: Direction = Direction.CALL,
) -> float:
    """Estimate the Black-Scholes delta of an option using a normal approximation.

    Args:
        underlying_price: Current price of the underlying.
        strike: Strike price of the option.
        days_to_expiry: Days until option expiration.
        iv: Implied volatility (default 0.20).
        direction: CALL or PUT direction.

    Returns:
        Estimated delta between -1.0 and 1.0.
    """
    if days_to_expiry < 1:
        days_to_expiry = 1
    sigma = iv * math.sqrt(days_to_expiry / 365.0)
    if sigma < 1e-6:
        sigma = 1e-6
    if underlying_price <= 0:
        return 0.0
    moneyness = (underlying_price - strike) / (underlying_price * sigma)
    try:
        delta_est = 0.5 * (1.0 + math.erf(moneyness / math.sqrt(2.0)))
    except (OverflowError, ValueError):
        delta_est = 0.5 if moneyness == 0 else (1.0 if moneyness > 0 else 0.0)
    if direction == Direction.PUT:
        delta_est = delta_est - 1.0
    return max(-1.0, min(1.0, delta_est))


def compute_premium_bounds(
    underlying_price: float,
    strike: float,
    direction: Direction,
    iv: float = 0.20,
) -> tuple[float, float]:
    """Compute estimated bid/ask bounds for an option premium.

    Args:
        underlying_price: Current price of the underlying.
        strike: Strike price of the option.
        direction: CALL or PUT direction.
        iv: Implied volatility (default 0.20).

    Returns:
        Tuple of (low_estimate, high_estimate) premium.
    """
    days = _days_to_expiry() + 1
    sigma = iv * math.sqrt(days / 365.0)
    intrinsic = max(
        0.0,
        (underlying_price - strike) if direction == Direction.CALL else (strike - underlying_price),
    )
    extrinsic = underlying_price * sigma * 0.4
    mid = intrinsic + extrinsic
    spread = mid * 0.1
    return round(mid - spread, 2), round(mid + spread, 2)
