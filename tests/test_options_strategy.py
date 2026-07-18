from src.engine.options_strategy import (
    compute_otm_strike,
    estimate_delta,
    compute_premium_bounds,
)
from src.models.recommendation import Direction


class TestComputeOtmStrike:
    def test_call_strike_above_spot(self) -> None:
        strike = compute_otm_strike(745.00, Direction.CALL, delta_target=0.30)
        assert strike > 745.00

    def test_put_strike_below_spot(self) -> None:
        strike = compute_otm_strike(745.00, Direction.PUT, delta_target=0.30)
        assert strike < 745.00

    def test_rounds_to_one_dollar_increment(self) -> None:
        strike = compute_otm_strike(745.30, Direction.CALL)
        assert strike == round(strike)


class TestEstimateDelta:
    def test_call_delta_positive(self) -> None:
        delta = estimate_delta(745.00, 746.00, 0, iv=0.20, direction=Direction.CALL)
        assert delta > 0

    def test_put_delta_negative(self) -> None:
        delta = estimate_delta(745.00, 746.00, 0, iv=0.20, direction=Direction.PUT)
        assert delta < 0

    def test_delta_bounds(self) -> None:
        delta = estimate_delta(745.00, 800.00, 0, iv=0.20, direction=Direction.CALL)
        assert -1.0 <= delta <= 1.0

    def test_atm_call_delta_approx_half(self) -> None:
        delta = estimate_delta(745.00, 745.00, 0, iv=0.20, direction=Direction.CALL)
        assert 0.3 <= delta <= 0.7


class TestComputePremiumBounds:
    def test_returns_positive_bounds(self) -> None:
        low, high = compute_premium_bounds(745.00, 746.00, Direction.CALL)
        assert low > 0
        assert high > low

    def test_put_premium(self) -> None:
        low, high = compute_premium_bounds(745.00, 744.00, Direction.PUT)
        assert low > 0
        assert high > low
