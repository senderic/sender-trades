from src.config import Settings, MCPDaemonConfig, MCPConfig
from src.mcp.client import MCPBrokerClient, parse_float
from src.mcp.schemas import (
    build_alpaca_execution,
    format_execution_json,
    occ_option_symbol,
)
from src.models.recommendation import Direction, PositionIntent, TradeRecommendation


class TestOccSymbol:
    def test_generates_correct_symbol(self) -> None:
        symbol = occ_option_symbol("SPY", "2026-07-17", 746.00, "C")
        assert symbol.startswith("SPY")
        assert "C" in symbol
        assert "746" in symbol

    def test_put_symbol(self) -> None:
        symbol = occ_option_symbol("QQQ", "2026-07-17", 715.00, "P")
        assert "P" in symbol


class TestBuildAlpacaExecution:
    def test_single_leg_includes_symbol(self) -> None:
        rec = TradeRecommendation(
            correlation_id="test",
            strategy_label="a",
            asset="SPY",
            direction=Direction.CALL,
            confidence=0.7,
            target_strike=746.0,
            contracts=2,
            order_type="market",
            position_intent=PositionIntent.BUY_TO_OPEN,
            rationale={},
            expires_at="2026-07-17",
            must_close_before="15:30",
        )
        cmd = build_alpaca_execution(rec, "SPY250717C00746000")
        assert cmd.action == "place_option_order"
        assert cmd.payload.symbol == "SPY250717C00746000"

    def test_dry_run_produces_valid_json(self) -> None:
        rec = TradeRecommendation(
            correlation_id="test",
            strategy_label="a",
            asset="QQQ",
            direction=Direction.PUT,
            confidence=0.6,
            target_strike=714.0,
            contracts=1,
            order_type="limit",
            limit_price=1.25,
            position_intent=PositionIntent.BUY_TO_OPEN,
            rationale={},
            expires_at="2026-07-17",
            must_close_before="15:30",
        )
        cmd = build_alpaca_execution(rec, "QQQ250717P00714000")
        payload = format_execution_json(cmd)
        assert "place_option_order" in payload
        assert "QQQ250717P00714000" in payload


class TestParseFloat:
    def test_parses_float(self) -> None:
        assert parse_float("3.14") == 3.14

    def test_parses_none(self) -> None:
        assert parse_float(None) is None

    def test_parses_invalid(self) -> None:
        assert parse_float("not-a-number") is None
