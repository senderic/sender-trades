"""Alpaca broker client using alpaca-py SDK with Tenacity retries."""

from __future__ import annotations

from typing import Any

import structlog
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)

from src.execution.models import ExecutionConfig, OrderResult
from src.execution.retry import al_api_retry
from src.mcp.schemas import occ_option_symbol
from src.models.recommendation import TradeRecommendation

logger = structlog.get_logger()


class AlpacaBrokerClient:
    """Direct Alpaca API client for order execution and market data.

    Uses ``alpaca-py`` SDK for all order actions and option chain
    queries. All API calls are wrapped in Tenacity retries configured
    via :class:`ExecutionConfig`.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        config: ExecutionConfig | None = None,
    ):
        """Initialize the Alpaca broker client.

        Args:
            api_key: Alpaca API key ID.
            secret_key: Alpaca API secret key.
            paper: Whether to use the paper trading environment.
            config: Execution configuration for retry parameters.
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.exec_config = config or ExecutionConfig()
        self._trading: TradingClient | None = None
        self._data: OptionHistoricalDataClient | None = None

    @property
    def trading(self) -> TradingClient:
        """Lazy-loaded Alpaca TradingClient."""
        if self._trading is None:
            self._trading = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        return self._trading

    @property
    def data(self) -> OptionHistoricalDataClient:
        """Lazy-loaded Alpaca OptionHistoricalDataClient."""
        if self._data is None:
            self._data = OptionHistoricalDataClient(self.api_key, self.secret_key)
        return self._data

    @al_api_retry()
    async def submit_order(self, order_data: dict[str, Any]) -> OrderResult:
        """Submit an order to Alpaca.

        Args:
            order_data: Dict with keys: symbol, qty, side, type,
                time_in_force, and optional limit_price/stop_price.

        Returns:
            An :class:`OrderResult` with the order details.
        """
        order_type = order_data.get("type", "market")
        side = OrderSide.BUY if order_data.get("side") == "buy" else OrderSide.SELL
        symbol = order_data["symbol"]
        qty = order_data["qty"]
        tif = TimeInForce.DAY
        if order_data.get("time_in_force") == "gtc":
            tif = TimeInForce.GTC

        if order_type == "limit":
            request = LimitOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=side,
                type=OrderType.LIMIT,
                limit_price=float(order_data["limit_price"]),
                time_in_force=tif,
            )
        elif order_type == "stop":
            request = StopOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=side,
                type=OrderType.STOP,
                stop_price=float(order_data["stop_price"]),
                time_in_force=tif,
            )
        else:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=side,
                type=OrderType.MARKET,
                time_in_force=tif,
            )

        response = self.trading.submit_order(request)
        logger.info("order_submitted", order_id=str(response.id), symbol=symbol, side=str(side))
        return _order_to_result(response)

    @al_api_retry()
    async def get_order(self, order_id: str) -> OrderResult:
        """Query an existing order by ID.

        Args:
            order_id: Alpaca order UUID.

        Returns:
            An :class:`OrderResult` with current order state.
        """
        response = self.trading.get_order_by_id(order_id)
        return _order_to_result(response)

    @al_api_retry()
    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order.

        Args:
            order_id: Alpaca order UUID.

        Returns:
            An :class:`OrderResult` reflecting the cancelled state.
        """
        self.trading.cancel_order_by_id(order_id)
        logger.info("order_cancelled", order_id=order_id)
        return await self.get_order(order_id)

    @al_api_retry()
    async def get_option_chain(
        self, underlying: str, expiry: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch the option chain for an underlying symbol.

        Args:
            underlying: Ticker symbol (e.g. SPY, QQQ).
            expiry: Expiration date in YYYY-MM-DD format.

        Returns:
            List of option contract dicts.
        """
        from src.timezone import today_local

        exp = expiry or today_local().isoformat()
        request = OptionChainRequest(
            underlying_symbol=underlying,
            expiration_date=exp,
        )
        response = self.data.get_option_chain(request)
        contracts = []
        if response and response.option_contracts:
            for c in response.option_contracts:
                contracts.append(
                    {
                        "symbol": c.symbol,
                        "strike_price": float(c.strike_price),
                        "type": c.type,
                        "expiration_date": c.expiration_date,
                    }
                )
        logger.info("option_chain_fetched", underlying=underlying, count=len(contracts))
        return contracts

    @al_api_retry()
    async def get_option_quote(self, occ_symbol: str) -> dict[str, Any] | None:
        """Fetch the latest quote for an option contract.

        Args:
            occ_symbol: OCC option symbol.

        Returns:
            Dict with bid/ask or None if not found.
        """
        request = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol)
        response = self.data.get_option_latest_quote(request)
        if response and occ_symbol in response:
            quote = response[occ_symbol]
            return {
                "symbol": occ_symbol,
                "bid": float(quote.bid_price) if quote.bid_price else None,
                "ask": float(quote.ask_price) if quote.ask_price else None,
                "bid_size": quote.bid_size,
                "ask_size": quote.ask_size,
            }
        return None

    def build_entry_order(
        self,
        rec: TradeRecommendation,
        occ_symbol: str | None = None,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Build an entry order dict from a trade recommendation.

        Args:
            rec: The trade recommendation to execute.
            occ_symbol: Pre-computed OCC symbol (computed if not provided).
            limit_price: Limit price override (uses mid+offset if None).

        Returns:
            Dict ready for :meth:`submit_order`.
        """
        if occ_symbol is None:
            expiry = rec.expires_at
            option_type = "C" if rec.direction.value == "CALL" else "P"
            occ_symbol = occ_option_symbol(rec.asset, expiry, rec.target_strike, option_type)

        order_type = rec.order_type or self.exec_config.entry.order_type
        side = "buy"

        order: dict[str, Any] = {
            "symbol": occ_symbol,
            "qty": rec.contracts,
            "side": side,
            "type": order_type,
            "time_in_force": "day",
        }

        if order_type == "limit":
            order["limit_price"] = (
                limit_price if limit_price else round(rec.target_strike * 0.005, 2)
            )

        return order


def _order_to_result(order: Any) -> OrderResult:
    return OrderResult(
        order_id=str(order.id),
        status=order.status,
        symbol=order.symbol or "",
        side=order.side,
        order_type=order.type,
        qty=order.qty or "0",
        filled_qty=order.filled_qty or "0",
        filled_avg_price=order.filled_avg_price,
        limit_price=getattr(order, "limit_price", None),
        created_at=getattr(order, "created_at", ""),
        updated_at=getattr(order, "updated_at", ""),
        raw=order.model_dump() if hasattr(order, "model_dump") else {},
    )
