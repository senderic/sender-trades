"""Alpaca broker client using alpaca-py SDK with Tenacity retries."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

import httpx
import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.execution.models import ExecutionConfig, OrderResult
from src.execution.retry import _is_retryable
from src.mcp.schemas import occ_option_symbol
from src.models.recommendation import TradeRecommendation

logger = structlog.get_logger()


class AlpacaBrokerClient:
    """Direct Alpaca API client for order execution and market data.

    Uses ``alpaca-py`` SDK for all order actions and direct REST
    calls for option chain/quote lookups. All API calls are wrapped
    in Tenacity retries configured via :class:`ExecutionConfig`.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        config: ExecutionConfig | None = None,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.exec_config = config or ExecutionConfig()
        self._trading: TradingClient | None = None
        self._http: httpx.AsyncClient | None = None
        self._data_http: httpx.AsyncClient | None = None
        self._base_url = (
            "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        )
        self._data_url = "https://data.alpaca.markets"

    @property
    def trading(self) -> TradingClient:
        if self._trading is None:
            self._trading = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        return self._trading

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                auth=httpx.BasicAuth(self.api_key, self.secret_key),
                timeout=30.0,
            )
        return self._http

    @property
    def data_http(self) -> httpx.AsyncClient:
        if self._data_http is None:
            self._data_http = httpx.AsyncClient(
                base_url=self._data_url,
                auth=httpx.BasicAuth(self.api_key, self.secret_key),
                timeout=30.0,
            )
        return self._data_http

    async def _with_retry(self, fn: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        cfg = self.exec_config.tenacity
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(cfg.max_attempts),
            wait=wait_exponential(
                multiplier=cfg.backoff_multiplier,
                min=cfg.min_wait_sec,
                max=cfg.max_wait_sec,
            ),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                return await fn()

    async def submit_order(self, order_data: dict[str, Any]) -> OrderResult:
        async def _do() -> OrderResult:
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

        return await self._with_retry(_do)

    async def get_order(self, order_id: str) -> OrderResult:
        async def _do() -> OrderResult:
            response = self.trading.get_order_by_id(order_id)
            return _order_to_result(response)

        return await self._with_retry(_do)

    async def cancel_order(self, order_id: str) -> OrderResult:
        async def _do() -> OrderResult:
            self.trading.cancel_order_by_id(order_id)
            logger.info("order_cancelled", order_id=order_id)
            return await self.get_order(order_id)

        return await self._with_retry(_do)

    async def get_option_chain(
        self, underlying: str, expiry: str | None = None
    ) -> list[dict[str, Any]]:
        from src.timezone import today_local

        async def _do() -> list[dict[str, Any]]:
            exp = expiry or today_local().isoformat()
            response = await self.http.get(
                "/v2/options/contracts",
                params={"underlying_symbols": underlying, "expiration_date": exp},
            )
            response.raise_for_status()
            data = response.json()
            raw = data.get("option_contracts", [])
            contracts = [
                {
                    "symbol": c["symbol"],
                    "strike_price": float(c["strike_price"]),
                    "type": c.get("type", ""),
                    "expiration_date": c.get("expiration_date", ""),
                }
                for c in raw
            ]
            logger.info("option_chain_fetched", underlying=underlying, count=len(contracts))
            return contracts

        return await self._with_retry(_do)

    async def get_option_quote(self, occ_symbol: str) -> dict[str, Any] | None:
        async def _do() -> dict[str, Any] | None:
            try:
                response = await self.data_http.get(
                    "/v1beta1/options/snapshots",
                    params={"symbols": occ_symbol},
                )
                response.raise_for_status()
                data = response.json()
                snapshots = data.get("snapshots", {})
                snap = snapshots.get(occ_symbol)
                if snap and snap.get("latestQuote"):
                    q = snap["latestQuote"]
                    return {
                        "symbol": occ_symbol,
                        "bid": float(q["bp"]) if q.get("bp") else None,
                        "ask": float(q["ap"]) if q.get("ap") else None,
                        "bid_size": q.get("bs"),
                        "ask_size": q.get("as"),
                    }
            except Exception:
                logger.debug("option_quote_unavailable", symbol=occ_symbol)
            return None

        return await self._with_retry(_do)

    async def get_underlying_quote(self, symbol: str) -> dict[str, Any] | None:
        """Fetch a live equity quote for the option's underlying.

        Options don't trade pre-market, so an option quote is typically
        unavailable at the 9:28 AM ET submission window. Equities do have
        pre-market quotes, so the underlying spot is used to price the
        entry limit in that window.

        Args:
            symbol: Underlying symbol (e.g. ``SPY``).

        Returns:
            Dict with ``last``/``bid``/``ask``, or None if unavailable.
        """

        async def _do() -> dict[str, Any] | None:
            try:
                response = await self.data_http.get(
                    "/v2/stocks/snapshots",
                    params={"symbols": symbol},
                )
                response.raise_for_status()
                data = response.json()
                snap = data.get(symbol)
                if not snap:
                    return None
                quote = snap.get("latestQuote") or {}
                trade = snap.get("latestTrade") or {}
                bid = float(quote["bp"]) if quote.get("bp") else None
                ask = float(quote["ap"]) if quote.get("ap") else None
                last = float(trade["p"]) if trade.get("p") else None
                return {"symbol": symbol, "last": last, "bid": bid, "ask": ask}
            except Exception:
                logger.debug("underlying_quote_unavailable", symbol=symbol)
                return None

        return await self._with_retry(_do)

    def build_entry_order(
        self,
        rec: TradeRecommendation,
        occ_symbol: str | None = None,
        limit_price: float | None = None,
        quote: dict[str, Any] | None = None,
        spot: float | None = None,
    ) -> dict[str, Any]:
        if occ_symbol is None:
            expiry = rec.expires_at
            option_type = "C" if rec.direction.value == "CALL" else "P"
            occ_symbol = occ_option_symbol(rec.asset, expiry, rec.target_strike, option_type)

        order_type = self.exec_config.entry.order_type
        side = "buy"

        order: dict[str, Any] = {
            "symbol": occ_symbol,
            "qty": rec.contracts,
            "side": side,
            "type": order_type,
            "time_in_force": "day",
        }

        if order_type == "limit":
            if limit_price is not None:
                order["limit_price"] = round(limit_price, 2)
            elif quote and quote.get("ask"):
                # Price off the live ask and lean slightly into the spread so a
                # bullish buy-to-open actually fills instead of expiring unfilled.
                offset = self.exec_config.entry.limit_offset_pct / 100.0
                order["limit_price"] = round(quote["ask"] * (1 + offset), 2)
            elif spot and spot > 0:
                # No live option quote (options don't trade pre-market). Use the
                # underlying spot to estimate a marketable premium: intrinsic
                # value plus a generous 0.5% of spot as 0DTE time value. Since
                # this is a BUY limit, it fills at the ask and never pays above
                # it; the price is a ceiling, so being generous guarantees entry
                # at the open without overpaying.
                offset = self.exec_config.entry.limit_offset_pct / 100.0
                intrinsic = (
                    max(0.0, spot - rec.target_strike)
                    if rec.direction.value == "CALL"
                    else max(0.0, rec.target_strike - spot)
                )
                est_price = intrinsic + (spot * 0.005)
                order["limit_price"] = round(est_price * (1 + offset), 2)
            else:
                delta = rec.rationale.get("delta", 0.3) if isinstance(rec.rationale, dict) else 0.3
                est_price = (
                    abs(delta)
                    * abs(rec.rationale.get("entry_price", rec.target_strike) - rec.target_strike)
                    + 0.15
                    if isinstance(rec.rationale, dict)
                    else 3.0
                )
                order["limit_price"] = round(max(est_price, 1.0), 2)

        return order


def _order_to_result(order: Any) -> OrderResult:
    def _str(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, datetime):
            return val.isoformat()
        return str(val)

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
        created_at=_str(getattr(order, "created_at", None)),
        updated_at=_str(getattr(order, "updated_at", None)),
        raw=order.model_dump() if hasattr(order, "model_dump") else {},
    )
