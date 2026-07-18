"""MCP message schemas and serialization utilities."""

from __future__ import annotations

import json
import uuid

from src.models.recommendation import (
    AlpacaOrderPayload,
    ExecutionCommand,
    RobinhoodOrderPayload,
    TradeRecommendation,
)


def build_alpaca_execution(
    rec: TradeRecommendation,
    occ_symbol: str,
    bid: float | None = None,
    ask: float | None = None,
) -> ExecutionCommand:
    """Build an execution command for the Alpaca MCP broker.

    Args:
        rec: The trade recommendation to execute.
        occ_symbol: OCC option symbol string.
        bid: Current bid price (optional, used for limit price inference).
        ask: Current ask price (optional, used for limit price inference).

    Returns:
        An ExecutionCommand targeting the Alpaca paper-trading environment.
    """
    qty = str(rec.contracts)
    payload: AlpacaOrderPayload
    if rec.legs and len(rec.legs) > 0:
        legs_dicts = []
        for leg in rec.legs:
            leg_dict = {"symbol": leg.symbol, "ratio_qty": leg.ratio_qty}
            if leg.side:
                leg_dict["side"] = leg.side
            if leg.position_intent:
                leg_dict["position_intent"] = leg.position_intent.value
            legs_dicts.append(leg_dict)
        payload = AlpacaOrderPayload(
            qty=qty,
            type=rec.order_type,
            time_in_force="day",
            order_class="mleg",
            legs=legs_dicts,
            limit_price=str(rec.limit_price) if rec.limit_price else None,
        )
    else:
        limit_price_str = str(rec.limit_price) if rec.limit_price else None
        if rec.order_type == "limit" and limit_price_str is None and ask and bid:
            limit_price_str = str(round((bid + ask) / 2, 2))
        payload = AlpacaOrderPayload(
            qty=qty,
            type=rec.order_type,
            time_in_force="day",
            symbol=occ_symbol,
            side="buy" if rec.direction.value == "CALL" else "sell",
            position_intent=rec.position_intent.value,
            limit_price=limit_price_str,
            client_order_id=str(uuid.uuid4()),
        )

    return ExecutionCommand(
        action="place_option_order",
        payload=payload,
        env_mode="PAPER_ALPACA",
    )


def build_robinhood_execution(
    rec: TradeRecommendation,
    occ_symbol: str,
) -> ExecutionCommand:
    """Build an execution command for the Robinhood MCP broker.

    Args:
        rec: The trade recommendation to execute.
        occ_symbol: OCC option symbol string.

    Returns:
        An ExecutionCommand targeting the Robinhood live environment.
    """
    payload = RobinhoodOrderPayload(
        symbol=occ_symbol,
        direction="buy" if rec.direction.value == "CALL" else "sell",
        quantity=rec.contracts,
        order_type=rec.order_type,
        price=rec.limit_price,
        time_in_force="day",
    )
    return ExecutionCommand(
        action="place_option_order",
        payload=payload,
        env_mode="LIVE_ROBINHOOD",
    )


def occ_option_symbol(
    underlying: str,
    expiry: str,
    strike: float,
    option_type: str,
) -> str:
    """Build an OCC option symbol from its components.

    Args:
        underlying: Underlying ticker symbol (e.g. SPY).
        expiry: Expiration date in ISO format (YYYY-MM-DD).
        strike: Strike price.
        option_type: 'CALL' or 'PUT'.

    Returns:
        OCC option symbol string.
    """
    date_part = expiry.replace("-", "")
    strike_int = round(strike * 1000)
    return f"{underlying}{date_part}{option_type.upper()[0]}{strike_int:08d}"


def format_execution_json(cmd: ExecutionCommand) -> str:
    """Format an execution command as a pretty-printed JSON string.

    Args:
        cmd: The execution command to serialise.

    Returns:
        Indented JSON string of the command payload.
    """
    if isinstance(cmd.payload, AlpacaOrderPayload):
        return json.dumps(
            {
                "action": cmd.action,
                "payload": cmd.payload.model_dump(exclude_none=True),
                "env_mode": cmd.env_mode,
            },
            indent=2,
        )
    return json.dumps(
        {
            "action": cmd.action,
            "payload": cmd.payload.model_dump(exclude_none=True),
            "env_mode": cmd.env_mode,
        },
        indent=2,
    )
