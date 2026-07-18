from __future__ import annotations

import asyncio
import json
import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional

import structlog

from src.config import Settings
from src.models.recommendation import (
    AlpacaOrderPayload,
    ExecutionCommand,
    TradeRecommendation,
)
from src.mcp.schemas import (
    build_alpaca_execution,
    format_execution_json,
    occ_option_symbol,
)

logger = structlog.get_logger()


class MCPClientError(Exception):
    """Raised when an MCP broker client encounters a non-recoverable error."""
    pass


class MCPBrokerClient:
    """Client for interacting with an MCP-based broker daemon (Alpaca)."""

    def __init__(self, config: Settings):
        self.config = config
        self.env_mode = config.general.env_mode

    async def query_option_chain(
        self, asset: str, expiry: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query the option chain for a given asset and expiry.

        Args:
            asset: Ticker symbol (e.g. SPY).
            expiry: Expiration date in ISO format (defaults to today).

        Returns:
            List of option contract dicts from the MCP daemon.
        """
        if expiry is None:
            expiry = date.today().isoformat()
        logger.info("mcp_chain_query", asset=asset, expiry=expiry)
        return await self._mcp_call("get_option_contracts", {
            "underlying_symbol": asset,
            "expiration_date": expiry,
        })

    async def query_option_quote(
        self, occ_symbol: str,
    ) -> Optional[dict[str, Any]]:
        """Query the latest quote for a specific OCC option symbol.

        Args:
            occ_symbol: The OCC option symbol to look up.

        Returns:
            Dict with quote data, or None if not found.
        """
        logger.info("mcp_quote_query", symbol=occ_symbol)
        return await self._mcp_single("get_option_latest_quote", {
            "symbol": occ_symbol,
        })

    async def execute(self, rec: TradeRecommendation) -> dict[str, Any]:
        """Execute a trade recommendation through the MCP broker.

        Verifies the contract exists in the chain, checks the bid/ask spread,
        and either dry-runs or submits the order.

        Args:
            rec: The trade recommendation to execute.

        Returns:
            Dict with execution status, occ_symbol, bid, ask, and result.
        """
        today_str = date.today().isoformat()
        option_type = "C" if rec.direction.value == "CALL" else "P"
        expiry = rec.expires_at or today_str
        occ_sym = occ_option_symbol(rec.asset, expiry, rec.target_strike, option_type)

        chain_data = await self.query_option_chain(rec.asset, expiry)
        chain_symbols = [c.get("symbol", "") for c in chain_data]
        if occ_sym not in chain_symbols:
            msg = f"Contract {occ_sym} not found in chain for {rec.asset} expiring {expiry}"
            logger.error("mcp_contract_not_found", occ_symbol=occ_sym)
            return {"error": msg, "occ_symbol": occ_sym, "found_symbols": chain_symbols[:10]}

        quote_data = await self.query_option_quote(occ_sym)
        bid = None
        ask = None
        if quote_data:
            bid = parse_float(quote_data.get("bid"))
            ask = parse_float(quote_data.get("ask"))
            if bid is not None and ask is not None and ask > 0:
                spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                max_spread = self.config.risk.max_bid_ask_spread_pct
                if spread_pct > max_spread:
                    msg = f"Bid/ask spread {spread_pct:.1f}% exceeds max {max_spread}%"
                    logger.warning("mcp_wide_spread", spread_pct=spread_pct)
                    return {"error": msg, "occ_symbol": occ_sym, "bid": bid, "ask": ask, "spread_pct": spread_pct}

        cmd = build_alpaca_execution(rec, occ_sym, bid=bid, ask=ask)
        execute_flag = self.config.general.execute

        if not execute_flag:
            execution_json = format_execution_json(cmd)
            logger.info("mcp_dry_run", execution_payload=json.loads(execution_json))
            return {
                "status": "dry_run",
                "occ_symbol": occ_sym,
                "bid": bid,
                "ask": ask,
                "execution_command": json.loads(execution_json),
            }

        result = await self._execute_mcp_command(cmd)
        return {
            "status": "executed",
            "occ_symbol": occ_sym,
            "bid": bid,
            "ask": ask,
            "execution_result": result,
        }

    async def _mcp_call(self, tool: str, params: dict) -> list[dict[str, Any]]:
        """Send a JSON-RPC tools/call request to the MCP daemon.

        Args:
            tool: Name of the tool to call.
            params: Arguments for the tool.

        Returns:
            List of result dicts from the daemon response.
        """
        daemon = self.config.mcp.alpaca
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool, "arguments": params},
            "id": str(uuid.uuid4()),
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                daemon.command,
                *daemon.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            request_bytes = (json.dumps(request) + "\n").encode()
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(request_bytes),
                timeout=daemon.timeout_sec,
            )
            if stderr_bytes:
                stderr = stderr_bytes.decode().strip()
                if stderr:
                    logger.warning("mcp_stderr", stderr=stderr)
            results = self._parse_mcp_response(stdout_bytes.decode())
            return results
        except asyncio.TimeoutError:
            logger.error("mcp_timeout", tool=tool)
            return []
        except FileNotFoundError:
            logger.error("mcp_not_found", command=daemon.command)
            return []
        except Exception as e:
            logger.error("mcp_error", tool=tool, error=str(e))
            return []

    async def _mcp_single(self, tool: str, params: dict) -> Optional[dict[str, Any]]:
        """Call an MCP tool and return the first result element.

        Args:
            tool: Name of the tool to call.
            params: Arguments for the tool.

        Returns:
            First result dict, or None if no results.
        """
        results = await self._mcp_call(tool, params)
        return results[0] if results else None

    async def _execute_mcp_command(self, cmd: ExecutionCommand) -> dict[str, Any]:
        """Send an order execution command to the MCP daemon.

        Args:
            cmd: The execution command to send.

        Returns:
            Dict containing parsed response and raw output.
        """
        daemon = self.config.mcp.alpaca
        payload = cmd.payload.model_dump(exclude_none=True) if isinstance(cmd.payload, AlpacaOrderPayload) else cmd.payload.model_dump(exclude_none=True)
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": cmd.action, "arguments": payload},
            "id": str(uuid.uuid4()),
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                daemon.command, *daemon.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate((json.dumps(request) + "\n").encode()),
                timeout=daemon.timeout_sec,
            )
            result_text = stdout_bytes.decode()
            if stderr_bytes:
                stderr_text = stderr_bytes.decode().strip()
                if stderr_text:
                    logger.warning("mcp_exec_stderr", stderr=stderr_text)
            parsed = self._parse_mcp_response(result_text)
            return {"response": parsed, "raw": result_text}
        except asyncio.TimeoutError:
            return {"error": "mcp_timeout"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _parse_mcp_response(output: str) -> list[dict[str, Any]]:
        """Parse JSON-RPC response lines from the MCP daemon output.

        Args:
            output: Raw stdout string from the daemon process.

        Returns:
            List of parsed result dicts, skipping errors.
        """
        results: list[dict[str, Any]] = []
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if "result" in msg:
                    content = msg["result"]
                    if isinstance(content, list):
                        results.extend(content)
                    elif isinstance(content, dict):
                        results.append(content)
                elif "error" in msg:
                    logger.error("mcp_rpc_error", error=msg["error"])
            except json.JSONDecodeError:
                continue
        return results


def parse_float(value: Any) -> Optional[float]:
    """Safely parse a value as a float, returning None on failure.

    Args:
        value: The value to parse.

    Returns:
        A float, or None if parsing fails.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
