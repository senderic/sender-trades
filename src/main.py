"""CLI entry point for the sender-trades application."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.config import Settings
from src.email_sender import send_email
from src.logging_setup import setup_logging
from src.pipeline import Pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the trading pipeline.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed namespace with config, dry-run, execute, and correlation-id options.
    """
    parser = argparse.ArgumentParser(
        description="0DTE Intraday Options Trading Recommendation System",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Override config: force dry-run (no MCP execution)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=None,
        help="Override config: force live execution via MCP",
    )
    parser.add_argument(
        "--correlation-id",
        type=str,
        default="",
        help="Correlation ID for this run (auto-generated if empty)",
    )
    parser.add_argument(
        "--email",
        action="store_true",
        default=None,
        help="Send forecast report via email",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        default=None,
        help="Skip sending forecast report via email",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    """Main entry point: parse args, load config, run pipeline, and print results.

    Args:
        argv: Optional argument list.

    Returns:
        Exit code (0 on success, 1 if errors occurred).
    """
    load_dotenv()
    args = parse_args(argv)
    correlation_id = args.correlation_id or uuid.uuid4().hex[:12]

    config_path = Path(args.config)
    settings = Settings.from_yaml(config_path)
    settings = settings.resolve_env_vars()

    if args.dry_run is True:
        settings.general.execute = False
    elif args.execute is True:
        settings.general.execute = True

    file_logger = setup_logging(settings, correlation_id)

    pipeline = Pipeline(settings, correlation_id, file_logger)
    result = await pipeline.run()

    print("\n" + "=" * 60)
    print(f"  Pipeline Complete — {correlation_id}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Errors:   {len(result.errors)}")
    if result.errors:
        for err in result.errors[:3]:
            print(f"    • {err}")

    if result.decision and result.decision.forecast:
        print(f"\n{result.decision.forecast.table()}")

    if result.decision and result.decision.recommendation:
        rec = result.decision.recommendation
        print(f"\n  Selected trade: {rec.asset} {rec.direction.value} @ {rec.target_strike}")
        print(f"  Confidence: {rec.confidence:.2f}  Contracts: {rec.contracts}")
        print(f"  Strategy: {rec.strategy_label}")
    else:
        print("\n  Trade: NONE — no recommendation passed all gates")
    print("=" * 60)

    should_email = (
        args.email
        if args.email is not None
        else (not args.no_email if args.no_email is not None else False)
    )
    if should_email and result.decision and result.decision.forecast:
        send_email(
            forecast=result.decision.forecast,
            dry_run=False,
            correlation_id=correlation_id,
            log_dir=settings.logging.json_dir,
        )

    return 0 if len(result.errors) == 0 else 1


def run() -> None:
    """Sync entry point — wraps the async main loop and exits with the return code."""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    run()
