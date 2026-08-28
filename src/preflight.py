"""Pre-flight LLM model availability probe.

Runs ahead of the trading pipeline — see ``scripts/preflight.sh``, a
separate cron entry that fires ~15 minutes before ``run_trades.sh`` —
probes every model in the configured chain concurrently, and writes the
result to :attr:`~src.config.PreflightConfig.file_path`.
:class:`src.llm.client.OpencodeLLMClient` reads that file to reorder its
model chain before a run (see ``_resolve_chain`` there).

Design rules carried over from
``~/atlas-morning-briefing/scripts/preflight_model_check.py`` (the same
rules are also stated on :class:`~src.config.PreflightConfig`, which is
the actual config contract this module consumes):

- **The roster comes from config, never a local table here.** A
  hardcoded copy drifts from ``config.yaml`` and silently swaps models
  out from under the trading pipeline. :func:`build_chain` reads
  straight from :class:`~src.config.LLMConfig`.
- **A probe must ask for real prose**, not a one-word reply, so a
  healthy-but-verbose model isn't mistaken for a dead one. Atlas's HTTP
  path additionally budgets ``max_tokens`` for this; the ``opencode`` CLI
  used here (for both providers, unlike atlas) exposes no such flag, so
  the safeguard is a generous ``probe_timeout_sec`` plus a prompt that
  forces genuine generation rather than a trivial completion.
- **Report what actually happened.** Every configured model gets its own
  record; a probe only claims total failure when every model in the
  chain failed, never when one model failed.
- **Stale results are ignored** — by the consumer
  (``OpencodeLLMClient``), via ``max_age_sec``. This module's only
  obligation to that rule is to always write an honest ``timestamp``.

This module deliberately does NOT replicate atlas's heavy/medium/light
tier system or its ``config/model_capabilities.yaml`` reasoning-control
registry. sender-trades has a single flat model chain shared by every
graph node — there is nothing to tier, and reasoning-suppression control
is not a concern for this workload.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src.config import LLMConfig, Settings
from src.llm.client import _parse_ndjson_response

logger = structlog.get_logger()

# Asks for a real sentence (not "reply OK") so an empty/truncated response
# is actual evidence of trouble, not just a model taking the trivial path.
_PROBE_PROMPT = (
    "In one sentence, explain what a 0DTE options trade is. Reply with the sentence only."
)


def probe_model(model: str, *, opencode_path: str, timeout_sec: int) -> dict[str, Any]:
    """Probe one model through the ``opencode`` CLI.

    Reuses the exact invocation shape and NDJSON parsing that
    :meth:`~src.llm.client.OpencodeLLMClient.invoke` uses at runtime, so a
    probe result reflects the real call path instead of a second,
    possibly-diverging one.

    Args:
        model: Model id, e.g. ``opencode-go/deepseek-v4-pro``.
        opencode_path: Path/name of the ``opencode`` binary
            (:attr:`~src.config.LLMConfig.opencode_path`).
        timeout_sec: Per-probe timeout in seconds
            (:attr:`~src.config.PreflightConfig.probe_timeout_sec`).

    Returns:
        A record with ``available`` (bool), ``latency_ms`` (int), and
        ``error`` (str | None).
    """
    cmd = [
        opencode_path,
        "run",
        "-m",
        model,
        "--format",
        "json",
        "--auto",
        "--dir",
        "/tmp",
        "--pure",
        _PROBE_PROMPT,
    ]
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        # A timeout marks the model unavailable rather than raising —
        # this probe must never take down the caller (main() or the
        # ThreadPoolExecutor driving the other probes).
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        return {
            "available": False,
            "latency_ms": elapsed_ms,
            "error": f"timeout after {timeout_sec}s",
        }
    except Exception as e:
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        return {
            "available": False,
            "latency_ms": elapsed_ms,
            "error": f"{type(e).__name__}: {e}"[:300],
        }

    elapsed_ms = round((time.monotonic() - t0) * 1000)
    if result.returncode != 0:
        error = (result.stderr or "")[:300] or f"exit code {result.returncode}"
        return {"available": False, "latency_ms": elapsed_ms, "error": error}

    text = _parse_ndjson_response(result.stdout)
    if not text.strip():
        error = (result.stderr or "")[:300] or "empty response"
        return {"available": False, "latency_ms": elapsed_ms, "error": error}

    return {"available": True, "latency_ms": elapsed_ms, "error": None}


def build_chain(llm_config: LLMConfig) -> list[str]:
    """Build the model chain to probe, straight from config.

    Never hardcode a roster in this module — see the module docstring's
    first design rule. Order is primary first, then fallbacks, deduped.

    Args:
        llm_config: The application's :class:`~src.config.LLMConfig`.

    Returns:
        Deduplicated model ids in configured order.
    """
    seen: set[str] = set()
    chain: list[str] = []
    for model in [llm_config.primary_model, *llm_config.fallback_models]:
        if model not in seen:
            seen.add(model)
            chain.append(model)
    return chain


def run_preflight(llm_config: LLMConfig) -> dict[str, Any]:
    """Probe every model in the configured chain concurrently.

    Args:
        llm_config: The application's :class:`~src.config.LLMConfig`.

    Returns:
        A dict with a top-level ``timestamp`` (UTC ISO 8601) and a
        ``models`` mapping of model id -> probe record (see
        :func:`probe_model`). Every model returned by :func:`build_chain`
        is guaranteed a record, even if its future raised — a crashed
        probe becomes an ``available: False`` record, not a silent gap
        that would misrepresent "what actually happened".
    """
    chain = build_chain(llm_config)
    models: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=max(1, len(chain))) as executor:
        futures = {
            executor.submit(
                probe_model,
                model,
                opencode_path=llm_config.opencode_path,
                timeout_sec=llm_config.preflight.probe_timeout_sec,
            ): model
            for model in chain
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                models[model] = future.result()
            except Exception as e:  # pragma: no cover - probe_model already catches broadly
                models[model] = {
                    "available": False,
                    "latency_ms": 0,
                    "error": f"{type(e).__name__}: {e}"[:300],
                }

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "models": models,
    }


def write_results(results: dict[str, Any], file_path: str) -> None:
    """Write probe results to ``file_path`` as JSON.

    Args:
        results: The dict returned by :func:`run_preflight`.
        file_path: Destination path (:attr:`~src.config.PreflightConfig.file_path`).
    """
    Path(file_path).write_text(json.dumps(results, indent=2))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: probe the configured chain and write the result file.

    Intended for ``python -m src.preflight`` from a cron entry run ahead
    of ``run_trades.sh`` (see ``scripts/preflight.sh``). Always exits 0,
    even when every model fails — this is advisory telemetry for
    :class:`~src.llm.client.OpencodeLLMClient` to use as a reordering
    hint, never a gate. The trading run must never be blocked by a
    preflight probe failure.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Always ``0``.
    """
    parser = argparse.ArgumentParser(description="Pre-flight LLM model availability probe")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_yaml(Path(args.config))
    llm_config = settings.llm

    if not llm_config.enabled:
        logger.info("preflight_skipped", reason="llm.enabled is false")
        return 0
    if not llm_config.preflight.enabled:
        logger.info("preflight_skipped", reason="llm.preflight.enabled is false")
        return 0

    chain = build_chain(llm_config)
    logger.info(
        "preflight_start",
        models=chain,
        probe_timeout_sec=llm_config.preflight.probe_timeout_sec,
    )

    results = run_preflight(llm_config)
    write_results(results, llm_config.preflight.file_path)

    for model in chain:
        record = results["models"].get(model, {})
        logger.info(
            "preflight_probe_result",
            model=model,
            available=record.get("available", False),
            latency_ms=record.get("latency_ms"),
            error=record.get("error"),
        )

    available = [m for m in chain if results["models"].get(m, {}).get("available")]
    unavailable = [m for m in chain if m not in available]
    if available:
        logger.info(
            "preflight_complete",
            available=available,
            unavailable=unavailable,
            file_path=llm_config.preflight.file_path,
        )
    else:
        # Only claim total failure when every model in the chain failed —
        # see the module docstring's "report what actually happened" rule.
        logger.warning(
            "preflight_all_models_failed",
            models=chain,
            file_path=llm_config.preflight.file_path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
