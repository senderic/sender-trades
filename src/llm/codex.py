"""Codex CLI backend for the LLM model chain.

Ported from ``~/atlas-morning-briefing/scripts/codex_client.py``. Model ids
of the form ``codex/<model>`` (e.g. ``codex/gpt-5.6-sol``) in
:attr:`~src.config.LLMConfig.primary_model` / ``fallback_models`` are served
by ``codex exec`` instead of ``opencode run``, so Codex slots into the same
ordered fallback chain, call budget, and graph deadlines as every other
model -- see :meth:`src.llm.client.OpencodeLLMClient.invoke`.

The CLI authenticates through the user's ChatGPT subscription (``codex
login``), not an API key, so calls carry no per-token charge.

Each call runs one isolated, non-interactive process: ``--ephemeral`` (no
session file), ``--ignore-user-config`` / ``--ignore-rules`` (no
``~/.codex`` config or AGENTS.md leaking into a trading prompt), a
read-only sandbox, and ``-C /tmp`` so the agent has no project checkout to
wander through. The prompt goes over stdin, so its size is not bounded by
argv limits.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from src.config import CodexConfig

CODEX_PREFIX = "codex/"


def is_codex_model(model_id: str) -> bool:
    """Return True for model ids served by the Codex CLI (``codex/<model>``)."""
    return model_id.startswith(CODEX_PREFIX)


def codex_model_name(model_id: str) -> str:
    """Strip the ``codex/`` routing prefix: ``codex/gpt-5.6-sol`` -> ``gpt-5.6-sol``."""
    return model_id[len(CODEX_PREFIX) :] if is_codex_model(model_id) else model_id


def build_command(config: CodexConfig, model_id: str) -> list[str]:
    """Build the ``codex exec`` argv for one call; the prompt is read from stdin."""
    return [
        config.executable,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "-C",
        "/tmp",
        "-m",
        codex_model_name(model_id),
        "-c",
        f'model_reasoning_effort="{config.reasoning_effort}"',
        "--json",
        "-",
    ]


def parse_jsonl(stdout: str) -> tuple[str | None, dict[str, Any] | None, bool, bool]:
    """Return ``(message, usage, completed, failed)`` from a ``codex exec --json`` stream.

    The final ``agent_message`` item is the response. A turn only counts
    when a ``turn.completed`` event closes it; any failure/error event (or
    a semantic event after completion) marks the stream failed. Non-JSON
    diagnostic lines are skipped. ``item.completed`` items of type
    ``error`` are CLI notices (e.g. "Exceeded skills context budget"), not
    turn failures, and are ignored like any other non-message item.
    """
    message: str | None = None
    usage: dict[str, Any] | None = None
    saw_completion = False
    saw_failure = False

    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "")).lower()
        if saw_completion and event_type:
            saw_failure = True
            continue
        if (
            event_type in {"error", "turn.failed", "turn.error", "item.failed", "item.error"}
            or event_type.endswith(".failed")
            or event_type.endswith(".error")
            or event.get("status") in {"failed", "error"}
            or event.get("error") is not None
        ):
            saw_failure = True

        item = event.get("item")
        if event_type == "item.completed" and isinstance(item, dict):
            if item.get("status") in {"failed", "error"}:
                saw_failure = True
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                message = item["text"]

        if event_type == "turn.completed":
            saw_completion = True
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]

    return message, usage, saw_completion, saw_failure


def run_codex(
    config: CodexConfig, model_id: str, prompt: str, timeout_sec: float
) -> tuple[str | None, dict[str, Any] | None, str]:
    """Run one Codex CLI call.

    Raises :class:`subprocess.TimeoutExpired` on timeout, exactly like the
    ``opencode`` path, so callers share one timeout handler.

    Returns:
        ``(response, usage, error)``. ``response`` is the stripped message
        text, or ``None`` on any failure, in which case ``error`` says why.
    """
    result = subprocess.run(
        build_command(config, model_id),
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip()[-300:] or (result.stdout or "")[-300:]
        return None, None, f"codex exit {result.returncode}: {detail}"

    message, usage, completed, failed = parse_jsonl(result.stdout)
    if failed or not completed or not message or not message.strip():
        return None, usage, "invalid codex response (failed or incomplete turn)"
    return message.strip(), usage, ""
