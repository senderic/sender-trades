"""OpenCode CLI LLM client with free-Zen-first + paid-Go fallback chain.

Models are organised into two tiers by provider namespace (see
:class:`src.config.LLMConfig`):

- Zen (``opencode/*``) — free tier, tried first in order.
- Go (``opencode-go/*``) — paid, tried only after every Zen model has
  been exhausted. :attr:`OpencodeLLMClient.paid_used` surfaces whether
  the last successful response came from a paid model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import structlog

from src.config import LLMConfig

logger = structlog.get_logger()

# Project root, resolved relative to this file (src/llm/client.py), used to
# locate ``.opencode/agent/<name>.md`` system-prompt files. This is
# deliberately independent of the ``--dir /tmp`` flag passed to the
# ``opencode`` CLI, which sets the CLI's own working directory and has
# nothing to do with where the agent definitions live on disk.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_paid_model(model_id: str) -> bool:
    """Return True for OpenCode Go (paid) model IDs.

    The OpenCode Go provider uses the ``opencode-go/`` namespace; the
    free Zen provider uses ``opencode/``. This is the only signal the
    CLI leaves us to distinguish cost tiers at config-entry time.

    Args:
        model_id: Model string as passed to ``opencode run -m ...``.

    Returns:
        True if the model lives in the paid Go namespace.
    """
    return model_id.startswith("opencode-go/")


class OpencodeLLMClient:
    """LLM client that shells out to the ``opencode`` CLI in headless mode.

    The free Zen models in :attr:`LLMConfig.zen_models` are tried
    first, in order, under a strict per-call timeout. If every Zen
    model fails (non-zero exit code, empty NDJSON response, or
    ``subprocess.TimeoutExpired``), the paid Go models in
    :attr:`LLMConfig.paid_go_models` are walked in order until one
    succeeds or the chain is exhausted.

    This mirrors the upstream atlas-morning-briefing
    ``scripts/opencode_client.py`` invocation pattern, intentionally
    pared down to the surface this project needs and extended with an
    explicit free/paid boundary.
    """

    def __init__(self, config: LLMConfig, reserved_calls_for_fallback: int = 0):
        """Initialize the client with an :class:`LLMConfig`.

        Args:
            config: LLM configuration (zen_models, paid_go_models, timeout).
            reserved_calls_for_fallback: Number of calls to hold back from
                ``config.max_calls_per_run`` for the exclusive use of
                :meth:`invoke`. :meth:`invoke_agent` (the graph path) is
                capped at ``max_calls_per_run - reserved_calls_for_fallback``
                so the monolithic fallback path called after a graph
                failure is never left with zero remaining budget. Defaults
                to 0 (no reservation), matching prior behaviour for
                existing callers and tests. The pipeline sets this from
                ``config.graph.reserved_calls_for_fallback``.
        """
        self.config = config
        self.reserved_calls_for_fallback = reserved_calls_for_fallback
        self._call_count = 0
        self._available: bool | None = None
        # Guards `_call_count` and every cumulative/last-* attribute below
        # so concurrent graph nodes (run via asyncio.to_thread, sharing one
        # client instance) can't lose increments or interleave writes.
        # Scope is kept tight: never held across subprocess.run.
        self._lock = threading.Lock()
        # Per-model outcome tracking surfaced for debug traces.
        self.last_served_by: str | None = None
        self.last_fallback_hit: bool = False
        self.paid_used: bool = False  # True iff last successful response came from opencode-go/*
        self.last_error: str = ""
        # Cumulative usage tracking (across all invocations).
        self.total_calls = 0
        self.total_failures = 0
        self.total_input_chars = 0
        self.total_output_chars = 0
        self.total_elapsed = 0.0
        self.fallback_hits = 0
        # Cache of agent-name -> system-prompt character count, so the
        # `.opencode/agent/<name>.md` file is read at most once per agent
        # per client lifetime (see `_agent_system_prompt_chars`).
        self._agent_prompt_chars_cache: dict[str, int] = {}

    @property
    def available(self) -> bool:
        """Check whether the opencode binary is on PATH and the client is enabled."""
        if self._available is not None:
            return self._available
        if not self.config.enabled:
            self._available = False
            return False
        self._available = shutil.which(self.config.opencode_path) is not None
        if self._available:
            logger.info("opencode_binary_found", path=self.config.opencode_path)
        else:
            logger.warning("opencode_binary_missing", path=self.config.opencode_path)
        return self._available

    def _try_reserve(self, reserve: int) -> tuple[bool, bool]:
        """Atomically check the call budget and claim one slot if available.

        This is the sole authority for admitting a call attempt; the
        check (against ``max_calls_per_run - reserve``) and the claim
        (``_call_count += 1``) happen under one lock acquisition so two
        threads racing for the last remaining slot cannot both succeed.
        Failed attempts must release their slot via :meth:`_record_failure`.

        Args:
            reserve: Calls to hold back from ``config.max_calls_per_run``
                (0 for :meth:`invoke`, ``reserved_calls_for_fallback`` for
                :meth:`invoke_agent`).

        Returns:
            ``(claimed, reserve_blocked)``. ``claimed`` is True if a slot
            was reserved. ``reserve_blocked`` is True only when the claim
            failed *solely* because of ``reserve`` — i.e. raw budget
            (``max_calls_per_run``) remained, but consuming it would eat
            into calls held back for the fallback path.
        """
        with self._lock:
            effective_max = self.config.max_calls_per_run - reserve
            if self._call_count < effective_max:
                self._call_count += 1
                return True, False
            reserve_blocked = reserve > 0 and self._call_count < self.config.max_calls_per_run
            return False, reserve_blocked

    def _record_failure(self) -> None:
        """Record a failed attempt and release the budget slot it claimed."""
        with self._lock:
            self.total_failures += 1
            self._call_count -= 1

    def _record_success(
        self,
        *,
        model: str,
        is_fallback: bool,
        is_paid: bool,
        input_chars: int,
        output_chars: int,
        elapsed: float,
    ) -> None:
        """Atomically record a successful call's usage and outcome state.

        Args:
            model: The model ID that served the response.
            is_fallback: True if this was not the first model in the chain.
            is_paid: True if ``model`` is in the paid Go namespace.
            input_chars: Character count of everything sent to the model.
            output_chars: Character count of the response text.
            elapsed: Wall-clock seconds the successful call took.
        """
        with self._lock:
            self.total_calls += 1
            self.total_input_chars += input_chars
            self.total_output_chars += output_chars
            self.total_elapsed += elapsed
            if is_fallback:
                self.fallback_hits += 1
            self.last_served_by = model
            self.last_fallback_hit = is_fallback
            self.paid_used = is_paid
            self.last_error = ""

    def _record_all_failed(self, last_error: str) -> None:
        """Atomically reset outcome state after every model in the chain failed."""
        with self._lock:
            self.last_error = last_error
            self.last_served_by = None
            self.last_fallback_hit = False
            self.paid_used = False

    def _agent_system_prompt_chars(self, agent_name: str) -> int:
        """Return the character length of an agent's system-prompt file.

        ``opencode run --agent <agent_name>`` loads a substantial system
        prompt from ``.opencode/agent/<agent_name>.md`` that is invisible
        to callers of :meth:`invoke_agent` (they only see ``prompt``), so
        cost telemetry built from ``len(prompt)`` alone systematically
        undercounts input size. This reads that file's length, relative to
        the project root -- NOT the ``--dir /tmp`` opencode working
        directory, which is unrelated to where agent definitions live.

        Results are cached per ``agent_name`` so the file is read at most
        once per agent for the life of this client. This is cost
        telemetry only: any failure to locate or read the file is
        swallowed and treated as zero extra chars so it never breaks the
        actual LLM call.

        Args:
            agent_name: Subagent name (matches the filename without ``.md``).

        Returns:
            Character count of the agent definition file, or 0 if it
            could not be found or read.
        """
        with self._lock:
            cached = self._agent_prompt_chars_cache.get(agent_name)
        if cached is not None:
            return cached

        chars = 0
        try:
            agent_path = _PROJECT_ROOT / ".opencode" / "agent" / f"{agent_name}.md"
            chars = len(agent_path.read_text())
        except OSError as e:
            logger.debug("opencode_agent_prompt_file_unreadable", agent=agent_name, error=str(e))
            chars = 0

        with self._lock:
            self._agent_prompt_chars_cache[agent_name] = chars
        return chars

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str | None:
        """Send a prompt via ``opencode run --format json`` with fallback.

        Tries every model in :attr:`LLMConfig.zen_models` (free) first,
        in order, under a ``timeout_sec``-second deadline. On timeout,
        non-zero exit, or empty response, walks
        :attr:`LLMConfig.paid_go_models` in order. Returns the first
        non-empty response text, or ``None`` if every model in both
        tiers failed.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions; prepended to
                the user prompt with a separator before passing to the
                CLI as a single positional argument.

        Returns:
            Response text from the first successful model, or ``None``.
        """
        if not self.available:
            return None

        chain = _dedupe(self.config.zen_models + self.config.paid_go_models)
        zen_set = set(self.config.zen_models)

        full_prompt = f"{system_prompt}\n\nUser Request: {prompt}" if system_prompt else prompt

        last_error = ""
        first_model = chain[0] if chain else ""

        for idx, model in enumerate(chain):
            is_fallback = idx > 0
            is_paid = is_paid_model(model)
            if is_fallback:
                logger.info(
                    "opencode_falling_back",
                    model=model,
                    paid=is_paid,
                    first=first_model,
                )

            # `invoke` is the monolithic fallback path itself, so it is
            # never subject to `reserved_calls_for_fallback` -- it gets
            # the full `max_calls_per_run` budget.
            claimed, _ = self._try_reserve(reserve=0)
            if not claimed:
                event = (
                    "opencode_budget_exhausted"
                    if idx == 0
                    else "opencode_budget_exhausted_during_fallback"
                )
                logger.warning(
                    event,
                    calls=self._call_count,
                    max=self.config.max_calls_per_run,
                    model=model,
                )
                break

            cmd = [
                self.config.opencode_path,
                "run",
                "-m",
                model,
                "--format",
                "json",
                "--auto",
                "--dir",
                "/tmp",
                "--pure",
                full_prompt,
            ]

            try:
                t0 = time.monotonic()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_sec,
                )
                elapsed = time.monotonic() - t0

                input_chars = len(full_prompt)
                if result.returncode != 0:
                    last_error = (result.stderr or "")[:300]
                    logger.debug(
                        "opencode_run_failed",
                        model=model,
                        paid=is_paid,
                        rc=result.returncode,
                        error=last_error,
                    )
                    self._record_failure()
                    continue

                response = _parse_ndjson_response(result.stdout)
                if not response:
                    last_error = "empty NDJSON response"
                    logger.debug(
                        "opencode_empty_response",
                        model=model,
                        paid=is_paid,
                        elapsed=round(elapsed, 2),
                    )
                    self._record_failure()
                    continue

                self._record_success(
                    model=model,
                    is_fallback=is_fallback,
                    is_paid=is_paid,
                    input_chars=input_chars,
                    output_chars=len(response),
                    elapsed=elapsed,
                )
                logger.info(
                    "opencode_invoke_ok",
                    model=model,
                    paid=is_paid,
                    fallback=is_fallback,
                    in_zen_tier=model in zen_set,
                    elapsed=round(elapsed, 2),
                    chars=len(response),
                )
                return response

            except subprocess.TimeoutExpired:
                self._record_failure()
                last_error = f"timeout after {self.config.timeout_sec}s"
                logger.warning(
                    "opencode_run_timed_out",
                    model=model,
                    paid=is_paid,
                    timeout=self.config.timeout_sec,
                )
                continue
            except Exception as e:
                self._record_failure()
                last_error = f"{type(e).__name__}: {e}"
                logger.debug("opencode_run_exception", model=model, error=str(e))
                continue

        self._record_all_failed(last_error)
        logger.warning(
            "opencode_all_models_failed",
            first=first_model,
            tried=len(chain),
            last_error=last_error,
        )
        return None

    def invoke_agent(
        self,
        agent_name: str,
        prompt: str,
        files: list[str] | None = None,
        timeout_sec: int | None = None,
        deadline_ts: float | None = None,
    ) -> str | None:
        """Invoke a named opencode subagent with the same fallback chain as :meth:`invoke`.

        Uses ``opencode run --agent <agent_name>`` to load the agent's system
        prompt from ``.opencode/agent/<agent_name>.md``. The model chain and
        fallback behaviour are identical to :meth:`invoke`.

        ``timeout_sec`` bounds a single *attempt*. With a 7-model chain a
        node's worst case is therefore ``7 * timeout_sec``, which on its
        own can overrun the caller's overall budget. ``deadline_ts`` bounds
        the whole retry chain: no further model is attempted once it
        passes, and the per-attempt timeout is clamped to the time left.

        Args:
            agent_name: Subagent name (matches the filename without ``.md``).
            prompt: User prompt passed as positional message to the agent.
            files: Optional list of file paths to attach via ``-f``.
            timeout_sec: Per-attempt timeout override.
            deadline_ts: Optional absolute :func:`time.monotonic` deadline
                for the entire fallback chain.

        Returns:
            Response text from the first successful model, or ``None``.
        """
        if not self.available:
            return None

        chain = _dedupe(self.config.zen_models + self.config.paid_go_models)
        timeout = timeout_sec if timeout_sec is not None else self.config.timeout_sec
        first_model = chain[0] if chain else ""
        # The agent's `.opencode/agent/<agent_name>.md` system prompt is
        # invisible to `prompt` but still sent to the CLI, so fold its
        # size into input-char accounting (read once, cached thereafter).
        agent_prompt_chars = self._agent_system_prompt_chars(agent_name)

        last_error = ""
        for idx, model in enumerate(chain):
            is_fallback = idx > 0
            is_paid = is_paid_model(model)

            attempt_timeout = timeout
            if deadline_ts is not None:
                remaining = deadline_ts - time.monotonic()
                if remaining <= 0:
                    last_error = "graph deadline exhausted"
                    logger.warning(
                        "opencode_agent_deadline_exhausted",
                        agent=agent_name,
                        model=model,
                        attempted=idx,
                    )
                    break
                attempt_timeout = max(1, min(timeout, int(remaining)))

            if is_fallback:
                logger.info(
                    "opencode_falling_back",
                    model=model,
                    paid=is_paid,
                    agent=agent_name,
                    first=first_model,
                )

            # `invoke_agent` is the graph path: it may only consume up to
            # `max_calls_per_run - reserved_calls_for_fallback`, leaving
            # the reserve for `invoke` (the monolithic fallback) so a
            # graph failure caused by budget exhaustion doesn't also
            # starve the fallback it triggers.
            claimed, reserve_blocked = self._try_reserve(reserve=self.reserved_calls_for_fallback)
            if not claimed:
                if reserve_blocked:
                    logger.warning(
                        "opencode_agent_reserve_blocked",
                        calls=self._call_count,
                        max=self.config.max_calls_per_run,
                        reserved=self.reserved_calls_for_fallback,
                        agent=agent_name,
                        model=model,
                    )
                else:
                    event = (
                        "opencode_budget_exhausted"
                        if idx == 0
                        else "opencode_budget_exhausted_during_fallback"
                    )
                    logger.warning(
                        event,
                        calls=self._call_count,
                        max=self.config.max_calls_per_run,
                        agent=agent_name,
                        model=model,
                    )
                break

            cmd = [
                self.config.opencode_path,
                "run",
                "--agent",
                agent_name,
                "-m",
                model,
                "--format",
                "json",
                "--auto",
                "--dir",
                "/tmp",
                "--pure",
            ]
            if files:
                for f in files:
                    cmd.extend(["-f", f])
            cmd.append(prompt)

            try:
                t0 = time.monotonic()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=attempt_timeout,
                )
                elapsed = time.monotonic() - t0

                input_chars = len(prompt) + agent_prompt_chars
                if result.returncode != 0:
                    last_error = (result.stderr or "")[:300]
                    logger.debug(
                        "opencode_agent_failed",
                        model=model,
                        agent=agent_name,
                        paid=is_paid,
                        rc=result.returncode,
                        error=last_error,
                    )
                    self._record_failure()
                    continue

                response = _parse_ndjson_response(result.stdout)
                if not response:
                    last_error = "empty NDJSON response"
                    logger.debug(
                        "opencode_agent_empty_response",
                        model=model,
                        agent=agent_name,
                        paid=is_paid,
                        elapsed=round(elapsed, 2),
                    )
                    self._record_failure()
                    continue

                self._record_success(
                    model=model,
                    is_fallback=is_fallback,
                    is_paid=is_paid,
                    input_chars=input_chars,
                    output_chars=len(response),
                    elapsed=elapsed,
                )
                logger.info(
                    "opencode_agent_ok",
                    model=model,
                    agent=agent_name,
                    paid=is_paid,
                    fallback=is_fallback,
                    elapsed=round(elapsed, 2),
                    chars=len(response),
                )
                return response

            except subprocess.TimeoutExpired:
                self._record_failure()
                last_error = f"timeout after {attempt_timeout}s"
                logger.warning(
                    "opencode_agent_timed_out",
                    model=model,
                    agent=agent_name,
                    paid=is_paid,
                    timeout=attempt_timeout,
                )
                continue
            except Exception as e:
                self._record_failure()
                last_error = f"{type(e).__name__}: {e}"
                logger.debug(
                    "opencode_agent_exception", model=model, agent=agent_name, error=str(e)
                )
                continue

        self._record_all_failed(last_error)
        logger.warning(
            "opencode_agent_all_models_failed",
            agent=agent_name,
            first=first_model,
            tried=len(chain),
            last_error=last_error,
        )
        return None

    def get_usage_summary_html(self) -> str:
        """Return an HTML snippet summarizing LLM usage for this run.

        Returns:
            An empty string if no LLM calls were made, otherwise an HTML
            <div> with a usage table that mirrors the atlas-briefing style.
        """
        if self.total_calls == 0 and self.total_failures == 0:
            return ""

        in_rate = 0.14
        out_rate = 0.28
        in_tok = int(self.total_input_chars / 4) if self.total_input_chars else 0
        out_tok = int(self.total_output_chars / 4) if self.total_output_chars else 0
        cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000

        model_str = self.last_served_by or "—"
        if self.fallback_hits:
            model_str += f" (<strong>{self.fallback_hits}</strong> fallback hit(s))"
        paid_note = (
            "<strong>paid</strong> model was used — actual cost may apply"
            if self.paid_used
            else "free tier — actual cost was $0.00"
        )

        return f"""<div style="margin-top:24px;padding-top:16px;border-top:1px solid #e1e4e8;font-size:12px;color:#57606a;">
<table style="border-collapse:collapse;width:100%;margin:8px 0;font-size:12px;">
<tr><td style="padding:4px 8px;font-weight:600;">LLM Calls</td><td style="padding:4px 8px;">{self.total_calls}</td>
    <td style="padding:4px 8px;font-weight:600;">Failed</td><td style="padding:4px 8px;">{self.total_failures}</td></tr>
<tr><td style="padding:4px 8px;font-weight:600;">Input (est.)</td><td style="padding:4px 8px;">{in_tok:,} tokens</td>
    <td style="padding:4px 8px;font-weight:600;">Output (est.)</td><td style="padding:4px 8px;">{out_tok:,} tokens</td></tr>
<tr><td style="padding:4px 8px;font-weight:600;">Elapsed</td><td style="padding:4px 8px;">{self.total_elapsed:.1f}s</td>
    <td style="padding:4px 8px;font-weight:600;">Est. Cost</td><td style="padding:4px 8px;">${cost:.6f}</td></tr>
<tr><td style="padding:4px 8px;font-weight:600;">Model</td><td colspan="3" style="padding:4px 8px;"><code>{model_str}</code></td></tr>
</table>
<p style="margin:4px 0;font-size:11px;color:#8b949e;">
Costs estimated at ${in_rate:.2f}/1M input and ${out_rate:.2f}/1M output (DeepSeek V4 Flash paid-tier rates).
Tokens estimated at ~4 bytes per token. This run used the {paid_note}.
</p>
</div>"""

    def get_usage_summary_text(self) -> str:
        """Return a plain-text usage summary suitable for the email body."""
        if self.total_calls == 0 and self.total_failures == 0:
            return ""

        in_rate = 0.14
        out_rate = 0.28
        in_tok = int(self.total_input_chars / 4) if self.total_input_chars else 0
        out_tok = int(self.total_output_chars / 4) if self.total_output_chars else 0
        cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000
        model_str = self.last_served_by or "—"
        paid_note = "paid model" if self.paid_used else "free tier"

        return (
            f"\n---\nOpencode Usage Summary\n"
            f"Calls: {self.total_calls}  Failures: {self.total_failures}  "
            f"Fallback hits: {self.fallback_hits}\n"
            f"Input (est.): {in_tok:,} tokens  Output (est.): {out_tok:,} tokens  "
            f"Elapsed: {self.total_elapsed:.1f}s\n"
            f"Est. Cost: ${cost:.6f}  Model: {model_str} ({paid_note})\n"
            f"Costs estimated at ${in_rate:.2f}/1M input, ${out_rate:.2f}/1M output. "
            f"Tokens estimated at ~4 bytes per token."
        )


def _dedupe(models: list[str]) -> list[str]:
    """De-duplicate a model list while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for m in models:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _parse_ndjson_response(stdout: str) -> str:
    """Parse ``opencode run --format json`` NDJSON output to response text.

    The CLI emits one JSON object per line (JSONL). Only events with
    ``type == "text"`` contribute to the response.

    Args:
        stdout: Raw stdout from the opencode process.

    Returns:
        Concatenated text from all ``text``-type events.
    """
    parts: list[str] = []
    for raw_line in stdout.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "text":
            text = ""
            part = event.get("part")
            if isinstance(part, dict):
                text = part.get("text", "") or ""
            elif isinstance(part, str):
                text = part
            if text:
                parts.append(text)
    return "".join(parts)
