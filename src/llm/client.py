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
import time
from typing import Any

import structlog

from src.config import LLMConfig

logger = structlog.get_logger()


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

    def __init__(self, config: LLMConfig):
        """Initialize the client with an :class:`LLMConfig`.

        Args:
            config: LLM configuration (zen_models, paid_go_models, timeout).
        """
        self.config = config
        self._call_count = 0
        self._available: bool | None = None
        # Per-model outcome tracking surfaced for debug traces.
        self.last_served_by: str | None = None
        self.last_fallback_hit: bool = False
        self.paid_used: bool = False  # True iff last successful response came from opencode-go/*
        self.last_error: str = ""

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

        if self._call_count >= self.config.max_calls_per_run:
            logger.warning(
                "opencode_budget_exhausted",
                calls=self._call_count,
                max=self.config.max_calls_per_run,
            )
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

            if self._call_count >= self.config.max_calls_per_run:
                logger.warning(
                    "opencode_budget_exhausted_during_fallback",
                    calls=self._call_count,
                    max=self.config.max_calls_per_run,
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

                if result.returncode != 0:
                    last_error = (result.stderr or "")[:300]
                    logger.debug(
                        "opencode_run_failed",
                        model=model,
                        paid=is_paid,
                        rc=result.returncode,
                        error=last_error,
                    )
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
                    continue

                self._call_count += 1
                # A call counts as a "fallback hit" if it was not the
                # very first model in the (zen+paid) chain.
                last_fallback_hit = is_fallback
                # `paid_used` is only true if the model that actually
                # served the response is from opencode-go/*. We do not
                # set it for free models simply because earlier paid
                # attempts failed.
                paid_used = is_paid
                logger.info(
                    "opencode_invoke_ok",
                    model=model,
                    paid=is_paid,
                    fallback=is_fallback,
                    in_zen_tier=model in zen_set,
                    elapsed=round(elapsed, 2),
                    chars=len(response),
                )
                self.last_served_by = model
                self.last_fallback_hit = last_fallback_hit
                self.paid_used = paid_used
                self.last_error = ""
                return response

            except subprocess.TimeoutExpired:
                last_error = f"timeout after {self.config.timeout_sec}s"
                logger.warning(
                    "opencode_run_timed_out",
                    model=model,
                    paid=is_paid,
                    timeout=self.config.timeout_sec,
                )
                continue
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.debug("opencode_run_exception", model=model, error=str(e))
                continue

        self.last_error = last_error
        self.last_served_by = None
        self.last_fallback_hit = False
        self.paid_used = False
        logger.warning(
            "opencode_all_models_failed",
            first=first_model,
            tried=len(chain),
            last_error=last_error,
        )
        return None


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
