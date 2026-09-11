"""OpenCode CLI LLM client with an ordered model fallback chain.

Models are tried in the order declared by :class:`src.config.LLMConfig`:
:attr:`~LLMConfig.primary_model` first, then
:attr:`~LLMConfig.fallback_models` in order (as of 2026-09-10: the
contributor-free Zen Muse primary, then nvidia-direct Nemotron 3 Ultra,
then DeepSeek V4 Pro via the paid OpenCode Go gateway
(``opencode-go/*``), then DeepSeek V4 Pro via OpenRouter
(``openrouter/*``) as a last resort). :attr:`OpencodeLLMClient.paid_used`
reflects whether the model that actually served the last successful
call is in a paid namespace -- see :func:`is_paid_model`.

When :attr:`~LLMConfig.preflight` is enabled, that configured order is a
starting point, not the final word: ``.model-availability.json`` (written
by ``src.preflight``, run ahead of the pipeline by a separate cron entry)
is consulted once per client instance to reorder — never shrink — the
chain. See :meth:`OpencodeLLMClient._resolve_chain`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime
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
    """Return True for paid provider namespaces.

    The OpenCode Go gateway (``opencode-go/*``), OpenRouter
    (``openrouter/*``), and nvidia-direct (``nvidia-direct/*``) incur real charges.
    """
    return (
        model_id.startswith("opencode-go/")
        or model_id.startswith("openrouter/")
        or model_id.startswith("nvidia-direct/")
    )


# Cache for `opencode models` output. `opencode models` takes a few
# seconds, so the roster is cached to disk with a timestamp and reused
# across processes (the pipeline, preflight, and any ad-hoc invocation
# within the same window) instead of being re-shelled-out-to on every
# client construction. A one-hour TTL is generous: the opencode model
# roster changes on the order of days, not minutes -- this incident was
# caused by a wrong *id*, not a roster change.
_KNOWN_MODELS_CACHE_PATH = ".opencode-models-cache.json"
_KNOWN_MODELS_CACHE_MAX_AGE_SEC = 3600


def _fetch_known_model_ids(opencode_path: str, timeout_sec: int = 20) -> set[str] | None:
    """Run ``opencode models`` and return the set of valid model ids.

    Returns ``None`` (never an empty set) when the command cannot be run,
    times out, or produces no usable output, so callers can distinguish
    "validation unavailable" from "everything is invalid" and fail open
    rather than dropping every configured model.

    Args:
        opencode_path: Path/name of the ``opencode`` binary.
        timeout_sec: Timeout for the ``opencode models`` subprocess.

    Returns:
        The set of model ids ``opencode`` currently recognizes, or
        ``None`` if the listing could not be obtained.
    """
    try:
        result = subprocess.run(
            [opencode_path, "models"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except Exception as e:
        logger.warning("opencode_models_list_failed", error=f"{type(e).__name__}: {e}")
        return None

    if result.returncode != 0:
        logger.warning(
            "opencode_models_list_failed",
            rc=result.returncode,
            error=(result.stderr or result.stdout or "")[:300],
        )
        return None

    ids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if not ids:
        logger.warning("opencode_models_list_empty")
        return None
    return ids


def get_known_model_ids(
    opencode_path: str,
    cache_path: str = _KNOWN_MODELS_CACHE_PATH,
    max_age_sec: int = _KNOWN_MODELS_CACHE_MAX_AGE_SEC,
) -> set[str] | None:
    """Return the known ``opencode`` model ids, cached on disk.

    ``opencode models`` takes a few seconds, so the result is cached to
    ``cache_path`` with a timestamp and reused for ``max_age_sec``. A
    failed refresh falls back to a stale cache (better than skipping
    validation) if one exists, else ``None``.

    Args:
        opencode_path: Path/name of the ``opencode`` binary.
        cache_path: Where to persist the cached roster.
        max_age_sec: Maximum cache age before a refresh is attempted.

    Returns:
        The set of known model ids, or ``None`` if it could not be
        determined (no cache and the live fetch failed) -- callers must
        treat ``None`` as "skip validation", not "nothing is valid".
    """
    path = Path(cache_path)
    cached_ids: set[str] | None = None
    if path.exists():
        try:
            data = json.loads(path.read_text())
            models = data.get("models")
            if isinstance(models, list):
                cached_ids = set(models)
            cached_at = datetime.fromisoformat(str(data["timestamp"]))
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=UTC)
            age_sec = (datetime.now(UTC) - cached_at).total_seconds()
            if age_sec <= max_age_sec and cached_ids:
                return cached_ids
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            cached_ids = None

    fresh = _fetch_known_model_ids(opencode_path)
    if fresh is not None:
        try:
            path.write_text(
                json.dumps({"timestamp": datetime.now(UTC).isoformat(), "models": sorted(fresh)})
            )
        except OSError as e:
            logger.debug("opencode_models_cache_write_failed", error=str(e))
        return fresh

    # Live refresh failed; fall back to a stale cache rather than
    # skipping validation outright.
    return cached_ids


def drop_unknown_models(chain: list[str], known: set[str] | None) -> list[str]:
    """Filter ``chain`` down to model ids ``opencode`` actually recognizes.

    Unknown ids are reported loudly (ERROR, since a typo'd/renamed model
    id is a silent-fallback trap identical to the 2026-09-05 Nemotron
    incident this validation exists to catch) and dropped. Never raises
    and never returns an empty chain when the input wasn't empty: if
    ``known`` is ``None`` (validation unavailable) or every model in
    ``chain`` is unknown (validation itself is probably unreliable, e.g.
    ``opencode models`` output changed shape), the original chain is
    returned unchanged so the pipeline still has something to try.

    Args:
        chain: Configured model ids, in order.
        known: Set of ids ``opencode models`` reports, or ``None`` if
            that could not be determined.

    Returns:
        ``chain`` with unknown ids removed, or ``chain`` unchanged when
        validation can't be trusted.
    """
    if known is None:
        return chain

    valid = [m for m in chain if m in known]
    unknown = [m for m in chain if m not in known]
    if unknown:
        logger.error(
            "unknown_model_id_dropped",
            unknown=unknown,
            remaining=valid,
        )
    if not valid:
        logger.error(
            "unknown_model_id_validation_suspect_failing_open",
            chain=chain,
        )
        return chain
    return valid


def validate_llm_config(llm_config: LLMConfig) -> LLMConfig:
    """Return ``llm_config`` with unknown model ids dropped from its chain.

    Intended for one-time use at pipeline startup (see
    :meth:`~src.pipeline.Pipeline.run`), *before* an
    :class:`OpencodeLLMClient` is constructed -- this is deliberately
    separate from :meth:`OpencodeLLMClient._resolve_chain`, which stays
    on the hot per-call path and must not add a subprocess call there
    (see its docstring). ``opencode models`` is a few seconds but is
    cached to disk (:func:`get_known_model_ids`), so calling this once
    per pipeline run is cheap.

    Never raises and never drops every model: if validation itself is
    unavailable or looks unreliable, the original config comes back
    unchanged (see :func:`drop_unknown_models`).

    Args:
        llm_config: The configured :class:`LLMConfig`.

    Returns:
        ``llm_config`` unchanged, or a copy with ``primary_model`` /
        ``fallback_models`` narrowed to ids ``opencode models`` reports.
    """
    if not llm_config.enabled:
        return llm_config

    configured = _dedupe([llm_config.primary_model, *llm_config.fallback_models])
    known = get_known_model_ids(llm_config.opencode_path)
    valid = drop_unknown_models(configured, known)
    if valid == configured:
        return llm_config

    return llm_config.model_copy(update={"primary_model": valid[0], "fallback_models": valid[1:]})


class OpencodeLLMClient:
    """LLM client that shells out to the ``opencode`` CLI in headless mode.

    Models are tried in order — :attr:`LLMConfig.primary_model` first,
    then :attr:`LLMConfig.fallback_models` — under a strict per-call
    timeout. A non-zero exit code, empty NDJSON response, or
    ``subprocess.TimeoutExpired`` advances to the next model until one
    succeeds or the chain is exhausted.
    """

    def __init__(self, config: LLMConfig, reserved_calls_for_fallback: int = 0):
        """Initialize the client with an :class:`LLMConfig`.

        Args:
            config: LLM configuration (primary_model, fallback_models, timeout).
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
        # Cache of agent-name -> stripped system-prompt body, so the
        # `.opencode/agent/<name>.md` file is read at most once per agent
        # per client lifetime (see `_agent_system_prompt`).
        self._agent_prompt_chars_cache: dict[str, str] = {}
        # The preflight-reordered model chain, resolved once and cached
        # (see `_resolve_chain`) so the `.model-availability.json` file is
        # read at most once per client lifetime rather than once per call.
        self._resolved_chain: list[str] | None = None

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

    def _resolve_chain(self) -> list[str]:
        """Return the model chain to try, reordered by the preflight probe if configured.

        Resolved once and cached on the instance under ``self._lock``, so
        concurrent graph nodes (:meth:`invoke_agent` runs via
        ``asyncio.to_thread``, several at once) share a single read of
        ``.model-availability.json`` instead of each re-reading and
        re-parsing it. :meth:`invoke` and :meth:`invoke_agent` both call
        this instead of building the chain independently, so the two
        paths cannot drift out of sync with each other.

        A preflight result may only ever REORDER the configured chain,
        never drop a model from it — a probe is a hint (the model could
        recover between probe time and run time), not a verdict. See
        :class:`~src.config.PreflightConfig` for the config contract and
        :func:`_reorder_chain` for the reordering rules themselves.

        Model-id validation against ``opencode models`` (see
        :func:`get_known_model_ids` / :func:`drop_unknown_models`) is
        deliberately NOT done here: this method is on the hot path for
        every :meth:`invoke`/:meth:`invoke_agent` call across the test
        suite and (via ``asyncio.to_thread``) concurrently from graph
        nodes, and shelling out to ``opencode models`` here would add a
        real subprocess call to every one of those call sites. Instead
        :func:`~src.pipeline.Pipeline.run` validates the configured
        chain once at startup, before any :class:`OpencodeLLMClient` is
        constructed, and :mod:`src.preflight` validates it separately
        before probing.

        Returns:
            The model chain, deduplicated, in the order calls should try
            it.
        """
        with self._lock:
            if self._resolved_chain is not None:
                return self._resolved_chain

            configured = _dedupe([self.config.primary_model, *self.config.fallback_models])
            preflight = self.config.preflight
            resolved = configured
            if preflight.enabled:
                data = _load_preflight_data(preflight.file_path, preflight.max_age_sec)
                if data is not None:
                    resolved = _reorder_chain(
                        configured, data.get("models", {}), preflight.select_by
                    )

            if resolved != configured:
                logger.info(
                    "preflight_chain_reordered",
                    select_by=preflight.select_by,
                    configured=configured,
                    resolved=resolved,
                    pinned=resolved[0],
                )
            elif preflight.enabled:
                logger.info(
                    "preflight_chain_unchanged",
                    select_by=preflight.select_by,
                    pinned=resolved[0] if resolved else None,
                )

            self._resolved_chain = resolved
            return self._resolved_chain

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
            is_paid: True if ``model`` is in a paid namespace
                (``opencode-go/*`` or ``openrouter/*``).
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

    def _agent_system_prompt(self, agent_name: str) -> str:
        """Return an agent's system-prompt body, frontmatter stripped.

        ``.opencode/agent/<agent_name>.md`` carries a YAML frontmatter
        block (``description``/``mode``/``permission``) followed by the
        agent's actual instructions. ``invoke_agent`` inlines those
        instructions directly into the prompt instead of relying on
        ``opencode run --agent``, because the paid Go/OpenRouter runtimes
        do not resolve project-local subagents and fail with
        ``agent "<name>" not found``.

        Results are cached per ``agent_name`` so the file is read at most
        once per agent for the life of this client. Any failure to
        locate or read the file is swallowed and returned as an empty
        string so it never breaks the actual LLM call.

        Args:
            agent_name: Subagent name (matches the filename without ``.md``).

        Returns:
            The stripped system-prompt body, or ``""`` if the file could
            not be found or read.
        """
        with self._lock:
            cached = self._agent_prompt_chars_cache.get(agent_name)
        if cached is not None:
            return cached

        body = ""
        try:
            agent_path = _PROJECT_ROOT / ".opencode" / "agent" / f"{agent_name}.md"
            body = _strip_frontmatter(agent_path.read_text())
        except OSError as e:
            logger.debug("opencode_agent_prompt_file_unreadable", agent=agent_name, error=str(e))
            body = ""

        with self._lock:
            self._agent_prompt_chars_cache[agent_name] = body
        return body

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str | None:
        """Send a prompt via ``opencode run --format json`` with fallback.

        Tries :attr:`LLMConfig.primary_model` first, then each model in
        :attr:`LLMConfig.fallback_models`, under a ``timeout_sec``-second
        deadline. Returns the first non-empty response text, or ``None``
        if every model in the chain failed.

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

        chain = self._resolve_chain()

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
                    # opencode writes its actual error to stdout, not
                    # stderr, so stderr is routinely empty here — fall
                    # back to the tail of stdout so the failure reason is
                    # never silently lost.
                    last_error = (result.stderr or "")[:300] or (result.stdout or "")[-300:]
                    logger.warning(
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
                if is_fallback:
                    # The primary model failed and a non-primary model in
                    # the chain served this response instead. This is a
                    # dead/degraded primary, not routine behaviour, so it
                    # is logged at WARNING even though the call itself
                    # succeeded -- see opencode_all_models_failed for the
                    # "nothing worked" case.
                    logger.warning(
                        "opencode_served_by_non_primary",
                        model=model,
                        primary=first_model,
                        paid=is_paid,
                        elapsed=round(elapsed, 2),
                        chars=len(response),
                    )
                logger.info(
                    "opencode_invoke_ok",
                    model=model,
                    paid=is_paid,
                    fallback=is_fallback,
                    elapsed=round(elapsed, 2),
                    chars=len(response),
                )
                return response

            except subprocess.TimeoutExpired:
                self._record_failure()
                elapsed = time.monotonic() - t0
                last_error = f"timeout after {self.config.timeout_sec}s"
                logger.warning(
                    "opencode_run_timed_out",
                    model=model,
                    paid=is_paid,
                    timeout=self.config.timeout_sec,
                    elapsed=round(elapsed, 2),
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

        The agent's system prompt is read from ``.opencode/agent/<agent_name>.md``
        (frontmatter stripped) and inlined into the prompt rather than passed
        via ``opencode run --agent``. The Go/OpenRouter runtimes do not
        resolve project-local subagents, so the ``--agent`` flag is avoided
        entirely. The model chain and fallback behaviour are otherwise
        identical to :meth:`invoke`.

        ``timeout_sec`` bounds a single *attempt*. ``deadline_ts`` bounds
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

        chain = self._resolve_chain()
        timeout = timeout_sec if timeout_sec is not None else self.config.timeout_sec
        first_model = chain[0] if chain else ""
        # Inline the agent's system prompt (frontmatter stripped) instead
        # of relying on `opencode run --agent`, which the paid runtimes
        # cannot resolve. Read once, cached thereafter.
        agent_system_prompt = self._agent_system_prompt(agent_name)
        full_prompt = (
            f"{agent_system_prompt}\n\nUser Request: {prompt}" if agent_system_prompt else prompt
        )

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
            cmd.append(full_prompt)

            try:
                t0 = time.monotonic()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=attempt_timeout,
                )
                elapsed = time.monotonic() - t0

                input_chars = len(full_prompt)
                if result.returncode != 0:
                    # opencode writes its actual error to stdout, not
                    # stderr, so stderr is routinely empty here — fall
                    # back to the tail of stdout so the failure reason is
                    # never silently lost.
                    last_error = (result.stderr or "")[:300] or (result.stdout or "")[-300:]
                    logger.warning(
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
                if is_fallback:
                    # The primary model failed and a non-primary model in
                    # the chain served this response instead -- a dead or
                    # degraded primary, logged at WARNING despite the call
                    # itself succeeding.
                    logger.warning(
                        "opencode_agent_served_by_non_primary",
                        model=model,
                        agent=agent_name,
                        primary=first_model,
                        paid=is_paid,
                        elapsed=round(elapsed, 2),
                        chars=len(response),
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
                elapsed = time.monotonic() - t0
                last_error = f"timeout after {attempt_timeout}s"
                logger.warning(
                    "opencode_agent_timed_out",
                    model=model,
                    agent=agent_name,
                    paid=is_paid,
                    timeout=attempt_timeout,
                    elapsed=round(elapsed, 2),
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

        # Surface a dead/degraded primary model in the email itself, not
        # just the logs -- the last successful call (if any) had to fall
        # back, or every model in the chain failed outright.
        warning_html = ""
        if self.last_served_by is None and self.total_failures > 0:
            warning_html = (
                '<p style="margin:4px 0 8px;font-size:12px;color:#cf222e;font-weight:600;">'
                "⚠ Every model in the LLM chain failed this run — check opencode logs."
                "</p>"
            )
        elif self.last_fallback_hit:
            warning_html = (
                '<p style="margin:4px 0 8px;font-size:12px;color:#9a6700;font-weight:600;">'
                f"⚠ Primary model did not respond — served by fallback ({self.last_served_by})."
                "</p>"
            )

        return f"""<div style="margin-top:24px;padding-top:16px;border-top:1px solid #e1e4e8;font-size:12px;color:#57606a;">
{warning_html}<table style="border-collapse:collapse;width:100%;margin:8px 0;font-size:12px;">
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

        warning_line = ""
        if self.last_served_by is None and self.total_failures > 0:
            warning_line = "WARNING: every model in the LLM chain failed this run.\n"
        elif self.last_fallback_hit:
            warning_line = (
                f"WARNING: primary model did not respond — served by fallback "
                f"({self.last_served_by}).\n"
            )

        return (
            f"\n---\nOpencode Usage Summary\n"
            f"{warning_line}"
            f"Calls: {self.total_calls}  Failures: {self.total_failures}  "
            f"Fallback hits: {self.fallback_hits}\n"
            f"Input (est.): {in_tok:,} tokens  Output (est.): {out_tok:,} tokens  "
            f"Elapsed: {self.total_elapsed:.1f}s\n"
            f"Est. Cost: ${cost:.6f}  Model: {model_str} ({paid_note})\n"
            f"Costs estimated at ${in_rate:.2f}/1M input, ${out_rate:.2f}/1M output. "
            f"Tokens estimated at ~4 bytes per token."
        )


def _strip_frontmatter(content: str) -> str:
    """Strip a leading YAML frontmatter block (``---`` ... ``---``).

    Agent ``.md`` files carry a frontmatter block before the actual
    system-prompt body. Returns the body (or the whole string when there
    is no frontmatter).
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return content
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :]).strip()
    return content


def _load_preflight_data(file_path: str, max_age_sec: int) -> dict[str, Any] | None:
    """Load and validate ``.model-availability.json``, written by ``src.preflight``.

    Returns ``None`` — meaning "use the configured chain order unchanged"
    — when the file is missing, unreadable, malformed, or older than
    ``max_age_sec``. Model health changes hour to hour (an outage clears,
    a rate limit resets), so a stale result is worse than none: it would
    keep pinning a model that failed hours ago and may be fine now. This
    never raises; a broken preflight file must never break a trading run.

    Args:
        file_path: Path to the preflight results file
            (:attr:`~src.config.PreflightConfig.file_path`).
        max_age_sec: Maximum age in seconds before the file is ignored
            (:attr:`~src.config.PreflightConfig.max_age_sec`).

    Returns:
        The parsed dict (with ``timestamp`` and ``models`` keys), or
        ``None`` if it should be ignored.
    """
    path = Path(file_path)
    if not path.exists():
        logger.info("preflight_file_missing", path=file_path)
        return None

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("preflight_file_unreadable", path=file_path, error=str(e))
        return None

    if not isinstance(data, dict) or "models" not in data or "timestamp" not in data:
        logger.warning("preflight_file_malformed", path=file_path)
        return None

    try:
        probed_at = datetime.fromisoformat(str(data["timestamp"]))
    except ValueError as e:
        logger.warning("preflight_file_bad_timestamp", path=file_path, error=str(e))
        return None
    if probed_at.tzinfo is None:
        probed_at = probed_at.replace(tzinfo=UTC)

    age_sec = (datetime.now(UTC) - probed_at).total_seconds()
    if age_sec > max_age_sec:
        logger.info(
            "preflight_file_stale", path=file_path, age_sec=round(age_sec), max_age_sec=max_age_sec
        )
        return None

    return data


def _reorder_chain(chain: list[str], models: dict[str, Any], select_by: str) -> list[str]:
    """Reorder ``chain`` using preflight probe records, without ever dropping a model.

    A probe is a hint, not a verdict — a model marked unavailable at
    probe time (network blip, rate limit) can easily recover before the
    run starts a few minutes later, so it is only ever demoted to the
    back of the chain, never removed. The returned list always contains
    every model in ``chain``.

    ``select_by`` diverges from atlas on purpose (see
    :class:`~src.config.PreflightConfig`): atlas pins the first model
    that answers because its failure mode is hard outages, so
    ``"order"`` (configured order, unavailable models demoted) matches
    its behaviour. Here the failure mode is a slow model burning a whole
    node's timeout budget rather than actually being down, so
    ``"latency"`` instead pins the fastest model that answered.

    Args:
        chain: The configured, deduplicated model chain.
        models: The ``models`` mapping from the preflight results file
            (model id -> probe record).
        select_by: ``"latency"`` or ``"order"``.

    Returns:
        ``chain`` reordered; same elements, same length.
    """

    def is_available(model: str) -> bool:
        record = models.get(model)
        return bool(isinstance(record, dict) and record.get("available"))

    available = [m for m in chain if is_available(m)]
    unavailable = [m for m in chain if not is_available(m)]

    if select_by == "latency":

        def latency_ms(model: str) -> float:
            record = models.get(model) or {}
            value = record.get("latency_ms")
            return float(value) if isinstance(value, int | float) else float("inf")

        available = sorted(available, key=latency_ms)

    # "order" (and any other value) falls through with `available` left in
    # configured order — unavailable models are still demoted to the back.
    return available + unavailable


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
