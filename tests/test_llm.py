from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog.testing

from src.config import LLMConfig, PreflightConfig
from src.llm.client import (
    OpencodeLLMClient,
    _load_preflight_data,
    _parse_ndjson_response,
    _reorder_chain,
    _strip_frontmatter,
    drop_unknown_models,
    get_known_model_ids,
    is_paid_model,
    validate_llm_config,
)
from src.llm.resynthesizer import resynthesize_briefing
from src.models.briefing import BriefingData, BriefingQuality


def _ndjson_output(text: str) -> str:
    """Build a synthetic NDJSON stream with one text event."""
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


class TestIsPaidModel:
    def test_go_namespace_is_paid(self) -> None:
        assert is_paid_model("opencode-go/deepseek-v4-pro") is True
        assert is_paid_model("opencode-go/glm-5.2") is True

    def test_openrouter_namespace_is_paid(self) -> None:
        assert is_paid_model("openrouter/deepseek/deepseek-v4-pro") is True

    def test_zen_namespace_is_free(self) -> None:
        assert is_paid_model("opencode/deepseek-v4-flash-free") is False
        assert is_paid_model("opencode/mimo-v2.5-free") is False

    def test_unknown_namespace_is_free(self) -> None:
        # Conservative default: not-paid unless we are certain.
        assert is_paid_model("deepinfra/foo") is False


class TestParseNdjsonResponse:
    def test_extracts_text_events(self) -> None:
        stdout = (
            json.dumps({"type": "session", "part": {"text": "ignored"}})
            + "\n"
            + json.dumps({"type": "text", "part": {"text": "Hello "}})
            + "\n"
            + json.dumps({"type": "text", "part": {"text": "world"}})
            + "\n"
        )
        assert _parse_ndjson_response(stdout) == "Hello world"

    def test_skips_non_json_lines(self) -> None:
        stdout = "not json\n" + _ndjson_output("ok")
        assert _parse_ndjson_response(stdout) == "ok"

    def test_empty_stdout_returns_empty(self) -> None:
        assert _parse_ndjson_response("") == ""


class TestOpencodeLLMClientAvailability:
    def test_disabled_unavailable(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=False))
        assert client.available is False
        assert client.invoke("hi") is None

    def test_enabled_but_binary_missing_returns_none(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="no-such-bin"))
        assert client.available is False
        assert client.invoke("hi") is None


class TestOpencodeLLMClientInvoke:
    def _success_completed(self, stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["opencode"], returncode=rc, stdout=stdout, stderr=""
        )

    def test_primary_model_success(self) -> None:
        client = OpencodeLLMClient(
            LLMConfig(enabled=True, opencode_path="opencode", timeout_sec=60)
        )
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("synthetic summary")),
            ) as mock_run,
        ):
            response = client.invoke("test prompt")
        assert response == "synthetic summary"
        # First successful model is the primary_model.
        assert client.last_served_by == "opencode/muse-spark-1.3-contributor-free"
        assert client.last_fallback_hit is False
        assert client.paid_used is False
        assert mock_run.call_args.kwargs["timeout"] == 60
        args = mock_run.call_args.args[0]
        assert "opencode/muse-spark-1.3-contributor-free" in args

    def test_primary_timeout_falls_back_to_fallback(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            timeout_sec=5,
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode-go/deepseek-v4-pro" in cmd:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 5))
            if "openrouter/deepseek/deepseek-v4-pro" in cmd:
                return self._success_completed(_ndjson_output("router ok"))
            return self._success_completed(_ndjson_output("unexpected"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "router ok"
        assert client.last_served_by == "openrouter/deepseek/deepseek-v4-pro"
        assert client.last_fallback_hit is True
        # Both primary and fallback are paid.
        assert client.paid_used is True

    def test_primary_fails_falls_back_to_router(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "openrouter/deepseek/deepseek-v4-pro" in cmd:
                return self._success_completed(_ndjson_output("router ok"))
            # Primary fails with non-zero rc.
            return subprocess.CompletedProcess(cmd, 1, "", "go fail")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "router ok"
        assert client.last_served_by == "openrouter/deepseek/deepseek-v4-pro"
        assert client.last_fallback_hit is True
        # Serving model is from openrouter/* so paid tracking must fire.
        assert client.paid_used is True

    def test_paid_model_failure_does_not_mark_paid_used(self) -> None:
        # If a paid model is tried but fails, and a subsequent free
        # model succeeds, paid_used should remain False (the response
        # didn't actually come from a paid model). The chain is
        # configurable, so a free fallback still exercises this path.
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["opencode/deepseek-v4-flash-free"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode-go/deepseek-v4-pro" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "fail")
            if "opencode/deepseek-v4-flash-free" in cmd:
                return self._success_completed(_ndjson_output("free ok"))

            return self._success_completed(_ndjson_output("unexpected"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "free ok"
        assert client.last_served_by == "opencode/deepseek-v4-flash-free"
        assert client.paid_used is False

    def test_all_models_fail(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
        )
        client = OpencodeLLMClient(cfg)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=subprocess.CompletedProcess(["opencode"], 1, "", "nope"),
            ),
        ):
            response = client.invoke("prompt")
        assert response is None
        assert client.last_served_by is None
        assert client.paid_used is False
        assert client.last_error != ""

    def test_dedup_across_primary_and_fallback(self) -> None:
        # A model appearing in both primary and fallback must only be
        # tried once (preserving first appearance, i.e. primary wins).
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=[
                "opencode-go/deepseek-v4-pro",
                "openrouter/deepseek/deepseek-v4-pro",
            ],
        )
        client = OpencodeLLMClient(cfg)
        call_count = {"n": 0}

        def run_side_effect(cmd, **kwargs):
            call_count["n"] += 1
            if "openrouter/deepseek/deepseek-v4-pro" in cmd:
                return self._success_completed(_ndjson_output("router ok"))
            # Primary fails.
            return subprocess.CompletedProcess(cmd, 1, "", "fail")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "router ok"
        # First call: go primary (fail). Second call: router (ok).
        # The duplicate opencode-go/deepseek-v4-pro in fallback_models
        # must NOT be retried.
        assert call_count["n"] == 2

    def test_empty_response_counts_as_failure(self) -> None:
        client = OpencodeLLMClient(
            LLMConfig(
                enabled=True,
                opencode_path="opencode",
                primary_model="opencode-go/deepseek-v4-pro",
                fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
            )
        )

        def run_side_effect(cmd, **kwargs):
            if "opencode-go/deepseek-v4-pro" in cmd:
                return self._success_completed("")
            return self._success_completed(_ndjson_output("fallback ok"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "fallback ok"
        assert client.last_fallback_hit is True
        assert client.paid_used is True

    def test_budget_exhausted_returns_none(self) -> None:
        client = OpencodeLLMClient(
            LLMConfig(enabled=True, opencode_path="opencode", max_calls_per_run=0)
        )
        with patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"):
            response = client.invoke("prompt")
        assert response is None


class TestResynthesize:
    def _degraded_briefing(self) -> BriefingData:
        return BriefingData(
            briefing_date=date(2026, 7, 18),
            executive_summary="Synthesis unavailable for today's briefing. See sections below.",
            news_items=[],
            briefing_quality=BriefingQuality.DEGRADED,
        )

    def test_full_briefing_skipped(self) -> None:
        briefing = BriefingData(
            briefing_date=date(2026, 7, 18),
            executive_summary="Market is bullish today.",
            briefing_quality=BriefingQuality.FULL,
        )
        client = OpencodeLLMClient(LLMConfig(enabled=True))
        # No subprocess calls should happen.
        result = resynthesize_briefing(briefing, client)
        assert result.executive_summary == "Market is bullish today."
        assert result.briefing_quality == BriefingQuality.FULL
        assert client.last_served_by is None

    def test_degraded_briefing_resynthed(self) -> None:
        briefing = self._degraded_briefing()
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["opencode"],
                    0,
                    _ndjson_output("Markets lean bullish on AI earnings beats."),
                    "",
                ),
            ),
        ):
            resynthesize_briefing(briefing, client)
        assert briefing.executive_summary == "Markets lean bullish on AI earnings beats."
        assert briefing.briefing_quality == BriefingQuality.FULL
        # macro_sentiment should now be a real float rather than None.
        assert briefing.macro_sentiment is not None

    def test_failed_briefing_no_feed_items(self) -> None:
        briefing = BriefingData(
            briefing_date=date(2026, 7, 18),
            briefing_quality=BriefingQuality.FAILED,
        )
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["opencode"], 0, _ndjson_output("Briefing unavailable."), ""
                ),
            ),
        ):
            resynthesize_briefing(briefing, client)
        assert briefing.executive_summary == "Briefing unavailable."
        assert briefing.briefing_quality == BriefingQuality.FULL

    def test_llm_failure_leaves_briefing_untouched(self) -> None:
        briefing = self._degraded_briefing()
        original_summary = briefing.executive_summary
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=subprocess.CompletedProcess(["opencode"], 1, "", "all dead"),
            ),
        ):
            resynthesize_briefing(briefing, client)
        assert briefing.executive_summary == original_summary
        assert briefing.briefing_quality == BriefingQuality.DEGRADED
        assert briefing.macro_sentiment is None


class TestOpencodeLLMClientInvokeAgent:
    def _success_completed(self, stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["opencode"], returncode=rc, stdout=stdout, stderr=""
        )

    def test_invoke_agent_inlines_system_prompt(self) -> None:
        client = OpencodeLLMClient(
            LLMConfig(enabled=True, opencode_path="opencode", timeout_sec=60)
        )
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("agent response")),
            ) as mock_run,
        ):
            response = client.invoke_agent("research-spy", "analyze SPY please")
        assert response == "agent response"
        args = mock_run.call_args.args[0]
        # The paid runtimes cannot resolve project-local subagents, so the
        # `--agent` flag must NOT be used — the system prompt is inlined.
        assert "--agent" not in args
        assert "--format" in args
        assert "json" in args
        # The final positional arg carries the agent instructions + prompt.
        full_prompt = args[-1]
        assert "analyze SPY please" in full_prompt
        assert "market research analyst" in full_prompt  # from research-spy.md body

    def test_invoke_agent_disabled_returns_none(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=False))
        assert client.invoke_agent("research-spy", "prompt") is None

    def test_invoke_agent_no_binary_returns_none(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="no-such-bin"))
        assert client.invoke_agent("research-spy", "prompt") is None

    def test_invoke_agent_with_files(self) -> None:
        client = OpencodeLLMClient(
            LLMConfig(enabled=True, opencode_path="opencode", timeout_sec=60)
        )
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ) as mock_run,
        ):
            response = client.invoke_agent("research-spy", "prompt", files=["/tmp/data.json"])
        assert response == "ok"
        args = mock_run.call_args.args[0]
        assert "-f" in args
        assert "/tmp/data.json" in args

    def test_invoke_agent_budget_exhausted(self) -> None:
        client = OpencodeLLMClient(
            LLMConfig(enabled=True, opencode_path="opencode", max_calls_per_run=0)
        )
        with patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"):
            response = client.invoke_agent("research-spy", "prompt")
        assert response is None

    def test_invoke_agent_custom_timeout(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode", timeout_sec=5))
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ) as mock_run,
        ):
            response = client.invoke_agent("predict-spy", "prompt", timeout_sec=99)
        assert response == "ok"
        assert mock_run.call_args.kwargs["timeout"] == 99

    def test_invoke_agent_fallback_chain(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode-go/deepseek-v4-pro" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "fail")
            return self._success_completed(_ndjson_output("router ok"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke_agent("checker", "validate please")
        assert response == "router ok"
        assert client.paid_used is True
        assert client.last_served_by == "openrouter/deepseek/deepseek-v4-pro"
        assert client.last_fallback_hit is True


class TestOpencodeLLMClientConcurrency:
    """Task 1: concurrent graph nodes share one client -- budget must hold."""

    def _success_completed(self, stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["opencode"], returncode=rc, stdout=stdout, stderr=""
        )

    def test_concurrent_invoke_agent_never_exceeds_budget(self) -> None:
        # 20 threads race for a 5-call budget. Without the lock in
        # `_try_reserve`, the non-atomic check-then-increment lets more
        # than 5 succeed (lost updates) or corrupts `_call_count`.
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode/deepseek-v4-flash-free",
            fallback_models=[],
            max_calls_per_run=5,
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            time.sleep(0.01)  # widen the race window between check and claim
            return self._success_completed(_ndjson_output("ok"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
            ThreadPoolExecutor(max_workers=20) as pool,
        ):
            results = list(
                pool.map(lambda _: client.invoke_agent("research-spy", "analyze"), range(20))
            )

        successes = [r for r in results if r is not None]
        assert len(successes) == 5
        assert client._call_count == 5
        assert client.total_calls == 5
        assert client.total_failures == 0

    def test_concurrent_invoke_agent_last_served_by_is_consistent(self) -> None:
        # `last_served_by`/`paid_used` must always reflect one coherent
        # attempt's fields, never a torn mix from two interleaved writers.
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode/deepseek-v4-flash-free",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
            max_calls_per_run=50,
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            time.sleep(0.005)
            if "openrouter/deepseek/deepseek-v4-pro" in cmd:
                return self._success_completed(_ndjson_output("paid ok"))
            return self._success_completed(_ndjson_output("free ok"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
            ThreadPoolExecutor(max_workers=10) as pool,
        ):
            list(pool.map(lambda _: client.invoke_agent("checker", "validate"), range(10)))

        # Whatever model served last, paid_used must agree with it -- not
        # a stale True left over from a differently-scheduled thread.
        assert client.paid_used == is_paid_model(client.last_served_by)


class TestOpencodeLLMClientReservedFallback:
    """Task 2: the graph path must not be able to starve the monolithic fallback."""

    def _success_completed(self, stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["opencode"], returncode=rc, stdout=stdout, stderr=""
        )

    def test_invoke_agent_refused_while_invoke_still_works(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode/deepseek-v4-flash-free",
            fallback_models=[],
            max_calls_per_run=2,
        )
        client = OpencodeLLMClient(cfg, reserved_calls_for_fallback=1)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ),
        ):
            # Effective graph budget is max(2) - reserve(1) = 1 call.
            first = client.invoke_agent("research-spy", "p1")
            assert first == "ok"

            # A second graph call would eat into the reserve -- refused,
            # even though raw `max_calls_per_run` (2) has not been hit.
            second = client.invoke_agent("research-spy", "p2")
            assert second is None
            assert client._call_count == 1

            # The monolithic fallback is not graph-scoped: it can still
            # spend the reserved call.
            third = client.invoke("prompt")
            assert third == "ok"
            assert client._call_count == 2

    def test_default_reserve_is_zero_and_unaffects_existing_callers(self) -> None:
        # Constructor default must not change behaviour for callers that
        # don't pass reserved_calls_for_fallback.
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))
        assert client.reserved_calls_for_fallback == 0

    def test_invoke_unaffected_by_reserve_up_to_full_budget(self) -> None:
        # `invoke` (the fallback path itself) is never reserve-limited --
        # it may use the entire max_calls_per_run.
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode/deepseek-v4-flash-free",
            fallback_models=[],
            max_calls_per_run=1,
        )
        client = OpencodeLLMClient(cfg, reserved_calls_for_fallback=1)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ),
        ):
            response = client.invoke("prompt")
        assert response == "ok"


class TestAgentSystemPromptCharAccounting:
    """Task 3: invoke_agent's cost telemetry must include the agent's system prompt."""

    def _success_completed(self, stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["opencode"], returncode=rc, stdout=stdout, stderr=""
        )

    def test_input_chars_include_agent_system_prompt(self) -> None:
        agent_file = Path(__file__).resolve().parents[1] / ".opencode" / "agent" / "research-spy.md"
        agent_body = _strip_frontmatter(agent_file.read_text())
        # Sanity check against the code-review finding (~1.5-3KB agent files).
        assert len(agent_body) > 1000

        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))
        prompt = "please research SPY today"
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ),
        ):
            response = client.invoke_agent("research-spy", prompt)
        assert response == "ok"
        # Inlined system prompt + separator + user prompt.
        expected = len(agent_body) + len("\n\nUser Request: ") + len(prompt)
        assert client.total_input_chars == expected

    def test_agent_system_prompt_chars_are_cached(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))
        real_read_text = Path.read_text
        read_calls = {"n": 0}

        def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
            read_calls["n"] += 1
            return real_read_text(self, *args, **kwargs)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ),
            patch.object(Path, "read_text", counting_read_text),
        ):
            client.invoke_agent("research-spy", "p1")
            client.invoke_agent("research-spy", "p2")

        # Second call must hit the cache, not re-read the file.
        assert read_calls["n"] == 1

    def test_missing_agent_file_falls_back_to_prompt_only(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))
        prompt = "some prompt text"
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ),
        ):
            response = client.invoke_agent("no-such-agent-definition", prompt)
        assert response == "ok"
        # Falls back to current (len(prompt)-only) behaviour, and the
        # call itself must not fail just because telemetry couldn't
        # resolve the agent file.
        assert client.total_input_chars == len(prompt)


class TestInvokeAgentEmptyResponse:
    """A model can exit 0 and still emit no text events. That is a node
    failure, not a success — the graph would otherwise treat an empty
    string as a parseable response and fail confusingly downstream.
    """

    def test_empty_response_falls_through_to_next_model(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode/a",
            fallback_models=["opencode/b"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode/a" in cmd:
                # Exit 0, but the NDJSON stream carries no text events.
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 0, _ndjson_output("second model ok"), "")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke_agent("research-spy", "prompt")

        assert response == "second model ok"
        assert client.last_served_by == "opencode/b"
        assert client.total_failures == 1
        # The empty attempt must release its budget slot.
        assert client.total_calls == 1

    def test_all_models_empty_returns_none(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode/a",
            fallback_models=["opencode/b"],
        )
        client = OpencodeLLMClient(cfg)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=subprocess.CompletedProcess(["opencode"], 0, "", ""),
            ),
        ):
            response = client.invoke_agent("research-spy", "prompt")

        assert response is None
        assert client.last_served_by is None
        assert client.total_calls == 0


class TestInvokeAgentChainDeadline:
    """``timeout_sec`` bounds one attempt; ``deadline_ts`` bounds the chain.

    Without a chain-level deadline a single graph node could walk all 7
    models at the per-attempt timeout (7 x 45s = 315s) and blow straight
    through the graph's 240s wall-clock budget, because the orchestrator
    only re-checks the budget *between* phases.
    """

    @staticmethod
    def _cfg() -> LLMConfig:
        return LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode/a",
            fallback_models=[
                "opencode/b",
                "opencode/c",
                "opencode/d",
                "opencode-go/x",
                "opencode-go/y",
                "opencode-go/z",
            ],
            max_calls_per_run=50,
        )

    def test_chain_stops_once_deadline_passes(self) -> None:
        client = OpencodeLLMClient(self._cfg())
        attempts: list[str] = []

        def failing_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            attempts.append(cmd[cmd.index("-m") + 1])
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=failing_run),
        ):
            # Deadline already in the past: not a single model should be tried.
            response = client.invoke_agent("research-spy", "p", deadline_ts=time.monotonic() - 1.0)

        assert response is None
        assert attempts == []

    def test_without_deadline_walks_the_whole_chain(self) -> None:
        """Control: the deadline is what stops it, not some other guard."""
        client = OpencodeLLMClient(self._cfg())
        attempts: list[str] = []

        def failing_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            attempts.append(cmd[cmd.index("-m") + 1])
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=failing_run),
        ):
            response = client.invoke_agent("research-spy", "p")

        assert response is None
        assert len(attempts) == 7

    def test_per_attempt_timeout_clamped_to_remaining_budget(self) -> None:
        """A node may not hand subprocess a timeout longer than the budget left."""
        client = OpencodeLLMClient(self._cfg())
        seen_timeouts: list[float] = []

        def failing_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            seen_timeouts.append(kwargs["timeout"])
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=failing_run),
        ):
            client.invoke_agent(
                "research-spy",
                "p",
                timeout_sec=45,
                deadline_ts=time.monotonic() + 5.0,
            )

        assert seen_timeouts, "expected at least one attempt"
        # Configured per-attempt timeout is 45s but only ~5s of budget
        # remains, so every attempt must be clamped below it.
        assert all(t <= 5 for t in seen_timeouts)


class TestLoadPreflightData:
    """`_load_preflight_data` must never raise -- a broken preflight file
    is a hint that's gone bad, not a reason to break a trading run.
    """

    @staticmethod
    def _write(tmp_path: Path, models: dict, *, timestamp: datetime | None = None) -> str:
        path = tmp_path / ".model-availability.json"
        data = {
            "timestamp": (timestamp or datetime.now(UTC)).isoformat(),
            "models": models,
        }
        path.write_text(json.dumps(data))
        return str(path)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert _load_preflight_data(str(tmp_path / "nope.json"), 21600) is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / ".model-availability.json"
        path.write_text("{not json")
        assert _load_preflight_data(str(path), 21600) is None

    def test_missing_required_keys_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / ".model-availability.json"
        path.write_text(json.dumps({"unrelated": True}))
        assert _load_preflight_data(str(path), 21600) is None

    def test_stale_file_returns_none(self, tmp_path: Path) -> None:
        old = datetime.now(UTC) - timedelta(hours=10)
        path = self._write(
            tmp_path, {"m": {"available": True, "latency_ms": 1, "error": None}}, timestamp=old
        )
        assert _load_preflight_data(path, max_age_sec=21600) is None

    def test_fresh_file_loads(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"m": {"available": True, "latency_ms": 1, "error": None}})
        data = _load_preflight_data(path, max_age_sec=21600)
        assert data is not None
        assert "m" in data["models"]


class TestReorderChain:
    """`_reorder_chain` may only reorder the configured chain, never drop
    a model from it -- a probe is a hint, not a verdict.
    """

    def test_order_keeps_configured_order_and_demotes_unavailable(self) -> None:
        chain = ["a", "b", "c"]
        models = {
            "a": {"available": False, "latency_ms": 10, "error": "down"},
            "b": {"available": True, "latency_ms": 500, "error": None},
            "c": {"available": True, "latency_ms": 100, "error": None},
        }
        # "b" then "c" is configured order among the available models;
        # "a" (unavailable) is demoted to the back.
        assert _reorder_chain(chain, models, "order") == ["b", "c", "a"]

    def test_latency_orders_available_ascending_then_appends_unavailable(self) -> None:
        chain = ["a", "b", "c"]
        models = {
            "a": {"available": True, "latency_ms": 900, "error": None},
            "b": {"available": False, "latency_ms": 5, "error": "down"},
            "c": {"available": True, "latency_ms": 100, "error": None},
        }
        assert _reorder_chain(chain, models, "latency") == ["c", "a", "b"]

    def test_unavailable_model_is_never_dropped(self) -> None:
        """The most important guarantee: reorder only, never remove."""
        chain = ["primary", "fallback1", "fallback2"]
        models = {
            "primary": {"available": False, "latency_ms": 0, "error": "down"},
            "fallback1": {"available": False, "latency_ms": 0, "error": "down"},
            "fallback2": {"available": False, "latency_ms": 0, "error": "down"},
        }
        result = _reorder_chain(chain, models, "latency")
        assert set(result) == set(chain)
        assert len(result) == len(chain)
        assert "primary" in result

    def test_model_missing_from_probe_data_treated_as_unavailable(self) -> None:
        chain = ["a", "b"]
        models = {"a": {"available": True, "latency_ms": 1, "error": None}}
        assert _reorder_chain(chain, models, "order") == ["a", "b"]


class TestClientConsumesPreflight:
    """Integration: `OpencodeLLMClient.invoke`/`invoke_agent` actually use
    the preflight-reordered chain, resolved once and shared between them.
    """

    @staticmethod
    def _success_completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["opencode"], returncode=rc, stdout=stdout, stderr=""
        )

    @staticmethod
    def _write_preflight(tmp_path: Path, models: dict) -> str:
        path = tmp_path / ".model-availability.json"
        path.write_text(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "models": models}))
        return str(path)

    def test_invoke_pins_fastest_available_model(self, tmp_path: Path) -> None:
        file_path = self._write_preflight(
            tmp_path,
            {
                "opencode-go/deepseek-v4-pro": {
                    "available": True,
                    "latency_ms": 900,
                    "error": None,
                },
                "openrouter/deepseek/deepseek-v4-pro": {
                    "available": True,
                    "latency_ms": 50,
                    "error": None,
                },
            },
        )
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
            preflight=PreflightConfig(enabled=True, file_path=file_path, select_by="latency"),
        )
        client = OpencodeLLMClient(cfg)
        attempted: list[str] = []

        def run_side_effect(cmd, **kwargs):  # type: ignore[no-untyped-def]
            attempted.append(cmd[cmd.index("-m") + 1])
            return self._success_completed(_ndjson_output("ok"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")

        assert response == "ok"
        # Despite being configured as the fallback, the faster model
        # (per the preflight probe) must be tried first.
        assert attempted[0] == "openrouter/deepseek/deepseek-v4-pro"

    def test_invoke_agent_pins_fastest_available_model(self, tmp_path: Path) -> None:
        file_path = self._write_preflight(
            tmp_path,
            {
                "opencode-go/deepseek-v4-pro": {
                    "available": True,
                    "latency_ms": 900,
                    "error": None,
                },
                "openrouter/deepseek/deepseek-v4-pro": {
                    "available": True,
                    "latency_ms": 50,
                    "error": None,
                },
            },
        )
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
            preflight=PreflightConfig(enabled=True, file_path=file_path, select_by="latency"),
        )
        client = OpencodeLLMClient(cfg)
        attempted: list[str] = []

        def run_side_effect(cmd, **kwargs):  # type: ignore[no-untyped-def]
            attempted.append(cmd[cmd.index("-m") + 1])
            return self._success_completed(_ndjson_output("ok"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke_agent("research-spy", "prompt")

        assert response == "ok"
        assert attempted[0] == "openrouter/deepseek/deepseek-v4-pro"

    def test_probed_unavailable_model_is_still_reachable(self, tmp_path: Path) -> None:
        """Both models probed unavailable, but neither may be dropped from
        the chain -- a model can recover between probe time and run time.
        """
        file_path = self._write_preflight(
            tmp_path,
            {
                "opencode-go/deepseek-v4-pro": {
                    "available": False,
                    "latency_ms": 0,
                    "error": "down",
                },
                "openrouter/deepseek/deepseek-v4-pro": {
                    "available": False,
                    "latency_ms": 0,
                    "error": "down",
                },
            },
        )
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
            preflight=PreflightConfig(enabled=True, file_path=file_path, select_by="order"),
        )
        client = OpencodeLLMClient(cfg)
        attempted: list[str] = []

        def run_side_effect(cmd, **kwargs):  # type: ignore[no-untyped-def]
            model = cmd[cmd.index("-m") + 1]
            attempted.append(model)
            if model == "opencode-go/deepseek-v4-pro":
                return subprocess.CompletedProcess(cmd, 1, "", "still down")
            # Simulates recovery between the preflight probe and the run:
            # this model was probed unavailable too, but must still be
            # reachable rather than dropped from the chain.
            return self._success_completed(_ndjson_output("recovered"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke_agent("research-spy", "prompt")

        assert attempted == [
            "opencode-go/deepseek-v4-pro",
            "openrouter/deepseek/deepseek-v4-pro",
        ]
        assert response == "recovered"

    def test_missing_preflight_file_falls_back_to_configured_order(self, tmp_path: Path) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
            preflight=PreflightConfig(
                enabled=True, file_path=str(tmp_path / "nope.json"), select_by="latency"
            ),
        )
        client = OpencodeLLMClient(cfg)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ) as mock_run,
        ):
            client.invoke("prompt")
        args = mock_run.call_args_list[0].args[0]
        assert args[args.index("-m") + 1] == "opencode-go/deepseek-v4-pro"

    def test_preflight_disabled_skips_reordering(self, tmp_path: Path) -> None:
        # File says the fallback is faster, but preflight is off -- must
        # still pin the configured primary.
        file_path = self._write_preflight(
            tmp_path,
            {
                "opencode-go/deepseek-v4-pro": {
                    "available": True,
                    "latency_ms": 900,
                    "error": None,
                },
                "openrouter/deepseek/deepseek-v4-pro": {
                    "available": True,
                    "latency_ms": 50,
                    "error": None,
                },
            },
        )
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
            preflight=PreflightConfig(enabled=False, file_path=file_path, select_by="latency"),
        )
        client = OpencodeLLMClient(cfg)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ) as mock_run,
        ):
            client.invoke("prompt")
        args = mock_run.call_args_list[0].args[0]
        assert args[args.index("-m") + 1] == "opencode-go/deepseek-v4-pro"

    def test_preflight_file_read_once_not_per_call(self, tmp_path: Path) -> None:
        file_path = self._write_preflight(
            tmp_path,
            {
                "opencode-go/deepseek-v4-pro": {"available": True, "latency_ms": 10, "error": None},
                "openrouter/deepseek/deepseek-v4-pro": {
                    "available": True,
                    "latency_ms": 5,
                    "error": None,
                },
            },
        )
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
            preflight=PreflightConfig(enabled=True, file_path=file_path, select_by="latency"),
        )
        client = OpencodeLLMClient(cfg)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client._load_preflight_data", wraps=_load_preflight_data) as spy,
            patch(
                "src.llm.client.subprocess.run",
                return_value=self._success_completed(_ndjson_output("ok")),
            ),
        ):
            client.invoke("first call")
            client.invoke_agent("research-spy", "second call")

        # Resolved once and cached -- invoke() and invoke_agent() must
        # share the same resolution rather than each re-reading the file.
        assert spy.call_count == 1


class TestGetKnownModelIds:
    """`opencode models` is a few seconds, so results are cached to disk --
    see `get_known_model_ids`. These tests never touch the real cache
    file path; each uses its own `tmp_path`."""

    def test_fetches_and_caches_on_first_call(self, tmp_path: Path) -> None:
        cache_path = str(tmp_path / "models-cache.json")
        with patch(
            "src.llm.client.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode", "models"], returncode=0, stdout="a/b\nc/d\n", stderr=""
            ),
        ) as mock_run:
            ids = get_known_model_ids("opencode", cache_path=cache_path)
        assert ids == {"a/b", "c/d"}
        assert mock_run.call_count == 1
        assert json.loads(Path(cache_path).read_text())["models"] == ["a/b", "c/d"]

    def test_fresh_cache_skips_live_fetch(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "models-cache.json"
        cache_path.write_text(
            json.dumps({"timestamp": datetime.now(UTC).isoformat(), "models": ["x/y"]})
        )
        with patch("src.llm.client.subprocess.run") as mock_run:
            ids = get_known_model_ids("opencode", cache_path=str(cache_path))
        assert ids == {"x/y"}
        mock_run.assert_not_called()

    def test_stale_cache_triggers_refresh(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "models-cache.json"
        old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        cache_path.write_text(json.dumps({"timestamp": old_ts, "models": ["stale/model"]}))
        with patch(
            "src.llm.client.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode", "models"], returncode=0, stdout="fresh/model\n", stderr=""
            ),
        ):
            ids = get_known_model_ids("opencode", cache_path=str(cache_path), max_age_sec=3600)
        assert ids == {"fresh/model"}

    def test_failed_refresh_falls_back_to_stale_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "models-cache.json"
        old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        cache_path.write_text(json.dumps({"timestamp": old_ts, "models": ["stale/model"]}))
        with patch("src.llm.client.subprocess.run", side_effect=OSError("no binary")):
            ids = get_known_model_ids("opencode", cache_path=str(cache_path), max_age_sec=3600)
        assert ids == {"stale/model"}

    def test_no_cache_and_failed_fetch_returns_none(self, tmp_path: Path) -> None:
        cache_path = str(tmp_path / "nope.json")
        with patch("src.llm.client.subprocess.run", side_effect=OSError("no binary")):
            assert get_known_model_ids("opencode", cache_path=cache_path) is None

    def test_nonzero_exit_returns_none(self, tmp_path: Path) -> None:
        cache_path = str(tmp_path / "nope.json")
        with patch(
            "src.llm.client.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode", "models"], returncode=1, stdout="", stderr="boom"
            ),
        ):
            assert get_known_model_ids("opencode", cache_path=cache_path) is None

    def test_empty_output_returns_none(self, tmp_path: Path) -> None:
        cache_path = str(tmp_path / "nope.json")
        with patch(
            "src.llm.client.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode", "models"], returncode=0, stdout="", stderr=""
            ),
        ):
            assert get_known_model_ids("opencode", cache_path=cache_path) is None


class TestDropUnknownModels:
    def test_no_validation_data_returns_chain_unchanged(self) -> None:
        chain = ["a/b", "c/d"]
        assert drop_unknown_models(chain, None) == chain

    def test_drops_unknown_and_logs(self) -> None:
        chain = ["a/b", "c/d", "e/f"]
        known = {"a/b", "e/f"}
        assert drop_unknown_models(chain, known) == ["a/b", "e/f"]

    def test_all_unknown_fails_open_to_original_chain(self) -> None:
        chain = ["a/b", "c/d"]
        known = {"z/z"}
        assert drop_unknown_models(chain, known) == chain

    def test_all_known_returns_unchanged(self) -> None:
        chain = ["a/b", "c/d"]
        assert drop_unknown_models(chain, set(chain)) == chain


class TestValidateLlmConfig:
    def test_disabled_config_returned_unchanged(self) -> None:
        cfg = LLMConfig(enabled=False, primary_model="a/b", fallback_models=["c/d"])
        with patch("src.llm.client.get_known_model_ids") as mock_known:
            result = validate_llm_config(cfg)
        assert result is cfg
        mock_known.assert_not_called()

    def test_unknown_fallback_dropped(self) -> None:
        cfg = LLMConfig(
            primary_model="opencode/muse-spark-1.3-contributor-free",
            fallback_models=["nvidia-direct/typo-model", "openrouter/deepseek/deepseek-v4-pro"],
        )
        with patch(
            "src.llm.client.get_known_model_ids",
            return_value={
                "opencode/muse-spark-1.3-contributor-free",
                "openrouter/deepseek/deepseek-v4-pro",
            },
        ):
            result = validate_llm_config(cfg)
        assert result.primary_model == "opencode/muse-spark-1.3-contributor-free"
        assert result.fallback_models == ["openrouter/deepseek/deepseek-v4-pro"]

    def test_all_known_returns_original_config(self) -> None:
        cfg = LLMConfig(primary_model="a/b", fallback_models=["c/d"])
        with patch("src.llm.client.get_known_model_ids", return_value={"a/b", "c/d"}):
            result = validate_llm_config(cfg)
        assert result is cfg

    def test_validation_unavailable_returns_original_config(self) -> None:
        cfg = LLMConfig(primary_model="a/b", fallback_models=["c/d"])
        with patch("src.llm.client.get_known_model_ids", return_value=None):
            result = validate_llm_config(cfg)
        assert result is cfg


class TestOpencodeFailureLogging:
    """The failure reason lives in stdout, not stderr -- opencode writes
    its actual error there. Both `invoke` and `invoke_agent` must log the
    rc!=0 path at WARNING (not DEBUG, which is easy to miss in
    production) and fall back to the tail of stdout when stderr is
    empty."""

    def test_run_failed_falls_back_to_stdout_tail_and_logs_warning(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            primary_model="opencode/muse-spark-1.3-contributor-free",
            fallback_models=[],
        )
        client = OpencodeLLMClient(cfg)
        completed = subprocess.CompletedProcess(
            args=["opencode"], returncode=1, stdout="some error text on stdout", stderr=""
        )
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", return_value=completed),
            structlog.testing.capture_logs() as logs,
        ):
            response = client.invoke("prompt")
        assert response is None
        failed = next(e for e in logs if e["event"] == "opencode_run_failed")
        assert failed["log_level"] == "warning"
        assert failed["error"] == "some error text on stdout"

    def test_run_failed_prefers_stderr_when_present(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            primary_model="opencode/muse-spark-1.3-contributor-free",
            fallback_models=[],
        )
        client = OpencodeLLMClient(cfg)
        completed = subprocess.CompletedProcess(
            args=["opencode"], returncode=1, stdout="stdout noise", stderr="real stderr error"
        )
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", return_value=completed) as mock_run,
        ):
            client.invoke("prompt")
        assert mock_run.called

    def test_served_by_non_primary_warning_on_fallback_success(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode-go/deepseek-v4-pro" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "primary down", "")
            return subprocess.CompletedProcess(cmd, 0, _ndjson_output("fallback ok"), "")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
            structlog.testing.capture_logs() as logs,
        ):
            response = client.invoke("prompt")
        assert response == "fallback ok"
        served = next(e for e in logs if e["event"] == "opencode_served_by_non_primary")
        assert served["log_level"] == "warning"
        assert served["model"] == "openrouter/deepseek/deepseek-v4-pro"
        assert served["primary"] == "opencode-go/deepseek-v4-pro"

    def test_no_warning_when_primary_succeeds(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
        )
        client = OpencodeLLMClient(cfg)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=subprocess.CompletedProcess(["opencode"], 0, _ndjson_output("ok"), ""),
            ),
            structlog.testing.capture_logs() as logs,
        ):
            client.invoke("prompt")
        assert not any(e["event"] == "opencode_served_by_non_primary" for e in logs)

    def test_timeout_logs_elapsed_at_warning(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            timeout_sec=1,
            primary_model="opencode/muse-spark-1.3-contributor-free",
            fallback_models=[],
        )
        client = OpencodeLLMClient(cfg)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["opencode"], timeout=1),
            ),
            structlog.testing.capture_logs() as logs,
        ):
            client.invoke("prompt")
        timed_out = next(e for e in logs if e["event"] == "opencode_run_timed_out")
        assert timed_out["log_level"] == "warning"
        assert "elapsed" in timed_out


class TestUsageSummaryWarnings:
    def test_html_warns_when_served_by_fallback(self) -> None:
        cfg = LLMConfig(enabled=True, primary_model="a/b", fallback_models=["c/d"])
        client = OpencodeLLMClient(cfg)
        client._record_success(
            model="c/d",
            is_fallback=True,
            is_paid=False,
            input_chars=10,
            output_chars=10,
            elapsed=1.0,
        )
        html = client.get_usage_summary_html()
        assert "Primary model did not respond" in html

    def test_html_warns_when_everything_failed(self) -> None:
        cfg = LLMConfig(enabled=True, primary_model="a/b", fallback_models=["c/d"])
        client = OpencodeLLMClient(cfg)
        client._record_failure()
        client._record_all_failed("boom")
        html = client.get_usage_summary_html()
        assert "Every model in the LLM chain failed" in html

    def test_html_has_no_warning_when_primary_succeeds(self) -> None:
        cfg = LLMConfig(enabled=True, primary_model="a/b", fallback_models=["c/d"])
        client = OpencodeLLMClient(cfg)
        client._record_success(
            model="a/b",
            is_fallback=False,
            is_paid=False,
            input_chars=10,
            output_chars=10,
            elapsed=1.0,
        )
        html = client.get_usage_summary_html()
        assert "⚠" not in html

    def test_text_summary_warns_when_served_by_fallback(self) -> None:
        cfg = LLMConfig(enabled=True, primary_model="a/b", fallback_models=["c/d"])
        client = OpencodeLLMClient(cfg)
        client._record_success(
            model="c/d",
            is_fallback=True,
            is_paid=False,
            input_chars=10,
            output_chars=10,
            elapsed=1.0,
        )
        text = client.get_usage_summary_text()
        assert "WARNING: primary model did not respond" in text


if __name__ == "__main__":
    pytest.main([__file__])
