from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import LLMConfig
from src.llm.client import OpencodeLLMClient, _parse_ndjson_response, is_paid_model
from src.llm.resynthesizer import resynthesize_briefing
from src.models.briefing import BriefingData, BriefingQuality


def _ndjson_output(text: str) -> str:
    """Build a synthetic NDJSON stream with one text event."""
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


class TestIsPaidModel:
    def test_go_namespace_is_paid(self) -> None:
        assert is_paid_model("opencode-go/glm-5.2") is True
        assert is_paid_model("opencode-go/kimi-k3") is True

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

    def test_first_zen_model_success(self) -> None:
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
        # First successful model is the first entry in zen_models.
        assert client.last_served_by == "opencode/deepseek-v4-flash-free"
        assert client.last_fallback_hit is False
        assert client.paid_used is False
        assert mock_run.call_args.kwargs["timeout"] == 60
        args = mock_run.call_args.args[0]
        assert "opencode/deepseek-v4-flash-free" in args

    def test_first_zen_timeout_falls_back_to_second_zen(self) -> None:
        # The chain should walk Zen models first before touching paid Go.
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            timeout_sec=5,
            zen_models=[
                "opencode/deepseek-v4-flash-free",
                "opencode/mimo-v2.5-free",
            ],
            paid_go_models=["opencode-go/glm-5.2"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode/deepseek-v4-flash-free" in cmd:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 5))
            if "opencode/mimo-v2.5-free" in cmd:
                return self._success_completed(_ndjson_output("mimo ok"))
            return self._success_completed(_ndjson_output("glm summary"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "mimo ok"
        assert client.last_served_by == "opencode/mimo-v2.5-free"
        assert client.last_fallback_hit is True
        # Second Zen model is still free.
        assert client.paid_used is False

    def test_all_zen_fail_falls_back_to_paid_go(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            zen_models=["opencode/deepseek-v4-flash-free", "opencode/mimo-v2.5-free"],
            paid_go_models=["opencode-go/glm-5.2"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode-go/glm-5.2" in cmd:
                return self._success_completed(_ndjson_output("paid glm ok"))
            # All Zen models fail with non-zero rc.
            return subprocess.CompletedProcess(cmd, 1, "", "zen fail")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "paid glm ok"
        assert client.last_served_by == "opencode-go/glm-5.2"
        assert client.last_fallback_hit is True
        # Serving model is from opencode-go/* so paid tracking must fire.
        assert client.paid_used is True

    def test_paid_model_failure_does_not_mark_paid_used(self) -> None:
        # If a paid model is tried but fails, and a subsequent free
        # Zen model succeeds, paid_used should remain False (the
        # response didn't actually come from a paid model).
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            zen_models=["opencode/deepseek-v4-flash-free"],
            paid_go_models=["opencode-go/glm-5.2", "opencode/mimo-v2.5-free"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode/deepseek-v4-flash-free" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "fail")
            if "opencode-go/glm-5.2" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "fail")
            if "opencode/mimo-v2.5-free" in cmd:
                return self._success_completed(_ndjson_output("zen mimo ok"))

            return self._success_completed(_ndjson_output("unexpected"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "zen mimo ok"
        assert client.last_served_by == "opencode/mimo-v2.5-free"
        assert client.paid_used is False

    def test_all_models_fail(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            zen_models=["opencode/deepseek-v4-flash-free"],
            paid_go_models=["opencode-go/glm-5.2"],
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

    def test_dedup_across_zen_and_paid(self) -> None:
        # A model appearing in both lists must only be tried once
        # (preserving first appearance, i.e. Zen tier wins).
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            zen_models=["opencode/deepseek-v4-flash-free"],
            paid_go_models=["opencode/deepseek-v4-flash-free", "opencode-go/glm-5.2"],
        )
        client = OpencodeLLMClient(cfg)
        call_count = {"n": 0}

        def run_side_effect(cmd, **kwargs):
            call_count["n"] += 1
            if "opencode-go/glm-5.2" in cmd:
                return self._success_completed(_ndjson_output("paid glm ok"))
            # Zen primary fails.
            return subprocess.CompletedProcess(cmd, 1, "", "fail")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "paid glm ok"
        # First call: zen deepseek (fail). Second call: paid glm (ok).
        # The duplicate opencode/deepseek-v4-flash-free in paid_go_models
        # must NOT be retried.
        assert call_count["n"] == 2

    def test_empty_response_counts_as_failure(self) -> None:
        client = OpencodeLLMClient(
            LLMConfig(
                enabled=True,
                opencode_path="opencode",
                zen_models=["opencode/deepseek-v4-flash-free"],
                paid_go_models=["opencode-go/glm-5.2"],
            )
        )

        def run_side_effect(cmd, **kwargs):
            if "opencode/deepseek-v4-flash-free" in cmd:
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

    def test_invoke_agent_adds_agent_flag(self) -> None:
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
        assert "--agent" in args
        assert "research-spy" in args
        assert "--format" in args
        assert "json" in args

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
            zen_models=["opencode/deepseek-v4-flash-free"],
            paid_go_models=["opencode-go/glm-5.2"],
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            if "opencode/deepseek-v4-flash-free" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "fail")
            return self._success_completed(_ndjson_output("paid ok"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke_agent("checker", "validate please")
        assert response == "paid ok"
        assert client.paid_used is True
        assert client.last_served_by == "opencode-go/glm-5.2"
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
            zen_models=["opencode/deepseek-v4-flash-free"],
            paid_go_models=[],
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
            zen_models=["opencode/deepseek-v4-flash-free"],
            paid_go_models=["opencode-go/glm-5.2"],
            max_calls_per_run=50,
        )
        client = OpencodeLLMClient(cfg)

        def run_side_effect(cmd, **kwargs):
            time.sleep(0.005)
            if "opencode-go/glm-5.2" in cmd:
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
            zen_models=["opencode/deepseek-v4-flash-free"],
            paid_go_models=[],
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
            zen_models=["opencode/deepseek-v4-flash-free"],
            paid_go_models=[],
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
        expected_agent_chars = len(agent_file.read_text())
        # Sanity check against the code-review finding (~1.5-3KB agent files).
        assert expected_agent_chars > 1000

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
        assert client.total_input_chars == len(prompt) + expected_agent_chars

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
            zen_models=["opencode/a", "opencode/b"],
            paid_go_models=[],
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
            zen_models=["opencode/a", "opencode/b"],
            paid_go_models=[],
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
            zen_models=["opencode/a", "opencode/b", "opencode/c", "opencode/d"],
            paid_go_models=["opencode-go/x", "opencode-go/y", "opencode-go/z"],
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


if __name__ == "__main__":
    pytest.main([__file__])
