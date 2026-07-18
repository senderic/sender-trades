from __future__ import annotations

import json
import subprocess
from datetime import date
from unittest.mock import patch

import pytest

from src.config import LLMConfig
from src.llm.client import OpencodeLLMClient, _parse_ndjson_response
from src.llm.resynthesizer import resynthesize_briefing
from src.models.briefing import BriefingData, BriefingQuality


def _ndjson_output(text: str) -> str:
    """Build a synthetic NDJSON stream with one text event."""
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


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

    def test_primary_success(self) -> None:
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
        assert client.last_served_by == "opencode/deepseek-v4-flash-free"
        assert client.last_fallback_hit is False
        # Verify the invocation used the primary model id.
        assert mock_run.call_args.kwargs["timeout"] == 60
        args = mock_run.call_args.args[0]
        assert "opencode/deepseek-v4-flash-free" in args

    def test_primary_timeout_falls_back_to_glm(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode", timeout_sec=5))

        def run_side_effect(cmd, **kwargs):
            # First call (primary) times out; second call (glm) succeeds.
            if "opencode/deepseek-v4-flash-free" in cmd:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 5))
            return self._success_completed(_ndjson_output("glm summary"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "glm summary"
        assert client.last_served_by == "opencode-go/glm-5.2"
        assert client.last_fallback_hit is True
        assert "timeout after 5s" not in client.last_error

    def test_primary_nonzero_rc_falls_back(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))

        def run_side_effect(cmd, **kwargs):
            if "opencode/deepseek-v4-flash-free" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "boom")
            return self._success_completed(_ndjson_output("glm ok"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            response = client.invoke("prompt")
        assert response == "glm ok"
        assert client.last_fallback_hit is True

    def test_all_models_fail(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            opencode_path="opencode",
            primary_model="opencode/deepseek-v4-flash-free",
            fallback_models=["opencode-go/glm-5.2"],
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
        assert client.last_error != ""

    def test_empty_response_counts_as_failure(self) -> None:
        client = OpencodeLLMClient(LLMConfig(enabled=True, opencode_path="opencode"))

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


if __name__ == "__main__":
    pytest.main([__file__])
