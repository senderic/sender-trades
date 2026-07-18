from __future__ import annotations

import json
import subprocess
from datetime import date
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


if __name__ == "__main__":
    pytest.main([__file__])
