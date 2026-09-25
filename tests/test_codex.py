"""Tests for the Codex CLI backend (``codex/<model>`` ids in the LLM chain)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from src.config import CodexConfig, LLMConfig
from src.llm.client import OpencodeLLMClient, drop_unknown_models
from src.llm.codex import build_command, codex_model_name, is_codex_model, parse_jsonl
from src.preflight import probe_model


def _stream(text: str | None = "hello", *, completed: bool = True, extra: list | None = None):
    events = [{"type": "thread.started"}, {"type": "turn.started"}]
    # A CLI notice seen in real output; must not count as a failure.
    events.append({"type": "item.completed", "item": {"type": "error", "message": "skills budget"}})
    if text is not None:
        events.append({"type": "item.completed", "item": {"type": "agent_message", "text": text}})
    events.extend(extra or [])
    if completed:
        events.append({"type": "turn.completed", "usage": {"input_tokens": 10}})
    return "\n".join(json.dumps(e) for e in events)


def _done(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["x"], returncode=rc, stdout=stdout, stderr="")


class TestParsing:
    def test_model_prefix(self) -> None:
        assert is_codex_model("codex/gpt-5.6-sol")
        assert not is_codex_model("opencode/muse-spark-1.3-contributor-free")
        assert codex_model_name("codex/gpt-5.6-sol") == "gpt-5.6-sol"

    def test_command_is_isolated_and_reads_stdin(self) -> None:
        cmd = build_command(CodexConfig(executable="/bin/codex"), "codex/gpt-5.6-sol")
        assert cmd[:2] == ["/bin/codex", "exec"]
        for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--json"):
            assert flag in cmd
        assert cmd[cmd.index("--sandbox") + 1] == "read-only"
        assert cmd[cmd.index("-m") + 1] == "gpt-5.6-sol"
        assert 'model_reasoning_effort="high"' in cmd
        assert cmd[-1] == "-"

    def test_parse_completed_turn(self) -> None:
        message, usage, completed, failed = parse_jsonl(_stream('{"a": 1}'))
        assert message == '{"a": 1}'
        assert usage == {"input_tokens": 10}
        assert completed
        assert not failed

    def test_parse_incomplete_turn(self) -> None:
        _, _, completed, _ = parse_jsonl(_stream(completed=False))
        assert not completed

    def test_parse_failed_turn(self) -> None:
        _, _, _, failed = parse_jsonl(_stream(extra=[{"type": "turn.failed"}]))
        assert failed

    def test_parse_skips_non_json_lines(self) -> None:
        message, _, completed, failed = parse_jsonl("warning: noise\n" + _stream("ok"))
        assert message == "ok"
        assert completed
        assert not failed


class TestClientRouting:
    def _cfg(self, **kw) -> LLMConfig:
        base = {
            "enabled": True,
            "primary_model": "codex/gpt-5.6-sol",
            "fallback_models": ["opencode/muse-spark-1.3-contributor-free"],
        }
        base.update(kw)
        return LLMConfig(**base)

    def test_codex_primary_serves_via_stdin(self) -> None:
        client = OpencodeLLMClient(self._cfg())
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/x"),
            patch("src.llm.client.subprocess.run", return_value=_done(_stream("UP"))) as run,
        ):
            assert client.invoke_agent("predict-spy", "the prompt") == "UP"
        cmd = run.call_args.args[0]
        assert cmd[1] == "exec"
        assert cmd[-1] == "-"
        assert "the prompt" in run.call_args.kwargs["input"]
        assert client.last_served_by == "codex/gpt-5.6-sol"
        assert not client.last_fallback_hit

    def test_codex_failure_falls_back_to_opencode(self) -> None:
        opencode_ok = _done(json.dumps({"type": "text", "part": {"text": "DOWN"}}))

        def side_effect(cmd, **kwargs):
            if cmd[1] == "exec":
                return _done(_stream(completed=False))
            return opencode_ok

        client = OpencodeLLMClient(self._cfg())
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/x"),
            patch("src.llm.client.subprocess.run", side_effect=side_effect),
        ):
            response = client.invoke("prompt", system_prompt="sys")
        assert response is not None
        assert client.last_served_by == "opencode/muse-spark-1.3-contributor-free"
        assert client.last_fallback_hit
        assert client.total_failures == 1

    def test_codex_timeout_falls_back(self) -> None:
        def side_effect(cmd, **kwargs):
            if cmd[1] == "exec":
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return _done(json.dumps({"type": "text", "part": {"text": "FLAT"}}))

        client = OpencodeLLMClient(self._cfg())
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/x"),
            patch("src.llm.client.subprocess.run", side_effect=side_effect),
        ):
            assert client.invoke_agent("checker", "p", timeout_sec=5) is not None
        assert client.last_served_by == "opencode/muse-spark-1.3-contributor-free"

    def test_available_when_only_codex_installed(self) -> None:
        cfg = self._cfg(fallback_models=[], codex=CodexConfig(executable="/opt/codex"))
        with patch(
            "src.llm.client.shutil.which",
            side_effect=lambda b: b if b == "/opt/codex" else None,
        ):
            assert OpencodeLLMClient(cfg).available


def test_drop_unknown_models_keeps_codex_ids() -> None:
    chain = ["codex/gpt-5.6-sol", "opencode/muse", "opencode/typo"]
    assert drop_unknown_models(chain, {"opencode/muse"}) == ["codex/gpt-5.6-sol", "opencode/muse"]


def test_preflight_probes_codex_models() -> None:
    with patch("src.preflight.subprocess.run", return_value=_done(_stream("A 0DTE..."))) as run:
        record = probe_model("codex/gpt-5.6-sol", opencode_path="opencode", timeout_sec=5)
    assert record["available"] is True
    assert run.call_args.args[0][1] == "exec"


def test_preflight_codex_invalid_stream_is_unavailable() -> None:
    with patch("src.preflight.subprocess.run", return_value=_done(_stream(completed=False))):
        record = probe_model("codex/gpt-5.6-sol", opencode_path="opencode", timeout_sec=5)
    assert record["available"] is False
