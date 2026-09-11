from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import LLMConfig, PreflightConfig
from src.preflight import build_chain, main, probe_model, run_preflight, write_results


def _ndjson_output(text: str) -> str:
    """Build a synthetic NDJSON stream with one text event, matching the
    ``opencode run --format json`` shape parsed by `_parse_ndjson_response`.
    """
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


class TestProbeModel:
    """`probe_model` must never raise -- a broken probe would take down the
    whole concurrent batch, not just the one model it's checking.
    """

    def test_success_records_availability_and_latency(self) -> None:
        with patch(
            "src.preflight.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode"], returncode=0, stdout=_ndjson_output("a real sentence"), stderr=""
            ),
        ):
            record = probe_model(
                "opencode-go/deepseek-v4-pro", opencode_path="opencode", timeout_sec=45
            )
        assert record["available"] is True
        assert record["error"] is None
        assert record["latency_ms"] >= 0
        assert isinstance(record["latency_ms"], int)

    def test_timeout_marks_unavailable_without_raising(self) -> None:
        with patch(
            "src.preflight.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["opencode"], timeout=45),
        ):
            record = probe_model(
                "opencode-go/deepseek-v4-pro", opencode_path="opencode", timeout_sec=45
            )
        assert record["available"] is False
        assert "timeout" in record["error"].lower()

    def test_nonzero_exit_marks_unavailable(self) -> None:
        with patch(
            "src.preflight.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode"], returncode=1, stdout="", stderr="auth failed"
            ),
        ):
            record = probe_model(
                "opencode-go/deepseek-v4-pro", opencode_path="opencode", timeout_sec=45
            )
        assert record["available"] is False
        assert "auth failed" in record["error"]

    def test_empty_response_marks_unavailable(self) -> None:
        # Exit 0 but no text events -- a node failure, not a success.
        with patch(
            "src.preflight.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode"], returncode=0, stdout="", stderr=""
            ),
        ):
            record = probe_model(
                "opencode-go/deepseek-v4-pro", opencode_path="opencode", timeout_sec=45
            )
        assert record["available"] is False
        assert record["error"] is not None

    def test_unexpected_exception_marks_unavailable_without_raising(self) -> None:
        with patch("src.preflight.subprocess.run", side_effect=OSError("no such binary")):
            record = probe_model(
                "opencode-go/deepseek-v4-pro", opencode_path="opencode", timeout_sec=45
            )
        assert record["available"] is False
        assert "no such binary" in record["error"]


class TestBuildChain:
    def test_builds_from_config_never_a_local_table(self) -> None:
        cfg = LLMConfig(
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["openrouter/deepseek/deepseek-v4-pro"],
        )
        assert build_chain(cfg) == [
            "opencode-go/deepseek-v4-pro",
            "openrouter/deepseek/deepseek-v4-pro",
        ]

    def test_dedupes_while_preserving_order(self) -> None:
        cfg = LLMConfig(
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=["opencode-go/deepseek-v4-pro", "openrouter/deepseek/deepseek-v4-pro"],
        )
        assert build_chain(cfg) == [
            "opencode-go/deepseek-v4-pro",
            "openrouter/deepseek/deepseek-v4-pro",
        ]


class TestRunPreflightAndWriteResults:
    def _cfg(self) -> LLMConfig:
        return LLMConfig(
            primary_model="opencode-go/deepseek-v4-pro",
            fallback_models=[
                "openrouter/deepseek/deepseek-v4-pro",
                "opencode/deepseek-v4-flash-free",
            ],
            preflight=PreflightConfig(probe_timeout_sec=45),
        )

    def test_every_configured_model_gets_a_record(self) -> None:
        cfg = self._cfg()
        with patch(
            "src.preflight.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode"], returncode=0, stdout=_ndjson_output("ok"), stderr=""
            ),
        ):
            results = run_preflight(cfg)
        assert set(results["models"].keys()) == set(build_chain(cfg))
        # Timestamp is real and parseable.
        datetime.fromisoformat(results["timestamp"])

    def test_probes_run_concurrently(self) -> None:
        # Each probe "takes" 0.15s; if run sequentially three of them
        # would take >=0.45s. Run concurrently, wall time should stay
        # well under the sequential sum.
        cfg = self._cfg()

        def slow_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            time.sleep(0.15)
            return subprocess.CompletedProcess(cmd, 0, _ndjson_output("ok"), "")

        with patch("src.preflight.subprocess.run", side_effect=slow_run):
            t0 = time.monotonic()
            run_preflight(cfg)
            elapsed = time.monotonic() - t0
        assert elapsed < 0.4

    def test_write_results_round_trips(self, tmp_path: Path) -> None:
        cfg = self._cfg()
        out_path = tmp_path / ".model-availability.json"
        with patch(
            "src.preflight.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode"], returncode=0, stdout=_ndjson_output("ok"), stderr=""
            ),
        ):
            results = run_preflight(cfg)
        write_results(results, str(out_path))

        reloaded = json.loads(out_path.read_text())
        assert reloaded["timestamp"] == results["timestamp"]
        assert set(reloaded["models"].keys()) == set(build_chain(cfg))
        for model in build_chain(cfg):
            record = reloaded["models"][model]
            assert record["available"] is True
            assert record["error"] is None
            assert isinstance(record["latency_ms"], int)


class TestMain:
    def _write_config(
        self, tmp_path: Path, *, llm_enabled: bool = True, preflight_enabled: bool = True
    ) -> Path:
        preflight_file = tmp_path / ".model-availability.json"
        config = {
            "llm": {
                "enabled": llm_enabled,
                "opencode_path": "opencode",
                "primary_model": "opencode-go/deepseek-v4-pro",
                "fallback_models": ["openrouter/deepseek/deepseek-v4-pro"],
                "preflight": {
                    "enabled": preflight_enabled,
                    "file_path": str(preflight_file),
                    "max_age_sec": 21600,
                    "probe_timeout_sec": 45,
                    "select_by": "latency",
                },
            }
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(json.dumps(config))  # valid YAML subset
        return config_path

    def test_exits_zero_and_writes_file_even_when_every_model_fails(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        with (
            # Model-id validation is exercised separately below; skip it
            # here (`None` = "couldn't determine the roster, don't drop
            # anything") so this test stays about probing, not validation,
            # and never shells out to `opencode models` for real.
            patch("src.preflight.get_known_model_ids", return_value=None),
            patch(
                "src.preflight.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["opencode"], returncode=1, stdout="", stderr="down"
                ),
            ),
        ):
            rc = main(["--config", str(config_path)])

        assert rc == 0
        preflight_file = tmp_path / ".model-availability.json"
        assert preflight_file.exists()
        data = json.loads(preflight_file.read_text())
        assert all(not record["available"] for record in data["models"].values())

    def test_exits_zero_and_reports_partial_success(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)

        def run_side_effect(cmd, **kwargs):  # type: ignore[no-untyped-def]
            if "opencode-go/deepseek-v4-pro" in cmd:
                return subprocess.CompletedProcess(cmd, 0, _ndjson_output("ok"), "")
            return subprocess.CompletedProcess(cmd, 1, "", "down")

        with (
            patch("src.preflight.get_known_model_ids", return_value=None),
            patch("src.preflight.subprocess.run", side_effect=run_side_effect),
        ):
            rc = main(["--config", str(config_path)])

        assert rc == 0
        data = json.loads((tmp_path / ".model-availability.json").read_text())
        assert data["models"]["opencode-go/deepseek-v4-pro"]["available"] is True
        assert data["models"]["openrouter/deepseek/deepseek-v4-pro"]["available"] is False

    def test_drops_unknown_model_id_before_probing(self, tmp_path: Path) -> None:
        """A model id `opencode models` doesn't recognize is dropped and
        logged loudly, and never reaches `probe_model` -- this is the
        fail-fast check that would have caught the 2026-09-05 Nemotron
        incident (a wrong id that failed every call in ~3s, silently)."""
        config_path = self._write_config(tmp_path)

        probed_cmds: list[list[str]] = []

        def run_side_effect(cmd, **kwargs):  # type: ignore[no-untyped-def]
            probed_cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, _ndjson_output("ok"), "")

        with (
            patch(
                "src.preflight.get_known_model_ids",
                return_value={"opencode-go/deepseek-v4-pro"},
            ),
            patch("src.preflight.subprocess.run", side_effect=run_side_effect),
        ):
            rc = main(["--config", str(config_path)])

        assert rc == 0
        data = json.loads((tmp_path / ".model-availability.json").read_text())
        assert "opencode-go/deepseek-v4-pro" in data["models"]
        assert "openrouter/deepseek/deepseek-v4-pro" not in data["models"]
        assert not any(
            "openrouter/deepseek/deepseek-v4-pro" in cmd for cmd in probed_cmds
        )

    def test_skips_and_does_not_write_when_preflight_disabled(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path, preflight_enabled=False)
        with (
            patch("src.preflight.get_known_model_ids") as mock_known,
            patch("src.preflight.subprocess.run") as mock_run,
        ):
            rc = main(["--config", str(config_path)])
        assert rc == 0
        mock_run.assert_not_called()
        mock_known.assert_not_called()
        assert not (tmp_path / ".model-availability.json").exists()

    def test_skips_and_does_not_write_when_llm_disabled(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path, llm_enabled=False)
        with (
            patch("src.preflight.get_known_model_ids") as mock_known,
            patch("src.preflight.subprocess.run") as mock_run,
        ):
            rc = main(["--config", str(config_path)])
        assert rc == 0
        mock_run.assert_not_called()
        mock_known.assert_not_called()
        assert not (tmp_path / ".model-availability.json").exists()


if __name__ == "__main__":
    pytest.main([__file__])
