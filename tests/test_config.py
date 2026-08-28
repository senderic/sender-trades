from pathlib import Path
from typing import ClassVar

from src.config import Settings


class TestSettings:
    def test_defaults_loaded(self) -> None:
        settings = Settings()
        assert settings.general.env_mode == "PAPER_ALPACA"
        assert "SPY" in settings.general.target_assets
        assert "QQQ" in settings.general.target_assets
        assert settings.general.execute is False

    def test_from_yaml_creates_settings(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text("""
general:
  env_mode: PAPER_ALPACA
  target_assets:
    - SPY
    - QQQ
  execute: false
risk:
  max_loss_per_trade_usd: 250.0
""")
        settings = Settings.from_yaml(yaml_path)
        assert settings.risk.max_loss_per_trade_usd == 250.0
        assert settings.general.env_mode == "PAPER_ALPACA"

    def test_from_yaml_missing_file_returns_default(self) -> None:
        settings = Settings.from_yaml("/nonexistent/config.yaml")
        assert isinstance(settings, Settings)

    def test_resolve_env_vars(self) -> None:
        settings = Settings()
        resolved = settings.resolve_env_vars()
        assert isinstance(resolved, Settings)


class TestGraphConfig:
    def test_defaults(self) -> None:
        gc = Settings().graph
        assert gc.enabled is False
        assert gc.fallback_to_monolithic is True
        assert gc.checker_contradiction_action == "veto"
        assert gc.checker_confidence_penalty == 0.15
        assert gc.research_timeout_sec == 45
        assert gc.prediction_timeout_sec == 30
        assert gc.checker_timeout_sec == 30
        assert gc.pick_trade_timeout_sec == 30
        assert gc.total_deadline_sec == 360
        assert gc.reserved_calls_for_fallback == 1
        assert gc.unsupported_confidence_cap == 0.35

    def test_deadline_leaves_room_in_entry_window(self) -> None:
        """The graph must not be able to eat the whole entry window.

        The pipeline fires ~2 min before the open and the entry window is
        ``execution.entry.entry_window_minutes`` long, so the graph's
        wall-clock ceiling has to leave time to actually place the order.
        """
        settings = Settings.from_yaml("config.yaml")
        window_sec = settings.execution.entry.entry_window_minutes * 60
        assert settings.graph.total_deadline_sec < window_sec

    def test_reserve_is_affordable(self) -> None:
        """The fallback reserve must fit inside the per-run call budget."""
        settings = Settings.from_yaml("config.yaml")
        assert 0 < settings.graph.reserved_calls_for_fallback < settings.llm.max_calls_per_run

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "graph_config.yaml"
        yaml_path.write_text("""
graph:
  enabled: true
  fallback_to_monolithic: true
  checker_contradiction_action: penalize
  checker_confidence_penalty: 0.25
""")
        settings = Settings.from_yaml(yaml_path)
        assert settings.graph.enabled is True
        assert settings.graph.checker_contradiction_action == "penalize"
        assert settings.graph.checker_confidence_penalty == 0.25


class TestObservedLatencyHeadroom:
    """Node timeouts must clear the worst latency actually observed.

    Sized from cron logs 2026-08-14..28. The original values gave the two
    SLOWEST nodes the least headroom — research 60s vs 58.0s observed,
    checker 45s vs 42.0s observed — which is why the checker node timed
    out on 5 of 11 runs and dropped the graph into its fallback path.
    These assertions exist so a future timeout edit cannot silently
    recreate that failure.
    """

    # p100 latency per node, in seconds, from opencode_agent_ok events.
    OBSERVED_PEAK: ClassVar[dict[str, float]] = {
        "research": 58.0,
        "prediction": 22.6,
        "checker": 42.1,
        "pick_trade": 28.8,
    }
    MIN_HEADROOM_SEC = 15.0

    def test_every_node_timeout_clears_observed_peak(self) -> None:
        gc = Settings.from_yaml("config.yaml").graph
        for node, peak in self.OBSERVED_PEAK.items():
            configured = getattr(gc, f"{node}_timeout_sec")
            assert configured - peak >= self.MIN_HEADROOM_SEC, (
                f"{node}: {configured}s leaves only {configured - peak:.1f}s over the "
                f"observed {peak}s peak (need >= {self.MIN_HEADROOM_SEC}s)"
            )

    def test_monolithic_fallback_timeout_clears_graph_node_peaks(self) -> None:
        """The fallback prompt is at least as heavy as any single node, so
        its timeout must not be tighter than theirs. On 2026-08-26 it was
        45s, timed out on both models, and left the run with no LLM output
        at all."""
        settings = Settings.from_yaml("config.yaml")
        assert settings.llm.timeout_sec >= max(self.OBSERVED_PEAK.values()) + self.MIN_HEADROOM_SEC

    def test_typical_full_graph_path_fits_the_deadline(self) -> None:
        """Every node succeeding on its first model must fit the wall-clock
        budget with room for one checker retry."""
        gc = Settings.from_yaml("config.yaml").graph
        typical = (
            self.OBSERVED_PEAK["research"]
            + self.OBSERVED_PEAK["prediction"]
            + self.OBSERVED_PEAK["checker"]
            + self.OBSERVED_PEAK["pick_trade"]
        )
        assert typical + self.OBSERVED_PEAK["checker"] < gc.total_deadline_sec


class TestPreflightConfig:
    def test_defaults_are_off(self) -> None:
        """Preflight is opt-in: absent config must not change behaviour."""
        pf = Settings().llm.preflight
        assert pf.enabled is False
        assert pf.select_by == "latency"
        assert pf.max_age_sec == 6 * 3600

    def test_enabled_in_project_config(self) -> None:
        pf = Settings.from_yaml("config.yaml").llm.preflight
        assert pf.enabled is True
        assert pf.file_path == ".model-availability.json"

    def test_probe_timeout_is_not_longer_than_the_run_it_protects(self) -> None:
        """A probe slower than the real call teaches nothing useful."""
        settings = Settings.from_yaml("config.yaml")
        assert settings.llm.preflight.probe_timeout_sec <= settings.llm.timeout_sec

    def test_staleness_window_covers_the_gap_to_the_run(self) -> None:
        """Preflight fires ~15 min before the pipeline; the max age must
        comfortably span that, but not so long it pins yesterday's winner."""
        pf = Settings.from_yaml("config.yaml").llm.preflight
        assert 15 * 60 < pf.max_age_sec <= 24 * 3600


class TestGapFadeConfig:
    def test_defaults(self) -> None:
        gf = Settings().gap_fade
        assert gf.threshold_for("SPY") == 1.5
        assert gf.threshold_for("QQQ") == 2.0
        assert gf.sentiment_magnitude_max == 0.20

    def test_unknown_asset_falls_back_to_default(self) -> None:
        """An asset with no explicit threshold gets the tighter default."""
        gf = Settings().gap_fade
        assert gf.threshold_for("IWM") == gf.default_threshold_pct

    def test_from_yaml_overrides(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "gap_fade.yaml"
        yaml_path.write_text("""
gap_fade:
  thresholds_pct:
    SPY: 1.0
    QQQ: 1.8
  default_threshold_pct: 1.2
  sentiment_magnitude_max: 0.30
""")
        settings = Settings.from_yaml(yaml_path)
        assert settings.gap_fade.threshold_for("SPY") == 1.0
        assert settings.gap_fade.threshold_for("QQQ") == 1.8
        assert settings.gap_fade.threshold_for("DIA") == 1.2
        assert settings.gap_fade.sentiment_magnitude_max == 0.30

    def test_project_config_matches_documented_thresholds(self) -> None:
        """config.yaml must carry the SPY/QQQ thresholds the prompts assume."""
        settings = Settings.from_yaml("config.yaml")
        assert settings.gap_fade.threshold_for("SPY") == 1.5
        assert settings.gap_fade.threshold_for("QQQ") == 2.0
