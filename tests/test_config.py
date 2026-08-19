from pathlib import Path

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
        assert gc.total_deadline_sec == 240
        assert gc.reserved_calls_for_fallback == 1

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
