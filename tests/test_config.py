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
