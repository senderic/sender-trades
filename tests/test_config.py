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
