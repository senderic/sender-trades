from datetime import date
from pathlib import Path

import pytest

from src.config import Settings
from src.models.briefing import BriefingData, NewsItem, TickerRow
from src.models.market import DataSource, MarketSnapshot, NewsHeadline, Quote
from src.models.recommendation import Direction, PositionIntent, TradeRecommendation


@pytest.fixture
def sample_briefing_markdown() -> str:
    return """# Atlas Morning Briefing
*Friday, July 17, 2026 | 06:50 AM PDT*

## Executive Summary
Today's briefing highlights a surge in AI-related stocks following positive earnings guidance from major tech firms. Market sentiment is bullish with strong momentum in semiconductor and cloud computing sectors.

## Financial Market Overview

| Ticker | Price | Change | Driver |
|--------|-------|--------|--------|
| **SPY** | $745.20 | +0.85% | Broad market strength |
| **QQQ** | $715.30 | +1.20% | Tech sector rally |
| **NVDA** | $235.40 | +2.10% | AI earnings optimism |

## AI & Tech News

### US Tech Stocks Surge on AI Demand
*Source: reuters.com*

[Read more](https://example.com/reuters-tech)

### Fed Signals Cautious Approach to Rate Cuts
*Source: bloomberg.com*

[Read more](https://example.com/bloomberg-fed)

## Blog Updates

### Understanding 0DTE Options Dynamics
*CBOE Blog*

A deep dive into 0DTE trading patterns and liquidity...

[Read more](https://example.com/cboe-0dte)

## Top Papers

### 1. Machine Learning for Options Pricing
**Authors**: Smith, Jones

Novel approach to options pricing using deep learning...

**Score**: 8.5 | **Difficulty**: M

[ArXiv](http://arxiv.org/abs/2403.00001)
"""


@pytest.fixture
def sample_briefing_data() -> BriefingData:
    return BriefingData(
        briefing_date=date(2026, 7, 17),
        executive_summary="Market sentiment is bullish following positive earnings guidance.",
        key_connections="Tech sector strength driven by AI demand.",
        tickers=[
            TickerRow(
                symbol="SPY", price=745.20, change_pct=0.85, likely_driver="Broad market strength"
            ),
            TickerRow(
                symbol="QQQ", price=715.30, change_pct=1.20, likely_driver="Tech sector rally"
            ),
        ],
        news_items=[
            NewsItem(
                title="US Tech Stocks Surge on AI Demand",
                source="reuters.com",
                url="https://example.com/reuters-tech",
            ),
        ],
    )


@pytest.fixture
def sample_market_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        quotes={
            "SPY": Quote(
                symbol="SPY",
                current_price=745.20,
                open_price=742.00,
                high_price=746.50,
                low_price=741.00,
                previous_close=739.00,
                change_pct=0.84,
                volume=45000000,
                source=DataSource.FINNHUB,
                timestamp=__import__("datetime").datetime.now(),
            ),
            "QQQ": Quote(
                symbol="QQQ",
                current_price=715.30,
                open_price=708.00,
                high_price=716.80,
                low_price=707.00,
                previous_close=708.00,
                change_pct=1.03,
                volume=32000000,
                source=DataSource.FINNHUB,
                timestamp=__import__("datetime").datetime.now(),
            ),
        },
        news=[
            NewsHeadline(
                title="Tech Stocks Rally on AI Optimism",
                source="reuters.com",
                url="https://example.com",
                snippet="Tech stocks up",
                polarity=0.6,
            ),
            NewsHeadline(
                title="Fed Rate Decision Looms",
                source="bloomberg.com",
                url="https://example.com/fed",
                snippet="Fed cautious",
                polarity=-0.2,
            ),
        ],
        rss_items=[],
    )


@pytest.fixture
def sample_trade_recommendation() -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="test-123",
        strategy_label="momentum",
        asset="SPY",
        direction=Direction.CALL,
        confidence=0.75,
        target_strike=746.0,
        contracts=1,
        order_type="market",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={"gap_pct": 0.41, "sentiment": 0.6, "strategy": "momentum"},
        expires_at="2026-07-17",
        must_close_before="15:30",
    )


@pytest.fixture
def sample_config() -> Settings:
    return Settings()


@pytest.fixture
def tmp_briefing_file(tmp_path: Path, sample_briefing_markdown: str) -> Path:
    file_path = tmp_path / "Atlas-Briefing-2026.07.17.md"
    file_path.write_text(sample_briefing_markdown)
    return file_path
