import pytest

from src.ingestion.fetcher import _estimate_polarity


class TestEstimatePolarity:
    def test_positive_text(self) -> None:
        score = _estimate_polarity("Stocks surge higher on strong growth")
        assert score > 0

    def test_negative_text(self) -> None:
        score = _estimate_polarity("Markets decline as selloff deepens")
        assert score < 0

    def test_neutral_text(self) -> None:
        score = _estimate_polarity("The meeting is scheduled for Tuesday")
        assert score == 0.0


class TestFinnhubFetcher:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_none(self) -> None:
        from src.ingestion.fetcher import FinnhubFetcher
        fetcher = FinnhubFetcher(api_key="")
        result = await fetcher.fetch_quote("SPY")
        assert result is None


class TestBraveFetcher:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self) -> None:
        from src.ingestion.fetcher import BraveFetcher
        fetcher = BraveFetcher(api_key="")
        results = await fetcher.fetch_news()
        assert results == []


class TestRSSFetcher:
    @pytest.mark.asyncio
    async def test_empty_urls_returns_empty(self) -> None:
        from src.ingestion.fetcher import RSSFetcher
        fetcher = RSSFetcher(feed_urls=[])
        results = await fetcher.fetch_all()
        assert results == []
