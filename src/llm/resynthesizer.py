"""Build a re-synthesis prompt from a degraded BriefingData and fold the
LLM response back into the model.

This module is only invoked when :attr:`BriefingData.briefing_quality`
is not :attr:`BriefingQuality.FULL` (i.e. the upstream LLM layer failed
or the briefing is missing). The happy path stays LLM-free.
"""

from __future__ import annotations

import structlog

from src.llm.client import OpencodeLLMClient
from src.models.briefing import BriefingData, BriefingQuality

logger = structlog.get_logger()


SYSTEM_PROMPT = (
    "You are the analytics engine for an intraday 0DTE options trading "
    "system. You are re-synthesising an executive summary because the "
    "upstream Atlas morning briefing's LLM layer failed (its summary was "
    "the deterministic fallback 'Synthesis unavailable' string). Using "
    "only the raw feed items provided below, write a tight 3-5 sentence "
    "executive summary of today's market backdrop: dominant themes, "
    "catalyst keywords (Fed, CPI, earnings, tariffs, etc.), and the "
    "directional lean (bullish/bearish/neutral) with one-line rationale. "
    "Do not invent tickers or numbers that are not in the inputs."
)


def _build_prompt(briefing: BriefingData) -> str:
    """Assemble a re-synthesis prompt from a degraded BriefingData.

    Args:
        briefing: Parsed briefing (will be DEGRADED or FAILED quality).

    Returns:
        A prompt string containing raw news/blog/ticker feed items.
    """
    sections: list[str] = []

    if briefing.executive_summary:
        sections.append(f"Degraded summary (do not trust):\n{briefing.executive_summary}")
    if briefing.key_connections:
        sections.append(f"Key connections raw:\n{briefing.key_connections}")

    if briefing.tickers:
        ticker_lines = [
            f"- {t.symbol}: ${t.price:.2f} ({t.change_pct:+.2f}%) "
            f"{('— ' + t.likely_driver) if t.likely_driver else ''}"
            for t in briefing.tickers
        ]
        sections.append("Tickers:\n" + "\n".join(ticker_lines))

    if briefing.news_items:
        news_lines = [
            f"- [{n.source}] {n.title}{(' ' + n.snippet) if n.snippet else ''}"
            for n in briefing.news_items
        ]
        sections.append("News headlines:\n" + "\n".join(news_lines))

    if briefing.blog_items:
        blog_lines = [
            f"- [{b.author}] {b.title}{(' ' + b.summary) if b.summary else ''}"
            for b in briefing.blog_items
        ]
        sections.append("Blog updates:\n" + "\n".join(blog_lines))

    if not sections:
        return "No feed items available. State that the briefing is unavailable."

    return "\n\n".join(sections)


def resynthesize_briefing(
    briefing: BriefingData,
    client: OpencodeLLMClient,
) -> BriefingData:
    """Re-synthesise a degraded BriefingData's executive summary via an LLM.

    Only fires when ``briefing_quality`` is not ``FULL`` and the LLM
    client is available. On success, overwrites
    :attr:`BriefingData.executive_summary` and promotes
    :attr:`BriefingData.briefing_quality` back to
    :attr:`BriefingQuality.FULL` so downstream strategies can trust the
    sentiment signal again. On failure, leaves the briefing untouched
    (quality stays DEGRADED/FAILED and ``macro_sentiment`` keeps
    returning ``None``).

    Args:
        briefing: Parsed briefing (quality matters; FULL briefings are
            returned unchanged).
        client: An :class:`OpencodeLLMClient` instance.

    Returns:
        The same ``briefing`` instance, mutated in place if
        re-synthesis succeeded.
    """
    if briefing.briefing_quality == BriefingQuality.FULL:
        return briefing

    prompt = _build_prompt(briefing)
    response = client.invoke(prompt=prompt, system_prompt=SYSTEM_PROMPT)

    if not response or not response.strip():
        logger.warning(
            "resynth_failed",
            quality=briefing.briefing_quality.value,
            served_by=client.last_served_by,
            error=client.last_error,
        )
        return briefing

    briefing.executive_summary = response.strip()
    briefing.briefing_quality = BriefingQuality.FULL
    logger.info(
        "resynth_ok",
        served_by=client.last_served_by,
        fallback_hit=client.last_fallback_hit,
        chars=len(response),
    )
    return briefing
