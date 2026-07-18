"""News article parsing and extraction utilities."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from src.models.briefing import (
    BlogItem,
    BriefingData,
    BriefingQuality,
    NewsItem,
    PaperItem,
    TickerRow,
)

# Literal deterministic fallback string emitted by
# atlas-morning-briefing's generate_markdown_briefing() when the LLM
# layer is skipped upstream. See LESSONS_LEARNED.md (2026-07-18 incident).
DEGRADED_SUMMARY_PREFIX = "Synthesis unavailable for today's briefing"


def parse_briefing_date(filename: str) -> date | None:
    """Extract the briefing date from a filename containing YYYY.MM.DD.

    Args:
        filename: The file name to search for a date pattern.

    Returns:
        The parsed date, or None if no pattern is found.
    """
    match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", filename)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return date(year, month, day)
    return None


def find_todays_briefing(
    directory: str | Path,
    briefings_subdir: str = "briefings",
) -> Path | None:
    """Find today's Atlas briefing markdown file, falling back to the most recent.

    Looks in ``<directory>/<briefings_subdir>`` first (current upstream
    layout) and falls back to ``<directory>`` itself if the subdir is
    missing or contains no ``Atlas-Briefing-*.md`` files. This handles
    both the modern layout (new briefings under ``briefings/``, stale
    ones left in root) and the legacy layout (all briefings in root).

    Within the resolved search directory, today's exact filename is
    preferred; otherwise the most recent file is returned.

    Args:
        directory: Path to the atlas-morning-briefing project root.
        briefings_subdir: Sub-directory containing briefing markdown
            files. Pass an empty string to search the root directly.

    Returns:
        Path to the briefing file, or None if none are found.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return None

    candidates: list[Path] = []
    if briefings_subdir:
        subdir = root / briefings_subdir
        if subdir.is_dir():
            candidates.append(subdir)
    candidates.append(root)

    today = date.today()
    expected_name = f"Atlas-Briefing-{today.year}.{today.month:02d}.{today.day:02d}.md"

    for search_dir in candidates:
        today_candidate = search_dir / expected_name
        if today_candidate.is_file():
            return today_candidate
        md_files = sorted(search_dir.glob("Atlas-Briefing-*.md"), reverse=True)
        if md_files:
            # Skip .epub (glob above is .md-only already); return newest.
            return md_files[0]
    return None


def _extract_ticker_section(text: str) -> str:
    """Extract the Financial Market Overview / Stock Watchlist section from markdown.

    Args:
        text: Full markdown content of the briefing.

    Returns:
        The raw text of the ticker section, or empty string.
    """
    match = re.search(
        r"(?:##\s*Financial Market Overview|##\s*Stock Watchlist)\s*\n(.*?)(?=\n##\s|\Z)",
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_executive_summary(text: str) -> str:
    """Extract the Executive Summary or Today's Key Connections section.

    Args:
        text: Full markdown content of the briefing.

    Returns:
        The raw text of the executive summary, or empty string.
    """
    match = re.search(
        r"(?:##\s*Executive Summary|##\s*Today's Key Connections)\s*\n(.*?)(?=\n##\s|\Z)",
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_tickers(ticker_section: str) -> list[TickerRow]:
    """Parse individual ticker rows from the ticker section text.

    Handles both pipe-table markdown format and plain bold-marker format.

    Args:
        ticker_section: Raw text of the ticker section.

    Returns:
        List of TickerRow objects parsed from the section.
    """
    tickers: list[TickerRow] = []
    lines = ticker_section.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("|"):
            if i < 2 or line.startswith("|---"):
                continue
            cells = [c.strip().strip("*") for c in line.split("|") if c.strip()]
            if not cells:
                continue
            symbol = cells[0]
            price = 0.0
            change_pct = 0.0
            driver = ""
            for cell in cells[1:]:
                price_match = re.search(r"\$?([0-9,]+\.\d{2})", cell)
                if price_match and price == 0.0:
                    price = float(price_match.group(1).replace(",", ""))
                pct_match = re.search(r"([+-]?\d+\.?\d*)%", cell)
                if pct_match and change_pct == 0.0:
                    change_pct = float(pct_match.group(1))
            driver_cells = [
                c
                for c in cells
                if not re.search(r"\$?[0-9,]+\.\d{2}", c) and not re.search(r"[+-]?\d+\.?\d*%", c)
            ]
            if driver_cells:
                driver = driver_cells[-1].strip() if len(driver_cells) > 1 else ""
            tickers.append(
                TickerRow(symbol=symbol, price=price, change_pct=change_pct, likely_driver=driver)
            )
            continue
        bold_match = re.match(r"\*\*(\w+)\*\*", line)
        if bold_match:
            symbol = bold_match.group(1)
            price_match = re.search(r"\$?([0-9,]+\.\d{2})", line)
            pct_match = re.search(r"([+-]?\d+\.?\d*)%", line)
            driver_match = re.search(
                r"(?:(?:Driver|driver)[:\s]+|Likely driver:\s*)(.+?)(?:\.\s|\.$|$)",
                line,
                re.IGNORECASE,
            )
            price = float(price_match.group(1).replace(",", "")) if price_match else 0.0
            change_pct = float(pct_match.group(1)) if pct_match else 0.0
            driver = driver_match.group(1).strip() if driver_match else ""
            tickers.append(
                TickerRow(symbol=symbol, price=price, change_pct=change_pct, likely_driver=driver)
            )
    return tickers


def _extract_news(text: str) -> list[NewsItem]:
    """Extract AI & Tech News / News items from the briefing markdown.

    Args:
        text: Full markdown content of the briefing.

    Returns:
        List of NewsItem objects.
    """
    items: list[NewsItem] = []
    section = _extract_section(text, "AI & Tech News")
    if not section:
        section = _extract_section(text, "News")
    if not section:
        return items
    lines = section.split("\n")
    current_title = ""
    current_source = ""
    current_url = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        url_match = re.search(r"\[Read more\]\((.+?)\)", line)
        if url_match:
            current_url = url_match.group(1)
            if current_title and current_source:
                items.append(NewsItem(title=current_title, source=current_source, url=current_url))
                current_title = ""
                current_source = ""
                current_url = ""
            continue
        source_match = re.match(r"\*Source:\s*(.+?)\*", line)
        if source_match:
            current_source = source_match.group(1).strip()
            continue
        if line.startswith("### ") or (line.startswith("**") and not line.startswith("***")):
            current_title = line.replace("### ", "").strip().strip("*").strip()
    return items


def _extract_blogs(text: str) -> list[BlogItem]:
    """Extract Blog Updates section from the briefing markdown.

    Args:
        text: Full markdown content of the briefing.

    Returns:
        List of BlogItem objects.
    """
    items: list[BlogItem] = []
    section = _extract_section(text, "Blog Updates")
    if not section:
        return items
    lines = section.split("\n")
    current_title = ""
    current_author = ""
    current_summary = ""
    current_rating = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("### ") or (
            line.startswith("**") and not line.startswith("***") and "*(" not in line
        ):
            if current_title and current_author:
                items.append(
                    BlogItem(
                        title=current_title,
                        author=current_author,
                        summary=current_summary,
                        rating=current_rating,
                    )
                )
            current_title = line.replace("### ", "").strip().strip("*").strip()
            current_author = ""
            current_summary = ""
            current_rating = 0
            continue
        author_match = re.match(r"\*([A-Za-z0-9\s&.]+)\*", line)
        if author_match and not current_author:
            current_author = author_match.group(1).strip()
            continue
        star_match = re.search(r"([★☆]+)", line)
        if star_match:
            current_rating = str(star_match.group(1)).count("★")
            continue
        url_match = re.search(r"\[Read more\]\((.+?)\)", line)
        if url_match:
            current_summary = current_summary.strip()
            continue
        if len(line) > 40 and not line.startswith("*Source"):
            current_summary = (current_summary + " " + line).strip()
    if current_title and current_author:
        items.append(
            BlogItem(
                title=current_title,
                author=current_author,
                summary=current_summary,
                rating=current_rating,
            )
        )
    return items


def _extract_papers(text: str) -> list[PaperItem]:
    """Extract Top Papers section from the briefing markdown.

    Args:
        text: Full markdown content of the briefing.

    Returns:
        List of PaperItem objects.
    """
    items: list[PaperItem] = []
    section = _extract_section(text, "Top Papers")
    if not section:
        return items
    lines = section.split("\n")
    current_title = ""
    current_authors = ""
    current_score = 0.0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        title_match = re.match(r"###\s+\d+\.\s+(.+?)(?:\s+[★☆]+\s*)?$", line)
        if title_match:
            if current_title:
                items.append(
                    PaperItem(
                        title=current_title,
                        authors=current_authors,
                        reproduction_score=current_score,
                    )
                )
            current_title = title_match.group(1).strip()
            current_authors = ""
            current_score = 0.0
            continue
        author_match = re.match(r"\*\*Authors?\*\*:\s*(.+)", line, re.IGNORECASE)
        if author_match:
            current_authors = author_match.group(1).strip()
            continue
        score_match = re.search(r"\*\*Score\*\*:\s*([0-9.]+)", line)
        if score_match:
            current_score = float(score_match.group(1))
            continue
        repro_match = re.search(r"Repro:\s*[🟡🟢🔴]\s*(\d+)/(\d+)", line)
        if repro_match:
            current_score = float(repro_match.group(1)) / float(repro_match.group(2)) * 10
            continue
    if current_title:
        items.append(
            PaperItem(
                title=current_title, authors=current_authors, reproduction_score=current_score
            )
        )
    return items


def _extract_key_connections(text: str) -> str:
    """Extract the Today's Key Connections section from the briefing.

    Args:
        text: Full markdown content of the briefing.

    Returns:
        The raw text of key connections, or empty string.
    """
    match = re.search(
        r"##\s*Today's Key Connections\s*\n(.*?)(?=\n##\s|\Z)",
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _classify_quality(briefing: BriefingData) -> BriefingQuality:
    """Classify briefing quality from parsed content.

    Mirrors the degradation signatures documented in
    ``LESSONS_LEARNED.md`` (2026-07-18 incident):

    - ``FAILED``: no executive summary AND no news items — the briefing
      is missing or unparsable.
    - ``DEGRADED``: either the executive summary is the literal upstream
      fallback string ("Synthesis unavailable for today's briefing"),
      or blog summaries are absent while news is present (blogs require
      an LLM pass; their absence with news present is a strong signal
      the LLM layer was skipped).
    - ``FULL``: otherwise.

    Args:
        briefing: Parsed BriefingData (quality field is ignored).

    Returns:
        The detected BriefingQuality tier.
    """
    if not briefing.executive_summary and not briefing.news_items:
        return BriefingQuality.FAILED
    if briefing.executive_summary.startswith(DEGRADED_SUMMARY_PREFIX):
        return BriefingQuality.DEGRADED
    if len(briefing.blog_items) == 0 and len(briefing.news_items) > 0:
        return BriefingQuality.DEGRADED
    return BriefingQuality.FULL


def read_briefing(path: str | Path) -> BriefingData:
    """Read a markdown briefing file and parse it into structured data.

    Args:
        path: Path to the briefing markdown file.

    Returns:
        Fully populated BriefingData instance with ``briefing_quality``
        populated by :func:`_classify_quality`.
    """
    path_obj = Path(path).expanduser().resolve()
    markdown = path_obj.read_text(encoding="utf-8")
    briefing_date = parse_briefing_date(path_obj.name) or date.today()
    exec_summary = _extract_executive_summary(markdown)
    key_connections = _extract_key_connections(markdown)
    ticker_section = _extract_ticker_section(markdown)
    tickers = _extract_tickers(ticker_section)
    news = _extract_news(markdown)
    blogs = _extract_blogs(markdown)
    papers = _extract_papers(markdown)
    briefing = BriefingData(
        briefing_date=briefing_date,
        executive_summary=exec_summary,
        key_connections=key_connections,
        tickers=tickers,
        news_items=news,
        blog_items=blogs,
        papers=papers,
        raw_markdown=markdown,
    )
    briefing.briefing_quality = _classify_quality(briefing)
    return briefing


def _extract_section(text: str, section_title: str) -> str:
    """Extract a named level-2 markdown section from the briefing text.

    Args:
        text: Full markdown content.
        section_title: Title of the section to extract.

    Returns:
        Content of the section, or empty string if not found.
    """
    pattern = rf"##\s*{re.escape(section_title)}\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""
