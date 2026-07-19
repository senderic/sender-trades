"""Local LLM fallback for degraded briefings.

Shells out to the ``opencode`` CLI (the same binary used by the
upstream atlas-morning-briefing pipeline) to re-synthesise an
executive summary from raw feed items when the upstream LLM layer
failed. The primary model is tried first under a strict timeout; on
timeout or failure the fallback chain is walked in order.

This mirrors the upstream ``scripts/opencode_client.py`` invocation
pattern (``opencode run -m <model> --format json --auto --dir /tmp
--pure <prompt>``) and NDJSON stream parser, but is intentionally
smaller Ã¢ no tier accounting, no cost tracking, just the fallback
behaviour needed by this project.
"""

from src.llm.client import OpencodeLLMClient
from src.llm.resynthesizer import resynthesize_briefing
from src.llm.trade_signal import LLMTradeStrategy

__all__ = ["LLMTradeStrategy", "OpencodeLLMClient", "resynthesize_briefing"]
