"""Utilities for tolerating legacy concatenated JSON log files.

Before 2026-09-10, ``src.execution.context.TradeContext`` wrote
``trade-<id>.json`` by appending one raw JSON object per line for the
life of a trade, replacing that with a single valid JSON document only
when the trade reached ``finalize()``. A trade that never finalized
(crash, kill, etc.) was left as several JSON objects concatenated in one
``.json`` file -- something ``json.load``/``json.loads`` cannot parse
(``json.JSONDecodeError: Extra data``).

``TradeContext`` now rewrites the whole file atomically on every event
(see ``src.execution.context.TradeContext._write_json_atomic``), so new
files are always a single JSON document. These helpers exist so readers
still tolerate files written before that fix, via a
``json.JSONDecoder.raw_decode`` loop instead of one ``json.load`` call.
"""

from __future__ import annotations

import json
from typing import Any


def raw_decode_all(text: str) -> list[Any]:
    """Decode every whitespace-separated top-level JSON value in ``text``.

    Returns one value for a normal, well-formed file. Returns more than
    one for a legacy file where several JSON objects were written
    back-to-back with no wrapping array or separator beyond a newline.
    Returns an empty list for blank/whitespace-only input.
    """
    decoder = json.JSONDecoder()
    values: list[Any] = []
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        value, end = decoder.raw_decode(text, idx)
        values.append(value)
        idx = end
    return values


def merge_concatenated_dicts(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold several concatenated per-line JSON objects into one audit dict.

    A single object is returned unchanged. For more than one (the legacy
    append-only shape), scalar fields are last-write-wins across the
    objects, and every object's own fields are collected into an
    ``entries`` list (extending any nested ``entries`` list found along
    the way) -- matching the shape a normal finalized/pending
    ``trade-*.json`` document already has, so downstream code
    (:func:`~src.execution.intraday_monitor.extract_open_trade`,
    :class:`~src.trade_tracker.TradeOutcome`, etc.) doesn't need to know
    which shape it received.

    Args:
        objects: Dicts decoded from one file via :func:`raw_decode_all`.

    Returns:
        A single merged dict, or ``{}`` if ``objects`` is empty.
    """
    if not objects:
        return {}
    if len(objects) == 1:
        return objects[0]

    merged: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    for obj in objects:
        nested = obj.get("entries")
        if isinstance(nested, list):
            entries.extend(nested)
        else:
            entries.append(obj)
        merged.update(obj)
    merged["entries"] = entries
    return merged


def load_json_tolerant(text: str) -> dict[str, Any]:
    """Parse ``text`` as one JSON object, tolerating legacy concatenation.

    Convenience wrapper combining :func:`raw_decode_all` and
    :func:`merge_concatenated_dicts` for the common case of loading one
    audit-style file. Non-dict top-level values are ignored.
    """
    objects = [obj for obj in raw_decode_all(text) if isinstance(obj, dict)]
    return merge_concatenated_dicts(objects)
