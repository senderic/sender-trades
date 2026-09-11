"""Unit tests for tolerant JSON loading of legacy concatenated log files."""

from __future__ import annotations

from src.json_utils import load_json_tolerant, merge_concatenated_dicts, raw_decode_all


class TestRawDecodeAll:
    def test_single_object(self) -> None:
        assert raw_decode_all('{"a": 1}') == [{"a": 1}]

    def test_multiple_concatenated_objects(self) -> None:
        text = '{"a": 1}\n{"b": 2}\n{"c": 3}\n'
        assert raw_decode_all(text) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_no_separator_between_objects(self) -> None:
        # json.JSONDecoder.raw_decode tolerates zero whitespace between
        # top-level values as long as each is itself well-formed.
        text = '{"a": 1}{"b": 2}'
        assert raw_decode_all(text) == [{"a": 1}, {"b": 2}]

    def test_blank_input_returns_empty_list(self) -> None:
        assert raw_decode_all("") == []
        assert raw_decode_all("   \n  ") == []

    def test_non_dict_top_level_values_preserved(self) -> None:
        assert raw_decode_all('[1, 2]\n"x"\n') == [[1, 2], "x"]


class TestMergeConcatenatedDicts:
    def test_empty_list_returns_empty_dict(self) -> None:
        assert merge_concatenated_dicts([]) == {}

    def test_single_object_returned_unchanged(self) -> None:
        obj = {"exit_reason": "take_profit", "entries": [{"event_type": "entry_submitted"}]}
        assert merge_concatenated_dicts([obj]) is obj

    def test_multiple_per_line_events_folded_into_entries(self) -> None:
        objects = [
            {"trade_id": "t1", "event_type": "entry_submitted", "occ_symbol": "SPY"},
            {"trade_id": "t1", "event_type": "entry_filled", "fill_price": 0.5},
            {"trade_id": "t1", "event_type": "exits_placed", "sl_level": 0.25},
        ]
        merged = merge_concatenated_dicts(objects)
        assert merged["trade_id"] == "t1"
        assert [e["event_type"] for e in merged["entries"]] == [
            "entry_submitted",
            "entry_filled",
            "exits_placed",
        ]

    def test_nested_entries_list_extended_not_nested(self) -> None:
        objects = [
            {"entries": [{"event_type": "a"}, {"event_type": "b"}]},
            {"event_type": "c"},
        ]
        merged = merge_concatenated_dicts(objects)
        assert [e["event_type"] for e in merged["entries"]] == ["a", "b", "c"]

    def test_scalar_fields_last_write_wins(self) -> None:
        objects = [
            {"exit_reason": None, "event_type": "entry_submitted"},
            {"exit_reason": "pending", "event_type": "entry_filled"},
        ]
        merged = merge_concatenated_dicts(objects)
        assert merged["exit_reason"] == "pending"


class TestLoadJsonTolerant:
    def test_normal_single_object_file(self) -> None:
        assert load_json_tolerant('{"exit_reason": "take_profit"}') == {
            "exit_reason": "take_profit"
        }

    def test_legacy_concatenated_file(self) -> None:
        text = (
            '{"trade_id": "t1", "event_type": "entry_submitted", "occ_symbol": "SPY"}\n'
            '{"trade_id": "t1", "event_type": "exits_placed", "sl_level": 0.4}\n'
        )
        data = load_json_tolerant(text)
        assert data["trade_id"] == "t1"
        assert len(data["entries"]) == 2
        assert data["entries"][1]["sl_level"] == 0.4

    def test_empty_file_returns_empty_dict(self) -> None:
        assert load_json_tolerant("") == {}
