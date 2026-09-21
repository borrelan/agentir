"""Tests for the provider-neutral published-row mapper."""

from __future__ import annotations

from agentir.hardening.frontend_mapping import (
    map_row,
    quality_gate,
    row_shape,
    tool_registry_from_row,
)
from agentir.ir.base import EventType


def _row() -> dict:
    return {
        "source_origin": {"source_file_sha256": "a" * 64, "extractor_version": "fixture/v1"},
        "messages": [
            {"role": "user", "content": "run the check"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "check",
                        "arguments": {"path": "src"},
                    }
                ],
                "thought": "must remain in loss accounting, not metadata",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "tool_name": "check",
                "content": "passed",
            },
            {"role": "assistant", "content": "done"},
        ],
    }


def test_row_shape_and_mapping_preserve_typed_tool_edge() -> None:
    row = _row()
    shape = row_shape(row)
    record, mapping = map_row(row, "fixture", 7)

    assert shape["dialogue"] is True
    assert shape["tool_bearing"] is True
    assert mapping["matched_tool_edges"] == 1
    assert mapping["unmatched_call_ids"] == 0
    assert mapping["unmatched_result_ids"] == 0
    assert mapping["dropped_message_fields"] == {"thought": 1}
    assert record.raw == {}
    events = record.episodes[0].events
    assert [event.event_type for event in events] == [
        EventType.USER_MESSAGE,
        EventType.ASSISTANT_MESSAGE,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.ASSISTANT_MESSAGE,
    ]
    assert events[2].action is not None
    assert events[3].action is not None
    assert events[2].action.tool_call_id == events[3].action.tool_call_id == "call-1"


def test_quality_gate_is_row_bound_and_defaults_to_unspecified() -> None:
    assert quality_gate({}) == "unspecified"
    assert quality_gate({"source_origin": {"quality_gate": "review_required"}}) == (
        "review_required"
    )


def test_tool_registry_requires_explicit_revision_and_schema() -> None:
    row = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read a file",
                    "parameters": {"type": "object"},
                },
            },
            {"name": "name-only"},
        ],
        "source_origin": {"tool_registry_revision": "tools/v1"},
    }

    tools, revision, malformed = tool_registry_from_row(row)

    assert revision == "tools/v1"
    assert malformed == 1
    assert [tool.name for tool in tools] == ["read"]
    assert tools[0].metadata["registry_revision"] == "tools/v1"


def test_mapper_retains_tool_registry_as_typed_candidate_metadata() -> None:
    row = _row()
    row["tools"] = [{"name": "check", "input_schema": {"type": "object"}}]
    row["tool_registry_revision"] = "tools/v1"

    record, mapping = map_row(row, "fixture", 7)

    assert mapping["tool_registry_count"] == 1
    assert mapping["tool_registry_revision"] == "tools/v1"
    assert record.tool_registry[0].name == "check"
