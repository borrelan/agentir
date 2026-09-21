"""Tests for content-free, deterministic review task generation."""

from __future__ import annotations

from agentir.hardening import REQUIRED_QUALITY_DIMENSIONS, build_review_tasks


def _envelope(unit_id: str, state: str, loss_code: str | None = None) -> dict:
    losses = [] if loss_code is None else [{"code": loss_code}]
    return {
        "unit_id": unit_id,
        "provider": "fixture",
        "source_line": int(unit_id.split("-")[-1]),
        "lineage": {
            "source_path": "frontend/fixture.jsonl",
            "source_sha256": "a" * 64,
            "source_size_bytes": 100,
            "byte_start": 0,
            "byte_end": 100,
            "parser_revision": "fixture/v1",
            "parent_snapshot": {
                "object_id": "raw-1",
                "source_sha256": "b" * 64,
                "source_size_bytes": 50,
                "manifest_verified": True,
            },
        },
        "admission": {"state": state, "rule_ids": [loss_code] if loss_code else []},
        "mapping": {"message_count": 2},
        "losses": losses,
        "projections": {
            "sft": {"state": state, "output": [], "losses": losses},
            "tool_use": {"state": state, "output": [], "losses": losses},
        },
    }


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def test_review_tasks_are_content_free_and_blank_by_default() -> None:
    result = build_review_tasks(
        [
            _envelope("unit-1", "candidate", "PRIVATE_VALUE"),
            _envelope("unit-2", "quarantined", "ORPHAN_TOOL_RESULT"),
        ],
        snapshot_revision="snapshot-1",
        manifest_revision="reviews/v1",
    )

    assert result["summary"] == {
        "tasks": 2,
        "assigned": 0,
        "accepted": 0,
        "quarantined": 0,
        "rejected": 0,
    }
    assert set(result["required_dimensions"]) == set(REQUIRED_QUALITY_DIMENSIONS)
    assert all(task["status"] == "unassigned" for task in result["tasks"])
    assert all(task["review"]["decision"] is None for task in result["tasks"])
    assert all(
        value is None for task in result["tasks"] for value in task["review"]["dimensions"].values()
    )
    assert not set(_keys(result)) & {"output", "messages", "content", "arguments"}


def test_review_task_selection_round_robins_admission_buckets() -> None:
    result = build_review_tasks(
        [
            _envelope("unit-1", "candidate", "PRIVATE_VALUE"),
            _envelope("unit-2", "candidate", "PRIVATE_VALUE"),
            _envelope("unit-3", "quarantined", "ORPHAN_TOOL_RESULT"),
        ],
        snapshot_revision="snapshot-1",
        manifest_revision="reviews/v1",
        max_tasks=2,
    )

    assert len(result["tasks"]) == 2
    assert {task["observed"]["admission_state"] for task in result["tasks"]} == {
        "candidate",
        "quarantined",
    }
    assert result["tasks"] == sorted(result["tasks"], key=lambda task: task["task_id"])
