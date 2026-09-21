from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentir.hardening import (
    HarnessRegistryError,
    acquire_harness_registry,
    harness_registry_digest,
)


def _event(
    *,
    ordinal: int,
    event_type: str,
    payload: dict,
    registry_revision: str = "tools/v1",
    episode_id: str = "episode-1",
) -> dict:
    return {
        "schema_version": "ai-data-extraction/harness-trace/v1",
        "trace_version": "1.0.0",
        "episode_id": episode_id,
        "event_id": f"event-{ordinal}",
        "ordinal": ordinal,
        "event_type": event_type,
        "registry_revision": registry_revision,
        "skill_revision": "skills/v1",
        "environment_revision": "env/v1",
        "verifier_revision": "verifier/v1",
        "privacy_state": "heuristic",
        "required_skills": [],
        "source": {"provider": "fixture"},
        "payload": payload,
    }


def _trace(path: Path, *, tamper_digest: bool = False, unknown_call: bool = False) -> None:
    tools = [
        {
            "name": "shell.exec",
            "schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            "trust_class": "sandboxed",
            "side_effect_class": "read_only",
        }
    ]
    digest = harness_registry_digest(registry_revision="tools/v1", tools=tools)
    if tamper_digest:
        digest = "sha256:" + "0" * 64
    call_name = "filesystem.write" if unknown_call else "shell.exec"
    rows = [
        _event(
            ordinal=0,
            event_type="tool_registry",
            payload={
                "registry_revision": "tools/v1",
                "registry_sha256": digest,
                "tools": tools,
            },
        ),
        _event(
            ordinal=1,
            event_type="tool_call",
            payload={
                "tool_name": call_name,
                "call_id": "call-1",
                "arguments": {"command": "true"},
                "permission_decision": "not_required",
            },
        ),
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_acquires_registry_in_one_streaming_pass(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _trace(trace)

    evidence = acquire_harness_registry(
        trace,
        source_revision="harness/fixture/v1",
        scope="fixture/default",
        require_tool_call=True,
    )

    assert evidence.episode_id == "episode-1"
    assert evidence.tool_call_count == 1
    assert evidence.harness_registry_sha256.startswith("sha256:")
    assert evidence.artifact.origin.value == "harness"
    assert evidence.artifact.definitions[0].name == "shell.exec"
    assert evidence.manifest()["trace_sha256"] == evidence.trace_sha256


def test_rejects_tampered_registry_digest(tmp_path: Path) -> None:
    trace = tmp_path / "tampered.jsonl"
    _trace(trace, tamper_digest=True)

    with pytest.raises(HarnessRegistryError, match="digest"):
        acquire_harness_registry(
            trace,
            source_revision="harness/fixture/v1",
            scope="fixture/default",
        )


def test_rejects_calls_missing_from_registry(tmp_path: Path) -> None:
    trace = tmp_path / "unknown-call.jsonl"
    _trace(trace, unknown_call=True)

    with pytest.raises(HarnessRegistryError, match="absent"):
        acquire_harness_registry(
            trace,
            source_revision="harness/fixture/v1",
            scope="fixture/default",
        )


def test_requires_runtime_registry_for_tool_calls(tmp_path: Path) -> None:
    trace = tmp_path / "missing-registry.jsonl"
    trace.write_text(
        json.dumps(
            _event(
                ordinal=0,
                event_type="tool_call",
                payload={
                    "tool_name": "shell.exec",
                    "call_id": "call-1",
                    "arguments": {},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HarnessRegistryError, match="precedes"):
        acquire_harness_registry(
            trace,
            source_revision="harness/fixture/v1",
            scope="fixture/default",
        )
