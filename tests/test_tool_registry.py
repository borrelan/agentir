"""Tests for the versioned tool-registry release artifact."""

from __future__ import annotations

import json

import pytest

from agentir.hardening import (
    RegistryOrigin,
    ToolDefinition,
    ToolRegistryArtifact,
    load_tool_registry,
    registry_tools,
    tool_registry_artifact,
    write_tool_registry,
)


def _definition(revision: str = "tools/v1", name: str = "read") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        registry_revision=revision,
    )


def _artifact() -> ToolRegistryArtifact:
    return tool_registry_artifact(
        registry_revision="tools/v1",
        source_revision="harness/tools/2026-09-17",
        scope="provider-neutral/default",
        origin=RegistryOrigin.HARNESS,
        definitions=(_definition(),),
    )


def test_registry_is_digest_bound_and_round_trips(tmp_path) -> None:
    artifact = _artifact()
    path = tmp_path / "tools.json"

    digest = write_tool_registry(path, artifact)
    loaded = load_tool_registry(
        path,
        expected_registry_revision="tools/v1",
        expected_source_revision="harness/tools/2026-09-17",
    )

    assert digest == artifact.artifact_sha256
    assert loaded == artifact
    assert registry_tools(loaded) == artifact.definitions


def test_registry_preserves_strict_function_semantics() -> None:
    definition = ToolDefinition(
        name="read",
        description="Read a file",
        parameters={"type": "object"},
        strict=True,
        registry_revision="tools/v1",
    )

    artifact = tool_registry_artifact(
        registry_revision="tools/v1",
        source_revision="harness/tools/2026-09-17",
        scope="provider-neutral/default",
        origin=RegistryOrigin.HARNESS,
        definitions=(definition,),
    )

    assert artifact.definitions[0].strict is True


def test_registry_rejects_tampering_and_identity_mismatch(tmp_path) -> None:
    artifact = _artifact()
    path = tmp_path / "tools.json"
    write_tool_registry(path, artifact)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["definitions"][0]["description"] = "changed"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_tool_registry(path)

    path.write_text(artifact.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="source revision"):
        load_tool_registry(path, expected_source_revision="harness/tools/other")


def test_registry_rejects_duplicate_names_and_revision_mismatch() -> None:
    with pytest.raises(ValueError, match="duplicate names"):
        tool_registry_artifact(
            registry_revision="tools/v1",
            source_revision="source/v1",
            scope="test",
            origin=RegistryOrigin.SOURCE_DECLARED,
            definitions=(_definition(), _definition()),
        )

    mismatched = _definition(revision="tools/v2")
    with pytest.raises(ValueError, match="different registry revision"):
        tool_registry_artifact(
            registry_revision="tools/v1",
            source_revision="source/v1",
            scope="test",
            origin=RegistryOrigin.SOURCE_DECLARED,
            definitions=(mismatched,),
        )


def test_registry_rejects_blocked_definition() -> None:
    with pytest.raises(ValueError, match="blocked release findings"):
        tool_registry_artifact(
            registry_revision="tools/v1",
            source_revision="source/v1",
            scope="test",
            origin=RegistryOrigin.EXTERNAL,
            definitions=(
                ToolDefinition(
                    name="read",
                    description="read /data-sea/private/file",
                    parameters={},
                    registry_revision="tools/v1",
                ),
            ),
        )
