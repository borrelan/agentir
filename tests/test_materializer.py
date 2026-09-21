"""Tests for bounded accepted-envelope materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentir.hardening import (
    DatasetSplit,
    DecisionSource,
    ModelTier,
    RegistryOrigin,
    ReleaseMetadata,
    ReleaseProvenance,
    ReleaseView,
    SourceRange,
    ToolDefinition,
    materialize_release_view,
    tool_registry_artifact,
)
from agentir.hardening.contracts import REQUIRED_QUALITY_DIMENSIONS
from agentir.hardening.streaming import CompileInterrupted


def _metadata() -> ReleaseMetadata:
    return ReleaseMetadata(
        release_revision="release/v1",
        example_id="example-1",
        split=DatasetSplit.TRAIN,
        provenance=ReleaseProvenance(
            unit_id="unit-1",
            snapshot_revision="snapshot-1",
            source_sha256="a" * 64,
            source_size_bytes=100,
            source_range=SourceRange(byte_start=0, byte_end=100, message_start=1, message_end=1),
            parser_revision="parser/v1",
            source_class="session",
            provider="fixture",
            parent_source_sha256="b" * 64,
            parent_snapshot_verified=True,
        ),
        decision_id="review-1",
        decision_source=DecisionSource.HUMAN_REVIEW,
        decision_revision="reviews/v1",
        evidence_ids=("unit-1",),
        quality_dimensions={
            dimension: "not_applicable" if dimension == "contamination" else "pass"
            for dimension in REQUIRED_QUALITY_DIMENSIONS
        },
        privacy_policy="redact_private_values",
        dedupe_group_id="group-1",
        model_tier=ModelTier.TIER1_FRONTIER,
        model_tier_registry_revision="tier-registry/v1",
    )


def _envelope(state: str = "accepted") -> dict:
    return {
        "unit_id": "unit-1",
        "admission": {"state": state},
        "projections": {
            "sft": {
                "state": state,
                "output": [
                    {"role": "user", "content": "check"},
                    {"role": "assistant", "content": "done"},
                ],
                "losses": [],
            },
            "tool_use": {
                "state": state,
                "output": [
                    {
                        "event_id": "event-1",
                        "context": [{"role": "user", "content": "read x"}],
                        "call_id": "call-1",
                        "tool_name": "read",
                        "arguments": {"path": "x"},
                        "observation_event_id": "result-1",
                        "observation": {
                            "status": "matched",
                            "content": "ok",
                            "exit_code": 0,
                            "error_type": None,
                            "truncated": False,
                        },
                    }
                ],
                "losses": [],
            },
        },
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="read",
        parameters={"type": "object"},
        registry_revision="tools/v1",
    )


def test_materializer_filters_unaccepted_and_writes_typed_sft(tmp_path: Path) -> None:
    source = tmp_path / "envelopes.jsonl"
    _write(source, [_envelope("candidate"), _envelope("accepted")])

    manifest = materialize_release_view(
        source,
        tmp_path / "release",
        target=ReleaseView.SFT,
        metadata_for=lambda _row, _target: _metadata(),
        release_revision="release/v1",
        shard_records=1,
    )

    assert manifest["records"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "release" / "shard-000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["view"] == "sft"
    assert rows[0]["metadata"]["provenance"]["snapshot_revision"] == "snapshot-1"


def test_materializer_requires_metadata_and_tool_registry(tmp_path: Path) -> None:
    source = tmp_path / "envelopes.jsonl"
    _write(source, [_envelope()])

    with pytest.raises(ValueError, match="no release metadata"):
        materialize_release_view(
            source,
            tmp_path / "missing-metadata",
            target=ReleaseView.SFT,
            metadata_for=lambda _row, _target: None,
            release_revision="release/v1",
        )

    with pytest.raises(ValueError, match="registry"):
        materialize_release_view(
            source,
            tmp_path / "missing-tools",
            target=ReleaseView.TOOL_USE,
            metadata_for=lambda _row, _target: _metadata(),
            release_revision="release/v1",
        )

    manifest = materialize_release_view(
        source,
        tmp_path / "tool-release",
        target=ReleaseView.TOOL_USE,
        metadata_for=lambda _row, _target: _metadata(),
        tools_for=lambda _row, _target: (_tool(),),
        release_revision="release/v1",
    )
    assert manifest["records"] == 1
    row = json.loads(
        (tmp_path / "tool-release" / "shard-000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["steps"][0]["observation_status"] == "matched"
    assert row["steps"][0]["observation"]["content"] == "ok"


def test_materializer_can_require_tier_one_for_gold_release(tmp_path: Path) -> None:
    source = tmp_path / "envelopes.jsonl"
    _write(source, [_envelope()])

    with pytest.raises(ValueError, match="model tier"):
        materialize_release_view(
            source,
            tmp_path / "tier-mismatch",
            target=ReleaseView.SFT,
            metadata_for=lambda _row, _target: _metadata().model_copy(
                update={"model_tier": ModelTier.TIER2_OPEN_SOURCE}
            ),
            release_revision="release/v1",
            required_model_tier=ModelTier.TIER1_FRONTIER,
        )

    manifest = materialize_release_view(
        source,
        tmp_path / "tier-one",
        target=ReleaseView.SFT,
        metadata_for=lambda _row, _target: _metadata(),
        release_revision="release/v1",
        required_model_tier=ModelTier.TIER1_FRONTIER,
    )
    assert manifest["records"] == 1


def test_materializer_accepts_only_a_digest_bound_registry_artifact(tmp_path: Path) -> None:
    source = tmp_path / "envelopes.jsonl"
    _write(source, [_envelope()])
    registry = tool_registry_artifact(
        registry_revision="tools/v1",
        source_revision="harness/tools/v1",
        scope="fixture",
        origin=RegistryOrigin.HARNESS,
        definitions=(_tool(),),
    )

    manifest = materialize_release_view(
        source,
        tmp_path / "artifact-tool-release",
        target=ReleaseView.TOOL_USE,
        metadata_for=lambda _row, _target: _metadata(),
        registry_for=lambda _row, _target: registry,
        release_revision="release/v1",
    )

    assert manifest["records"] == 1


def test_materializer_rejects_ambiguous_registry_resolvers(tmp_path: Path) -> None:
    source = tmp_path / "envelopes.jsonl"
    _write(source, [_envelope()])
    with pytest.raises(ValueError, match="either tools_for or registry_for"):
        materialize_release_view(
            source,
            tmp_path / "ambiguous",
            target=ReleaseView.TOOL_USE,
            metadata_for=lambda _row, _target: _metadata(),
            tools_for=lambda _row, _target: (_tool(),),
            registry_for=lambda _row, _target: None,
            release_revision="release/v1",
        )


def test_materializer_resume_matches_clean_release_manifest(tmp_path: Path) -> None:
    source = tmp_path / "envelopes.jsonl"
    _write(source, [_envelope()])
    clean = materialize_release_view(
        source,
        tmp_path / "clean",
        target="sft",
        metadata_for=lambda _row, _target: _metadata(),
        release_revision="release/v1",
    )

    with pytest.raises(CompileInterrupted):
        materialize_release_view(
            source,
            tmp_path / "resumed",
            target=ReleaseView.SFT,
            metadata_for=lambda _row, _target: _metadata(),
            release_revision="release/v1",
            stop_after_source_records=1,
        )
    resumed = materialize_release_view(
        source,
        tmp_path / "resumed",
        target=ReleaseView.SFT,
        metadata_for=lambda _row, _target: _metadata(),
        release_revision="release/v1",
        resume=True,
    )

    assert resumed == clean
