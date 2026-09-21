"""Tests for the streaming frontend-to-hardening bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentir.hardening import (
    REQUIRED_QUALITY_DIMENSIONS,
    AdmissionState,
    ParentSnapshotRef,
    QualityReviewManifest,
    SourceClass,
    build_source_lineage,
    compile_frontend_jsonl,
    load_review_manifest,
    review_map,
    write_review_manifest,
)
from agentir.hardening.contracts import QualityReview
from agentir.hardening.streaming import CompileInterrupted
from agentir.ir.base import EventType, IRLevel, MessageRole
from agentir.ir.content import ContentBlock
from agentir.ir.episode import Episode
from agentir.ir.event import Event
from agentir.ir.record import AgentIRRecord
from agentir.ir.source import SourceRef
from agentir.ir.task import TaskSpec


def _record() -> AgentIRRecord:
    return AgentIRRecord(
        record_id="unit-1",
        level=IRLevel.PARSED,
        source=SourceRef(dataset="fixture", row_id="unit-1"),
        task=TaskSpec(task_id="unit-1", instruction="check the change"),
        episodes=[
            Episode(
                episode_id="episode-1",
                events=[
                    Event(
                        event_id="user-1",
                        idx=0,
                        event_type=EventType.USER_MESSAGE,
                        role=MessageRole.USER,
                        content=[ContentBlock(type="text", text="check the change")],
                    ),
                    Event(
                        event_id="assistant-1",
                        idx=1,
                        event_type=EventType.ASSISTANT_MESSAGE,
                        role=MessageRole.ASSISTANT,
                        content=[ContentBlock(type="text", text="done")],
                    ),
                ],
            )
        ],
    )


def _review() -> QualityReview:
    return QualityReview(
        unit_id="unit-1",
        review_id="review-1",
        manifest_revision="reviews/v1",
        snapshot_revision="snapshot-1",
        reviewer_id="reviewer-1",
        rubric_revision="quality/v1",
        reviewed_at="2026-09-17T00:00:00Z",
        decision="accepted",
        dimensions={name: "pass" for name in REQUIRED_QUALITY_DIMENSIONS},
        evidence_ids=("unit-1",),
    )


def _lineage(row: dict, record: AgentIRRecord, context):
    return build_source_lineage(
        row,
        record,
        context,
        source_class=SourceClass.SESSION,
        parser_name="fixture.frontend",
        parser_revision="fixture/v1",
        snapshot_revision="snapshot-1",
        parent_snapshot=ParentSnapshotRef(
            snapshot_revision="snapshot-1",
            source_uri="snapshot://raw/source.jsonl",
            source_path="raw/source.jsonl",
            source_sha256="b" * 64,
            source_size_bytes=10,
            object_id="raw-1",
            manifest_verified=True,
        ),
    )


def _mapper(_row: dict, _provider: str, _source_line: int):
    return _record(), {"message_count": 2}


def _write_source(path: Path) -> None:
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "check the change"}]}) + "\n",
        encoding="utf-8",
    )


def _read_rows(output: Path) -> list[dict]:
    rows: list[dict] = []
    for shard in sorted(output.glob("shard-*.jsonl")):
        rows.extend(json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines())
    return rows


def _compile(source: Path, output: Path, *, reviews=None, resume=False, stop=None):
    return compile_frontend_jsonl(
        source,
        output,
        provider="fixture",
        map_row=_mapper,
        build_lineage=_lineage,
        parser_revision="fixture/v1",
        pass_config={"projection": "sft"},
        quality_state=lambda _row, _mapping: AdmissionState.ACCEPTED,
        reviews=reviews,
        targets=("sft",),
        shard_records=1,
        shard_bytes=1024,
        resume=resume,
        stop_after_source_records=stop,
    )


def test_unreviewed_frontend_row_is_streamed_as_non_release(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    manifest = _compile(source, tmp_path / "output")
    row = _read_rows(tmp_path / "output")[0]

    assert manifest["records"] == 1
    assert row["admission"]["state"] == "candidate"
    assert row["projections"]["sft"]["state"] == "candidate"
    assert row["projections"]["sft"]["output"] == []
    assert row["lineage"]["byte_start"] == 0
    assert row["lineage"]["byte_end"] == source.stat().st_size
    assert row["mapping"] == {"message_count": 2}


def test_reviewed_frontend_row_emits_only_allowlisted_projection(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    _compile(source, tmp_path / "output", reviews={"unit-1": _review()})
    row = _read_rows(tmp_path / "output")[0]

    assert row["admission"] == {
        "state": "accepted",
        "rule_ids": [],
        "review_id": "review-1",
    }
    assert row["projections"]["sft"]["state"] == "accepted"
    assert [message["role"] for message in row["projections"]["sft"]["output"]] == [
        "user",
        "assistant",
    ]
    assert "messages" not in row


def test_frontend_preserves_bounded_model_tier_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)

    compile_frontend_jsonl(
        source,
        tmp_path / "output",
        provider="fixture",
        map_row=lambda _row, _provider, _line: (
            _record(),
            {
                "message_count": 2,
                "model_tier": "tier1_frontier",
                "model_tier_basis": "fixture_registry",
                "model_tier_registry_revision": "fixture-v1",
            },
        ),
        build_lineage=_lineage,
        parser_revision="fixture/v1",
        pass_config={"projection": "sft"},
        targets=("sft",),
        shard_records=1,
        shard_bytes=1024,
    )
    row = _read_rows(tmp_path / "output")[0]

    assert row["mapping"] == {
        "message_count": 2,
        "model_tier": "tier1_frontier",
        "model_tier_basis": "fixture_registry",
        "model_tier_registry_revision": "fixture-v1",
    }


def test_loaded_review_artifact_is_the_stream_review_boundary(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    review_path = tmp_path / "reviews.json"
    _write_source(source)
    write_review_manifest(
        review_path,
        QualityReviewManifest(
            manifest_revision="reviews/v1",
            snapshot_revision="snapshot-1",
            rubric_revision="quality/v1",
        ),
    )

    reviews = review_map(
        load_review_manifest(
            review_path,
            expected_manifest_revision="reviews/v1",
            expected_snapshot_revision="snapshot-1",
            expected_rubric_revision="quality/v1",
        )
    )
    _compile(source, tmp_path / "output", reviews=reviews)
    row = _read_rows(tmp_path / "output")[0]

    assert reviews == {}
    assert row["projections"]["sft"]["output"] == []


def test_frontend_resume_has_same_semantic_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    clean = _compile(source, tmp_path / "clean", reviews={"unit-1": _review()})

    with pytest.raises(CompileInterrupted):
        _compile(source, tmp_path / "resumed", reviews={"unit-1": _review()}, stop=1)
    resumed = _compile(
        source,
        tmp_path / "resumed",
        reviews={"unit-1": _review()},
        resume=True,
    )

    assert resumed == clean
    assert len(_read_rows(tmp_path / "resumed")) == 1
