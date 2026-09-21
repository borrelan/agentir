"""Tests for the fail-closed hardening boundary."""

from __future__ import annotations

import pytest

from agentir.hardening import (
    REQUIRED_QUALITY_DIMENSIONS,
    AdmissionState,
    LossAction,
    ParentSnapshotRef,
    PrivacyPolicy,
    QualityReview,
    QualityReviewManifest,
    SnapshotManifestIndex,
    SourceClass,
    SourceLineage,
    assess_record,
    build_review_context,
    project,
    scan_value,
    verify_parent_snapshot,
)
from agentir.ir.action import Action
from agentir.ir.base import ActionKind, EventType, IRLevel, MessageRole, ObservationKind
from agentir.ir.content import ContentBlock
from agentir.ir.episode import Episode
from agentir.ir.event import Event
from agentir.ir.observation import Observation
from agentir.ir.record import AgentIRRecord
from agentir.ir.source import SourceRef


def _lineage(unit_id: str = "unit-1") -> SourceLineage:
    return SourceLineage(
        source_uri="snapshot://source.jsonl",
        source_path="source.jsonl",
        source_class=SourceClass.SESSION,
        provider="fixture",
        snapshot_revision="snapshot-1",
        source_sha256="a" * 64,
        source_size_bytes=512,
        byte_start=10,
        byte_end=80,
        message_start=1,
        message_end=3,
        parser_name="fixture.frontend",
        parser_revision="fixture/v1",
        unit_id=unit_id,
        parent_snapshot=ParentSnapshotRef(
            snapshot_revision="snapshot-1",
            source_uri="snapshot://raw/source.jsonl",
            source_path="raw/source.jsonl",
            source_sha256="b" * 64,
            source_size_bytes=256,
            object_id="raw-object-1",
            root_label="primary",
            message_start=1,
            message_end=3,
            manifest_verified=True,
        ),
    )


def _event(
    event_id: str,
    idx: int,
    event_type: EventType,
    *,
    role: MessageRole | None = None,
    text: str | None = None,
    action: Action | None = None,
    observation: Observation | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        idx=idx,
        event_type=event_type,
        role=role,
        content=[ContentBlock(type="text", text=text)] if text is not None else [],
        action=action,
        observation=observation,
    )


def _record(events: list[Event], *, raw: dict | None = None) -> AgentIRRecord:
    return AgentIRRecord(
        record_id="unit-1",
        level=IRLevel.PARSED,
        source=SourceRef(dataset="fixture", row_id="unit-1"),
        episodes=[Episode(episode_id="episode-1", events=events)],
        raw=raw or {},
    )


def _valid_record() -> AgentIRRecord:
    return _record(
        [
            _event("u", 0, EventType.USER_MESSAGE, role=MessageRole.USER, text="run the check"),
            _event(
                "c",
                1,
                EventType.TOOL_CALL,
                role=MessageRole.ASSISTANT,
                action=Action(
                    kind=ActionKind.GENERIC_TOOL,
                    tool_name="check",
                    tool_call_id="call-1",
                    arguments={"path": "src"},
                ),
            ),
            _event(
                "r",
                2,
                EventType.TOOL_RESULT,
                role=MessageRole.TOOL,
                action=Action(
                    kind=ActionKind.GENERIC_TOOL,
                    tool_name="check",
                    tool_call_id="call-1",
                ),
                observation=Observation(
                    kind=ObservationKind.TOOL_JSON,
                    content=[ContentBlock(type="text", text="passed")],
                ),
            ),
            _event("a", 3, EventType.ASSISTANT_MESSAGE, role=MessageRole.ASSISTANT, text="done"),
            _event("p", 4, EventType.PLAN, role=MessageRole.ASSISTANT, text="private plan"),
        ]
    )


def _accepted_review() -> QualityReview:
    return QualityReview(
        unit_id="unit-1",
        review_id="review-1",
        manifest_revision="reviews/2026-09-17/v1",
        snapshot_revision="snapshot-1",
        reviewer_id="reviewer-1",
        rubric_revision="quality/v1",
        reviewed_at="2026-09-17T00:00:00Z",
        decision="accepted",
        dimensions={name: "pass" for name in REQUIRED_QUALITY_DIMENSIONS},
        evidence_ids=("unit-1", "source-range-1"),
    )


def test_lineage_is_required_and_normalizes_sha256_prefix() -> None:
    lineage = _lineage()
    assert lineage.source_sha256 == "a" * 64
    with pytest.raises(ValueError):
        SourceLineage(
            **{
                **lineage.model_dump(),
                "source_sha256": "not-a-digest",
            }
        )


def test_accepted_sft_is_allowlisted_and_drops_typed_reasoning() -> None:
    hardened = assess_record(
        _valid_record(),
        _lineage(),
        declared_state=AdmissionState.ACCEPTED,
        quality_review=_accepted_review(),
    )
    result = project(hardened, "sft")

    assert result.state is AdmissionState.ACCEPTED
    assert [message["role"] for message in result.output] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert all("thinking" not in str(message).lower() for message in result.output)
    assert any(loss.code == "TYPED_REASONING_DROPPED" for loss in result.losses)
    assert all(loss.action is not LossAction.REJECT for loss in result.losses)


def test_accepted_tool_use_preserves_context_and_matched_observation() -> None:
    hardened = assess_record(
        _valid_record(),
        _lineage(),
        declared_state=AdmissionState.ACCEPTED,
        quality_review=_accepted_review(),
    )
    result = project(hardened, "tool_use")

    assert result.state is AdmissionState.ACCEPTED
    assert len(result.output) == 1
    step = result.output[0]
    assert [message["role"] for message in step["context"]] == ["user"]
    assert step["call_id"] == "call-1"
    assert step["observation_event_id"] == "r"
    assert step["observation"]["status"] == "matched"
    assert step["observation"]["content"] == "passed"


def test_tool_use_without_visible_context_is_quarantined() -> None:
    record = _record(
        [
            _event(
                "c",
                0,
                EventType.TOOL_CALL,
                role=MessageRole.ASSISTANT,
                action=Action(
                    kind=ActionKind.GENERIC_TOOL,
                    tool_name="check",
                    tool_call_id="call-1",
                    arguments={"path": "src"},
                ),
            ),
            _event(
                "r",
                1,
                EventType.TOOL_RESULT,
                role=MessageRole.TOOL,
                action=Action(
                    kind=ActionKind.GENERIC_TOOL,
                    tool_name="check",
                    tool_call_id="call-1",
                ),
                observation=Observation(
                    kind=ObservationKind.TOOL_JSON,
                    content=[ContentBlock(type="text", text="passed")],
                ),
            ),
        ]
    )
    hardened = assess_record(
        record,
        _lineage(),
        declared_state=AdmissionState.ACCEPTED,
        quality_review=_accepted_review(),
    )
    result = project(hardened, "tool_use")

    assert result.state is AdmissionState.QUARANTINED
    assert result.output == []
    assert any(loss.code == "MISSING_TOOL_CONTEXT" for loss in result.losses)


def test_candidate_preview_is_visible_but_not_release_accepted() -> None:
    hardened = assess_record(
        _valid_record(),
        _lineage(),
        declared_state=AdmissionState.ACCEPTED,
        privacy_policy=PrivacyPolicy.REDACT_PRIVATE_VALUES,
    )

    normal = project(hardened, "sft")
    preview = project(hardened, "sft", preview=True)
    context = build_review_context(
        task_id="review-task-1",
        mapping={"message_count": 5, "call_count": 1},
        hardened=hardened,
        projections={"sft": preview},
        registry=None,
    )

    assert normal.state is AdmissionState.CANDIDATE
    assert normal.output == []
    assert preview.state is AdmissionState.CANDIDATE
    assert [message["role"] for message in preview.output] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert context["review_only"] is True
    assert context["decision"] is None
    assert all(value is None for value in context["quality_dimensions"].values())
    assert context["projections"]["sft"]["output"]
    assert "raw" not in context


def test_untyped_hidden_content_quarantines_before_lowering() -> None:
    record = _record(
        [
            _event(
                "a",
                0,
                EventType.ASSISTANT_MESSAGE,
                role=MessageRole.ASSISTANT,
                text="<thinking>secret</thinking>",
            )
        ]
    )
    hardened = assess_record(record, _lineage(), declared_state=AdmissionState.ACCEPTED)
    result = project(hardened, "sft")

    assert hardened.quality.state is AdmissionState.QUARANTINED
    assert result.state is AdmissionState.QUARANTINED
    assert result.output == []
    assert any(loss.code == "HIDDEN_MARKER" for loss in result.losses)


def test_raw_and_orphan_tool_result_cannot_be_trainer_data() -> None:
    record = _record(
        [
            _event(
                "r",
                0,
                EventType.TOOL_RESULT,
                role=MessageRole.TOOL,
                action=Action(
                    kind=ActionKind.GENERIC_TOOL,
                    tool_name="check",
                    tool_call_id="orphan",
                ),
                observation=Observation(kind=ObservationKind.TOOL_JSON),
            )
        ],
        raw={"provider_payload": "must not cross the boundary"},
    )
    hardened = assess_record(record, _lineage(), declared_state=AdmissionState.ACCEPTED)
    result = project(hardened, "tool_use")

    assert result.state is AdmissionState.QUARANTINED
    assert result.output == []
    assert {loss.code for loss in result.losses} >= {"UNTRUSTED_FIELD", "ORPHAN_TOOL_RESULT"}


def test_candidate_quality_never_emits_a_release_row() -> None:
    hardened = assess_record(_valid_record(), _lineage(), declared_state=AdmissionState.CANDIDATE)
    result = project(hardened, "sft")

    assert result.state is AdmissionState.CANDIDATE
    assert result.output == []


def test_source_accepted_assertion_without_review_stays_candidate() -> None:
    hardened = assess_record(_valid_record(), _lineage(), declared_state=AdmissionState.ACCEPTED)
    result = project(hardened, "sft")

    assert hardened.quality.state is AdmissionState.CANDIDATE
    assert hardened.quality.review is None
    assert result.output == []


def test_accepted_review_requires_all_dimensions() -> None:
    with pytest.raises(ValueError, match="accepted quality reviews must cover"):
        QualityReview(
            unit_id="unit-1",
            review_id="review-1",
            manifest_revision="reviews/2026-09-17/v1",
            snapshot_revision="snapshot-1",
            reviewer_id="reviewer-1",
            rubric_revision="quality/v1",
            reviewed_at="2026-09-17T00:00:00Z",
            decision="accepted",
            dimensions={"structural": "pass"},
            evidence_ids=("unit-1",),
        )


def test_quality_review_must_join_the_lineage_unit() -> None:
    review = _accepted_review().model_copy(
        update={"unit_id": "other-unit", "evidence_ids": ("other-unit",)}
    )
    hardened = assess_record(_valid_record(), _lineage(), quality_review=review)

    assert hardened.quality.state is AdmissionState.QUARANTINED
    assert any(loss.code == "QUALITY_REVIEW_MISMATCH" for loss in hardened.losses)


def test_quality_review_must_join_the_lineage_snapshot() -> None:
    review = _accepted_review().model_copy(update={"snapshot_revision": "snapshot-2"})
    hardened = assess_record(_valid_record(), _lineage(), quality_review=review)

    assert hardened.quality.state is AdmissionState.QUARANTINED
    assert any(loss.code == "QUALITY_REVIEW_SNAPSHOT_MISMATCH" for loss in hardened.losses)


def test_review_manifest_rejects_duplicate_units_and_mismatched_versions() -> None:
    review = _accepted_review()
    with pytest.raises(ValueError, match="duplicate unit_id"):
        QualityReviewManifest(
            manifest_revision="reviews/2026-09-17/v1",
            snapshot_revision="snapshot-1",
            rubric_revision="quality/v1",
            reviews=(review, review.model_copy(update={"review_id": "review-2"})),
        )
    with pytest.raises(ValueError, match="revision does not match"):
        QualityReviewManifest(
            manifest_revision="reviews/2026-09-17/v2",
            snapshot_revision="snapshot-1",
            rubric_revision="quality/v1",
            reviews=(review,),
        )


def test_accepted_review_requires_parent_snapshot_binding() -> None:
    lineage = _lineage().model_copy(update={"parent_snapshot": None})
    hardened = assess_record(_valid_record(), lineage, quality_review=_accepted_review())

    assert hardened.quality.state is AdmissionState.QUARANTINED
    assert any(loss.code == "PARENT_SNAPSHOT_UNBOUND" for loss in hardened.losses)


def test_accepted_review_requires_manifest_verified_parent() -> None:
    lineage = _lineage().model_copy(
        update={
            "parent_snapshot": _lineage().parent_snapshot.model_copy(
                update={"manifest_verified": False}
            )
        }
    )
    hardened = assess_record(_valid_record(), lineage, quality_review=_accepted_review())

    assert hardened.quality.state is AdmissionState.QUARANTINED
    assert any(loss.code == "PARENT_SNAPSHOT_UNVERIFIED" for loss in hardened.losses)


def test_parent_snapshot_verification_matches_one_stable_manifest_object() -> None:
    parent = _lineage().parent_snapshot
    assert parent is not None
    verified = verify_parent_snapshot(
        parent.model_copy(update={"manifest_verified": False}),
        {
            "snapshot_revision": "snapshot-1",
            "files": [
                {
                    "sha256": "b" * 64,
                    "bytes": 256,
                    "status": "stable",
                    "root_label": "primary",
                }
            ],
        },
    )

    assert verified.manifest_verified is True

    with pytest.raises(ValueError, match="ambiguous"):
        verify_parent_snapshot(
            parent.model_copy(update={"manifest_verified": False, "root_label": None}),
            {
                "snapshot_revision": "snapshot-1",
                "files": [
                    {"sha256": "b" * 64, "bytes": 256, "status": "stable"},
                    {"sha256": "b" * 64, "bytes": 256, "status": "stable", "root_label": "backup"},
                ],
            },
        )


def test_indexed_parent_snapshot_verification_matches_mapping_contract() -> None:
    parent = _lineage().parent_snapshot
    assert parent is not None
    manifest = {
        "snapshot_revision": "snapshot-1",
        "files": [
            {
                "sha256": "b" * 64,
                "bytes": 256,
                "status": "stable",
                "root_label": "primary",
            }
        ],
    }
    index = SnapshotManifestIndex.from_manifest(manifest)

    verified = verify_parent_snapshot(
        parent.model_copy(update={"manifest_verified": False}),
        index,
    )

    assert verified.manifest_verified is True


def test_explicit_private_value_policy_redacts_without_releasing_the_value() -> None:
    record = _valid_record()
    assert record.episodes[0].events[1].action is not None
    record.episodes[0].events[1].action.arguments["path"] = "/home/alice/private-repo"

    hardened = assess_record(
        record,
        _lineage(),
        declared_state=AdmissionState.ACCEPTED,
        privacy_policy=PrivacyPolicy.REDACT_PRIVATE_VALUES,
        quality_review=_accepted_review(),
    )
    result = project(hardened, "tool_use")

    assert result.state is AdmissionState.ACCEPTED
    assert result.output[0]["arguments"]["path"] == "<PRIVATE_PATH>"
    assert all(code != "PRIVATE_VALUE" for code, _ in scan_value(result.output))
    assert any(
        loss.code == "PRIVATE_VALUE" and loss.action is LossAction.TRANSFORM
        for loss in result.losses
    )


def test_data_mount_private_value_is_not_missed() -> None:
    record = _valid_record()
    assert record.episodes[0].events[1].action is not None
    record.episodes[0].events[1].action.arguments["path"] = "/data-sea/project"

    hardened = assess_record(
        record,
        _lineage(),
        declared_state=AdmissionState.ACCEPTED,
        quality_review=_accepted_review(),
    )

    assert any(loss.code == "PRIVATE_VALUE" for loss in hardened.losses)


def test_data_mount_private_value_is_redacted_before_tool_release() -> None:
    record = _valid_record()
    assert record.episodes[0].events[1].action is not None
    record.episodes[0].events[1].action.arguments["path"] = "/data-sea/project"

    hardened = assess_record(
        record,
        _lineage(),
        declared_state=AdmissionState.ACCEPTED,
        privacy_policy=PrivacyPolicy.REDACT_PRIVATE_VALUES,
        quality_review=_accepted_review(),
    )
    result = project(hardened, "tool_use")

    assert result.state is AdmissionState.ACCEPTED
    assert result.output[0]["arguments"]["path"] == "<PRIVATE_PATH>"
    assert all(code != "PRIVATE_VALUE" for code, _ in scan_value(result.output))
