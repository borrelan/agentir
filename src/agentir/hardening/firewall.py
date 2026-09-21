"""Recursive fail-closed screening before trainer projection."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from agentir.hardening.contracts import (
    AdmissionState,
    HardenedRecord,
    LossAction,
    LossDecision,
    PrivacyPolicy,
    QualityDecision,
    QualityDimension,
    QualityReview,
    SourceLineage,
    SourceRange,
)
from agentir.ir.base import EventType
from agentir.ir.record import AgentIRRecord

_HIDDEN_KEY_RE = re.compile(
    r"(?:^|[_-])(thoughts?|thinking|reasoning|analysis|deliberation|"
    r"scratchpad|cot|chain[_-]?of[_-]?thought|internal[_-]?monologue)(?:$|[_-])",
    re.IGNORECASE,
)
_MARKER_RE = re.compile(
    r"<\s*/?\s*(?:thinking|analysis|reasoning|deliberation|scratchpad|"
    r"chain[_ -]?of[_ -]?thought)\b|"
    r"\b(?:chain of thought|hidden reasoning|internal reasoning|"
    r"private scratchpad)\b",
    re.IGNORECASE,
)
_RAW_KEYS = frozenset({"raw", "extra", "debug", "stack_trace", "stacktrace"})
_SECRET_RE = re.compile(
    r"-----BEGIN [^-]{2,80} PRIVATE KEY-----|"
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|"
    r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b",
    re.IGNORECASE,
)
PRIVATE_PATH_PATTERN = (
    r"(?<![A-Za-z0-9_])/(?:home|root|Users|private|"
    r"data(?:-[A-Za-z0-9._-]+)?|tmp|var|opt|mnt|workspace|srv)/[^\s\"']+"
)
_PRIVATE_VALUE_RE = re.compile(
    PRIVATE_PATH_PATTERN + "|"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)


def _iter_findings(value: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    """Yield code/path pairs without retaining the matched value."""

    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            normalized = key_text.strip().lower()
            if _HIDDEN_KEY_RE.search(normalized):
                yield "HIDDEN_FIELD", child_path
            if normalized in _RAW_KEYS:
                yield "UNTRUSTED_FIELD", child_path
            yield from _iter_findings(child, child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_findings(child, f"{path}[{index}]")
        return
    if isinstance(value, str):
        if _MARKER_RE.search(value):
            yield "HIDDEN_MARKER", path
        if _SECRET_RE.search(value):
            yield "SECRET_PATTERN", path
        if _PRIVATE_VALUE_RE.search(value):
            yield "PRIVATE_VALUE", path


def scan_value(value: Any) -> tuple[tuple[str, str], ...]:
    """Return countable finding locations; never return protected values."""

    return tuple(_iter_findings(value))


def _record_content_values(record: AgentIRRecord) -> Iterator[tuple[str, Any]]:
    if record.raw:
        yield "record.raw", {"raw": record.raw}
    if record.model_extra:
        yield "record.extra", {"extra": record.model_extra}
    if record.task is not None:
        yield "task.instruction", record.task.instruction
        yield "task.metadata", record.task.metadata
    for index, tool in enumerate(record.tool_registry):
        yield f"tool_registry[{index}]", tool.model_dump(mode="python")
    for episode_index, episode in enumerate(record.episodes):
        for event_index, event in enumerate(episode.events):
            prefix = f"episodes[{episode_index}].events[{event_index}]"
            if event.raw:
                yield f"{prefix}.raw", {"raw": event.raw}
            if event.model_extra:
                yield f"{prefix}.extra", {"extra": event.model_extra}
            yield f"{prefix}.content", [block.model_dump(mode="python") for block in event.content]
            yield (
                f"{prefix}.action",
                event.action.model_dump(mode="python") if event.action else None,
            )
            if event.observation is not None:
                yield f"{prefix}.observation", event.observation.model_dump(mode="python")
            yield f"{prefix}.metadata", event.metadata
    if record.outcome is not None:
        yield "outcome", record.outcome.model_dump(mode="python")


def _edge_losses(record: AgentIRRecord) -> list[LossDecision]:
    calls: dict[str, str] = {}
    results: dict[str, str] = {}
    losses: list[LossDecision] = []
    for episode in record.episodes:
        for event in episode.events:
            action = event.action
            if event.event_type == EventType.TOOL_CALL:
                if action is None or not action.tool_call_id:
                    losses.append(
                        LossDecision(
                            code="MISSING_TOOL_CALL_ID",
                            action=LossAction.QUARANTINE,
                            event_id=event.event_id,
                            field="action.tool_call_id",
                            message="Tool call identity is required before trainer lowering.",
                        )
                    )
                    continue
                call_id = action.tool_call_id
                if call_id in calls:
                    losses.append(
                        LossDecision(
                            code="DUPLICATE_TOOL_CALL_ID",
                            action=LossAction.QUARANTINE,
                            event_id=event.event_id,
                            field="action.tool_call_id",
                            message="Tool call identity is duplicated within the unit.",
                        )
                    )
                calls[call_id] = event.event_id
            elif event.event_type == EventType.TOOL_RESULT:
                if action is None or not action.tool_call_id:
                    losses.append(
                        LossDecision(
                            code="MISSING_TOOL_RESULT_ID",
                            action=LossAction.QUARANTINE,
                            event_id=event.event_id,
                            field="action.tool_call_id",
                            message="Tool result identity is required before trainer lowering.",
                        )
                    )
                    continue
                result_id = action.tool_call_id
                if result_id in results:
                    losses.append(
                        LossDecision(
                            code="DUPLICATE_TOOL_RESULT_ID",
                            action=LossAction.QUARANTINE,
                            event_id=event.event_id,
                            field="action.tool_call_id",
                            message="Tool result identity is duplicated within the unit.",
                        )
                    )
                results[result_id] = event.event_id
    for call_id, event_id in sorted(calls.items()):
        if call_id not in results:
            losses.append(
                LossDecision(
                    code="MISSING_TOOL_RESULT",
                    action=LossAction.QUARANTINE,
                    event_id=event_id,
                    field="action.tool_call_id",
                    message="Tool call has no linked result in the unit.",
                )
            )
    for result_id, event_id in sorted(results.items()):
        if result_id not in calls:
            losses.append(
                LossDecision(
                    code="ORPHAN_TOOL_RESULT",
                    action=LossAction.QUARANTINE,
                    event_id=event_id,
                    field="action.tool_call_id",
                    message="Tool result has no linked call in the unit.",
                )
            )
    return losses


def assess_record(
    record: AgentIRRecord,
    lineage: SourceLineage,
    *,
    declared_state: AdmissionState = AdmissionState.CANDIDATE,
    dimensions: dict[str, QualityDimension] | None = None,
    privacy_policy: PrivacyPolicy = PrivacyPolicy.QUARANTINE,
    quality_review: QualityReview | None = None,
) -> HardenedRecord:
    """Attach typed lineage and a fail-closed, provider-independent decision."""

    losses: list[LossDecision] = []
    for _field, value in _record_content_values(record):
        for code, path in scan_value(value):
            action = LossAction.QUARANTINE
            if code == "PRIVATE_VALUE" and privacy_policy == PrivacyPolicy.REDACT_PRIVATE_VALUES:
                action = LossAction.TRANSFORM
            losses.append(
                LossDecision(
                    code=code,
                    action=action,
                    field=path,
                    message="Protected or untrusted content was detected before projection.",
                )
            )
    losses.extend(_edge_losses(record))

    if quality_review is not None and quality_review.unit_id != lineage.unit_id:
        losses.append(
            LossDecision(
                code="QUALITY_REVIEW_MISMATCH",
                action=LossAction.QUARANTINE,
                field="quality_review.unit_id",
                message="Quality review evidence is bound to a different unit.",
            )
        )
    if quality_review is not None and quality_review.snapshot_revision != lineage.snapshot_revision:
        losses.append(
            LossDecision(
                code="QUALITY_REVIEW_SNAPSHOT_MISMATCH",
                action=LossAction.QUARANTINE,
                field="quality_review.snapshot_revision",
                message="Quality review evidence belongs to a different source snapshot.",
            )
        )
    if quality_review is not None and quality_review.decision == "accepted":
        parent_error = (
            "Accepted rows must bind to an immutable raw snapshot object."
            if lineage.parent_snapshot is None
            else (
                "Accepted rows must bind to an object verified by the snapshot manifest."
                if not lineage.parent_snapshot.manifest_verified
                else None
            )
        )
        parent_code = (
            "PARENT_SNAPSHOT_UNBOUND"
            if lineage.parent_snapshot is None
            else "PARENT_SNAPSHOT_UNVERIFIED"
        )
        if parent_error is not None:
            losses.append(
                LossDecision(
                    code=parent_code,
                    action=LossAction.QUARANTINE,
                    field="lineage.parent_snapshot",
                    message=parent_error,
                )
            )

    event_indices = {
        event.event_id: event.provenance.row_index
        for episode in record.episodes
        for event in episode.events
        if event.provenance is not None
    }
    ranged_losses = [
        loss.model_copy(
            update={
                "source_range": SourceRange(
                    byte_start=lineage.byte_start,
                    byte_end=lineage.byte_end,
                    message_start=(
                        event_indices.get(loss.event_id) if loss.event_id is not None else None
                    ),
                    message_end=(
                        event_indices.get(loss.event_id) if loss.event_id is not None else None
                    ),
                )
            }
        )
        for loss in losses
    ]
    blocking_losses = [
        loss for loss in ranged_losses if loss.action in {LossAction.QUARANTINE, LossAction.REJECT}
    ]
    if declared_state == AdmissionState.REJECTED:
        state = AdmissionState.REJECTED
    elif blocking_losses:
        state = AdmissionState.QUARANTINED
    elif quality_review is not None and quality_review.decision == "rejected":
        state = AdmissionState.REJECTED
    elif quality_review is not None and quality_review.decision == "quarantined":
        state = AdmissionState.QUARANTINED
    elif quality_review is not None and quality_review.decision == "accepted":
        state = (
            AdmissionState.QUARANTINED
            if declared_state == AdmissionState.QUARANTINED
            else AdmissionState.ACCEPTED
        )
    elif declared_state == AdmissionState.ACCEPTED:
        # A source/provider assertion is not review evidence.
        state = AdmissionState.CANDIDATE
    else:
        state = declared_state

    rule_ids = tuple(sorted({loss.code for loss in ranged_losses}))
    evidence_ids = quality_review.evidence_ids if quality_review is not None else (lineage.unit_id,)
    quality = QualityDecision(
        state=state,
        dimensions={
            key: value
            for key, value in (
                quality_review.dimensions if quality_review is not None else (dimensions or {})
            ).items()
        },
        rule_ids=rule_ids,
        evidence_ids=evidence_ids,
        review=quality_review,
    )
    return HardenedRecord(
        record=record,
        lineage=lineage,
        quality=quality,
        privacy_policy=privacy_policy,
        losses=tuple(ranged_losses),
    )


def admission_losses(hardened: HardenedRecord) -> tuple[LossDecision, ...]:
    """Recompute non-content admission losses for projection reporting."""

    return hardened.losses
