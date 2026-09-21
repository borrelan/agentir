"""Metadata-only review task generation for projected decision envelopes."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from agentir.hardening.contracts import REQUIRED_QUALITY_DIMENSIONS

RUBRIC_REVISION = "quality/v1"
TASK_SCHEMA = "agentir/hardening/quality-review-tasks/v1"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Narrow untrusted JSON containers before reading their fields."""

    return value if isinstance(value, Mapping) else {}


def _as_items(value: Any) -> Iterable[Any]:
    """Return a safe iterable for optional JSON arrays."""

    return value if isinstance(value, (list, tuple)) else ()


def _task_id(snapshot_revision: str, envelope: Mapping[str, Any]) -> str:
    lineage = _as_mapping(envelope.get("lineage"))
    source_range = _as_mapping(lineage.get("source_range"))
    identity = "|".join(
        str(value)
        for value in (
            snapshot_revision,
            envelope.get("unit_id", ""),
            lineage.get("source_sha256", ""),
            source_range.get("byte_start", ""),
            source_range.get("byte_end", ""),
            lineage.get("parser_revision", ""),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _loss_codes(envelope: Mapping[str, Any]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for loss in _as_items(envelope.get("losses", ())):
        if isinstance(loss, Mapping) and isinstance(loss.get("code"), str):
            counts[loss["code"]] += 1
    projections = envelope.get("projections")
    if isinstance(projections, Mapping):
        for target, projection in projections.items():
            if not isinstance(target, str) or not isinstance(projection, Mapping):
                continue
            for loss in _as_items(projection.get("losses", ())):
                if isinstance(loss, Mapping) and isinstance(loss.get("code"), str):
                    counts[f"{target}:{loss['code']}"] += 1
    return dict(sorted(counts.items()))


def _source_locator(snapshot_revision: str, envelope: Mapping[str, Any]) -> dict[str, Any]:
    lineage = _as_mapping(envelope.get("lineage"))
    parent = lineage.get("parent_snapshot")
    parent_summary: dict[str, Any] = {}
    if isinstance(parent, Mapping):
        parent_summary = {
            "object_id": parent.get("object_id"),
            "source_sha256": parent.get("source_sha256"),
            "source_size_bytes": parent.get("source_size_bytes"),
            "manifest_verified": parent.get("manifest_verified", False),
            "root_label": parent.get("root_label"),
        }
    return {
        "unit_id": envelope.get("unit_id"),
        "provider": envelope.get("provider"),
        "source_path": lineage.get("source_path"),
        "source_line": envelope.get("source_line"),
        "byte_start": lineage.get("byte_start"),
        "byte_end": lineage.get("byte_end"),
        "source_sha256": lineage.get("source_sha256"),
        "source_size_bytes": lineage.get("source_size_bytes"),
        "snapshot_revision": snapshot_revision,
        "parser_revision": lineage.get("parser_revision"),
        "parent_snapshot": parent_summary,
    }


def _task(
    snapshot_revision: str,
    manifest_revision: str,
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    admission = _as_mapping(envelope.get("admission"))
    projections = _as_mapping(envelope.get("projections"))
    projection_states = {
        target: projection.get("state")
        for target, projection in projections.items()
        if isinstance(target, str) and isinstance(projection, Mapping)
    }
    return {
        "task_id": _task_id(snapshot_revision, envelope),
        "status": "unassigned",
        "source_locator": _source_locator(snapshot_revision, envelope),
        "observed": {
            "admission_state": admission.get("state"),
            "rule_ids": sorted(admission.get("rule_ids", ()))
            if isinstance(admission.get("rule_ids"), (list, tuple))
            else [],
            "mapping": envelope.get("mapping", {}),
            "loss_codes": _loss_codes(envelope),
            "projection_states": projection_states,
            "model_tier": envelope.get("model_tier"),
            "model_tier_basis": envelope.get("model_tier_basis"),
            "model_tier_registry_revision": envelope.get(
                "model_tier_registry_revision"
            ),
        },
        "review": {
            "manifest_revision": manifest_revision,
            "snapshot_revision": snapshot_revision,
            "rubric_revision": RUBRIC_REVISION,
            "decision": None,
            "reviewer_id": None,
            "evidence_ids": [],
            "dimensions": {dimension: None for dimension in sorted(REQUIRED_QUALITY_DIMENSIONS)},
        },
        "release_rule": "unassigned_review_is_not_trainer_data",
    }


def build_review_tasks(
    envelopes: Iterable[Mapping[str, Any]],
    *,
    snapshot_revision: str,
    manifest_revision: str,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Build deterministic, content-free tasks from decision envelopes.

    Selection is round-robin across admission/loss buckets so a bounded queue
    does not silently become a provider-only or clean-only sample.  No review
    decision is inferred from the observed state.
    """

    buckets: defaultdict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for envelope in envelopes:
        admission = _as_mapping(envelope.get("admission"))
        state = str(admission.get("state", "unknown"))
        losses = tuple(sorted(_loss_codes(envelope)))
        buckets[(state, losses)].append(envelope)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: str(item.get("unit_id", "")))

    selected: list[Mapping[str, Any]] = []
    keys = sorted(buckets)
    while keys and (max_tasks is None or len(selected) < max_tasks):
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket and (max_tasks is None or len(selected) < max_tasks):
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys

    tasks = [_task(snapshot_revision, manifest_revision, envelope) for envelope in selected]
    tasks.sort(key=lambda task: task["task_id"])
    return {
        "schema_version": TASK_SCHEMA,
        "snapshot_revision": snapshot_revision,
        "manifest_revision": manifest_revision,
        "rubric_revision": RUBRIC_REVISION,
        "retention": "metadata_only_no_session_content",
        "required_dimensions": sorted(REQUIRED_QUALITY_DIMENSIONS),
        "summary": {
            "tasks": len(tasks),
            "assigned": 0,
            "accepted": 0,
            "quarantined": 0,
            "rejected": 0,
        },
        "tasks": tasks,
    }
