"""Bounded, redacted review contexts that cannot become release rows.

Review tasks intentionally contain no session content.  This module provides
an on-demand bridge for an authorized reviewer: it accepts already-mapped
hardening outputs, keeps only the allowlisted projections and registry, and
marks any bounded content as review-only.  It does not create a quality
decision or call the release materializer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from agentir.hardening.contracts import (
    REQUIRED_QUALITY_DIMENSIONS,
    HardenedRecord,
    ProjectionResult,
)
from agentir.hardening.firewall import scan_value
from agentir.hardening.projection import redact_value
from agentir.hardening.tool_registry import ToolRegistryArtifact

REVIEW_CONTEXT_SCHEMA = "agentir/hardening/review-context/v1"
DEFAULT_MAX_STRING_BYTES = 256 * 1024
DEFAULT_MAX_CONTEXT_BYTES = 2 * 1024 * 1024
_BLOCKING_PREVIEW_CODES = frozenset(
    {"HIDDEN_FIELD", "UNTRUSTED_FIELD", "HIDDEN_MARKER", "SECRET_PATTERN", "PRIVATE_VALUE"}
)
_SAFE_MAPPING_NUMBERS = frozenset(
    {
        "message_count",
        "call_count",
        "result_count",
        "missing_call_ids",
        "missing_result_ids",
        "unmatched_call_ids",
        "unmatched_result_ids",
        "matched_tool_edges",
        "unknown_roles",
        "tool_registry_count",
        "malformed_tool_definitions",
    }
)
_SAFE_MAPPING_COUNTERS = frozenset({"role_counts", "dropped_message_fields"})
_SAFE_MAPPING_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SAFE_MODEL_TIER = frozenset(
    {"tier1_frontier", "tier2_open_source", "tier3_local", "unclassified"}
)
_SAFE_MODEL_METADATA = frozenset(
    {"model_tier_basis", "model_tier_registry_revision"}
)


class ReviewContextError(ValueError):
    """Raised when a review preview cannot satisfy the bounded contract."""


class _Bounder:
    def __init__(self, *, max_string_bytes: int, max_total_bytes: int) -> None:
        if max_string_bytes <= 0 or max_total_bytes <= 0:
            raise ValueError("review context bounds must be positive")
        self.max_string_bytes = max_string_bytes
        self.max_total_bytes = max_total_bytes
        self.used_bytes = 0
        self.truncated = False
        self.truncated_paths: list[str] = []

    def _marker(self, value: str, path: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        self.truncated = True
        if len(self.truncated_paths) < 100:
            self.truncated_paths.append(path)
        return f"<REVIEW_TRUNCATED sha256:{digest}>"

    def value(self, value: Any, path: str = "$") -> Any:
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            remaining = self.max_total_bytes - self.used_bytes
            if len(encoded) > self.max_string_bytes or len(encoded) > remaining:
                bounded = self._marker(value, path)
                self.used_bytes += len(bounded.encode("utf-8"))
                return bounded
            self.used_bytes += len(encoded)
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self.value(child, f"{path}.{key}")
                for key, child in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.value(child, f"{path}[{index}]") for index, child in enumerate(value)]
        return value


def _safe_losses(losses: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Keep loss identity while dropping free-form diagnostic messages."""

    return [
        {
            key: value
            for key, value in loss.model_dump(mode="json", exclude_none=True).items()
            if key != "message"
        }
        for loss in losses
    ]


def _safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Retain bounded adapter counters, never arbitrary provider metadata."""

    result: dict[str, Any] = {}
    for key in _SAFE_MAPPING_NUMBERS:
        value = mapping.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = value
    for key in _SAFE_MAPPING_COUNTERS:
        value = mapping.get(key)
        if not isinstance(value, Mapping):
            continue
        counter: dict[str, int] = {}
        for counter_key, counter_value in value.items():
            if (
                isinstance(counter_key, str)
                and _SAFE_MAPPING_KEY.fullmatch(counter_key)
                and isinstance(counter_value, int)
                and not isinstance(counter_value, bool)
                and counter_value >= 0
            ):
                counter[counter_key] = counter_value
        if counter:
            result[key] = dict(sorted(counter.items()))
    revision = mapping.get("tool_registry_revision")
    if isinstance(revision, str) and revision.strip():
        result["tool_registry_revision"] = revision.strip()
    tier = mapping.get("model_tier")
    if isinstance(tier, str) and tier in _SAFE_MODEL_TIER:
        result["model_tier"] = tier
    for key in _SAFE_MODEL_METADATA:
        value = mapping.get(key)
        if (
            isinstance(value, str)
            and value.strip()
            and len(value) <= 160
            and _SAFE_MAPPING_KEY.fullmatch(value.strip())
        ):
            result[key] = value.strip()
    return dict(sorted(result.items()))


def _safe_lineage(hardened: HardenedRecord) -> dict[str, Any]:
    lineage = hardened.lineage
    parent = lineage.parent_snapshot
    result: dict[str, Any] = {
        "unit_id": lineage.unit_id,
        "provider": lineage.provider,
        "source_class": lineage.source_class.value,
        "snapshot_revision": lineage.snapshot_revision,
        "source_sha256": lineage.source_sha256,
        "source_size_bytes": lineage.source_size_bytes,
        "source_range": {
            "byte_start": lineage.byte_start,
            "byte_end": lineage.byte_end,
            "message_start": lineage.message_start,
            "message_end": lineage.message_end,
        },
        "parser_name": lineage.parser_name,
        "parser_revision": lineage.parser_revision,
        "parent_snapshot": None,
    }
    if parent is not None:
        result["parent_snapshot"] = {
            "snapshot_revision": parent.snapshot_revision,
            "source_sha256": parent.source_sha256,
            "source_size_bytes": parent.source_size_bytes,
            "root_label": parent.root_label,
            "message_start": parent.message_start,
            "message_end": parent.message_end,
            "manifest_verified": parent.manifest_verified,
            "object_id_sha256": hashlib.sha256(parent.object_id.encode("utf-8")).hexdigest(),
        }
    return result


def _record_summary(hardened: HardenedRecord) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    event_count = 0
    for episode in hardened.record.episodes:
        for event in episode.events:
            counts[event.event_type.value] += 1
            event_count += 1
    return {
        "episode_count": len(hardened.record.episodes),
        "event_count": event_count,
        "event_type_counts": dict(sorted(counts.items())),
        "tool_registry_count": len(hardened.record.tool_registry),
    }


def _bounded_projection(
    result: ProjectionResult,
    bounder: _Bounder,
    *,
    path: str,
) -> dict[str, Any]:
    findings = [
        finding
        for finding in scan_value(result.output)
        if finding[0] in _BLOCKING_PREVIEW_CODES
    ]
    if findings:
        raise ReviewContextError(
            "projection contains blocked review content: "
            + ", ".join(sorted({code for code, _path in findings}))
        )
    return {
        "state": result.state.value,
        "output": bounder.value(redact_value(result.output), f"{path}.output"),
        "losses": _safe_losses(result.losses),
        "emitted_events": result.emitted_events,
        "omitted_events": result.omitted_events,
    }


def build_review_context(
    *,
    task_id: str,
    mapping: Mapping[str, Any],
    hardened: HardenedRecord,
    projections: Mapping[str, ProjectionResult],
    registry: ToolRegistryArtifact | None,
    max_string_bytes: int = DEFAULT_MAX_STRING_BYTES,
    max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
) -> dict[str, Any]:
    """Build one review-only context from already-validated hardening results.

    The returned object intentionally has no source path, raw record, model
    reasoning, provider envelope, or inferred quality labels.  A caller may
    persist it under a task-local directory for authorized review; it is not
    a trainer release and has no release metadata.
    """

    if not task_id.strip():
        raise ValueError("task_id must be non-empty")
    if hardened.quality.review is not None:
        raise ReviewContextError("review context must not contain an existing quality decision")
    if set(projections).difference({"sft", "tool_use"}):
        raise ReviewContextError("review context received an unsupported projection")
    if registry is not None:
        definitions = registry.definitions
        registry_payload: dict[str, Any] | None = {
            "registry_revision": registry.registry_revision,
            "artifact_sha256": registry.artifact_sha256,
            "source_revision": registry.source_revision,
            "scope": registry.scope,
            "origin": registry.origin.value,
            "definitions": [definition.model_dump(mode="json") for definition in definitions],
        }
    else:
        registry_payload = None

    bounder = _Bounder(
        max_string_bytes=max_string_bytes,
        max_total_bytes=max_context_bytes,
    )
    result: dict[str, Any] = {
        "schema_version": REVIEW_CONTEXT_SCHEMA,
        "task_id": task_id,
        "review_only": True,
        "decision": None,
        "quality_dimensions": {
            dimension: None for dimension in sorted(REQUIRED_QUALITY_DIMENSIONS)
        },
        "lineage": _safe_lineage(hardened),
        "record_summary": _record_summary(hardened),
        "mapping": _safe_mapping(mapping),
        "admission": {
            "state": hardened.quality.state.value,
            "rule_ids": list(hardened.quality.rule_ids),
            "losses": _safe_losses(hardened.losses),
        },
        "tool_registry": registry_payload,
        "projections": {},
        "content_policy": {
            "reasoning": "typed_and_hidden_reasoning_excluded",
            "privacy": "private_values_redacted_or_blocked",
            "raw_fields": "excluded",
            "provider_envelope": "excluded",
        },
    }
    for target in ("sft", "tool_use"):
        projection = projections.get(target)
        if projection is not None:
            result["projections"][target] = _bounded_projection(
                projection,
                bounder,
                path=f"$.projections.{target}",
            )
    result["bounds"] = {
        "max_string_bytes": max_string_bytes,
        "max_context_bytes": max_context_bytes,
        "content_truncated": bounder.truncated,
        "truncated_paths": bounder.truncated_paths,
    }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(encoded) > max_context_bytes:
        raise ReviewContextError("review context exceeded max_context_bytes after bounding")
    return result
