"""Streaming bridge from published frontend rows to the hardening boundary.

Provider adapters own parsing and row-shape interpretation.  This module owns
the shared boundary after a row has been parsed: byte-ranged lineage, review
lookup, fail-closed admission, and allowlisted projection.  It deliberately
emits a safe decision envelope for every source row.  A trainer payload is
present only when the review, parent snapshot, privacy, and projection gates
all admit the unit.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentir.hardening.contracts import (
    AdmissionState,
    ParentSnapshotRef,
    PrivacyPolicy,
    QualityDimension,
    QualityReview,
    SourceClass,
    SourceLineage,
)
from agentir.hardening.firewall import assess_record
from agentir.hardening.projection import project
from agentir.hardening.streaming import compile_jsonl
from agentir.ir.record import AgentIRRecord

FrontendMapper = Callable[[dict[str, Any], str, int], tuple[AgentIRRecord, Mapping[str, Any]]]
QualityStateResolver = Callable[[dict[str, Any], Mapping[str, Any]], AdmissionState]
QualityDimensionResolver = Callable[
    [dict[str, Any], Mapping[str, Any]], dict[str, QualityDimension]
]
ReviewResolver = Callable[[str], QualityReview | None]
LineageBuilder = Callable[[dict[str, Any], AgentIRRecord, "FrontendCompileContext"], SourceLineage]


@dataclass(frozen=True)
class FrontendCompileContext:
    """Immutable source facts available to a row adapter."""

    provider: str
    source_line: int
    byte_start: int
    byte_end: int
    source: Mapping[str, Any]


_SAFE_MAPPING_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SAFE_MAPPING_COUNTERS = frozenset({"role_counts", "dropped_message_fields"})
_SAFE_MAPPING_TEXT = frozenset(
    {
        "tool_registry_revision",
        "model_tier",
        "model_tier_basis",
        "model_tier_registry_revision",
    }
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


def build_source_lineage(
    row: dict[str, Any],
    record: AgentIRRecord,
    context: FrontendCompileContext,
    *,
    source_class: SourceClass,
    parser_name: str,
    parser_revision: str,
    snapshot_revision: str,
    parent_snapshot: ParentSnapshotRef | None = None,
    source_path: str | None = None,
    source_uri: str | None = None,
) -> SourceLineage:
    """Build the shared lineage join for a mapped frontend row.

    A provider adapter may supply a more specific ``source_path`` or parent
    snapshot.  The frontend file itself is always fingerprinted by the
    compiler and ranged by source bytes; no row content is copied here.
    """

    del row
    path = source_path or str(context.source["path"])
    uri = source_uri or path
    return SourceLineage(
        source_uri=uri,
        source_path=path,
        source_class=source_class,
        provider=context.provider,
        snapshot_revision=snapshot_revision,
        source_sha256=str(context.source["sha256"]),
        source_size_bytes=int(context.source["size_bytes"]),
        byte_start=context.byte_start,
        byte_end=context.byte_end,
        message_start=context.source_line,
        message_end=context.source_line,
        parser_name=parser_name,
        parser_revision=parser_revision,
        unit_id=record.record_id,
        parent_snapshot=parent_snapshot,
    )


def _default_state(_row: dict[str, Any], _mapping: Mapping[str, Any]) -> AdmissionState:
    return AdmissionState.CANDIDATE


def _default_dimensions(
    _row: dict[str, Any], _mapping: Mapping[str, Any]
) -> dict[str, QualityDimension]:
    return {}


def _review_resolver(
    reviews: Mapping[str, QualityReview] | None,
) -> ReviewResolver:
    if reviews is None:
        return lambda _unit_id: None
    return lambda unit_id: reviews.get(unit_id)


def _safe_mapping_summary(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only bounded counters from an adapter diagnostic summary."""

    summary: dict[str, Any] = {}
    for key in _SAFE_MAPPING_NUMBERS:
        value = mapping.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            summary[key] = value
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
            summary[key] = dict(sorted(counter.items()))
    for key in _SAFE_MAPPING_TEXT:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip() and len(value) <= 256:
            summary[key] = value.strip()
    return dict(sorted(summary.items()))


def _losses(losses: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [loss.model_dump(mode="json", exclude_none=True) for loss in losses]


def _lineage_summary(lineage: SourceLineage) -> dict[str, Any]:
    return lineage.model_dump(mode="json", exclude_none=True)


def _failure_envelope(
    *,
    context: FrontendCompileContext,
    stage: str,
    error: Exception,
) -> dict[str, Any]:
    error_type = type(error).__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", error_type):
        error_type = "UnhandledError"
    return {
        "schema_version": "agentir/hardening/projected-unit/v1",
        "provider": context.provider,
        "source_line": context.source_line,
        "source_range": {
            "byte_start": context.byte_start,
            "byte_end": context.byte_end,
            "message_start": context.source_line,
            "message_end": context.source_line,
        },
        "source": {
            "path": context.source["path"],
            "sha256": context.source["sha256"],
            "size_bytes": context.source["size_bytes"],
        },
        "admission": {
            "state": AdmissionState.QUARANTINED.value,
            "failure_stage": stage,
            "failure_type": error_type,
        },
        "mapping": {},
        "projections": {},
    }


def compile_frontend_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    provider: str,
    map_row: FrontendMapper,
    build_lineage: LineageBuilder,
    parser_revision: str,
    pass_config: Mapping[str, Any],
    quality_state: QualityStateResolver | None = None,
    quality_dimensions: QualityDimensionResolver | None = None,
    reviews: Mapping[str, QualityReview] | None = None,
    privacy_policy: PrivacyPolicy = PrivacyPolicy.QUARANTINE,
    targets: tuple[str, ...] = ("sft", "tool_use"),
    shard_records: int = 1000,
    shard_bytes: int = 64 * 1024 * 1024,
    resume: bool = False,
    stop_after_source_records: int | None = None,
    source_record_limit: int | None = None,
    source_line_start: int = 1,
) -> dict[str, Any]:
    """Stream mapped frontend rows through the canonical hardening funnel.

    Mapping and lineage failures are represented as quarantined decision rows
    so a single malformed provider record does not silently become trainer
    data or force a whole-corpus in-memory retry.  Invalid JSON still fails at
    the source reader because there is no trustworthy row identity to attach.
    """

    if not provider.strip():
        raise ValueError("provider must be non-empty")
    if not targets:
        raise ValueError("at least one projection target is required")
    resolve_state = quality_state or _default_state
    resolve_dimensions = quality_dimensions or _default_dimensions
    resolve_review = _review_resolver(reviews)

    def transform(
        row: dict[str, Any],
        source_line: int,
        byte_start: int,
        byte_end: int,
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        context = FrontendCompileContext(
            provider=provider,
            source_line=source_line,
            byte_start=byte_start,
            byte_end=byte_end,
            source=source,
        )
        try:
            record, mapping = map_row(row, provider, source_line)
        except Exception as error:  # fail closed at the row boundary
            return _failure_envelope(context=context, stage="mapping", error=error)

        try:
            lineage = build_lineage(row, record, context)
            review = resolve_review(record.record_id)
            hardened = assess_record(
                record,
                lineage,
                declared_state=resolve_state(row, mapping),
                dimensions=resolve_dimensions(row, mapping),
                privacy_policy=privacy_policy,
                quality_review=review,
            )
            projections: dict[str, Any] = {}
            for target in targets:
                result = project(hardened, target)
                projections[target] = {
                    "state": result.state.value,
                    "output": result.output if result.state is AdmissionState.ACCEPTED else [],
                    "losses": _losses(result.losses),
                    "emitted_events": result.emitted_events,
                    "omitted_events": result.omitted_events,
                }
        except Exception as error:  # fail closed at the release boundary
            return _failure_envelope(context=context, stage="hardening", error=error)

        return {
            "schema_version": "agentir/hardening/projected-unit/v1",
            "unit_id": record.record_id,
            "provider": provider,
            "source_line": source_line,
            "lineage": _lineage_summary(lineage),
            "admission": {
                "state": hardened.quality.state.value,
                "rule_ids": list(hardened.quality.rule_ids),
                "review_id": review.review_id if review is not None else None,
            },
            "mapping": _safe_mapping_summary(mapping),
            "losses": _losses(hardened.losses),
            "projections": projections,
        }

    config = dict(pass_config)
    config.update(
        {
            "provider": provider,
            "parser_revision": parser_revision,
            "privacy_policy": privacy_policy.value,
            "targets": list(targets),
        }
    )
    return compile_jsonl(
        input_path,
        output_dir,
        lambda _row, _source_line: None,
        transform_with_context=transform,
        parser_revision=parser_revision,
        pass_config=config,
        shard_records=shard_records,
        shard_bytes=shard_bytes,
        resume=resume,
        stop_after_source_records=stop_after_source_records,
        source_record_limit=source_record_limit,
        source_line_start=source_line_start,
    )
