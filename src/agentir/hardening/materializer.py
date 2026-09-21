"""Disk-backed materialization from decision envelopes to typed release rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agentir.hardening.contracts import ProjectionResult
from agentir.hardening.release_views import (
    ModelTier,
    ReleaseMetadata,
    ReleaseView,
    SFTRelease,
    ToolDefinition,
    ToolUseRelease,
    sft_release_from_projection,
    tool_use_release_from_projection,
)
from agentir.hardening.streaming import compile_jsonl
from agentir.hardening.tool_registry import ToolRegistryArtifact, registry_tools

RELEASE_MATERIALIZER_REVISION = "agentir/hardening/release-materializer/v1"
MetadataResolver = Callable[[Mapping[str, Any], ReleaseView], ReleaseMetadata | None]
ToolResolver = Callable[[Mapping[str, Any], ReleaseView], tuple[ToolDefinition, ...]]
RegistryResolver = Callable[[Mapping[str, Any], ReleaseView], ToolRegistryArtifact | None]


def _empty_tools(_envelope: Mapping[str, Any], _target: ReleaseView) -> tuple[ToolDefinition, ...]:
    return ()


def _projection_result(envelope: Mapping[str, Any], target: ReleaseView) -> ProjectionResult | None:
    admission = envelope.get("admission")
    if not isinstance(admission, Mapping) or admission.get("state") != "accepted":
        return None
    projections = envelope.get("projections")
    if not isinstance(projections, Mapping):
        raise ValueError("accepted envelope has no projection map")
    value = projections.get(target.value)
    if not isinstance(value, Mapping):
        raise ValueError(f"accepted envelope has no {target.value} projection")
    if value.get("state") != "accepted":
        return None
    return ProjectionResult.model_validate(
        {
            "target": target.value,
            "state": value.get("state"),
            "output": value.get("output", []),
            "losses": value.get("losses", []),
            "emitted_events": value.get("emitted_events", 0),
            "omitted_events": value.get("omitted_events", 0),
        }
    )


def materialize_release_view(
    input_path: Path,
    output_dir: Path,
    *,
    target: ReleaseView,
    metadata_for: MetadataResolver,
    tools_for: ToolResolver | None = None,
    registry_for: RegistryResolver | None = None,
    release_revision: str,
    required_model_tier: ModelTier | None = None,
    shard_records: int = 1000,
    shard_bytes: int = 64 * 1024 * 1024,
    resume: bool = False,
    stop_after_source_records: int | None = None,
    source_record_limit: int | None = None,
    source_line_start: int = 1,
) -> dict[str, Any]:
    """Stream accepted decision envelopes into one typed release view.

    Non-accepted envelopes are filtered without retaining their payload. An
    accepted envelope without release metadata or a required tool registry
    raises before any trainer row is emitted.
    """

    try:
        target = ReleaseView(target)
    except ValueError as error:
        raise ValueError("unsupported release view") from error
    if target not in {ReleaseView.SFT, ReleaseView.TOOL_USE}:
        raise ValueError("historical envelope materialization supports only sft and tool_use")
    if not release_revision.strip():
        raise ValueError("release_revision must be non-empty")
    if required_model_tier is not None:
        try:
            required_model_tier = ModelTier(required_model_tier)
        except ValueError as error:
            raise ValueError("unsupported required model tier") from error
    if tools_for is not None and registry_for is not None:
        raise ValueError("provide either tools_for or registry_for, not both")

    def resolve_tools(row: Mapping[str, Any], view: ReleaseView) -> tuple[ToolDefinition, ...]:
        if registry_for is not None:
            artifact = registry_for(row, view)
            return registry_tools(artifact) if artifact is not None else ()
        return tools_for(row, view) if tools_for is not None else _empty_tools(row, view)

    def transform(row: dict[str, Any], _source_line: int) -> dict[str, Any] | None:
        projection = _projection_result(row, target)
        if projection is None:
            return None
        metadata = metadata_for(row, target)
        if metadata is None:
            raise ValueError("accepted envelope has no release metadata decision")
        if metadata.release_revision != release_revision:
            raise ValueError("release metadata revision does not match materializer")
        if required_model_tier is not None and metadata.model_tier != required_model_tier:
            raise ValueError(
                "release metadata model tier does not match materializer: "
                f"expected {required_model_tier.value}, got {metadata.model_tier.value}"
            )
        tools = resolve_tools(row, target)
        release: SFTRelease | ToolUseRelease
        if target is ReleaseView.SFT:
            release = sft_release_from_projection(
                projection,
                metadata=metadata,
                tools=tools,
            )
        else:
            release = tool_use_release_from_projection(
                projection,
                metadata=metadata,
                tools=tools,
            )
        return release.model_dump(mode="json")

    return compile_jsonl(
        input_path,
        output_dir,
        transform,
        parser_revision=RELEASE_MATERIALIZER_REVISION,
        pass_config={
            "release_revision": release_revision,
            "required_model_tier": (
                required_model_tier.value if required_model_tier is not None else None
            ),
            "target": target.value,
            "schema_version": f"agentir/hardening/release/{target.value}/v1",
        },
        shard_records=shard_records,
        shard_bytes=shard_bytes,
        resume=resume,
        stop_after_source_records=stop_after_source_records,
        source_record_limit=source_record_limit,
        source_line_start=source_line_start,
    )
