"""Acquire a digest-bound tool registry from a harness trace.

The historical session adapters cannot reconstruct a tool schema from a tool
name.  A live harness trace is different: it records the exact registry that
was exposed for one episode.  This adapter validates that evidence in one
streaming pass and converts it to the hardening package's canonical registry
artifact.

This module is intentionally an adapter, not a second registry owner.  The
harness owns capture; :mod:`tool_registry` owns the release artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentir.hardening.release_views import ToolDefinition
from agentir.hardening.tool_registry import (
    RegistryOrigin,
    ToolRegistryArtifact,
    tool_registry_artifact,
)

HARNESS_TRACE_SCHEMA = "ai-data-extraction/harness-trace/v1"
HARNESS_REGISTRY_EVIDENCE_SCHEMA = "agentir/hardening/harness-registry-evidence/v1"
HARNESS_REGISTRY_ADAPTER_REVISION = "agentir/hardening/harness-registry-adapter/v1"
DEFAULT_MAX_TRACE_LINE_BYTES = 64 * 1024 * 1024

_EVENT_TYPES = {
    "skill_preflight",
    "tool_registry",
    "mcp_health",
    "decision",
    "tool_call",
    "tool_observation",
    "state_delta",
    "verification",
    "terminal",
    "loop_guard",
}


class HarnessRegistryError(ValueError):
    """Raised when a trace cannot prove one coherent runtime registry."""


@dataclass(frozen=True)
class HarnessRegistryEvidence:
    """Validated registry evidence and its trace-level join identities."""

    episode_id: str
    trace_sha256: str
    registry_event_id: str
    registry_revision: str
    harness_registry_sha256: str
    tool_call_count: int
    artifact: ToolRegistryArtifact

    def manifest(self) -> dict[str, Any]:
        """Return metadata safe to persist beside the registry artifact."""

        return {
            "schema_version": HARNESS_REGISTRY_EVIDENCE_SCHEMA,
            "adapter_revision": HARNESS_REGISTRY_ADAPTER_REVISION,
            "episode_id": self.episode_id,
            "trace_sha256": self.trace_sha256,
            "registry_event_id": self.registry_event_id,
            "registry_revision": self.registry_revision,
            "harness_registry_sha256": self.harness_registry_sha256,
            "registry_artifact_sha256": self.artifact.artifact_sha256,
            "tool_count": len(self.artifact.definitions),
            "tool_call_count": self.tool_call_count,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessRegistryError(f"{field} must be a non-empty string")
    return value.strip()


def _required_sha256(value: Any, field: str) -> str:
    normalized = _required_text(value, field)
    digest = normalized[7:] if normalized.startswith("sha256:") else normalized
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise HarnessRegistryError(f"{field} must be a SHA-256 digest")
    return "sha256:" + digest.lower()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessRegistryError(f"{field} must be an object")
    return value


def _normalized_tools(value: Any) -> tuple[dict[str, Any], ...]:
    raw_tools = value if isinstance(value, (list, tuple)) else None
    if raw_tools is None:
        raise HarnessRegistryError("tool_registry.tools must be an array")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_tool in enumerate(raw_tools):
        tool = _mapping(raw_tool, f"tool_registry.tools[{index}]")
        name = _required_text(tool.get("name"), f"tool_registry.tools[{index}].name")
        if name in names:
            raise HarnessRegistryError(f"duplicate tool name in registry: {name}")
        names.add(name)
        schema = tool.get("schema", tool.get("inputSchema"))
        if not isinstance(schema, Mapping):
            raise HarnessRegistryError(f"tool {name} schema must be an object")
        trust_class = _required_text(
            tool.get("trust_class", "unknown"),
            f"tool_registry.tools[{index}].trust_class",
        )
        side_effect_class = _required_text(
            tool.get("side_effect_class", "read_only"),
            f"tool_registry.tools[{index}].side_effect_class",
        )
        normalized.append(
            {
                "name": name,
                "schema": dict(schema),
                "trust_class": trust_class,
                "side_effect_class": side_effect_class,
            }
        )
    if not normalized:
        raise HarnessRegistryError("tool registry must expose at least one tool")
    normalized.sort(key=lambda item: item["name"])
    return tuple(normalized)


def harness_registry_digest(*, registry_revision: str, tools: Iterable[Mapping[str, Any]]) -> str:
    """Recompute the capture-side registry digest used by ``HarnessTrace``."""

    normalized = _normalized_tools(list(tools))
    return _sha256(
        _canonical_json(
            {
                "registry_revision": _required_text(registry_revision, "registry_revision"),
                "tools": list(normalized),
            }
        )
    )


def acquire_harness_registry(
    trace_path: Path,
    *,
    source_revision: str,
    scope: str,
    expected_episode_id: str | None = None,
    require_tool_call: bool = False,
    max_trace_line_bytes: int = DEFAULT_MAX_TRACE_LINE_BYTES,
) -> HarnessRegistryEvidence:
    """Stream one trace and return its validated canonical registry.

    ``source_revision`` is deliberately caller-supplied.  It must identify
    the exact harness/provider/environment source revision used for the join;
    this function never invents that identity from a filename or tool name.
    """

    trace_path = Path(trace_path)
    source_revision = _required_text(source_revision, "source_revision")
    scope = _required_text(scope, "scope")
    if expected_episode_id is not None:
        expected_episode_id = _required_text(expected_episode_id, "expected_episode_id")
    if max_trace_line_bytes <= 0:
        raise ValueError("max_trace_line_bytes must be positive")

    trace_digest = hashlib.sha256()
    episode_id: str | None = None
    registry_revision: str | None = None
    registry_event_id: str | None = None
    harness_digest: str | None = None
    registry_tools: tuple[dict[str, Any], ...] | None = None
    registry_names: frozenset[str] | None = None
    seen_call_ids: set[str] = set()
    tool_call_count = 0
    event_count = 0

    try:
        source = trace_path.open("rb")
    except OSError as error:
        raise HarnessRegistryError(f"unable to open trace: {trace_path}") from error

    with source:
        for line_number, raw_line in enumerate(source, start=1):
            trace_digest.update(raw_line)
            if len(raw_line) > max_trace_line_bytes:
                raise HarnessRegistryError(f"trace line {line_number} exceeds max_trace_line_bytes")
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise HarnessRegistryError(f"invalid JSON at trace line {line_number}") from error
            event_object = _mapping(event, f"trace line {line_number}")
            if event_object.get("schema_version") != HARNESS_TRACE_SCHEMA:
                raise HarnessRegistryError(
                    f"trace line {line_number} has an unsupported schema_version"
                )
            current_episode = _required_text(
                event_object.get("episode_id"),
                f"trace line {line_number}.episode_id",
            )
            if episode_id is None:
                episode_id = current_episode
                if expected_episode_id is not None and episode_id != expected_episode_id:
                    raise HarnessRegistryError("trace episode does not match expected_episode_id")
            elif current_episode != episode_id:
                raise HarnessRegistryError("trace contains multiple episode IDs")
            event_type = _required_text(
                event_object.get("event_type"),
                f"trace line {line_number}.event_type",
            )
            if event_type not in _EVENT_TYPES:
                raise HarnessRegistryError(f"unsupported event type: {event_type}")
            ordinal = event_object.get("ordinal")
            if ordinal != event_count:
                raise HarnessRegistryError(f"trace ordinal discontinuity at line {line_number}")
            event_count += 1
            event_registry_revision = _required_text(
                event_object.get("registry_revision"),
                f"trace line {line_number}.registry_revision",
            )
            if registry_revision is None:
                registry_revision = event_registry_revision
            elif event_registry_revision != registry_revision:
                raise HarnessRegistryError("trace contains multiple registry revisions")
            event_id = _required_text(
                event_object.get("event_id"),
                f"trace line {line_number}.event_id",
            )
            payload = _mapping(event_object.get("payload"), f"trace line {line_number}.payload")

            if event_type == "tool_registry":
                if registry_tools is not None:
                    raise HarnessRegistryError("trace contains multiple tool_registry events")
                payload_revision = _required_text(
                    payload.get("registry_revision"),
                    f"trace line {line_number}.payload.registry_revision",
                )
                if payload_revision != event_registry_revision:
                    raise HarnessRegistryError(
                        "tool registry payload revision does not match envelope"
                    )
                normalized = _normalized_tools(payload.get("tools"))
                observed_digest = _required_sha256(
                    payload.get("registry_sha256"),
                    f"trace line {line_number}.payload.registry_sha256",
                )
                expected_digest = harness_registry_digest(
                    registry_revision=payload_revision,
                    tools=normalized,
                )
                if observed_digest != expected_digest:
                    raise HarnessRegistryError("harness registry digest does not match its content")
                registry_tools = normalized
                registry_names = frozenset(item["name"] for item in normalized)
                harness_digest = observed_digest
                registry_event_id = event_id
                continue

            if event_type == "tool_call":
                if registry_tools is None:
                    raise HarnessRegistryError("tool call precedes the runtime registry")
                tool_name = _required_text(
                    payload.get("tool_name"),
                    f"trace line {line_number}.payload.tool_name",
                )
                call_id = _required_text(
                    payload.get("call_id"),
                    f"trace line {line_number}.payload.call_id",
                )
                if registry_names is None or tool_name not in registry_names:
                    raise HarnessRegistryError(
                        f"tool call is absent from the runtime registry: {tool_name}"
                    )
                if call_id in seen_call_ids:
                    raise HarnessRegistryError(f"duplicate tool call ID: {call_id}")
                seen_call_ids.add(call_id)
                tool_call_count += 1

    if event_count == 0 or episode_id is None:
        raise HarnessRegistryError("trace is empty")
    if registry_tools is None or registry_revision is None:
        raise HarnessRegistryError("trace has no runtime tool registry")
    if registry_event_id is None or harness_digest is None:
        raise HarnessRegistryError("runtime registry evidence is incomplete")
    if require_tool_call and tool_call_count == 0:
        raise HarnessRegistryError("trace has a registry but no tool call")

    definitions = tuple(
        ToolDefinition(
            name=tool["name"],
            parameters=tool["schema"],
            registry_revision=registry_revision,
        )
        for tool in registry_tools
    )
    artifact = tool_registry_artifact(
        registry_revision=registry_revision,
        source_revision=source_revision,
        scope=scope,
        origin=RegistryOrigin.HARNESS,
        definitions=definitions,
    )
    return HarnessRegistryEvidence(
        episode_id=episode_id,
        trace_sha256="sha256:" + trace_digest.hexdigest(),
        registry_event_id=registry_event_id,
        registry_revision=registry_revision,
        harness_registry_sha256=harness_digest,
        tool_call_count=tool_call_count,
        artifact=artifact,
    )


__all__ = [
    "DEFAULT_MAX_TRACE_LINE_BYTES",
    "HARNESS_REGISTRY_ADAPTER_REVISION",
    "HARNESS_REGISTRY_EVIDENCE_SCHEMA",
    "HARNESS_TRACE_SCHEMA",
    "HarnessRegistryError",
    "HarnessRegistryEvidence",
    "acquire_harness_registry",
    "harness_registry_digest",
]
