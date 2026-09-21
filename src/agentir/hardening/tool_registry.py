"""Versioned, digest-bound tool-registry artifacts.

Tool names found in a transcript are not enough to train tool use.  A release
must join calls to the exact schemas that were available to the agent.  This
module owns that join artifact; it deliberately does not infer a registry from
an unversioned tool name.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentir.hardening.release_views import ToolDefinition

TOOL_REGISTRY_SCHEMA: Literal["agentir/hardening/tool-registry/v1"] = (
    "agentir/hardening/tool-registry/v1"
)
DEFAULT_MAX_REGISTRY_BYTES = 16 * 1024 * 1024


class RegistryOrigin(StrEnum):
    """Authority that supplied the versioned tool definitions."""

    HARNESS = "harness"
    SOURCE_DECLARED = "source_declared"
    EXTERNAL = "external"


class ToolRegistryArtifact(BaseModel):
    """A trainer-safe registry whose definitions share one revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agentir/hardening/tool-registry/v1"] = TOOL_REGISTRY_SCHEMA
    registry_revision: str
    source_revision: str
    scope: str
    origin: RegistryOrigin
    definitions: tuple[ToolDefinition, ...] = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("registry_revision", "source_revision", "scope", mode="before")
    @classmethod
    def require_identity_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("tool registry identity fields must be non-empty")

    @model_validator(mode="after")
    def validate_definitions_and_digest(self) -> ToolRegistryArtifact:
        names = [definition.name for definition in self.definitions]
        if len(names) != len(set(names)):
            raise ValueError("tool registry contains duplicate names")
        mismatched = sorted(
            {
                definition.name
                for definition in self.definitions
                if definition.registry_revision != self.registry_revision
            }
        )
        if mismatched:
            raise ValueError(
                "tool definitions use a different registry revision: " + ", ".join(mismatched)
            )
        expected = tool_registry_digest(
            registry_revision=self.registry_revision,
            source_revision=self.source_revision,
            scope=self.scope,
            origin=self.origin,
            definitions=self.definitions,
        )
        if self.artifact_sha256 != expected:
            raise ValueError("tool registry artifact digest does not match its content")
        return self


def _canonical_registry_bytes(
    *,
    registry_revision: str,
    source_revision: str,
    scope: str,
    origin: RegistryOrigin,
    definitions: tuple[ToolDefinition, ...],
) -> bytes:
    value = {
        "registry_revision": registry_revision,
        "source_revision": source_revision,
        "scope": scope,
        "origin": origin.value,
        "definitions": [definition.model_dump(mode="json") for definition in definitions],
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def tool_registry_digest(
    *,
    registry_revision: str,
    source_revision: str,
    scope: str,
    origin: RegistryOrigin,
    definitions: tuple[ToolDefinition, ...],
) -> str:
    """Return the stable content digest for a validated registry body."""

    return hashlib.sha256(
        _canonical_registry_bytes(
            registry_revision=registry_revision,
            source_revision=source_revision,
            scope=scope,
            origin=origin,
            definitions=definitions,
        )
    ).hexdigest()


def tool_registry_artifact(
    *,
    registry_revision: str,
    source_revision: str,
    scope: str,
    origin: RegistryOrigin,
    definitions: tuple[ToolDefinition, ...],
) -> ToolRegistryArtifact:
    """Create a digest-bound artifact from already validated definitions."""

    artifact_sha256 = tool_registry_digest(
        registry_revision=registry_revision,
        source_revision=source_revision,
        scope=scope,
        origin=origin,
        definitions=definitions,
    )
    return ToolRegistryArtifact(
        registry_revision=registry_revision,
        source_revision=source_revision,
        scope=scope,
        origin=origin,
        definitions=definitions,
        artifact_sha256=artifact_sha256,
    )


def _artifact_payload(artifact: ToolRegistryArtifact) -> bytes:
    return json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def write_tool_registry(path: Path, artifact: ToolRegistryArtifact) -> str:
    """Atomically publish a validated registry artifact and return its digest."""

    payload = _artifact_payload(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except Exception:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise
    return artifact.artifact_sha256


def load_tool_registry(
    path: Path,
    *,
    expected_registry_revision: str | None = None,
    expected_source_revision: str | None = None,
    max_bytes: int = DEFAULT_MAX_REGISTRY_BYTES,
) -> ToolRegistryArtifact:
    """Load one bounded, digest-verified registry artifact."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    payload = path.read_bytes()
    if len(payload) > max_bytes:
        raise ValueError("tool registry exceeds max_bytes")
    artifact = ToolRegistryArtifact.model_validate_json(payload)
    if (
        expected_registry_revision is not None
        and artifact.registry_revision != expected_registry_revision
    ):
        raise ValueError("tool registry revision does not match expected identity")
    if (
        expected_source_revision is not None
        and artifact.source_revision != expected_source_revision
    ):
        raise ValueError("tool registry source revision does not match expected identity")
    return artifact


def registry_tools(artifact: ToolRegistryArtifact) -> tuple[ToolDefinition, ...]:
    """Return definitions for the materializer's explicit registry resolver."""

    return artifact.definitions
