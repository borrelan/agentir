"""Versioned, content-addressed quality-review manifest artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentir.hardening.contracts import QualityReview, QualityReviewManifest

REVIEW_MANIFEST_SCHEMA: Literal["agentir/hardening/quality-review-manifest/v1"] = (
    "agentir/hardening/quality-review-manifest/v1"
)
DEFAULT_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class ReviewManifestArtifact(BaseModel):
    """On-disk wrapper that binds review decisions to one manifest digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agentir/hardening/quality-review-manifest/v1"] = REVIEW_MANIFEST_SCHEMA
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest: QualityReviewManifest

    @model_validator(mode="after")
    def validate_digest(self) -> ReviewManifestArtifact:
        expected = review_manifest_digest(self.manifest)
        if self.manifest_sha256 != expected:
            raise ValueError("review manifest digest does not match its content")
        return self


def _canonical_manifest_bytes(manifest: QualityReviewManifest) -> bytes:
    return json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def review_manifest_digest(manifest: QualityReviewManifest) -> str:
    """Return the stable SHA-256 of the validated manifest body."""

    return hashlib.sha256(_canonical_manifest_bytes(manifest)).hexdigest()


def review_manifest_artifact(manifest: QualityReviewManifest) -> ReviewManifestArtifact:
    """Wrap a validated manifest in its versioned, digest-bound artifact."""

    return ReviewManifestArtifact(
        manifest_sha256=review_manifest_digest(manifest),
        manifest=manifest,
    )


def _validate_expected_identity(
    manifest: QualityReviewManifest,
    *,
    expected_manifest_revision: str | None,
    expected_snapshot_revision: str | None,
    expected_rubric_revision: str | None,
) -> None:
    expected = (
        ("manifest_revision", expected_manifest_revision),
        ("snapshot_revision", expected_snapshot_revision),
        ("rubric_revision", expected_rubric_revision),
    )
    for field_name, expected_value in expected:
        if expected_value is not None and getattr(manifest, field_name) != expected_value:
            raise ValueError(f"review manifest {field_name} does not match expected identity")


def load_review_manifest(
    path: Path,
    *,
    expected_manifest_revision: str | None = None,
    expected_snapshot_revision: str | None = None,
    expected_rubric_revision: str | None = None,
    max_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
) -> QualityReviewManifest:
    """Load one bounded, digest-verified review artifact from JSON."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError("review manifest exceeds max_bytes")
    artifact = ReviewManifestArtifact.model_validate_json(data)
    _validate_expected_identity(
        artifact.manifest,
        expected_manifest_revision=expected_manifest_revision,
        expected_snapshot_revision=expected_snapshot_revision,
        expected_rubric_revision=expected_rubric_revision,
    )
    return artifact.manifest


def review_map(manifest: QualityReviewManifest) -> dict[str, QualityReview]:
    """Return the row-bound review lookup used by the streaming bridge."""

    return {review.unit_id: review for review in manifest.reviews}


def write_review_manifest(path: Path, manifest: QualityReviewManifest) -> str:
    """Atomically publish a canonical review artifact and return its digest."""

    artifact = review_manifest_artifact(manifest)
    payload = json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".part",
        dir=path.parent,
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
    return artifact.manifest_sha256


def artifact_from_json(value: Any) -> ReviewManifestArtifact:
    """Validate an already-decoded artifact without weakening its contract."""

    return ReviewManifestArtifact.model_validate(value)
