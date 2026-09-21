"""Tests for versioned, digest-bound review manifest artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentir.hardening import (
    QualityReviewManifest,
    artifact_from_json,
    load_review_manifest,
    review_manifest_artifact,
    review_manifest_digest,
    review_map,
    write_review_manifest,
)


def _manifest() -> QualityReviewManifest:
    return QualityReviewManifest(
        manifest_revision="reviews/v1",
        snapshot_revision="snapshot-1",
        rubric_revision="quality/v1",
    )


def test_review_artifact_digest_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    manifest = _manifest()
    artifact = review_manifest_artifact(manifest)
    path = tmp_path / "reviews.json"

    digest = write_review_manifest(path, manifest)

    assert digest == review_manifest_digest(manifest)
    assert (
        load_review_manifest(
            path,
            expected_manifest_revision="reviews/v1",
            expected_snapshot_revision="snapshot-1",
            expected_rubric_revision="quality/v1",
        )
        == manifest
    )
    assert artifact.manifest_sha256 == digest
    assert not list(tmp_path.glob("*.part"))


def test_review_artifact_rejects_tampering_and_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "reviews.json"
    write_review_manifest(path, _manifest())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest"]["snapshot_revision"] = "snapshot-2"

    with pytest.raises(ValueError, match="digest"):
        artifact_from_json(payload)

    write_review_manifest(
        path,
        QualityReviewManifest(
            manifest_revision="reviews/v1",
            snapshot_revision="snapshot-2",
            rubric_revision="quality/v1",
        ),
    )
    with pytest.raises(ValueError, match="digest"):
        artifact_from_json(payload)

    with pytest.raises(ValueError, match="snapshot_revision"):
        load_review_manifest(path, expected_snapshot_revision="snapshot-1")


def test_review_map_is_row_bound() -> None:
    manifest = _manifest()
    assert review_map(manifest) == {}
