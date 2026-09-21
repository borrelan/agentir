"""Manifest-backed verification for parent raw-snapshot references."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentir.hardening.contracts import ParentSnapshotRef


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Narrow an optional JSON object at the provenance boundary."""

    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class SnapshotManifestIndex:
    """Indexed view of one immutable snapshot manifest.

    Parent identity is checked once against this index rather than scanning
    every manifest file for every normalized row. The plain mapping API is
    retained for small callers and compatibility tests; real streaming jobs
    should construct this index once before row processing.
    """

    snapshot_revision: str
    _objects: Mapping[tuple[str, int], tuple[Mapping[str, Any], ...]]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> SnapshotManifestIndex:
        revision = manifest.get("snapshot_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("snapshot manifest revision must be non-empty")
        objects: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
        entries = manifest.get("files", ())
        for entry in entries if isinstance(entries, (list, tuple)) else ():
            if not isinstance(entry, Mapping):
                continue
            digest_value = entry.get("sha256")
            size_value = entry.get("bytes")
            if not isinstance(digest_value, str) or not isinstance(size_value, int):
                continue
            digest = digest_value.lower().removeprefix("sha256:")
            objects.setdefault((digest, size_value), []).append(entry)
        return cls(
            snapshot_revision=revision,
            _objects={key: tuple(value) for key, value in objects.items()},
        )

    def matches(self, parent: ParentSnapshotRef) -> tuple[Mapping[str, Any], ...]:
        """Return manifest objects matching digest, size, and root identity."""

        candidates = self._objects.get((parent.source_sha256, parent.source_size_bytes), ())
        return tuple(
            entry
            for entry in candidates
            if parent.root_label is None or entry.get("root_label") == parent.root_label
        )


def verify_parent_snapshot(
    parent: ParentSnapshotRef,
    manifest: Mapping[str, Any] | SnapshotManifestIndex,
) -> ParentSnapshotRef:
    """Return a verified reference only when one stable manifest object matches."""

    manifest_revision: Any
    matches: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]
    if isinstance(manifest, SnapshotManifestIndex):
        manifest_revision = manifest.snapshot_revision
        matches = manifest.matches(parent)
    else:
        manifest_revision = manifest.get("snapshot_revision")
        matches = []
        entries = manifest.get("files", ())
        for entry in entries if isinstance(entries, (list, tuple)) else ():
            if not isinstance(entry, Mapping):
                continue
            digest = str(entry.get("sha256", "")).lower().removeprefix("sha256:")
            if (
                digest == parent.source_sha256
                and entry.get("bytes") == parent.source_size_bytes
                and (parent.root_label is None or entry.get("root_label") == parent.root_label)
            ):
                matches.append(entry)

    if manifest_revision != parent.snapshot_revision:
        raise ValueError("parent snapshot revision does not match the manifest")
    if not matches:
        raise ValueError("parent object is absent from the snapshot manifest")
    if len(matches) != 1:
        raise ValueError("parent object identity is ambiguous in the snapshot manifest")
    if matches[0].get("status") != "stable":
        raise ValueError("parent object is not stable in the snapshot manifest")
    return parent.model_copy(update={"manifest_verified": True})


def parent_snapshot_from_row(
    row: Mapping[str, Any],
    snapshot_revision: str,
    manifest: Mapping[str, Any] | SnapshotManifestIndex,
) -> ParentSnapshotRef | None:
    """Resolve and verify a parent raw object from published row metadata.

    The row metadata is treated as a lookup hint only.  A returned reference is
    marked verified only when exactly one stable manifest object matches its
    digest, byte size, and optional source-root label.  Ambiguous or missing
    objects remain explicit, unverified references for quarantine decisions.
    """

    origin = _as_mapping(row.get("source_origin"))
    database = _as_mapping(origin.get("database"))
    source_path = (
        row.get("source_file")
        or row.get("session_file")
        or origin.get("source_file_name")
        or database.get("file_name")
    )
    source_sha256 = origin.get("source_file_sha256") or database.get("sha256")
    if not isinstance(source_path, str) or not source_path.strip():
        return None
    if not isinstance(source_sha256, str) or not source_sha256.strip():
        return None

    source_size = origin.get("source_size_bytes") or database.get("bytes")
    if not isinstance(source_size, int):
        candidate = Path(source_path)
        source_size = candidate.stat().st_size if candidate.is_file() else 0

    message_range: tuple[int, int] | None = None
    for key in ("source_event_line_range", "message_line_range", "message_index_range"):
        value = origin.get(key)
        if (
            isinstance(value, Mapping)
            and isinstance(value.get("start"), int)
            and isinstance(value.get("end"), int)
        ):
            message_range = (value["start"], value["end"])
            break
    if (
        message_range is None
        and isinstance(row.get("_chunk_message_start"), int)
        and isinstance(row.get("_chunk_message_end"), int)
    ):
        message_range = (row["_chunk_message_start"], row["_chunk_message_end"])

    normalized_sha = source_sha256.removeprefix("sha256:").lower()
    source_uri = source_path if Path(source_path).is_absolute() else f"snapshot://raw/{source_path}"
    parent = ParentSnapshotRef(
        snapshot_revision=snapshot_revision,
        source_uri=source_uri,
        source_path=source_path,
        source_sha256=normalized_sha,
        source_size_bytes=source_size,
        object_id=f"{source_path}#{normalized_sha}",
        root_label=(
            origin.get("source_root_label")
            if isinstance(origin.get("source_root_label"), str)
            else None
        ),
        message_start=message_range[0] if message_range else None,
        message_end=message_range[1] if message_range else None,
    )
    try:
        return verify_parent_snapshot(parent, manifest)
    except ValueError:
        return parent
