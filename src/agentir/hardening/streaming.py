"""Bounded, checkpointed JSONL compilation primitives."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any


class CompileInterrupted(RuntimeError):
    """Raised by the deterministic test hook after leaving a resumable checkpoint."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError):
        return None


def _read_jsonl_with_ranges(
    path: Path,
) -> Iterator[tuple[int, dict[str, Any], int, int]]:
    with path.open("rb") as source:
        byte_offset = 0
        for line_number, raw_line in enumerate(source, start=1):
            byte_start = byte_offset
            byte_offset += len(raw_line)
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"source line {line_number} is not a JSON object")
            yield line_number, value, byte_start, byte_offset


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read JSONL without exposing byte ranges to legacy callers."""

    for line_number, value, _byte_start, _byte_end in _read_jsonl_with_ranges(path):
        yield line_number, value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object at a file boundary with an explicit type."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return {str(key): item for key, item in value.items()}


def _checkpoint_payload(
    *,
    source: dict[str, Any],
    parser_revision: str,
    pass_config_hash: str,
    completed_source_line: int,
    completed_source_records: int,
    completed_records: int,
    shards: list[dict[str, Any]],
    status: str,
    input_manifest_hash: str,
    source_record_limit: int | None,
    source_line_start: int,
) -> dict[str, Any]:
    return {
        "schema_version": "agentir/hardening-checkpoint/v1",
        "status": status,
        "source": source,
        "input_manifest_hash": input_manifest_hash,
        "parser_revision": parser_revision,
        "pass_config_hash": pass_config_hash,
        "completed_source_line": completed_source_line,
        "completed_source_records": completed_source_records,
        "completed_records": completed_records,
        "source_record_limit": source_record_limit,
        "source_line_start": source_line_start,
        "shards": shards,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    source: dict[str, Any],
    parser_revision: str,
    pass_config_hash: str,
    output_dir: Path,
    source_record_limit: int | None,
    source_line_start: int,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    if checkpoint.get("source") != source:
        raise ValueError("checkpoint source fingerprint does not match input")
    if checkpoint.get("parser_revision") != parser_revision:
        raise ValueError("checkpoint parser revision does not match requested run")
    if checkpoint.get("pass_config_hash") != pass_config_hash:
        raise ValueError("checkpoint pass configuration does not match requested run")
    if checkpoint.get("source_record_limit") != source_record_limit:
        raise ValueError("checkpoint source record limit does not match requested run")
    if checkpoint.get("source_line_start", 1) != source_line_start:
        raise ValueError("checkpoint source line start does not match requested run")
    if checkpoint.get("input_manifest_hash") != source.get("input_manifest_hash"):
        raise ValueError("checkpoint input manifest hash does not match input")
    shards = checkpoint.get("shards")
    if not isinstance(shards, list):
        raise ValueError("checkpoint shards are malformed")
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("checkpoint contains a malformed shard")
        shard_path = output_dir / str(shard.get("name", ""))
        if not shard_path.is_file() or _sha256_file(shard_path) != shard.get("sha256"):
            raise ValueError(f"checkpoint shard is missing or changed: {shard_path.name}")
    return (
        int(checkpoint.get("completed_source_line", 0)),
        int(checkpoint.get("completed_source_records", 0)),
        int(checkpoint.get("completed_records", 0)),
        shards,
    )


def compile_jsonl(
    input_path: Path,
    output_dir: Path,
    transform: Callable[[dict[str, Any], int], dict[str, Any] | None],
    *,
    transform_with_context: Callable[
        [dict[str, Any], int, int, int, Mapping[str, Any]], dict[str, Any] | None
    ]
    | None = None,
    parser_revision: str,
    pass_config: dict[str, Any],
    shard_records: int = 1000,
    shard_bytes: int = 64 * 1024 * 1024,
    resume: bool = False,
    stop_after_source_records: int | None = None,
    source_record_limit: int | None = None,
    source_line_start: int = 1,
) -> dict[str, Any]:
    """Compile JSONL with bounded memory and atomic, restartable shards.

    A transform is called once per source object after the input line is
    decoded. ``None`` means the source object is intentionally filtered. If
    ``transform_with_context`` is supplied, it receives the source byte range
    and immutable input descriptor as well; this is the provenance-aware path
    for release-bound adapters. A checkpoint advances only after a complete
    shard is atomically renamed, so a crash or deterministic interruption can
    discard one partial shard and resume without duplicating committed output.
    """

    if transform_with_context is not None and not callable(transform_with_context):
        raise TypeError("transform_with_context must be callable")

    if shard_records <= 0 or shard_bytes <= 0:
        raise ValueError("shard_records and shard_bytes must be positive")
    if stop_after_source_records is not None and stop_after_source_records <= 0:
        raise ValueError("stop_after_source_records must be positive")
    if source_record_limit is not None and source_record_limit <= 0:
        raise ValueError("source_record_limit must be positive")
    if source_line_start <= 0:
        raise ValueError("source_line_start must be positive")
    if source_line_start != 1 and source_record_limit is None:
        raise ValueError("source_line_start requires source_record_limit")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    manifest_path = output_dir / "manifest.json"
    source_descriptor: dict[str, Any] = {
        "path": str(input_path),
        "sha256": _sha256_file(input_path),
        "size_bytes": input_path.stat().st_size,
    }
    source: dict[str, Any] = {
        **source_descriptor,
        "input_manifest_hash": _canonical_hash(source_descriptor),
    }
    pass_config_hash = _canonical_hash(pass_config)

    if manifest_path.is_file() and resume:
        manifest = _read_json_object(manifest_path)
        if manifest.get("source") != source:
            raise ValueError("completed manifest source fingerprint does not match input")
        if manifest.get("parser_revision") != parser_revision:
            raise ValueError("completed manifest parser revision does not match requested run")
        if manifest.get("pass_config_hash") != pass_config_hash:
            raise ValueError("completed manifest pass configuration does not match requested run")
        if manifest.get("source_record_limit") != source_record_limit:
            raise ValueError("completed manifest source record limit does not match requested run")
        if manifest.get("source_line_start", 1) != source_line_start:
            raise ValueError("completed manifest source line start does not match requested run")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if _canonical_hash(payload) != manifest.get("manifest_sha256"):
            raise ValueError("completed manifest hash is invalid")
        return manifest

    if checkpoint_path.is_file():
        if not resume:
            raise FileExistsError("checkpoint exists; pass resume=True to continue")
        checkpoint = _read_json_object(checkpoint_path)
        completed_line, completed_source_records, completed_records, shards = _validate_resume(
            checkpoint,
            source=source,
            parser_revision=parser_revision,
            pass_config_hash=pass_config_hash,
            output_dir=output_dir,
            source_record_limit=source_record_limit,
            source_line_start=source_line_start,
        )
    else:
        if resume:
            raise FileNotFoundError("resume requested but checkpoint is absent")
        completed_line = source_line_start - 1
        completed_source_records, completed_records, shards = 0, 0, []
        for stale in output_dir.glob("*.part"):
            stale.unlink()

    for stale in output_dir.glob("*.part"):
        stale.unlink()

    batch: list[bytes] = []
    batch_bytes = 0
    batch_source_start: int | None = None
    batch_source_end = completed_line
    batch_source_record_end = completed_source_records
    processed_since_resume = 0
    processed_source_records = completed_source_records
    last_source_line_seen = completed_line
    committed_source_line = completed_line
    committed_source_records = completed_source_records

    def commit_batch(
        *,
        source_line_end: int | None = None,
        source_record_count: int | None = None,
    ) -> None:
        nonlocal \
            batch_bytes, \
            batch_source_start, \
            batch_source_end, \
            batch_source_record_end, \
            completed_line, \
            completed_source_records, \
            committed_source_line, \
            committed_source_records, \
            completed_records
        if not batch:
            return
        index = len(shards)
        name = f"shard-{index:06d}.jsonl"
        partial = output_dir / f"{name}.part"
        final = output_dir / name
        with partial.open("wb") as destination:
            for encoded in batch:
                destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        partial.replace(final)
        shard = {
            "name": name,
            "sha256": _sha256_file(final),
            "records": len(batch),
            "source_line_start": batch_source_start,
            "source_line_end": batch_source_end,
        }
        shards.append(shard)
        committed_source_line = batch_source_end if source_line_end is None else source_line_end
        committed_source_records = (
            batch_source_record_end if source_record_count is None else source_record_count
        )
        completed_line = committed_source_line
        completed_source_records = committed_source_records
        completed_records += len(batch)
        _write_json(
            checkpoint_path,
            _checkpoint_payload(
                source=source,
                parser_revision=parser_revision,
                pass_config_hash=pass_config_hash,
                completed_source_line=completed_line,
                completed_source_records=completed_source_records,
                completed_records=completed_records,
                shards=shards,
                status="running",
                input_manifest_hash=source["input_manifest_hash"],
                source_record_limit=source_record_limit,
                source_line_start=source_line_start,
            ),
        )
        batch.clear()
        batch_bytes = 0
        batch_source_start = None
        batch_source_record_end = completed_source_records

    for source_line, row, byte_start, byte_end in _read_jsonl_with_ranges(input_path):
        if source_line <= completed_line:
            continue
        if source_record_limit is not None and processed_source_records >= source_record_limit:
            break
        processed_since_resume += 1
        processed_source_records += 1
        last_source_line_seen = source_line
        if transform_with_context is not None:
            transformed = transform_with_context(
                row,
                source_line,
                byte_start,
                byte_end,
                source,
            )
        else:
            transformed = transform(row, source_line)
        if transformed is not None:
            encoded = (
                json.dumps(transformed, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            )
            if batch and (len(batch) >= shard_records or batch_bytes + len(encoded) > shard_bytes):
                commit_batch()
            if batch_source_start is None:
                batch_source_start = source_line
            batch.append(encoded)
            batch_bytes += len(encoded)
            batch_source_end = source_line
            batch_source_record_end = processed_source_records
            if len(batch) >= shard_records or batch_bytes >= shard_bytes:
                commit_batch(
                    source_line_end=last_source_line_seen,
                    source_record_count=processed_source_records,
                )
        if (
            stop_after_source_records is not None
            and processed_since_resume >= stop_after_source_records
        ):
            _write_json(
                checkpoint_path,
                _checkpoint_payload(
                    source=source,
                    parser_revision=parser_revision,
                    pass_config_hash=pass_config_hash,
                    completed_source_line=completed_line,
                    completed_source_records=completed_source_records,
                    completed_records=completed_records,
                    shards=shards,
                    status="interrupted",
                    input_manifest_hash=source["input_manifest_hash"],
                    source_record_limit=source_record_limit,
                    source_line_start=source_line_start,
                ),
            )
            raise CompileInterrupted("deterministic interruption after source record bound")

    commit_batch(
        source_line_end=last_source_line_seen,
        source_record_count=processed_source_records,
    )
    completed_line = last_source_line_seen
    completed_source_records = processed_source_records
    manifest_payload = {
        "schema_version": "agentir/hardening-release-manifest/v1",
        "status": "bounded"
        if source_record_limit is not None or source_line_start != 1
        else "complete",
        "source": source,
        "input_manifest_hash": source["input_manifest_hash"],
        "parser_revision": parser_revision,
        "pass_config_hash": pass_config_hash,
        "completed_source_line": completed_line,
        "completed_source_records": completed_source_records,
        "source_record_limit": source_record_limit,
        "source_line_start": source_line_start,
        "records": completed_records,
        "shards": shards,
    }
    manifest = {
        **manifest_payload,
        "manifest_sha256": _canonical_hash(manifest_payload),
    }
    _write_json(manifest_path, manifest)
    _write_json(
        checkpoint_path,
        _checkpoint_payload(
            source=source,
            parser_revision=parser_revision,
            pass_config_hash=pass_config_hash,
            completed_source_line=completed_line,
            completed_source_records=completed_source_records,
            completed_records=completed_records,
            shards=shards,
            status=manifest["status"],
            input_manifest_hash=source["input_manifest_hash"],
            source_record_limit=source_record_limit,
            source_line_start=source_line_start,
        ),
    )
    return manifest
