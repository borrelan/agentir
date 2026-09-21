"""Lossless, bounded-memory Parquet mirrors for hardened JSONL releases."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from agentir.hardening.streaming import (
    _canonical_hash,
    _peak_rss_bytes,
    _read_json_object,
    _sha256_file,
)

PARQUET_MIRROR_SCHEMA_VERSION = "agentir/hardening-parquet-mirror/v1"
PARQUET_MIRROR_SCHEMA = pa.schema(
    [
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source_line", pa.int64(), nullable=False),
        pa.field("record_sha256", pa.string(), nullable=False),
        pa.field("unit_id", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("admission_state", pa.string()),
        pa.field("record_json", pa.large_string(), nullable=False),
    ]
)


class MirrorInterrupted(RuntimeError):
    """Raised by the deterministic test hook after a complete mirror shard."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_signed_json(path: Path, payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    signed = {**payload, hash_field: _canonical_hash(payload)}
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(signed, destination, ensure_ascii=False, sort_keys=True, indent=2)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)
    return signed


def _validate_signed_json(value: Mapping[str, Any], hash_field: str, label: str) -> None:
    declared = value.get(hash_field)
    if not isinstance(declared, str):
        raise ValueError(f"{label} has no {hash_field}")
    payload = {key: item for key, item in value.items() if key != hash_field}
    if _canonical_hash(payload) != declared:
        raise ValueError(f"{label} hash is invalid")


def _source_manifest(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = input_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = _read_json_object(path)
    _validate_signed_json(manifest, "manifest_sha256", "source manifest")
    if manifest.get("schema_version") != "agentir/hardening-release-manifest/v1":
        raise ValueError("source manifest is not a compile_jsonl release manifest")
    if manifest.get("status") not in {"complete", "bounded"}:
        raise ValueError("source manifest is not complete")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("source manifest shards are malformed")
    declared_records = manifest.get("records")
    if not isinstance(declared_records, int) or declared_records < 0:
        raise ValueError("source manifest record count is malformed")

    names: set[str] = set()
    shard_records = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("source manifest contains a malformed shard")
        name = shard.get("name")
        sha256 = shard.get("sha256")
        records = shard.get("records")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".jsonl")
            or name in names
        ):
            raise ValueError("source manifest contains an invalid or duplicate shard name")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"source manifest shard hash is malformed: {name}")
        try:
            int(sha256, 16)
        except ValueError as error:
            raise ValueError(f"source manifest shard hash is malformed: {name}") from error
        if not isinstance(records, int) or records < 0:
            raise ValueError(f"source manifest shard record count is malformed: {name}")
        names.add(name)
        shard_records += records
    if shard_records != declared_records:
        raise ValueError("source manifest record count does not equal its shard counts")

    descriptor = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "manifest_sha256": manifest["manifest_sha256"],
        "schema_version": manifest.get("schema_version"),
        "records": declared_records,
        "shards": len(shards),
    }
    return manifest, descriptor


def _canonical_record(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _string_at(row: Mapping[str, Any], *paths: tuple[str, ...]) -> str | None:
    for path in paths:
        value: Any = row
        for part in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if isinstance(value, str) and value:
            return value
    return None


def _mirror_row(row: Mapping[str, Any], source_shard: str, source_line: int) -> dict[str, Any]:
    record_json = _canonical_record(row)
    return {
        "source_shard": source_shard,
        "source_line": source_line,
        "record_sha256": hashlib.sha256(record_json.encode("utf-8")).hexdigest(),
        "unit_id": _string_at(
            row,
            ("unit_id",),
            ("metadata", "provenance", "unit_id"),
        ),
        "provider": _string_at(
            row,
            ("provider",),
            ("metadata", "provenance", "provider"),
        ),
        "admission_state": _string_at(row, ("admission", "state"), ("state",)),
        "record_json": record_json,
    }


def _iter_source_rows(
    path: Path,
) -> Iterator[tuple[int, dict[str, Any], bytes]]:
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                yield line_number, {}, raw_line
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"source shard {path.name} line {line_number} is not an object")
            yield line_number, value, raw_line


def _verify_parquet(path: Path, expected_records: int, row_group_records: int) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != PARQUET_MIRROR_SCHEMA:
        raise ValueError(f"Parquet schema mismatch: {path.name}")
    if parquet.metadata.num_rows != expected_records:
        raise ValueError(f"Parquet row count mismatch: {path.name}")
    for index in range(parquet.metadata.num_row_groups):
        if parquet.metadata.row_group(index).num_rows > row_group_records:
            raise ValueError(f"Parquet row group exceeds configured bound: {path.name}")


def _stage_mirror_shard(
    source_path: Path,
    output_path: Path,
    *,
    source_name: str,
    expected_sha256: str,
    expected_records: int,
    row_group_records: int,
    compression: str,
) -> dict[str, Any]:
    partial = output_path.with_name(f"{output_path.name}.part")
    digest = hashlib.sha256()
    records = 0
    batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(partial, PARQUET_MIRROR_SCHEMA, compression=compression)
        for line_number, row, raw_line in _iter_source_rows(source_path):
            digest.update(raw_line)
            if not row and not raw_line.strip():
                continue
            batch.append(_mirror_row(row, source_name, line_number))
            records += 1
            if len(batch) == row_group_records:
                writer.write_table(
                    pa.Table.from_pylist(batch, schema=PARQUET_MIRROR_SCHEMA),
                    row_group_size=row_group_records,
                )
                batch.clear()
        if batch:
            writer.write_table(
                pa.Table.from_pylist(batch, schema=PARQUET_MIRROR_SCHEMA),
                row_group_size=row_group_records,
            )
            batch.clear()
        writer.close()
        writer = None

        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"source shard is missing or changed: {source_name}")
        if records != expected_records:
            raise ValueError(f"source shard record count changed: {source_name}")
        _verify_parquet(partial, records, row_group_records)
        descriptor = os.open(partial, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise

    return {
        "name": output_path.name,
        "sha256": _sha256_file(partial),
        "size_bytes": partial.stat().st_size,
        "records": records,
        "source_name": source_name,
        "source_sha256": expected_sha256,
    }


def _checkpoint_payload(
    *,
    source_manifest: Mapping[str, Any],
    config_sha256: str,
    shards: list[dict[str, Any]],
    pending_shard: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": PARQUET_MIRROR_SCHEMA_VERSION,
        "status": status,
        "source_manifest": dict(source_manifest),
        "config_sha256": config_sha256,
        "records": sum(int(shard["records"]) for shard in shards),
        "shards": shards,
        "pending_shard": pending_shard,
    }


def _verify_source_shard(input_dir: Path, shard: Mapping[str, Any]) -> None:
    name = str(shard["name"])
    path = input_dir / name
    if not path.is_file() or _sha256_file(path) != shard["sha256"]:
        raise ValueError(f"source shard is missing or changed: {name}")


def _verify_output_path(
    path: Path,
    shard: Mapping[str, Any],
    row_group_records: int,
) -> None:
    name = str(shard.get("name", ""))
    if (
        Path(name).name != name
        or not path.is_file()
        or _sha256_file(path) != shard.get("sha256")
        or path.stat().st_size != shard.get("size_bytes")
    ):
        raise ValueError(f"mirror shard is missing or changed: {name}")
    _verify_parquet(path, int(shard["records"]), row_group_records)


def _verify_output_shard(
    output_dir: Path,
    shard: Mapping[str, Any],
    row_group_records: int,
) -> None:
    _verify_output_path(output_dir / str(shard.get("name", "")), shard, row_group_records)


def _publish_staged_shard(
    output_dir: Path,
    shard: Mapping[str, Any],
    row_group_records: int,
) -> None:
    final = output_dir / str(shard["name"])
    partial = final.with_name(f"{final.name}.part")
    if final.exists():
        raise FileExistsError(f"mirror shard already exists: {final.name}")
    _verify_output_path(partial, shard, row_group_records)
    partial.replace(final)
    _fsync_directory(output_dir)


def _validate_completed_shards(
    completed: list[dict[str, Any]],
    source_shards: list[dict[str, Any]],
) -> None:
    if len(completed) > len(source_shards):
        raise ValueError("mirror contains more shards than its source manifest")
    output_names: set[str] = set()
    for output, source in zip(completed, source_shards, strict=False):
        output_name = output.get("name")
        expected_output_name = f"{Path(str(source['name'])).stem}.parquet"
        if output_name != expected_output_name or output_name in output_names:
            raise ValueError("mirror shard names do not match the source shard order")
        if output.get("source_name") != source["name"]:
            raise ValueError("mirror shard source names do not match the source manifest")
        if output.get("source_sha256") != source["sha256"]:
            raise ValueError("mirror shard source hashes do not match the source manifest")
        if output.get("records") != source["records"]:
            raise ValueError("mirror shard record counts do not match the source manifest")
        output_names.add(str(output_name))


def _discard_unjournaled_partial(output_dir: Path, expected_name: str | None) -> None:
    partials = list(output_dir.glob("*.parquet.part"))
    if not partials:
        return
    expected = output_dir / f"{expected_name}.part" if expected_name is not None else None
    if len(partials) != 1 or partials[0] != expected:
        raise ValueError("unexpected unjournaled Parquet partial exists")
    partials[0].unlink()
    _fsync_directory(output_dir)


def mirror_jsonl_shards_to_parquet(
    input_dir: Path,
    output_dir: Path,
    *,
    row_group_records: int = 1000,
    compression: str = "zstd",
    resume: bool = False,
    stop_after_shards: int | None = None,
) -> dict[str, Any]:
    """Mirror a completed ``compile_jsonl`` release without changing its semantics.

    JSONL remains authoritative. Each non-empty source object is canonicalized
    into ``record_json`` and accompanied only by provenance/index columns. The
    writer retains at most ``row_group_records`` decoded rows and publishes a
    shard only after source digest/count and Parquet metadata verification.
    """

    if row_group_records <= 0:
        raise ValueError("row_group_records must be positive")
    if not compression.strip():
        raise ValueError("compression must be non-empty")
    if stop_after_shards is not None and stop_after_shards <= 0:
        raise ValueError("stop_after_shards must be positive")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input and output directories must differ")

    source, source_descriptor = _source_manifest(input_dir)
    source_shards = source["shards"]
    config = {
        "compression": compression,
        "row_group_records": row_group_records,
        "schema": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in PARQUET_MIRROR_SCHEMA
        ],
    }
    config_sha256 = _canonical_hash(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    manifest_path = output_dir / "manifest.json"

    if manifest_path.is_file():
        if not resume:
            raise FileExistsError("completed mirror exists; pass resume=True to verify it")
        manifest = _read_json_object(manifest_path)
        _validate_signed_json(manifest, "manifest_sha256", "mirror manifest")
        if manifest.get("source_manifest") != source_descriptor:
            raise ValueError("mirror manifest source fingerprint does not match input")
        if manifest.get("config_sha256") != config_sha256:
            raise ValueError("mirror manifest configuration does not match requested run")
        if manifest.get("config") != config:
            raise ValueError("mirror manifest configuration payload is invalid")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise ValueError("mirror manifest shards are malformed")
        if not all(isinstance(shard, dict) for shard in shards):
            raise ValueError("mirror manifest contains a malformed shard")
        typed_shards = [dict(shard) for shard in shards]
        _validate_completed_shards(typed_shards, source_shards)
        if len(typed_shards) != len(source_shards):
            raise ValueError("completed mirror does not cover every source shard")
        if manifest.get("records") != source_descriptor["records"]:
            raise ValueError("mirror manifest record count does not match its source")
        for source_shard in source_shards:
            _verify_source_shard(input_dir, source_shard)
        for shard in typed_shards:
            _verify_output_shard(output_dir, shard, row_group_records)
        return manifest

    completed: list[dict[str, Any]]
    pending: dict[str, Any] | None = None
    if checkpoint_path.is_file():
        if not resume:
            raise FileExistsError("mirror checkpoint exists; pass resume=True to continue")
        checkpoint = _read_json_object(checkpoint_path)
        _validate_signed_json(checkpoint, "checkpoint_sha256", "mirror checkpoint")
        if checkpoint.get("source_manifest") != source_descriptor:
            raise ValueError("mirror checkpoint source fingerprint does not match input")
        if checkpoint.get("config_sha256") != config_sha256:
            raise ValueError("mirror checkpoint configuration does not match requested run")
        if checkpoint.get("schema_version") != PARQUET_MIRROR_SCHEMA_VERSION:
            raise ValueError("mirror checkpoint schema version is unsupported")
        if checkpoint.get("status") not in {"running", "publishing", "complete"}:
            raise ValueError("mirror checkpoint status is invalid")
        checkpoint_shards = checkpoint.get("shards")
        if not isinstance(checkpoint_shards, list):
            raise ValueError("mirror checkpoint shards are malformed")
        completed = []
        for shard in checkpoint_shards:
            if not isinstance(shard, dict):
                raise ValueError("mirror checkpoint contains a malformed shard")
            _verify_output_shard(output_dir, shard, row_group_records)
            completed.append(shard)
        pending_value = checkpoint.get("pending_shard")
        if pending_value is not None:
            if not isinstance(pending_value, dict):
                raise ValueError("mirror checkpoint pending shard is malformed")
            pending = pending_value
    else:
        partials = list(output_dir.glob("*.parquet.part"))
        if resume and not partials:
            raise FileNotFoundError("resume requested but mirror checkpoint is absent")
        completed = []
        existing = list(output_dir.glob("*.parquet"))
        if existing:
            raise FileExistsError("untracked Parquet shards exist in output directory")

    _validate_completed_shards(completed, source_shards)
    for source_shard in source_shards[: len(completed)]:
        _verify_source_shard(input_dir, source_shard)

    if pending is not None:
        if len(completed) >= len(source_shards):
            raise ValueError("mirror checkpoint has a pending shard after source completion")
        _validate_completed_shards(completed + [pending], source_shards)
        pending_source = source_shards[len(completed)]
        _verify_source_shard(input_dir, pending_source)
        pending_final = output_dir / str(pending["name"])
        pending_partial = pending_final.with_name(f"{pending_final.name}.part")
        if pending_final.exists() and pending_partial.exists():
            raise ValueError("pending mirror shard exists as both partial and final")
        if pending_partial.exists():
            _publish_staged_shard(output_dir, pending, row_group_records)
        elif pending_final.exists():
            _verify_output_shard(output_dir, pending, row_group_records)
        else:
            raise ValueError("pending mirror shard is missing")
        completed.append(pending)
        _write_signed_json(
            checkpoint_path,
            _checkpoint_payload(
                source_manifest=source_descriptor,
                config_sha256=config_sha256,
                shards=completed,
                pending_shard=None,
                status="running",
            ),
            "checkpoint_sha256",
        )

    next_output_name = None
    if len(completed) < len(source_shards):
        next_source_name = str(source_shards[len(completed)]["name"])
        next_output_name = f"{Path(next_source_name).stem}.parquet"
    _discard_unjournaled_partial(output_dir, next_output_name)

    for written_this_run, source_shard in enumerate(
        source_shards[len(completed) :],
        start=1,
    ):
        source_name = str(source_shard["name"])
        output_name = f"{Path(source_name).stem}.parquet"
        output_path = output_dir / output_name
        if output_path.exists():
            raise FileExistsError(f"untracked mirror shard exists: {output_name}")
        descriptor = _stage_mirror_shard(
            input_dir / source_name,
            output_path,
            source_name=source_name,
            expected_sha256=str(source_shard["sha256"]),
            expected_records=int(source_shard["records"]),
            row_group_records=row_group_records,
            compression=compression,
        )
        _write_signed_json(
            checkpoint_path,
            _checkpoint_payload(
                source_manifest=source_descriptor,
                config_sha256=config_sha256,
                shards=completed,
                pending_shard=descriptor,
                status="publishing",
            ),
            "checkpoint_sha256",
        )
        _publish_staged_shard(output_dir, descriptor, row_group_records)
        completed.append(descriptor)
        _write_signed_json(
            checkpoint_path,
            _checkpoint_payload(
                source_manifest=source_descriptor,
                config_sha256=config_sha256,
                shards=completed,
                pending_shard=None,
                status="running",
            ),
            "checkpoint_sha256",
        )
        if stop_after_shards is not None and written_this_run >= stop_after_shards:
            raise MirrorInterrupted("deterministic interruption after complete mirror shard")

    records = sum(int(shard["records"]) for shard in completed)
    if records != source_descriptor["records"]:
        raise ValueError("mirror record count does not match source manifest")
    manifest_payload = {
        "schema_version": PARQUET_MIRROR_SCHEMA_VERSION,
        "status": "complete",
        "source_manifest": source_descriptor,
        "config": config,
        "config_sha256": config_sha256,
        "records": records,
        "shards": completed,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    manifest = _write_signed_json(manifest_path, manifest_payload, "manifest_sha256")
    _write_signed_json(
        checkpoint_path,
        _checkpoint_payload(
            source_manifest=source_descriptor,
            config_sha256=config_sha256,
            shards=completed,
            pending_shard=None,
            status="complete",
        ),
        "checkpoint_sha256",
    )
    return manifest
