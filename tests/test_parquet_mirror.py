"""Tests for the bounded, lossless hardening Parquet mirror."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import agentir.hardening.parquet_mirror as parquet_mirror
from agentir.hardening import (
    PARQUET_MIRROR_SCHEMA,
    MirrorInterrupted,
    mirror_jsonl_shards_to_parquet,
)
from agentir.hardening.streaming import compile_jsonl


def _write_input(path: Path, count: int = 5) -> list[dict]:
    rows = [
        {
            "unit_id": f"unit-{index}",
            "provider": "fixture",
            "admission": {"state": "accepted"},
            "text": f"row-{index}",
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def _compile_source(tmp_path: Path, *, count: int = 5) -> tuple[Path, list[dict]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jsonl"
    rows = _write_input(source, count=count)
    release = tmp_path / "release"
    compile_jsonl(
        source,
        release,
        lambda row, _line: row,
        parser_revision="fixture/v1",
        pass_config={"view": "decision"},
        shard_records=2,
    )
    return release, rows


def _read_mirror(output: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        assert parquet.schema_arrow == PARQUET_MIRROR_SCHEMA
        for batch in parquet.iter_batches(batch_size=2):
            rows.extend(batch.to_pylist())
    return rows


def test_mirror_round_trips_canonical_records_with_bounded_row_groups(tmp_path: Path) -> None:
    release, source_rows = _compile_source(tmp_path)
    output = tmp_path / "mirror"

    manifest = mirror_jsonl_shards_to_parquet(
        release,
        output,
        row_group_records=1,
    )

    rows = _read_mirror(output)
    decoded = [json.loads(row["record_json"]) for row in rows]
    assert decoded == source_rows
    assert manifest["records"] == len(source_rows)
    assert len(manifest["shards"]) == 3
    assert [row["source_line"] for row in rows] == [1, 2, 1, 2, 1]
    assert all(row["unit_id"] == f"unit-{index}" for index, row in enumerate(rows))
    for row in rows:
        assert (
            row["record_sha256"] == hashlib.sha256(row["record_json"].encode("utf-8")).hexdigest()
        )
    for path in output.glob("*.parquet"):
        parquet = pq.ParquetFile(path)
        assert all(
            parquet.metadata.row_group(index).num_rows <= 1
            for index in range(parquet.metadata.num_row_groups)
        )


def test_interrupted_mirror_resumes_without_rewriting_complete_shards(tmp_path: Path) -> None:
    release, _rows = _compile_source(tmp_path)
    output = tmp_path / "mirror"
    with pytest.raises(MirrorInterrupted):
        mirror_jsonl_shards_to_parquet(
            release,
            output,
            row_group_records=2,
            stop_after_shards=1,
        )
    first = output / "shard-000000.parquet"
    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    first_mtime = first.stat().st_mtime_ns

    manifest = mirror_jsonl_shards_to_parquet(
        release,
        output,
        row_group_records=2,
        resume=True,
    )
    resumed = mirror_jsonl_shards_to_parquet(
        release,
        output,
        row_group_records=2,
        resume=True,
    )

    assert resumed == manifest
    assert hashlib.sha256(first.read_bytes()).hexdigest() == first_hash
    assert first.stat().st_mtime_ns == first_mtime
    assert not list(output.glob("*.part"))


@pytest.mark.parametrize("crash_after_rename", [False, True])
def test_resume_recovers_both_pending_publication_crash_positions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after_rename: bool,
) -> None:
    release, source_rows = _compile_source(tmp_path)
    output = tmp_path / "mirror"

    if crash_after_rename:
        original_write = parquet_mirror._write_signed_json
        checkpoint_writes = 0

        def fail_committed_checkpoint(
            path: Path,
            payload: dict,
            hash_field: str,
        ) -> dict:
            nonlocal checkpoint_writes
            if path.name == "checkpoint.json":
                checkpoint_writes += 1
                if checkpoint_writes == 2:
                    raise OSError("injected crash after rename")
            return original_write(path, payload, hash_field)

        monkeypatch.setattr(parquet_mirror, "_write_signed_json", fail_committed_checkpoint)
    else:
        original_publish = parquet_mirror._publish_staged_shard

        def fail_publish(*_args: object, **_kwargs: object) -> None:
            raise OSError("injected crash before rename")

        monkeypatch.setattr(parquet_mirror, "_publish_staged_shard", fail_publish)

    with pytest.raises(OSError, match="injected crash"):
        mirror_jsonl_shards_to_parquet(release, output, row_group_records=2)

    if crash_after_rename:
        monkeypatch.setattr(parquet_mirror, "_write_signed_json", original_write)
    else:
        monkeypatch.setattr(parquet_mirror, "_publish_staged_shard", original_publish)

    manifest = mirror_jsonl_shards_to_parquet(
        release,
        output,
        row_group_records=2,
        resume=True,
    )

    assert manifest["records"] == len(source_rows)
    assert [json.loads(row["record_json"]) for row in _read_mirror(output)] == source_rows
    assert not list(output.glob("*.part"))


def test_resume_rejects_changed_source_and_output_shards(tmp_path: Path) -> None:
    source_release, _rows = _compile_source(tmp_path / "source-case")
    source_output = tmp_path / "source-case" / "mirror"
    mirror_jsonl_shards_to_parquet(source_release, source_output)
    source_shard = source_release / "shard-000000.jsonl"
    source_shard.write_text(source_shard.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source shard is missing or changed"):
        mirror_jsonl_shards_to_parquet(source_release, source_output, resume=True)

    output_release, _rows = _compile_source(tmp_path / "output-case")
    output = tmp_path / "output-case" / "mirror"
    mirror_jsonl_shards_to_parquet(output_release, output)
    output_shard = output / "shard-000000.parquet"
    output_shard.write_bytes(output_shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="mirror shard is missing or changed"):
        mirror_jsonl_shards_to_parquet(output_release, output, resume=True)


def test_mirror_rejects_changed_source_manifest_and_configuration(tmp_path: Path) -> None:
    release, _rows = _compile_source(tmp_path)
    output = tmp_path / "mirror"
    mirror_jsonl_shards_to_parquet(release, output, row_group_records=2)

    with pytest.raises(ValueError, match="configuration"):
        mirror_jsonl_shards_to_parquet(
            release,
            output,
            row_group_records=3,
            resume=True,
        )

    source_manifest = release / "manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["records"] += 1
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest hash is invalid"):
        mirror_jsonl_shards_to_parquet(release, output, row_group_records=2, resume=True)


def test_empty_release_produces_complete_empty_mirror(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    release = tmp_path / "release"
    compile_jsonl(
        source,
        release,
        lambda row, _line: row,
        parser_revision="fixture/v1",
        pass_config={},
    )

    manifest = mirror_jsonl_shards_to_parquet(release, tmp_path / "mirror")

    assert manifest["status"] == "complete"
    assert manifest["records"] == 0
    assert manifest["shards"] == []
    assert not list((tmp_path / "mirror").glob("*.parquet"))
