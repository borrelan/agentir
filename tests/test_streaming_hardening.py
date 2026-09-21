from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentir.hardening.streaming import CompileInterrupted, compile_jsonl


def _write_input(path: Path) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for index in range(9):
            destination.write(json.dumps({"id": index, "text": f"row-{index}"}) + "\n")


def _transform(row: dict, source_line: int) -> dict:
    return {"unit_id": f"unit-{row['id']}", "source_line": source_line, "text": row["text"]}


def _filter_odd_rows(row: dict, source_line: int) -> dict | None:
    if row["id"] % 2:
        return None
    return _transform(row, source_line)


def test_interrupted_and_resumed_compile_match_clean_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_input(source)
    clean = compile_jsonl(
        source,
        tmp_path / "clean",
        _transform,
        parser_revision="fixture/v1",
        pass_config={"projection": "sft"},
        shard_records=2,
        shard_bytes=1024,
    )

    with pytest.raises(CompileInterrupted):
        compile_jsonl(
            source,
            tmp_path / "resumed",
            _transform,
            parser_revision="fixture/v1",
            pass_config={"projection": "sft"},
            shard_records=2,
            shard_bytes=1024,
            stop_after_source_records=5,
        )
    resumed = compile_jsonl(
        source,
        tmp_path / "resumed",
        _transform,
        parser_revision="fixture/v1",
        pass_config={"projection": "sft"},
        shard_records=2,
        shard_bytes=1024,
        resume=True,
    )

    assert resumed == clean
    assert resumed["records"] == 9
    assert resumed["input_manifest_hash"]
    ranges = [(shard["source_line_start"], shard["source_line_end"]) for shard in resumed["shards"]]
    assert ranges == [(1, 2), (3, 4), (5, 6), (7, 8), (9, 9)]
    assert not list((tmp_path / "resumed").glob("*.part"))
    checkpoint = json.loads((tmp_path / "resumed" / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["input_manifest_hash"] == resumed["input_manifest_hash"]
    assert checkpoint["peak_rss_bytes"] is None or checkpoint["peak_rss_bytes"] > 0


def test_resume_rejects_changed_pass_configuration(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_input(source)
    with pytest.raises(CompileInterrupted):
        compile_jsonl(
            source,
            tmp_path / "output",
            _transform,
            parser_revision="fixture/v1",
            pass_config={"projection": "sft"},
            stop_after_source_records=1,
        )

    with pytest.raises(ValueError, match="pass configuration"):
        compile_jsonl(
            source,
            tmp_path / "output",
            _transform,
            parser_revision="fixture/v1",
            pass_config={"projection": "tool_use"},
            resume=True,
        )


def test_resume_rejects_changed_committed_shard(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_input(source)
    output = tmp_path / "output"
    compile_jsonl(
        source,
        output,
        _transform,
        parser_revision="fixture/v1",
        pass_config={},
        shard_records=2,
    )
    shard = output / "shard-000000.jsonl"
    shard.write_text(shard.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    (output / "manifest.json").unlink()

    with pytest.raises(ValueError, match="shard is missing or changed"):
        compile_jsonl(
            source,
            output,
            _transform,
            parser_revision="fixture/v1",
            pass_config={},
            resume=True,
        )


def test_bounded_limit_tracks_filtered_source_rows_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_input(source)
    clean = compile_jsonl(
        source,
        tmp_path / "clean",
        _filter_odd_rows,
        parser_revision="fixture/v1",
        pass_config={"projection": "decision"},
        shard_records=2,
        source_record_limit=5,
    )

    with pytest.raises(CompileInterrupted):
        compile_jsonl(
            source,
            tmp_path / "resumed",
            _filter_odd_rows,
            parser_revision="fixture/v1",
            pass_config={"projection": "decision"},
            shard_records=2,
            source_record_limit=5,
            stop_after_source_records=4,
        )
    resumed = compile_jsonl(
        source,
        tmp_path / "resumed",
        _filter_odd_rows,
        parser_revision="fixture/v1",
        pass_config={"projection": "decision"},
        shard_records=2,
        source_record_limit=5,
        resume=True,
    )

    assert resumed == clean
    assert resumed["status"] == "bounded"
    assert resumed["completed_source_records"] == 5
    assert resumed["completed_source_line"] == 5
    assert resumed["records"] == 3


def test_bounded_line_range_selects_middle_without_claiming_full_source(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_input(source)
    result = compile_jsonl(
        source,
        tmp_path / "middle",
        _transform,
        parser_revision="fixture/v1",
        pass_config={"projection": "decision"},
        shard_records=2,
        source_record_limit=3,
        source_line_start=4,
    )

    assert result["status"] == "bounded"
    assert result["source_line_start"] == 4
    assert result["completed_source_line"] == 6
    assert result["completed_source_records"] == 3
    assert result["records"] == 3
    rows = []
    for shard in sorted((tmp_path / "middle").glob("shard-*.jsonl")):
        rows.extend(json.loads(line) for line in shard.read_text().splitlines())
    assert [row["unit_id"] for row in rows] == ["unit-3", "unit-4", "unit-5"]


def test_resume_does_not_skip_row_after_preappend_byte_commit(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_input(source)
    kwargs = {
        "parser_revision": "fixture/v1",
        "pass_config": {"projection": "decision"},
        "shard_records": 100,
        "shard_bytes": 80,
        "source_record_limit": 4,
    }
    clean = compile_jsonl(source, tmp_path / "clean", _transform, **kwargs)

    with pytest.raises(CompileInterrupted):
        compile_jsonl(
            source,
            tmp_path / "resumed",
            _transform,
            stop_after_source_records=2,
            **kwargs,
        )
    resumed = compile_jsonl(source, tmp_path / "resumed", _transform, resume=True, **kwargs)

    assert resumed == clean
    assert resumed["completed_source_records"] == 4
    assert resumed["records"] == 4
