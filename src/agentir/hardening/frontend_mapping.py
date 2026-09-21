"""Provider-neutral mapping from published frontend rows into AgentIR.

Provider-specific extractors are expected to publish a bounded row with a
``messages`` list.  This adapter owns only the common message/tool contract;
source identity and release admission remain owned by ``frontend_stream`` and
the hardening firewall.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from agentir.ir.action import Action
from agentir.ir.base import (
    ActionKind,
    CallStyle,
    ContentType,
    EventType,
    IRLevel,
    MessageRole,
    ObservationKind,
    OutcomeStatus,
)
from agentir.ir.content import ContentBlock
from agentir.ir.episode import Episode
from agentir.ir.event import Event
from agentir.ir.observation import Observation
from agentir.ir.outcome import Outcome
from agentir.ir.provenance import Provenance
from agentir.ir.record import AgentIRRecord
from agentir.ir.source import SourceRef
from agentir.ir.task import TaskSpec
from agentir.ir.tool import ToolSpec

PARSER_REVISION = "agentir-hardening-frontend-mapper/v1"
FRONTEND_MAPPER_REVISION = PARSER_REVISION
_TOOL_REGISTRY_FIELDS = (
    "tools",
    "tool_registry",
    "tool_definitions",
    "tools_json",
    "tool_registry_json",
)
_TOOL_REGISTRY_REVISION_FIELDS = (
    "tool_registry_revision",
    "registry_revision",
)
_MAX_TOOL_REGISTRY_JSON_BYTES = 2 * 1024 * 1024


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Narrow untrusted JSON containers before reading their fields."""

    return value if isinstance(value, Mapping) else {}


def _json_value(value: Any) -> Any:
    """Decode one bounded JSON string without accepting arbitrary raw text."""

    if not isinstance(value, str):
        return value
    if len(value.encode("utf-8")) > _MAX_TOOL_REGISTRY_JSON_BYTES:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _tool_registry_revision(row: dict[str, Any]) -> str | None:
    for container in (row, _as_mapping(row.get("source_origin"))):
        for key in _TOOL_REGISTRY_REVISION_FIELDS:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def tool_registry_from_row(row: dict[str, Any]) -> tuple[list[ToolSpec], str | None, int]:
    """Extract only explicitly declared, schema-bearing tools from a row.

    The returned revision is metadata only.  A caller must still join it to a
    digest-verified :class:`ToolRegistryArtifact` before trainer release.
    The final integer counts malformed declarations; they are never silently
    converted into tool definitions.
    """

    declared: Any = None
    for container in (row, _as_mapping(row.get("source_origin"))):
        for key in _TOOL_REGISTRY_FIELDS:
            if key in container:
                declared = _json_value(container[key])
                break
        if declared is not None:
            break
    if isinstance(declared, dict):
        for key in ("tools", "definitions", "tool_definitions"):
            if key in declared:
                declared = _json_value(declared[key])
                break
    if not isinstance(declared, list):
        return [], _tool_registry_revision(row), 0

    revision = _tool_registry_revision(row)
    tools: list[ToolSpec] = []
    malformed = 0
    for ordinal, item in enumerate(declared):
        if not isinstance(item, dict):
            malformed += 1
            continue
        function = _as_mapping(item.get("function")) or item
        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if parameters is None:
            parameters = function.get("input_schema")
        if not isinstance(name, str) or not name.strip() or not isinstance(parameters, dict):
            malformed += 1
            continue
        normalized_name = name.strip().lower()
        identity = json.dumps(
            {
                "name": name.strip(),
                "parameters": parameters,
                "revision": revision or "unversioned",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        tools.append(
            ToolSpec(
                tool_id=stable_id("tool", identity),
                name=name.strip(),
                description=description.strip() if isinstance(description, str) else None,
                input_schema=parameters,
                source="row_declared",
                normalized_name=normalized_name,
                metadata={
                    "declaration_ordinal": ordinal,
                    "registry_revision": revision,
                },
            )
        )
    return tools, revision, malformed


def role_of(value: Any) -> str:
    if isinstance(value, Mapping):
        role = value.get("role")
        if isinstance(role, str):
            return role.lower()
    return "unknown"


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [text_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "output", "result", "value"):
            if key in value:
                result = text_content(value[key])
                if result:
                    return result
    return ""


def action_kind(name: str) -> ActionKind:
    lowered = name.lower()
    if lowered in {"shell", "bash", "terminal", "run_command", "exec"}:
        return ActionKind.TERMINAL
    if lowered in {"read", "read_file", "cat", "glob", "search", "search_file_content"}:
        return ActionKind.FILE_READ
    if lowered in {"write", "write_file", "edit", "apply_patch", "replace"}:
        return ActionKind.FILE_WRITE
    return ActionKind.GENERIC_TOOL


def call_items(message: dict[str, Any], message_index: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in ("tool_calls", "tool_call", "tool_uses", "tool_use", "function_call"):
        for ordinal, item in enumerate(_as_list(message.get(field))):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower().replace("-", "_")
            if "result" in item_type or "output" in item_type:
                continue
            function = _as_mapping(item.get("function"))
            call_id = item.get("id") or item.get("call_id") or item.get("callID")
            name = item.get("name") or function.get("name") or "unknown"
            arguments = item.get("input")
            if arguments is None:
                arguments = item.get("arguments", function.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"value": arguments}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            result.append(
                {
                    "id": str(call_id) if call_id is not None else None,
                    "name": str(name),
                    "arguments": arguments,
                    "field": field,
                    "ordinal": ordinal,
                    "message_index": message_index,
                }
            )
    return result


def result_items(message: dict[str, Any], message_index: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in ("tool_results", "tool_result", "toolResult", "function_call_output"):
        for ordinal, item in enumerate(_as_list(message.get(field))):
            if not isinstance(item, dict):
                continue
            call_id = (
                item.get("tool_call_id")
                or item.get("call_id")
                or item.get("callID")
                or item.get("id")
            )
            name = item.get("tool_name") or item.get("name") or item.get("tool") or "unknown"
            output = item.get("output", item.get("result", item.get("content", "")))
            result.append(
                {
                    "id": str(call_id) if call_id is not None else None,
                    "name": str(name),
                    "output": output,
                    "field": field,
                    "ordinal": ordinal,
                    "message_index": message_index,
                }
            )
    if role_of(message) == "tool":
        call_id = message.get("tool_call_id") or message.get("call_id") or message.get("callID")
        result.append(
            {
                "id": str(call_id) if call_id is not None else None,
                "name": str(message.get("tool_name") or "unknown"),
                "output": message.get("content", ""),
                "field": "role_tool",
                "ordinal": 0,
                "message_index": message_index,
            }
        )
    content = message.get("content")
    for ordinal, item in enumerate(_as_list(content)):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower().replace("-", "_")
        if item_type not in {
            "tool_result",
            "tool_response",
            "function_call_output",
            "custom_tool_call_output",
        }:
            continue
        call_id = item.get("tool_call_id") or item.get("call_id") or item.get("tool_use_id")
        result.append(
            {
                "id": str(call_id) if call_id is not None else None,
                "name": str(item.get("name") or item.get("tool_name") or "unknown"),
                "output": item.get("content", item.get("output", item.get("result", ""))),
                "field": "content_part",
                "ordinal": ordinal,
                "message_index": message_index,
            }
        )
    return result


def row_shape(row: dict[str, Any]) -> dict[str, Any]:
    message_value = row.get("messages")
    messages: list[Any] = message_value if isinstance(message_value, list) else []
    roles = Counter(role_of(message) for message in messages if isinstance(message, dict))
    calls = [
        call
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        for call in call_items(message, index)
    ]
    results = [
        result
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        for result in result_items(message, index)
    ]
    return {
        "messages": messages,
        "roles": dict(roles),
        "calls": calls,
        "results": results,
        "dialogue": roles.get("user", 0) > 0 and roles.get("assistant", 0) > 0,
        "tool_bearing": bool(calls or results),
    }


def quality_gate(row: dict[str, Any]) -> str:
    for container in (row, _as_mapping(row.get("source_origin"))):
        for key in ("quality_gate", "gate"):
            value = container.get(key)
            if isinstance(value, str):
                return value
        assessment = _as_mapping(container.get("quality_assessment"))
        gate = assessment.get("gate")
        if isinstance(gate, str):
            return gate
    return "unspecified"


def _content_blocks(value: Any) -> list[ContentBlock]:
    text = text_content(value)
    return [ContentBlock(type=ContentType.TEXT, text=text)] if text else []


def _message_role(role: str) -> MessageRole:
    return {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
        "tool": MessageRole.TOOL,
        "developer": MessageRole.DEVELOPER,
    }.get(role, MessageRole.UNKNOWN)


def map_row(
    row: dict[str, Any], provider: str, source_line: int
) -> tuple[AgentIRRecord, dict[str, Any]]:
    """Map one published frontend row without retaining its provider envelope."""

    shape = row_shape(row)
    tool_registry, tool_registry_revision, malformed_tool_definitions = tool_registry_from_row(row)
    source_origin = _as_mapping(row.get("source_origin"))
    row_id = stable_id(
        "row",
        f"{provider}:{source_line}:{source_origin.get('source_file_sha256', '')}",
    )
    parser = f"agentir.frontend.{provider}"
    source = SourceRef(
        dataset="session-corpus",
        row_id=row_id,
        row_index=source_line,
        framework="ai-data-extraction",
        framework_version=str(source_origin.get("extractor_version", "unknown")),
        format="frontend-jsonl",
        original_source=provider,
    )
    events: list[Event] = []
    call_events: dict[str, str] = {}
    missing_call_ids = 0
    missing_result_ids = 0
    unknown_roles = 0
    dropped_message_fields: Counter[str] = Counter()
    call_count = 0
    result_count = 0
    call_ids: set[str] = set()
    result_ids: set[str] = set()

    for message_index, message in enumerate(shape["messages"]):
        if not isinstance(message, dict):
            unknown_roles += 1
            continue
        role = role_of(message)
        if role not in {"system", "user", "assistant", "developer", "tool"}:
            unknown_roles += 1
            continue
        allowed_message_fields = {
            "role",
            "content",
            "timestamp",
            "tool_calls",
            "tool_call",
            "tool_uses",
            "tool_use",
            "function_call",
            "tool_results",
            "tool_result",
            "toolResult",
            "function_call_output",
            "tool_call_id",
            "call_id",
            "callID",
            "tool_name",
            "status",
        }
        for field_name in message:
            if field_name not in allowed_message_fields:
                dropped_message_fields[field_name] += 1

        calls = call_items(message, message_index)
        results = result_items(message, message_index)
        if role != "tool":
            visible = text_content(message.get("content"))
            if visible:
                event_type = (
                    EventType.SYSTEM_MESSAGE
                    if role in {"system", "developer"}
                    else EventType.USER_MESSAGE
                    if role == "user"
                    else EventType.ASSISTANT_MESSAGE
                )
                events.append(
                    Event(
                        event_id=stable_id("event", f"{row_id}:message:{message_index}"),
                        idx=len(events),
                        event_type=event_type,
                        role=_message_role(role),
                        content=_content_blocks(message.get("content")),
                        timestamp=(
                            str(message.get("timestamp"))
                            if message.get("timestamp") is not None
                            else None
                        ),
                        provenance=Provenance(
                            row_id=row_id,
                            row_index=message_index,
                            source_field=f"messages[{message_index}].content",
                            parser=parser,
                        ),
                    )
                )
        for ordinal, call in enumerate(calls):
            call_id = call["id"] or stable_id("call", f"{row_id}:{message_index}:{ordinal}")
            if call["id"] is None:
                missing_call_ids += 1
            else:
                call_ids.add(call_id)
            event_id = stable_id("event", f"{row_id}:call:{message_index}:{ordinal}")
            call_events[call_id] = event_id
            events.append(
                Event(
                    event_id=event_id,
                    idx=len(events),
                    event_type=EventType.TOOL_CALL,
                    actor_id="assistant",
                    role=MessageRole.ASSISTANT,
                    action=Action(
                        kind=action_kind(call["name"]),
                        tool_name=call["name"],
                        tool_call_id=call_id,
                        arguments=call["arguments"],
                        normalized_arguments=call["arguments"],
                        call_style=CallStyle.NATIVE_TOOL_CALL,
                    ),
                    provenance=Provenance(
                        row_id=row_id,
                        row_index=message_index,
                        source_field=f"messages[{message_index}].{call['field']}[{ordinal}]",
                        parser=parser,
                    ),
                )
            )
            call_count += 1
        for ordinal, result in enumerate(results):
            result_id = result["id"]
            if result_id is None:
                missing_result_ids += 1
                result_id = stable_id("result", f"{row_id}:{message_index}:{ordinal}")
            else:
                result_ids.add(result_id)
            action_event_id = call_events.get(result_id)
            output = result["output"]
            output_text = text_content(output)
            events.append(
                Event(
                    event_id=stable_id("event", f"{row_id}:result:{message_index}:{ordinal}"),
                    idx=len(events),
                    event_type=EventType.TOOL_RESULT,
                    actor_id="tool",
                    role=MessageRole.TOOL,
                    parent_event_ids=[action_event_id] if action_event_id else [],
                    action=Action(
                        kind=action_kind(result["name"]),
                        tool_name=result["name"],
                        tool_call_id=result_id,
                    ),
                    observation=Observation(
                        kind=ObservationKind.TOOL_JSON,
                        content=_content_blocks(output),
                        stdout=output_text if isinstance(output, str) else None,
                        metadata={
                            "tool_call_id": result_id,
                            "source_status": message.get("status"),
                        },
                    ),
                    provenance=Provenance(
                        row_id=row_id,
                        row_index=message_index,
                        source_field=f"messages[{message_index}].{result['field']}[{ordinal}]",
                        parser=parser,
                    ),
                )
            )
            result_count += 1

    first_user = next(
        (
            message.get("content")
            for message in shape["messages"]
            if isinstance(message, dict) and role_of(message) == "user"
        ),
        None,
    )
    record = AgentIRRecord(
        record_id=row_id,
        level=IRLevel.PARSED,
        source=source,
        task=TaskSpec(task_id=row_id, instruction=text_content(first_user)),
        episodes=[
            Episode(
                episode_id=stable_id("episode", row_id),
                task_id=row_id,
                attempt_id=row_id,
                events=events,
            )
        ],
        outcome=Outcome(status=OutcomeStatus.UNKNOWN),
        tool_registry=tool_registry,
        metadata={"provider": provider, "parser_revision": PARSER_REVISION},
    )
    source_ref_missing = [
        field
        for field in (
            "source_sha256",
            "snapshot_revision",
            "byte_start",
            "byte_end",
            "parser_revision",
        )
        if field not in SourceRef.model_fields
    ]
    return record, {
        "message_count": len(shape["messages"]),
        "role_counts": shape["roles"],
        "call_count": call_count,
        "result_count": result_count,
        "missing_call_ids": missing_call_ids,
        "missing_result_ids": missing_result_ids,
        "unmatched_call_ids": len(call_ids - result_ids),
        "unmatched_result_ids": len(result_ids - call_ids),
        "matched_tool_edges": len(call_ids & result_ids),
        "unknown_roles": unknown_roles,
        "dropped_message_fields": dict(dropped_message_fields),
        "source_ref_fields_missing": source_ref_missing,
        "source_digest_present": bool(
            source_origin.get("source_file_sha256") or source_origin.get("source_fingerprint")
        ),
        "source_range_present": any(
            key in source_origin
            for key in ("message_line_range", "message_index_range", "source_event_line_range")
        ),
        "tool_registry_count": len(tool_registry),
        "tool_registry_revision": tool_registry_revision,
        "malformed_tool_definitions": malformed_tool_definitions,
    }
