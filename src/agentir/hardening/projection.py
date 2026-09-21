"""Allowlisted, reasoning-free trainer projections."""

from __future__ import annotations

import json
import re
from typing import Any

from agentir.hardening.contracts import (
    AdmissionState,
    HardenedRecord,
    LossAction,
    LossDecision,
    ProjectionResult,
)
from agentir.hardening.firewall import PRIVATE_PATH_PATTERN, admission_losses, scan_value
from agentir.ir.base import EventType, ObservationKind


def _text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.text is not None:
            parts.append(block.text)
        elif block.json_value is not None:
            parts.append(json.dumps(block.json_value, ensure_ascii=False, sort_keys=True))
    return redact_text("\n".join(part for part in parts if part))


def redact_text(value: str) -> str:
    """Apply only deterministic replacements approved by the privacy policy."""

    value = re.sub(
        PRIVATE_PATH_PATTERN,
        "<PRIVATE_PATH>",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "<PRIVATE_EMAIL>",
        value,
    )


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_value(child) for child in value]
    if isinstance(value, tuple):
        return [redact_value(child) for child in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _observation_text(event: Any) -> str:
    if event.observation is None:
        return ""
    parts: list[str] = []
    if event.observation.stdout:
        parts.append(event.observation.stdout)
    if event.observation.stderr:
        parts.append(event.observation.stderr)
    parts.append(_text(event.observation.content))
    return redact_text("\n".join(part for part in parts if part))


def _observation_status(event: Any) -> str:
    """Classify one observed result without treating failure as success."""

    observation = event.observation
    if observation is None:
        return "not_observed"
    if (
        observation.error_type
        or observation.kind is ObservationKind.ERROR
        or (observation.exit_code is not None and observation.exit_code != 0)
    ):
        return "error"
    if observation.truncated:
        return "truncated"
    return "matched"


def _observation_payload(event: Any) -> dict[str, Any] | None:
    """Return bounded, trainer-safe observation fields for one result event."""

    observation = event.observation
    if observation is None:
        return None
    return {
        "status": _observation_status(event),
        "content": _observation_text(event) or None,
        "exit_code": observation.exit_code,
        "error_type": observation.error_type,
        "truncated": observation.truncated,
    }


def _context_message(event: Any) -> dict[str, Any] | None:
    """Convert one visible message event to the tool-use context wire shape."""

    if event.event_type == EventType.SYSTEM_MESSAGE:
        role = "system"
    elif event.event_type == EventType.USER_MESSAGE:
        role = "user"
    elif event.event_type == EventType.ASSISTANT_MESSAGE:
        role = "assistant"
    else:
        return None
    content = _text(event.content)
    return {"role": role, "content": content} if content else None


def _blocking(result: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        finding
        for finding in scan_value(result)
        if finding[0]
        in {"HIDDEN_FIELD", "UNTRUSTED_FIELD", "HIDDEN_MARKER", "SECRET_PATTERN", "PRIVATE_VALUE"}
    )


def _loss(
    code: str,
    action: LossAction,
    message: str,
    *,
    event_id: str | None = None,
    field: str | None = None,
) -> LossDecision:
    return LossDecision(code=code, action=action, event_id=event_id, field=field, message=message)


def _project_sft(hardened: HardenedRecord, *, preview: bool = False) -> ProjectionResult:
    losses = list(admission_losses(hardened))
    if hardened.quality.state != AdmissionState.ACCEPTED and not preview:
        return ProjectionResult(
            target="sft",
            state=hardened.quality.state,
            losses=tuple(losses),
        )

    output: list[dict[str, Any]] = []
    omitted = 0
    for episode in hardened.record.episodes:
        for event in episode.events:
            event_type = event.event_type
            if event_type in {EventType.REASONING, EventType.PLAN}:
                omitted += 1
                losses.append(
                    _loss(
                        "TYPED_REASONING_DROPPED",
                        LossAction.DROP,
                        "Typed reasoning/plan event is excluded from the SFT view.",
                        event_id=event.event_id,
                    )
                )
                continue
            if event_type in {
                EventType.SYSTEM_MESSAGE,
                EventType.USER_MESSAGE,
                EventType.ASSISTANT_MESSAGE,
            }:
                content = _text(event.content)
                if content:
                    output.append(
                        {"role": event.role.value if event.role else "user", "content": content}
                    )
                continue
            if event_type == EventType.TOOL_CALL:
                if event.action is None or not event.action.tool_call_id:
                    omitted += 1
                    losses.append(
                        _loss(
                            "MISSING_TOOL_CALL_ID",
                            LossAction.QUARANTINE,
                            "Tool call identity is required for SFT lowering.",
                            event_id=event.event_id,
                        )
                    )
                    continue
                content = _text(event.content)
                output.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": event.action.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": event.action.tool_name or "unknown",
                                    "arguments": json.dumps(
                                        redact_value(event.action.arguments),
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    ),
                                },
                            }
                        ],
                    }
                )
                continue
            if event_type == EventType.TOOL_RESULT:
                if event.action is None or not event.action.tool_call_id:
                    omitted += 1
                    continue
                output.append(
                    {
                        "role": "tool",
                        "tool_call_id": event.action.tool_call_id,
                        "content": _observation_text(event),
                    }
                )
                continue
            omitted += 1
            losses.append(
                _loss(
                    "EVENT_NOT_IN_SFT_ALLOWLIST",
                    LossAction.DROP,
                    "Event type is not part of the SFT projection allowlist.",
                    event_id=event.event_id,
                    field="event_type",
                )
            )

    findings = _blocking(output)
    if findings:
        return ProjectionResult(
            target="sft",
            state=AdmissionState.QUARANTINED,
            losses=tuple(
                losses
                + [
                    _loss(
                        code,
                        LossAction.QUARANTINE,
                        "Trainer output failed the recursive release firewall.",
                        field=path,
                    )
                    for code, path in findings
                ]
            ),
            omitted_events=omitted,
        )
    return ProjectionResult(
        target="sft",
        state=hardened.quality.state if preview else AdmissionState.ACCEPTED,
        output=output,
        losses=tuple(losses),
        emitted_events=len(output),
        omitted_events=omitted,
    )


def _project_tool_use(hardened: HardenedRecord, *, preview: bool = False) -> ProjectionResult:
    losses = list(admission_losses(hardened))
    if hardened.quality.state != AdmissionState.ACCEPTED and not preview:
        return ProjectionResult(
            target="tool_use", state=hardened.quality.state, losses=tuple(losses)
        )

    output: list[dict[str, Any]] = []
    omitted = 0
    blocking_losses: list[LossDecision] = []
    observations_by_call_id: dict[str, Any] = {}
    events = [
        event
        for episode in hardened.record.episodes
        for event in episode.events
    ]
    for event in events:
        if (
            event.event_type == EventType.TOOL_RESULT
            and event.action is not None
            and event.action.tool_call_id
        ):
            observations_by_call_id.setdefault(event.action.tool_call_id, event)

    context: list[dict[str, Any]] = []
    for event in events:
        message = _context_message(event)
        if message is not None:
            context.append(message)
            continue
        if event.event_type == EventType.TOOL_CALL:
            if event.action is None or not event.action.tool_call_id:
                omitted += 1
                continue
            if not context:
                blocking_losses.append(
                    _loss(
                        "MISSING_TOOL_CONTEXT",
                        LossAction.QUARANTINE,
                        "Tool-use training requires visible conversation context before the call.",
                        event_id=event.event_id,
                    )
                )
                omitted += 1
                continue
            call_id = event.action.tool_call_id
            observation_event = observations_by_call_id.get(call_id)
            output.append(
                {
                    "event_id": event.event_id,
                    "context": [dict(item) for item in context],
                    "call_id": call_id,
                    "tool_name": event.action.tool_name or "unknown",
                    "arguments": redact_value(event.action.arguments),
                    "observation_event_id": (
                        observation_event.event_id if observation_event is not None else None
                    ),
                    "observation": _observation_payload(observation_event)
                    if observation_event is not None
                    else None,
                }
            )
            context.append(
                {
                    "role": "assistant",
                    "content": _text(event.content) or None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": event.action.tool_name or "unknown",
                                "arguments": redact_value(event.action.arguments),
                            },
                        }
                    ],
                }
            )
            continue
        if event.event_type == EventType.TOOL_RESULT:
            if event.action is None or not event.action.tool_call_id:
                continue
            context.append(
                {
                    "role": "tool",
                    "tool_call_id": event.action.tool_call_id,
                    "content": _observation_text(event),
                }
            )
    findings = _blocking(output)
    if findings or blocking_losses:
        return ProjectionResult(
            target="tool_use",
            state=AdmissionState.QUARANTINED,
            losses=tuple(
                losses
                + blocking_losses
                + [
                    _loss(
                        code,
                        LossAction.QUARANTINE,
                        "Tool-use output failed the recursive release firewall.",
                        field=path,
                    )
                    for code, path in findings
                ]
            ),
            omitted_events=omitted,
        )
    return ProjectionResult(
        target="tool_use",
        state=hardened.quality.state if preview else AdmissionState.ACCEPTED,
        output=output,
        losses=tuple(losses),
        emitted_events=len(output),
        omitted_events=omitted,
    )


def project(
    hardened: HardenedRecord, target: str, *, preview: bool = False
) -> ProjectionResult:
    """Lower to an allowlisted release view or an explicitly non-release preview.

    A preview may show the deterministic, reasoning-free lowering for a
    candidate row so a reviewer can inspect it.  Its state remains candidate
    or quarantined, and the normal release materializer never accepts preview
    output because it requires an ``accepted`` projection.
    """

    if target == "sft":
        return _project_sft(hardened, preview=preview)
    if target == "tool_use":
        return _project_tool_use(hardened, preview=preview)
    return ProjectionResult(
        target=target,
        state=AdmissionState.REJECTED,
        losses=(
            _loss(
                "UNSUPPORTED_PROJECTION",
                LossAction.REJECT,
                "Projection target is not implemented by this hardening gate.",
            ),
        ),
    )
