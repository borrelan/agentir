"""Strict, backend-neutral release views for post-training datasets.

These models describe what may cross the canonical release boundary. They do
not infer preferences, rewards, or verifier outcomes from a transcript.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from agentir.hardening.contracts import (
    REQUIRED_QUALITY_DIMENSIONS,
    PrivacyPolicy,
    ProjectionResult,
    QualityDimension,
    SourceClass,
    SourceRange,
)
from agentir.hardening.firewall import scan_value


class ReleaseView(StrEnum):
    """Finite set of release products understood by downstream trainers."""

    SFT = "sft"
    TOOL_USE = "tool_use"
    PREFERENCE = "preference"
    RL_PROMPT = "rl_prompt"
    RL_ROLLOUT = "rl_rollout"


class DatasetSplit(StrEnum):
    """Source-independent dataset split labels."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DecisionSource(StrEnum):
    """Authority that produced the release admission decision."""

    HUMAN_REVIEW = "human_review"
    AUTOMATED_AUDIT = "automated_audit"
    VERIFIER = "verifier"
    EXPLICIT_SOURCE = "explicit_source"
    TASK_MANIFEST = "task_manifest"


class ModelTier(StrEnum):
    """Finite provenance/sampling tier for a trainer release row."""

    TIER1_FRONTIER = "tier1_frontier"
    TIER2_OPEN_SOURCE = "tier2_open_source"
    TIER3_LOCAL = "tier3_local"
    UNCLASSIFIED = "unclassified"


class IdentityStatus(StrEnum):
    """Evidence state for an agent, harness, or model identity."""

    OBSERVED = "observed"
    REVIEWED = "reviewed"
    NOT_OBSERVED = "not_observed"


class ReleaseIdentity(BaseModel):
    """Evidence-bound identity; provider names never fill this implicitly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: IdentityStatus = IdentityStatus.NOT_OBSERVED
    name: str | None = None
    version: str | None = None
    evidence_ids: tuple[str, ...] = ()

    @field_validator("name", "version", mode="before")
    @classmethod
    def normalize_optional_identity_text(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("release identity name/version must be non-empty when supplied")

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_identity_evidence(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("release identity evidence_ids must be a sequence")
        normalized = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
        if len(normalized) != len(value):
            raise ValueError("release identity evidence_ids must contain non-empty strings")
        return normalized

    @model_validator(mode="after")
    def validate_identity_state(self) -> ReleaseIdentity:
        if self.status is IdentityStatus.NOT_OBSERVED:
            if self.name is not None or self.version is not None or self.evidence_ids:
                raise ValueError("not-observed identity cannot carry values or evidence")
            return self
        if self.name is None:
            raise ValueError("observed/reviewed identity requires a name")
        if not self.evidence_ids:
            raise ValueError("observed/reviewed identity requires evidence_ids")
        return self


class ObservationStatus(StrEnum):
    """Explicit tool-observation state; absence is never implicit success."""

    MATCHED = "matched"
    ERROR = "error"
    TRUNCATED = "truncated"
    NOT_OBSERVED = "not_observed"


class VerifierStatus(StrEnum):
    """Terminal status emitted by an independently replayable verifier."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ReleaseProvenance(BaseModel):
    """Trainer-safe lineage without raw filesystem paths or source content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str
    parent_unit_id: str | None = None
    snapshot_revision: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(ge=0)
    source_range: SourceRange
    parser_revision: str
    source_class: SourceClass
    provider: str
    parent_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_snapshot_verified: bool

    @field_validator(
        "unit_id",
        "snapshot_revision",
        "parser_revision",
        "provider",
        mode="before",
    )
    @classmethod
    def require_identity_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("release provenance identity fields must be non-empty")

    @model_validator(mode="after")
    def require_verified_parent(self) -> ReleaseProvenance:
        if not self.parent_snapshot_verified:
            raise ValueError("release provenance requires a verified parent snapshot")
        return self


class ReleaseMetadata(BaseModel):
    """Shared admission, quality, split, and lineage tags for every view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    release_revision: str
    example_id: str
    split: DatasetSplit
    provenance: ReleaseProvenance
    decision_id: str
    decision_source: DecisionSource
    decision_revision: str
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    quality_dimensions: dict[str, QualityDimension]
    privacy_policy: PrivacyPolicy
    dedupe_group_id: str
    model_tier: ModelTier
    model_tier_registry_revision: str
    agent_identity: ReleaseIdentity = Field(default_factory=ReleaseIdentity)
    harness_identity: ReleaseIdentity = Field(default_factory=ReleaseIdentity)
    model_identity: ReleaseIdentity = Field(default_factory=ReleaseIdentity)

    @field_validator(
        "release_revision",
        "example_id",
        "decision_id",
        "decision_revision",
        "dedupe_group_id",
        "model_tier_registry_revision",
        mode="before",
    )
    @classmethod
    def require_metadata_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("release metadata identity fields must be non-empty")

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("release evidence_ids must be a non-empty sequence")
        normalized = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
        if not normalized:
            raise ValueError("release evidence_ids must be a non-empty sequence")
        return normalized

    @model_validator(mode="after")
    def validate_quality_dimensions(self) -> ReleaseMetadata:
        unknown = set(self.quality_dimensions).difference(REQUIRED_QUALITY_DIMENSIONS)
        if unknown:
            raise ValueError("release quality dimensions unknown: " + ", ".join(sorted(unknown)))
        missing = REQUIRED_QUALITY_DIMENSIONS.difference(self.quality_dimensions)
        if missing:
            raise ValueError("release quality dimensions missing: " + ", ".join(sorted(missing)))
        invalid = {
            name
            for name in REQUIRED_QUALITY_DIMENSIONS
            if self.quality_dimensions[name] not in {"pass", "not_applicable"}
        }
        if invalid:
            raise ValueError(
                "release quality dimensions cannot be fail/unknown: " + ", ".join(sorted(invalid))
            )
        if self.model_tier is ModelTier.UNCLASSIFIED:
            raise ValueError("release metadata requires a classified model tier")
        if self.provenance.unit_id not in self.evidence_ids:
            raise ValueError("release evidence_ids must include provenance.unit_id")
        identity_evidence = {
            evidence_id
            for identity in (
                self.agent_identity,
                self.harness_identity,
                self.model_identity,
            )
            for evidence_id in identity.evidence_ids
        }
        unknown_identity_evidence = identity_evidence.difference(self.evidence_ids)
        if unknown_identity_evidence:
            raise ValueError(
                "release identity evidence must be present in metadata evidence_ids: "
                + ", ".join(sorted(unknown_identity_evidence))
            )
        return self


def _reject_blocked(value: Any, label: str) -> None:
    findings = scan_value(value)
    if findings:
        codes = sorted({code for code, _path in findings})
        raise ValueError(f"{label} contains blocked release findings: {', '.join(codes)}")


class TrainingToolCall(BaseModel):
    """Provider-neutral function call accepted by a release view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]

    @field_validator("id", "name", mode="before")
    @classmethod
    def require_call_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("tool call id and name must be non-empty")

    @model_validator(mode="after")
    def screen_call(self) -> TrainingToolCall:
        _reject_blocked(self.model_dump(mode="json"), "tool call")
        return self


class TrainingMessage(BaseModel):
    """Minimal conversational message; hidden reasoning has no valid role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[TrainingToolCall, ...] = ()
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_message(self) -> TrainingMessage:
        if self.content is None and not self.tool_calls:
            raise ValueError("training message must contain content or tool_calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        _reject_blocked(self.model_dump(mode="json"), "training message")
        return self


class ToolDefinition(BaseModel):
    """Versioned function schema supplied to a tool-bearing trainer row.

    ``type`` is a provider transport envelope field and belongs in the raw
    registry sidecar.  ``strict`` changes function-call validation semantics,
    so it is retained in the canonical provider-neutral definition when the
    source supplies it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str | None = None
    parameters: dict[str, Any]
    strict: bool | None = None
    registry_revision: str

    @field_validator("name", "registry_revision", mode="before")
    @classmethod
    def require_tool_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("tool definition identity fields must be non-empty")

    @model_validator(mode="after")
    def screen_definition(self) -> ToolDefinition:
        _reject_blocked(self.model_dump(mode="json"), "tool definition")
        return self


class ToolObservation(BaseModel):
    """Bounded, explicit observation attached to a tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ObservationStatus
    content: str | dict[str, Any] | list[Any] | None = None
    exit_code: int | None = None
    error_type: str | None = None
    truncated: bool = False

    @model_validator(mode="after")
    def screen_observation(self) -> ToolObservation:
        _reject_blocked(self.model_dump(mode="json"), "tool observation")
        return self


class ToolUseStep(BaseModel):
    """One action/observation edge with absence explicitly represented."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    context: tuple[TrainingMessage, ...] = Field(min_length=1)
    call: TrainingToolCall
    observation_status: ObservationStatus = ObservationStatus.NOT_OBSERVED
    observation: ToolObservation | None = None
    observation_event_id: str | None = None

    @field_validator("event_id", mode="before")
    @classmethod
    def require_event_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("tool-use event_id must be non-empty")

    @model_validator(mode="after")
    def validate_observation_edge(self) -> ToolUseStep:
        if self.observation is None:
            if self.observation_status is not ObservationStatus.NOT_OBSERVED:
                raise ValueError("missing observation must be marked not_observed")
        elif self.observation_status is ObservationStatus.NOT_OBSERVED:
            raise ValueError("an attached observation cannot be marked not_observed")
        elif self.observation_status is not self.observation.status:
            raise ValueError("observation_status must match attached observation status")
        elif self.observation_event_id is None or not self.observation_event_id.strip():
            raise ValueError("attached observations require observation_event_id")
        return self


def _tool_names(tools: tuple[ToolDefinition, ...]) -> set[str]:
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError("tool registry contains duplicate names")
    return set(names)


def _validate_tool_registry(
    messages: tuple[TrainingMessage, ...], tools: tuple[ToolDefinition, ...]
) -> None:
    names = _tool_names(tools)
    calls = [call.name for message in messages for call in message.tool_calls]
    missing = sorted(set(calls).difference(names))
    if missing:
        raise ValueError("messages contain tools absent from the registry: " + ", ".join(missing))


class _ReleaseBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    view: ReleaseView
    metadata: ReleaseMetadata


class SFTRelease(_ReleaseBase):
    """Reasoning-free conversational SFT row."""

    schema_version: Literal["agentir/hardening/release/sft/v1"] = "agentir/hardening/release/sft/v1"
    view: Literal[ReleaseView.SFT] = ReleaseView.SFT
    messages: tuple[TrainingMessage, ...] = Field(min_length=1)
    tools: tuple[ToolDefinition, ...] = ()

    @model_validator(mode="after")
    def validate_tool_registry(self) -> SFTRelease:
        _validate_tool_registry(self.messages, self.tools)
        return self


class ToolUseRelease(_ReleaseBase):
    """Tool-selection/call dataset with explicit observation status."""

    schema_version: Literal["agentir/hardening/release/tool-use/v1"] = (
        "agentir/hardening/release/tool-use/v1"
    )
    view: Literal[ReleaseView.TOOL_USE] = ReleaseView.TOOL_USE
    steps: tuple[ToolUseStep, ...] = Field(min_length=1)
    tools: tuple[ToolDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tool_registry(self) -> ToolUseRelease:
        names = _tool_names(self.tools)
        missing = sorted({step.call.name for step in self.steps}.difference(names))
        if missing:
            raise ValueError("tool-use calls lack registry definitions: " + ", ".join(missing))
        return self


class PreferenceRelease(_ReleaseBase):
    """Explicit chosen/rejected pair; no inferred preference is representable."""

    schema_version: Literal["agentir/hardening/release/preference/v1"] = (
        "agentir/hardening/release/preference/v1"
    )
    view: Literal[ReleaseView.PREFERENCE] = ReleaseView.PREFERENCE
    prompt: tuple[TrainingMessage, ...] = Field(min_length=1)
    chosen: tuple[TrainingMessage, ...] = Field(min_length=1)
    rejected: tuple[TrainingMessage, ...] = Field(min_length=1)
    label_source: DecisionSource
    label_revision: str

    @field_validator("label_revision", mode="before")
    @classmethod
    def require_label_revision(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("preference label_revision must be non-empty")

    @model_validator(mode="after")
    def validate_pair(self) -> PreferenceRelease:
        if self.label_source is DecisionSource.TASK_MANIFEST:
            raise ValueError("task manifests cannot create preference labels")
        chosen = json.dumps(
            [message.model_dump(mode="json") for message in self.chosen],
            sort_keys=True,
            separators=(",", ":"),
        )
        rejected = json.dumps(
            [message.model_dump(mode="json") for message in self.rejected],
            sort_keys=True,
            separators=(",", ":"),
        )
        if chosen == rejected:
            raise ValueError("preference chosen and rejected responses must differ")
        return self


class RLPromptRelease(_ReleaseBase):
    """Prompt seed for a live, resettable, verifier-backed rollout."""

    schema_version: Literal["agentir/hardening/release/rl-prompt/v1"] = (
        "agentir/hardening/release/rl-prompt/v1"
    )
    view: Literal[ReleaseView.RL_PROMPT] = ReleaseView.RL_PROMPT
    prompt: tuple[TrainingMessage, ...] = Field(min_length=1)
    environment_id: str
    environment_revision: str
    verifier_revision: str
    tool_registry_revision: str
    max_turns: int = Field(gt=0)

    @field_validator(
        "environment_id",
        "environment_revision",
        "verifier_revision",
        "tool_registry_revision",
        mode="before",
    )
    @classmethod
    def require_environment_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("RL prompt environment identities must be non-empty")


class VerifierOutcome(BaseModel):
    """Replay-backed terminal signal; reward is invalid without replayability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VerifierStatus
    reward: float
    components: dict[str, float]
    verifier_revision: str
    replay_id: str
    replayable: bool

    @field_validator("reward", mode="before")
    @classmethod
    def require_finite_reward(cls, value: Any) -> Any:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError("verifier reward must be finite")
        return float(value)

    @field_validator("verifier_revision", "replay_id", mode="before")
    @classmethod
    def require_verifier_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("verifier identity fields must be non-empty")

    @field_validator("components")
    @classmethod
    def require_finite_components(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(component) for component in value.values()):
            raise ValueError("verifier components must be finite")
        return value

    @model_validator(mode="after")
    def require_replay(self) -> VerifierOutcome:
        if not self.replayable:
            raise ValueError("verifier outcome must be replayable for RL release")
        _reject_blocked(self.model_dump(mode="json"), "verifier outcome")
        return self


class RLStep(BaseModel):
    """Replayable state/action/observation transition for offline analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=0)
    action: TrainingToolCall | None = None
    observation_status: ObservationStatus = ObservationStatus.NOT_OBSERVED
    observation: ToolObservation | None = None
    reward: float
    done: bool
    state_hash_before: str
    state_hash_after: str

    @field_validator("reward", mode="before")
    @classmethod
    def require_finite_step_reward(cls, value: Any) -> Any:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError("RL step reward must be finite")
        return float(value)

    @field_validator("state_hash_before", "state_hash_after", mode="before")
    @classmethod
    def require_state_hash(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("RL step state hashes must be non-empty")

    @model_validator(mode="after")
    def validate_observation_edge(self) -> RLStep:
        if self.observation is None:
            if self.observation_status is not ObservationStatus.NOT_OBSERVED:
                raise ValueError("missing RL observation must be marked not_observed")
        elif self.observation_status is ObservationStatus.NOT_OBSERVED:
            raise ValueError("an attached RL observation cannot be marked not_observed")
        elif self.observation_status is not self.observation.status:
            raise ValueError("RL observation_status must match attached observation status")
        return self


class RLRolloutRelease(_ReleaseBase):
    """Complete verifier-backed rollout, separate from a historical transcript."""

    schema_version: Literal["agentir/hardening/release/rl-rollout/v1"] = (
        "agentir/hardening/release/rl-rollout/v1"
    )
    view: Literal[ReleaseView.RL_ROLLOUT] = ReleaseView.RL_ROLLOUT
    prompt: tuple[TrainingMessage, ...] = Field(min_length=1)
    steps: tuple[RLStep, ...] = Field(min_length=1)
    outcome: VerifierOutcome
    environment_id: str
    environment_revision: str
    verifier_revision: str
    tool_registry_revision: str

    @field_validator(
        "environment_id",
        "environment_revision",
        "verifier_revision",
        "tool_registry_revision",
        mode="before",
    )
    @classmethod
    def require_rollout_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("RL rollout environment identities must be non-empty")


ReleaseRow: TypeAlias = Annotated[
    SFTRelease | ToolUseRelease | PreferenceRelease | RLPromptRelease | RLRolloutRelease,
    Field(discriminator="view"),
]
_RELEASE_ROW_ADAPTER: TypeAdapter[ReleaseRow] = TypeAdapter(ReleaseRow)


def validate_release_row(value: Any) -> ReleaseRow:
    """Validate one trainer-facing row through the single release funnel."""

    return _RELEASE_ROW_ADAPTER.validate_python(value)


def _require_accepted_projection(result: ProjectionResult, target: str) -> None:
    if result.target != target:
        raise ValueError(f"projection target must be {target}")
    if result.state.value != "accepted":
        raise ValueError("only accepted projections can cross the release boundary")
    if not result.output:
        raise ValueError("accepted projection must contain at least one output item")


def _projection_tool_call(value: Any) -> TrainingToolCall:
    if not isinstance(value, dict):
        raise ValueError("projected tool call must be an object")
    function = value.get("function")
    if not isinstance(function, dict) or value.get("type") != "function":
        raise ValueError("projected tool call must use the function wire shape")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError("projected tool arguments must be valid JSON") from error
    if not isinstance(arguments, dict):
        raise ValueError("projected tool arguments must be a JSON object")
    return TrainingToolCall(
        id=value.get("id"),
        name=function.get("name"),
        arguments=arguments,
    )


def _messages_from_projection(output: list[dict[str, Any]]) -> tuple[TrainingMessage, ...]:
    if not isinstance(output, list):
        raise ValueError("projected messages must be a list")
    messages: list[TrainingMessage] = []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("projected SFT message must be an object")
        role = item.get("role")
        if role == "assistant":
            calls = item.get("tool_calls", ())
            if not isinstance(calls, (list, tuple)):
                raise ValueError("assistant tool_calls must be a sequence")
            messages.append(
                TrainingMessage(
                    role="assistant",
                    content=item.get("content"),
                    tool_calls=tuple(_projection_tool_call(call) for call in calls),
                )
            )
            continue
        if role == "tool":
            messages.append(
                TrainingMessage(
                    role="tool",
                    content=item.get("content"),
                    tool_call_id=item.get("tool_call_id"),
                )
            )
            continue
        if role in {"system", "developer", "user"}:
            messages.append(TrainingMessage(role=role, content=item.get("content")))
            continue
        raise ValueError("projected SFT message has unsupported role")
    return tuple(messages)


def sft_release_from_projection(
    result: ProjectionResult,
    *,
    metadata: ReleaseMetadata,
    tools: tuple[ToolDefinition, ...] = (),
) -> SFTRelease:
    """Convert one accepted SFT projection through the typed release funnel."""

    _require_accepted_projection(result, "sft")
    return SFTRelease(
        metadata=metadata,
        messages=_messages_from_projection(result.output),
        tools=tools,
    )


def tool_use_release_from_projection(
    result: ProjectionResult,
    *,
    metadata: ReleaseMetadata,
    tools: tuple[ToolDefinition, ...] = (),
) -> ToolUseRelease:
    """Convert tool actions with context and explicit observation state."""

    _require_accepted_projection(result, "tool_use")
    if not tools:
        raise ValueError("tool-use release requires a tool registry")
    steps = []
    for item in result.output:
        if not isinstance(item, dict):
            raise ValueError("projected tool-use item must be an object")
        raw_context = item.get("context")
        if not isinstance(raw_context, list):
            raise ValueError("projected tool-use context must be a list")
        context = _messages_from_projection(raw_context)
        observation_value = item.get("observation")
        observation = None
        if observation_value is not None:
            if not isinstance(observation_value, dict):
                raise ValueError("projected tool observation must be an object or null")
            try:
                observation = ToolObservation(
                    status=observation_value.get("status"),
                    content=observation_value.get("content"),
                    exit_code=observation_value.get("exit_code"),
                    error_type=observation_value.get("error_type"),
                    truncated=observation_value.get("truncated", False),
                )
            except (TypeError, ValueError) as error:
                raise ValueError("projected tool observation is invalid") from error
        observation_status = (
            observation.status if observation is not None else ObservationStatus.NOT_OBSERVED
        )
        steps.append(
            ToolUseStep(
                event_id=item.get("event_id"),
                context=context,
                call=TrainingToolCall(
                    id=item.get("call_id"),
                    name=item.get("tool_name"),
                    arguments=item.get("arguments"),
                ),
                observation_status=observation_status,
                observation=observation,
                observation_event_id=item.get("observation_event_id"),
            )
        )
    return ToolUseRelease(metadata=metadata, steps=tuple(steps), tools=tools)


def preference_pair_id(
    prompt: tuple[TrainingMessage, ...], chosen: tuple[TrainingMessage, ...]
) -> str:
    """Compute a stable pair-side digest for deduplication and audit joins."""

    payload = {
        "prompt": [message.model_dump(mode="json") for message in prompt],
        "chosen": [message.model_dump(mode="json") for message in chosen],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
