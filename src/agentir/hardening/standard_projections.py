"""Typed, loss-aware projections from admitted releases to external contracts.

This module starts after the strict :class:`ReleaseRow` boundary. It does not
read provider records, decide admission, assign splits, authorize training, or
invent rewards and policy-token data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentir.hardening.contracts import LossAction, LossDecision
from agentir.hardening.firewall import scan_value
from agentir.hardening.release_views import (
    DatasetSplit,
    IdentityStatus,
    PreferenceRelease,
    ReleaseIdentity,
    ReleaseMetadata,
    ReleaseRow,
    ReleaseView,
    RLPromptRelease,
    SFTRelease,
    ToolDefinition,
    TrainingMessage,
    TrainingToolCall,
    validate_release_row,
)

STANDARD_PROJECTION_SCHEMA: Literal["agentir/hardening/standard-projection/v1"] = (
    "agentir/hardening/standard-projection/v1"
)
ATIF_SCHEMA_VERSION: Literal["ATIF-v1.8"] = "ATIF-v1.8"
ATIF_CONTRACT_REVISION = "harbor/ATIF-v1.8@71c77fdd119df12eb6ab56e5bc0f29bf62fad338"
TRL_CONTRACT_REVISION = "huggingface/trl/v1.13.0/dataset-formats"
AGENT_LIGHTNING_CONTRACT_REVISION = (
    "microsoft/agent-lightning/v1.0.1@8435586d147b4cf7bff33e687d7317149e79cbb8"
)


class ProjectionProfile(StrEnum):
    """Finite external contracts emitted by the standards boundary."""

    ATIF_V1_8 = "atif_v1_8"
    TRL_SFT = "trl_sft"
    TRL_PREFERENCE = "trl_preference"
    TRL_PROMPT = "trl_prompt"
    AGENT_LIGHTNING_ROLLOUT = "agent_lightning_rollout"


class _WireBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TRLFunctionCall(_WireBase):
    name: str
    arguments: dict[str, Any]

    @field_validator("name", mode="before")
    @classmethod
    def require_name(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("tool function name must be non-empty")


class TRLToolCall(_WireBase):
    id: str
    type: Literal["function"] = "function"
    function: TRLFunctionCall

    @field_validator("id", mode="before")
    @classmethod
    def require_id(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("tool call ID must be non-empty")


class TRLMessage(_WireBase):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[TRLToolCall, ...] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_message(self) -> TRLMessage:
        if self.content is None and not self.tool_calls:
            raise ValueError("TRL message must contain content or tool calls")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("TRL tool message requires tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("TRL tool_call_id is valid only on tool messages")
        return self


class TRLFunctionDefinition(_WireBase):
    name: str
    description: str | None = None
    parameters: dict[str, Any]
    strict: bool | None = None


class TRLToolDefinition(_WireBase):
    type: Literal["function"] = "function"
    function: TRLFunctionDefinition


class TRLSFTRow(_WireBase):
    messages: tuple[TRLMessage, ...] = Field(min_length=1)
    tools: tuple[TRLToolDefinition, ...] | None = None


class TRLPreferenceRow(_WireBase):
    prompt: tuple[TRLMessage, ...] = Field(min_length=1)
    chosen: tuple[TRLMessage, ...] = Field(min_length=1)
    rejected: tuple[TRLMessage, ...] = Field(min_length=1)


class TRLPromptRow(_WireBase):
    prompt: tuple[TRLMessage, ...] = Field(min_length=1)


class ATIFToolCall(_WireBase):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


class ATIFObservationResult(_WireBase):
    source_call_id: str
    content: str


class ATIFObservation(_WireBase):
    results: tuple[ATIFObservationResult, ...] = Field(min_length=1)


class ATIFStep(_WireBase):
    step_id: int = Field(gt=0)
    source: Literal["system", "user", "agent"]
    message: str
    tool_calls: tuple[ATIFToolCall, ...] | None = None
    observation: ATIFObservation | None = None


class ATIFAgent(_WireBase):
    name: str
    version: str
    model_name: str | None = None
    tool_definitions: tuple[TRLToolDefinition, ...] | None = None


class ATIFTrajectory(_WireBase):
    schema_version: Literal["ATIF-v1.8"] = ATIF_SCHEMA_VERSION
    trajectory_id: str
    session_id: str
    agent: ATIFAgent
    steps: tuple[ATIFStep, ...] = Field(min_length=1)


class AgentLightningEnvironment(_WireBase):
    environment_id: str
    environment_revision: str
    verifier_revision: str
    tool_registry_revision: str


class AgentLightningBudget(_WireBase):
    max_turns: int = Field(gt=0)


class AgentLightningTaskInput(_WireBase):
    schema_version: Literal["agentir/hardening/agent-lightning-task/v1"] = (
        "agentir/hardening/agent-lightning-task/v1"
    )
    task_id: str
    prompt: tuple[TRLMessage, ...] = Field(min_length=1)
    environment: AgentLightningEnvironment
    budget: AgentLightningBudget


class AgentLightningMetadata(_WireBase):
    agentir_example_id: str
    agentir_release_revision: str
    agentir_split: DatasetSplit
    agentir_source_sha256: str
    agentir_snapshot_revision: str


class AgentLightningRolloutRow(_WireBase):
    input: AgentLightningTaskInput
    is_train: bool
    metadata: AgentLightningMetadata
    rollout_id: str


ProjectionRow: TypeAlias = (
    TRLSFTRow | TRLPreferenceRow | TRLPromptRow | ATIFTrajectory | AgentLightningRolloutRow
)


_PROFILE_ROW_TYPES: dict[ProjectionProfile, type[BaseModel]] = {
    ProjectionProfile.TRL_SFT: TRLSFTRow,
    ProjectionProfile.TRL_PREFERENCE: TRLPreferenceRow,
    ProjectionProfile.TRL_PROMPT: TRLPromptRow,
    ProjectionProfile.ATIF_V1_8: ATIFTrajectory,
    ProjectionProfile.AGENT_LIGHTNING_ROLLOUT: AgentLightningRolloutRow,
}
_PROFILE_CONTRACTS = {
    ProjectionProfile.TRL_SFT: TRL_CONTRACT_REVISION,
    ProjectionProfile.TRL_PREFERENCE: TRL_CONTRACT_REVISION,
    ProjectionProfile.TRL_PROMPT: TRL_CONTRACT_REVISION,
    ProjectionProfile.ATIF_V1_8: ATIF_CONTRACT_REVISION,
    ProjectionProfile.AGENT_LIGHTNING_ROLLOUT: AGENT_LIGHTNING_CONTRACT_REVISION,
}
_PROFILE_VIEWS = {
    ProjectionProfile.TRL_SFT: ReleaseView.SFT,
    ProjectionProfile.TRL_PREFERENCE: ReleaseView.PREFERENCE,
    ProjectionProfile.TRL_PROMPT: ReleaseView.RL_PROMPT,
    ProjectionProfile.ATIF_V1_8: ReleaseView.SFT,
    ProjectionProfile.AGENT_LIGHTNING_ROLLOUT: ReleaseView.RL_PROMPT,
}


class StandardProjection(BaseModel):
    """One external row plus the complete canonical sidecar and loss record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agentir/hardening/standard-projection/v1"] = STANDARD_PROJECTION_SCHEMA
    profile: ProjectionProfile
    contract_revision: str
    source_view: ReleaseView
    row: ProjectionRow
    metadata: ReleaseMetadata
    losses: tuple[LossDecision, ...] = ()

    @model_validator(mode="after")
    def validate_profile_contract(self) -> StandardProjection:
        expected_type = _PROFILE_ROW_TYPES[self.profile]
        if not isinstance(self.row, expected_type):
            raise ValueError(f"{self.profile.value} requires {expected_type.__name__}")
        if self.contract_revision != _PROFILE_CONTRACTS[self.profile]:
            raise ValueError("projection contract revision does not match profile")
        if self.source_view is not _PROFILE_VIEWS[self.profile]:
            raise ValueError("projection source view does not match profile")
        findings = scan_value(self.external_row())
        if findings:
            codes = sorted({code for code, _path in findings})
            raise ValueError("external row contains blocked findings: " + ", ".join(codes))
        return self

    def external_row(self) -> dict[str, Any]:
        """Return only the strict trainer/interchange row, without its sidecar."""

        return self.row.model_dump(mode="json", exclude_none=True)


def _tool_call(call: TrainingToolCall) -> TRLToolCall:
    return TRLToolCall(
        id=call.id,
        function=TRLFunctionCall(name=call.name, arguments=call.arguments),
    )


def _message(message: TrainingMessage) -> TRLMessage:
    return TRLMessage(
        role=message.role,
        content=message.content,
        tool_calls=tuple(_tool_call(call) for call in message.tool_calls) or None,
        tool_call_id=message.tool_call_id,
    )


def _tool_definition(tool: ToolDefinition) -> TRLToolDefinition:
    return TRLToolDefinition(
        function=TRLFunctionDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            strict=tool.strict,
        )
    )


def _identity_for_atif(metadata: ReleaseMetadata) -> tuple[ReleaseIdentity, bool]:
    agent = metadata.agent_identity
    if agent.status is not IdentityStatus.NOT_OBSERVED and agent.version is not None:
        return agent, False
    harness = metadata.harness_identity
    if harness.status is not IdentityStatus.NOT_OBSERVED and harness.version is not None:
        return harness, True
    raise ValueError("ATIF projection requires a versioned agent or harness identity")


def _atif_projection(row: SFTRelease) -> tuple[ATIFTrajectory, tuple[LossDecision, ...]]:
    identity, used_harness = _identity_for_atif(row.metadata)
    losses: list[LossDecision] = []
    if used_harness:
        losses.append(
            LossDecision(
                code="atif_harness_identity_used_as_agent",
                action=LossAction.TRANSFORM,
                field="agent",
                message="ATIF agent identity uses the evidence-bound harness identity",
            )
        )

    steps: list[ATIFStep] = []
    call_step: dict[str, int] = {}
    observed_calls: set[str] = set()
    last_non_tool_step: int | None = None
    for message in row.messages:
        if message.role == "tool":
            call_id = message.tool_call_id
            if call_id is None or call_id not in call_step:
                raise ValueError("ATIF tool observation has no preceding call")
            if call_id in observed_calls:
                raise ValueError("ATIF tool call has duplicate observations")
            step_index = call_step[call_id]
            if last_non_tool_step != step_index:
                raise ValueError("ATIF cannot attach a delayed tool observation without reordering")
            step = steps[step_index]
            result = ATIFObservationResult(source_call_id=call_id, content=message.content or "")
            results = () if step.observation is None else step.observation.results
            steps[step_index] = step.model_copy(
                update={"observation": ATIFObservation(results=(*results, result))}
            )
            observed_calls.add(call_id)
            continue

        if message.tool_calls and message.role != "assistant":
            raise ValueError("ATIF tool calls require an assistant source message")
        source = "agent" if message.role == "assistant" else message.role
        if message.role == "developer":
            source = "system"
            losses.append(
                LossDecision(
                    code="developer_role_to_atif_system",
                    action=LossAction.TRANSFORM,
                    field="messages.role",
                    message="ATIF has no developer source; mapped visible developer text to system",
                )
            )
        atif_calls = tuple(
            ATIFToolCall(
                tool_call_id=call.id,
                function_name=call.name,
                arguments=call.arguments,
            )
            for call in message.tool_calls
        )
        for call in atif_calls:
            if call.tool_call_id in call_step:
                raise ValueError("ATIF trajectory has duplicate tool call IDs")
        steps.append(
            ATIFStep(
                step_id=len(steps) + 1,
                source=source,
                message=message.content or "",
                tool_calls=atif_calls or None,
            )
        )
        last_non_tool_step = len(steps) - 1
        for call in atif_calls:
            call_step[call.tool_call_id] = last_non_tool_step

    model_identity = row.metadata.model_identity
    model_name = (
        model_identity.name if model_identity.status is not IdentityStatus.NOT_OBSERVED else None
    )
    trajectory = ATIFTrajectory(
        trajectory_id=row.metadata.example_id,
        session_id=(row.metadata.provenance.parent_unit_id or row.metadata.provenance.unit_id),
        agent=ATIFAgent(
            name=identity.name or "",
            version=identity.version or "",
            model_name=model_name,
            tool_definitions=tuple(_tool_definition(tool) for tool in row.tools) or None,
        ),
        steps=tuple(steps),
    )
    return trajectory, tuple(losses)


def _agent_lightning_projection(row: RLPromptRelease) -> AgentLightningRolloutRow:
    identity = "\0".join(
        (
            row.metadata.release_revision,
            row.metadata.example_id,
            row.environment_id,
            row.environment_revision,
            row.verifier_revision,
        )
    )
    rollout_id = "agentir-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return AgentLightningRolloutRow(
        input=AgentLightningTaskInput(
            task_id=row.metadata.example_id,
            prompt=tuple(_message(message) for message in row.prompt),
            environment=AgentLightningEnvironment(
                environment_id=row.environment_id,
                environment_revision=row.environment_revision,
                verifier_revision=row.verifier_revision,
                tool_registry_revision=row.tool_registry_revision,
            ),
            budget=AgentLightningBudget(max_turns=row.max_turns),
        ),
        is_train=row.metadata.split is DatasetSplit.TRAIN,
        metadata=AgentLightningMetadata(
            agentir_example_id=row.metadata.example_id,
            agentir_release_revision=row.metadata.release_revision,
            agentir_split=row.metadata.split,
            agentir_source_sha256=row.metadata.provenance.source_sha256,
            agentir_snapshot_revision=row.metadata.provenance.snapshot_revision,
        ),
        rollout_id=rollout_id,
    )


def project_release(
    row: ReleaseRow | Mapping[str, Any],
    profile: ProjectionProfile | str,
) -> StandardProjection:
    """Project one admitted release through the sole standards funnel."""

    release = validate_release_row(row) if isinstance(row, Mapping) else row
    try:
        selected = ProjectionProfile(profile)
    except ValueError as error:
        raise ValueError("unsupported projection profile") from error

    external: ProjectionRow
    losses: tuple[LossDecision, ...] = ()
    if selected is ProjectionProfile.TRL_SFT and isinstance(release, SFTRelease):
        external = TRLSFTRow(
            messages=tuple(_message(message) for message in release.messages),
            tools=tuple(_tool_definition(tool) for tool in release.tools) or None,
        )
    elif selected is ProjectionProfile.TRL_PREFERENCE and isinstance(release, PreferenceRelease):
        messages = (*release.prompt, *release.chosen, *release.rejected)
        if any(message.tool_calls or message.role == "tool" for message in messages):
            raise ValueError("tool-bearing preferences require an exact canonical tool registry")
        external = TRLPreferenceRow(
            prompt=tuple(_message(message) for message in release.prompt),
            chosen=tuple(_message(message) for message in release.chosen),
            rejected=tuple(_message(message) for message in release.rejected),
        )
    elif selected is ProjectionProfile.TRL_PROMPT and isinstance(release, RLPromptRelease):
        external = TRLPromptRow(prompt=tuple(_message(message) for message in release.prompt))
    elif selected is ProjectionProfile.ATIF_V1_8 and isinstance(release, SFTRelease):
        external, losses = _atif_projection(release)
    elif selected is ProjectionProfile.AGENT_LIGHTNING_ROLLOUT and isinstance(
        release, RLPromptRelease
    ):
        external = _agent_lightning_projection(release)
    else:
        raise ValueError(
            f"projection profile {selected.value} is incompatible with release view "
            f"{release.view.value}"
        )

    return StandardProjection(
        profile=selected,
        contract_revision=_PROFILE_CONTRACTS[selected],
        source_view=release.view,
        row=external,
        metadata=release.metadata,
        losses=losses,
    )
