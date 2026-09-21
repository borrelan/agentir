"""Contract tests for SFT, tool-use, preference, and verifier-backed RL views."""

from __future__ import annotations

import pytest

from agentir.hardening import (
    AdmissionState,
    DatasetSplit,
    DecisionSource,
    IdentityStatus,
    ModelTier,
    ObservationStatus,
    PreferenceRelease,
    ReleaseIdentity,
    ReleaseMetadata,
    ReleaseProvenance,
    ReleaseView,
    RLPromptRelease,
    RLRolloutRelease,
    RLStep,
    SFTRelease,
    SourceRange,
    ToolDefinition,
    ToolObservation,
    ToolUseRelease,
    ToolUseStep,
    TrainingMessage,
    TrainingToolCall,
    VerifierOutcome,
    VerifierStatus,
    preference_pair_id,
    sft_release_from_projection,
    tool_use_release_from_projection,
    validate_release_row,
)
from agentir.hardening.contracts import ProjectionResult


def _metadata() -> ReleaseMetadata:
    return ReleaseMetadata(
        release_revision="release/v1",
        example_id="example-1",
        split=DatasetSplit.TRAIN,
        provenance=ReleaseProvenance(
            unit_id="unit-1",
            snapshot_revision="snapshot-1",
            source_sha256="a" * 64,
            source_size_bytes=100,
            source_range=SourceRange(byte_start=0, byte_end=100, message_start=1, message_end=1),
            parser_revision="parser/v1",
            source_class="session",
            provider="fixture",
            parent_source_sha256="b" * 64,
            parent_snapshot_verified=True,
        ),
        decision_id="review-1",
        decision_source=DecisionSource.HUMAN_REVIEW,
        decision_revision="reviews/v1",
        evidence_ids=("unit-1",),
        quality_dimensions={
            "structural": "pass",
            "tool_integrity": "pass",
            "tool_correctness": "pass",
            "observation_grounding": "pass",
            "privacy": "pass",
            "reasoning_exclusion": "pass",
            "provenance": "pass",
            "task_quality": "pass",
            "human_review": "pass",
            "contamination": "not_applicable",
            "deduplication": "pass",
        },
        privacy_policy="redact_private_values",
        dedupe_group_id="group-1",
        model_tier=ModelTier.TIER1_FRONTIER,
        model_tier_registry_revision="tier-registry/v1",
    )


def _message(text: str = "complete") -> TrainingMessage:
    return TrainingMessage(role="assistant", content=text)


def _tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="read",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        registry_revision="tools/v1",
    )


def test_sft_release_is_strict_and_discriminated() -> None:
    row = SFTRelease(
        metadata=_metadata(),
        messages=(TrainingMessage(role="user", content="check"), _message()),
    )

    validated = validate_release_row(row.model_dump(mode="json"))

    assert isinstance(validated, SFTRelease)
    assert validated.view is ReleaseView.SFT


def test_release_metadata_rejects_unclassified_model_tier() -> None:
    payload = _metadata().model_dump()
    payload["model_tier"] = ModelTier.UNCLASSIFIED
    with pytest.raises(ValueError, match="classified model tier"):
        ReleaseMetadata(**payload)


def test_environment_authored_rl_prompt_allows_not_applicable_model_tier() -> None:
    metadata = _metadata().model_copy(
        update={
            "model_tier": ModelTier.NOT_APPLICABLE,
            "model_tier_registry_revision": "not-applicable/environment-authored-task/v1",
        }
    )
    row = RLPromptRelease(
        metadata=metadata,
        prompt=(TrainingMessage(role="user", content="repair the service"),),
        environment_id="env-1",
        environment_revision="env/v1",
        verifier_revision="verifier/v1",
        tool_registry_revision="tools/v1",
        max_turns=8,
    )

    assert row.metadata.model_tier is ModelTier.NOT_APPLICABLE


def test_model_generated_views_reject_not_applicable_model_tier() -> None:
    metadata = _metadata().model_copy(
        update={
            "model_tier": ModelTier.NOT_APPLICABLE,
            "model_tier_registry_revision": "not-applicable/environment-authored-task/v1",
        }
    )
    with pytest.raises(ValueError, match="model-generated release views require a model tier"):
        SFTRelease(
            metadata=metadata,
            messages=(TrainingMessage(role="user", content="check"), _message()),
        )


def test_release_identity_is_explicit_and_evidence_bound() -> None:
    assert ReleaseIdentity().status is IdentityStatus.NOT_OBSERVED
    with pytest.raises(ValueError, match="cannot carry"):
        ReleaseIdentity(status=IdentityStatus.NOT_OBSERVED, name="guessed-agent")
    with pytest.raises(ValueError, match="requires evidence"):
        ReleaseIdentity(status=IdentityStatus.OBSERVED, name="codex")

    payload = _metadata().model_dump()
    payload["agent_identity"] = {
        "status": "observed",
        "name": "codex",
        "version": "1.2.3",
        "evidence_ids": ["identity-1"],
    }
    with pytest.raises(ValueError, match="metadata evidence_ids"):
        ReleaseMetadata(**payload)


def test_release_messages_reject_reasoning_markers_and_private_values() -> None:
    with pytest.raises(ValueError, match="blocked release findings"):
        TrainingMessage(role="assistant", content="<thinking>secret</thinking>")
    with pytest.raises(ValueError, match="blocked release findings"):
        TrainingMessage(role="assistant", content="read /data-sea/private/file")


def test_tool_use_requires_explicit_observation_state() -> None:
    call = TrainingToolCall(id="call-1", name="read", arguments={"path": "file.txt"})
    context = (TrainingMessage(role="user", content="read the file"),)
    row = ToolUseRelease(
        metadata=_metadata(),
        steps=(ToolUseStep(event_id="event-1", context=context, call=call),),
        tools=(_tool_definition(),),
    )

    assert row.steps[0].observation is None
    assert ObservationStatus.NOT_OBSERVED.value == "not_observed"

    empty_result = ToolUseRelease(
        metadata=_metadata(),
        steps=(
            ToolUseStep(
                event_id="event-1",
                context=context,
                call=call,
                observation_event_id="result-1",
            ),
        ),
        tools=(_tool_definition(),),
    )
    assert empty_result.steps[0].observation_event_id == "result-1"

    with pytest.raises(ValueError, match="not_observed"):
        ToolUseStep(
            event_id="event-1",
            context=context,
            call=call,
            observation=ToolObservation(status=ObservationStatus.NOT_OBSERVED),
        )

    with pytest.raises(ValueError, match="missing observation"):
        ToolUseStep(
            event_id="event-1",
            context=context,
            call=call,
            observation_status=ObservationStatus.MATCHED,
        )


def test_preference_requires_distinct_explicit_sides() -> None:
    prompt = (TrainingMessage(role="user", content="fix it"),)
    with pytest.raises(ValueError, match="must differ"):
        PreferenceRelease(
            metadata=_metadata(),
            prompt=prompt,
            chosen=(_message("same"),),
            rejected=(_message("same"),),
            label_source=DecisionSource.HUMAN_REVIEW,
            label_revision="reviews/v1",
        )

    row = PreferenceRelease(
        metadata=_metadata(),
        prompt=prompt,
        chosen=(_message("patched and tested"),),
        rejected=(_message("claimed success"),),
        label_source=DecisionSource.VERIFIER,
        label_revision="verifier/v1",
    )
    assert preference_pair_id(prompt, row.chosen)


def test_rl_prompt_has_environment_contract_and_no_reward_field() -> None:
    row = RLPromptRelease(
        metadata=_metadata(),
        prompt=(TrainingMessage(role="user", content="repair the service"),),
        environment_id="env-1",
        environment_revision="env/v1",
        verifier_revision="verifier/v1",
        tool_registry_revision="tools/v1",
        max_turns=8,
    )
    assert row.view is ReleaseView.RL_PROMPT

    with pytest.raises(ValueError):
        validate_release_row(row.model_dump(mode="json") | {"reward": 1.0})


def test_rl_rollout_requires_replayable_verifier_outcome() -> None:
    outcome = VerifierOutcome(
        status=VerifierStatus.SUCCESS,
        reward=1.0,
        components={"tests": 1.0},
        verifier_revision="verifier/v1",
        replay_id="replay-1",
        replayable=True,
    )
    row = RLRolloutRelease(
        metadata=_metadata(),
        prompt=(TrainingMessage(role="user", content="repair"),),
        steps=(
            RLStep(
                step_index=0,
                reward=1.0,
                done=True,
                state_hash_before="state-a",
                state_hash_after="state-b",
            ),
        ),
        outcome=outcome,
        environment_id="env-1",
        environment_revision="env/v1",
        verifier_revision="verifier/v1",
        tool_registry_revision="tools/v1",
    )
    assert row.outcome.replayable is True

    with pytest.raises(ValueError, match="replayable"):
        VerifierOutcome(
            status=VerifierStatus.SUCCESS,
            reward=1.0,
            components={},
            verifier_revision="verifier/v1",
            replay_id="replay-1",
            replayable=False,
        )


def test_projection_egress_adapters_preserve_wire_contracts() -> None:
    sft_result = ProjectionResult(
        target="sft",
        state=AdmissionState.ACCEPTED,
        output=[
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path": "x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ],
    )
    sft = sft_release_from_projection(sft_result, metadata=_metadata(), tools=(_tool_definition(),))
    assert sft.messages[1].tool_calls[0].arguments == {"path": "x"}

    tool_result = ProjectionResult(
        target="tool_use",
        state=AdmissionState.ACCEPTED,
        output=[
            {
                "event_id": "event-1",
                "context": [{"role": "user", "content": "read the file"}],
                "call_id": "call-1",
                "tool_name": "read",
                "arguments": {"path": "x"},
                "observation_event_id": "result-1",
                "observation": {
                    "status": "matched",
                    "content": "ok",
                    "exit_code": 0,
                    "error_type": None,
                    "truncated": False,
                },
            }
        ],
    )
    tool = tool_use_release_from_projection(
        tool_result, metadata=_metadata(), tools=(_tool_definition(),)
    )
    assert tool.steps[0].observation_status is ObservationStatus.MATCHED
    assert tool.steps[0].observation is not None
    assert tool.steps[0].observation.content == "ok"
    assert tool.steps[0].context[0].role == "user"
    assert tool.steps[0].observation_event_id == "result-1"
