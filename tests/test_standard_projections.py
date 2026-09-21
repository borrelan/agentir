"""Contract tests for the sole external standards-projection boundary."""

from __future__ import annotations

import pytest

from agentir.hardening import (
    DatasetSplit,
    DecisionSource,
    IdentityStatus,
    ModelTier,
    PreferenceRelease,
    ProjectionProfile,
    ReleaseIdentity,
    ReleaseMetadata,
    ReleaseProvenance,
    RLPromptRelease,
    RLRolloutRelease,
    RLStep,
    SFTRelease,
    SourceRange,
    ToolDefinition,
    TrainingMessage,
    TrainingToolCall,
    VerifierOutcome,
    VerifierStatus,
    project_release,
)
from agentir.hardening.contracts import REQUIRED_QUALITY_DIMENSIONS


def _identity(name: str, version: str | None = None) -> ReleaseIdentity:
    return ReleaseIdentity(
        status=IdentityStatus.OBSERVED,
        name=name,
        version=version,
        evidence_ids=("unit-1",),
    )


def _metadata(split: DatasetSplit = DatasetSplit.TRAIN) -> ReleaseMetadata:
    return ReleaseMetadata(
        release_revision="release/v1",
        example_id="example-1",
        split=split,
        provenance=ReleaseProvenance(
            unit_id="unit-1",
            parent_unit_id="session-1",
            snapshot_revision="snapshot-1",
            source_sha256="a" * 64,
            source_size_bytes=100,
            source_range=SourceRange(byte_start=0, byte_end=100, message_start=1, message_end=3),
            parser_revision="parser/v1",
            source_class="session",
            provider="fixture-provider",
            parent_source_sha256="b" * 64,
            parent_snapshot_verified=True,
        ),
        decision_id="review-1",
        decision_source=DecisionSource.HUMAN_REVIEW,
        decision_revision="reviews/v1",
        evidence_ids=("unit-1",),
        quality_dimensions={
            dimension: "not_applicable" if dimension == "contamination" else "pass"
            for dimension in REQUIRED_QUALITY_DIMENSIONS
        },
        privacy_policy="redact_private_values",
        dedupe_group_id="group-1",
        model_tier=ModelTier.TIER1_FRONTIER,
        model_tier_registry_revision="tier-registry/v1",
        agent_identity=_identity("coding-agent", "2.1.0"),
        harness_identity=_identity("agent-harness", "4.0.0"),
        model_identity=_identity("frontier-model", "2026-09"),
    )


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        strict=True,
        registry_revision="tools/v1",
    )


def _tool_sft(*, developer: bool = False, unmatched: bool = False) -> SFTRelease:
    messages = []
    if developer:
        messages.append(TrainingMessage(role="developer", content="Use the repository tools."))
    messages.extend(
        [
            TrainingMessage(role="user", content="Read the file."),
            TrainingMessage(
                role="assistant",
                tool_calls=(
                    TrainingToolCall(
                        id="call-1",
                        name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ),
            ),
            TrainingMessage(
                role="tool",
                tool_call_id="missing-call" if unmatched else "call-1",
                content="contents",
            ),
            TrainingMessage(role="assistant", content="The file is valid."),
        ]
    )
    return SFTRelease(metadata=_metadata(), messages=tuple(messages), tools=(_tool(),))


def _rl_prompt(split: DatasetSplit = DatasetSplit.TRAIN) -> RLPromptRelease:
    return RLPromptRelease(
        metadata=_metadata(split),
        prompt=(TrainingMessage(role="user", content="Repair the failing service."),),
        environment_id="env-1",
        environment_revision="env/v3",
        verifier_revision="verifier/v2",
        tool_registry_revision="tools/v1",
        max_turns=12,
    )


def test_trl_sft_preserves_tool_contract_and_omits_sidecar_columns() -> None:
    projection = project_release(_tool_sft(), ProjectionProfile.TRL_SFT)
    row = projection.external_row()

    assert set(row) == {"messages", "tools"}
    assert row["messages"][1]["tool_calls"][0] == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "read_file", "arguments": {"path": "README.md"}},
    }
    assert row["messages"][2]["tool_call_id"] == "call-1"
    assert row["tools"][0]["function"]["strict"] is True
    assert projection.metadata.provenance.source_sha256 == "a" * 64


def test_trl_preference_and_prompt_profiles_are_exact() -> None:
    preference = PreferenceRelease(
        metadata=_metadata(),
        prompt=(TrainingMessage(role="user", content="Fix it."),),
        chosen=(TrainingMessage(role="assistant", content="Patched and tested."),),
        rejected=(TrainingMessage(role="assistant", content="Claimed success."),),
        label_source=DecisionSource.VERIFIER,
        label_revision="verifier/v2",
    )
    preference_row = project_release(preference, "trl_preference").external_row()
    prompt_row = project_release(_rl_prompt(), "trl_prompt").external_row()

    assert set(preference_row) == {"prompt", "chosen", "rejected"}
    assert set(prompt_row) == {"prompt"}
    assert prompt_row["prompt"][0]["content"] == "Repair the failing service."


def test_atif_v18_correlates_tools_omits_reasoning_and_reports_role_loss() -> None:
    projection = project_release(_tool_sft(developer=True), "atif_v1_8")
    row = projection.external_row()

    assert row["schema_version"] == "ATIF-v1.8"
    assert row["trajectory_id"] == "example-1"
    assert row["session_id"] == "session-1"
    assert row["agent"]["name"] == "coding-agent"
    assert row["agent"]["version"] == "2.1.0"
    assert row["agent"]["model_name"] == "frontier-model"
    call_step = next(step for step in row["steps"] if step.get("tool_calls"))
    assert call_step["observation"]["results"] == [
        {"source_call_id": "call-1", "content": "contents"}
    ]
    assert "reasoning_content" not in str(row)
    assert [loss.code for loss in projection.losses] == ["developer_role_to_atif_system"]


def test_atif_requires_identity_and_rejects_unmatched_or_delayed_observations() -> None:
    missing_identity = _tool_sft().model_copy(
        update={
            "metadata": _metadata().model_copy(
                update={
                    "agent_identity": ReleaseIdentity(),
                    "harness_identity": ReleaseIdentity(),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="versioned agent or harness"):
        project_release(missing_identity, "atif_v1_8")
    with pytest.raises(ValueError, match="no preceding call"):
        project_release(_tool_sft(unmatched=True), "atif_v1_8")

    delayed = _tool_sft().model_copy(
        update={
            "messages": (
                TrainingMessage(role="user", content="Read."),
                _tool_sft().messages[1],
                TrainingMessage(role="user", content="Anything else?"),
                _tool_sft().messages[2],
            )
        }
    )
    with pytest.raises(ValueError, match="delayed tool observation"):
        project_release(delayed, "atif_v1_8")


def test_agent_lightning_v101_projection_is_task_only_and_deterministic() -> None:
    train = project_release(_rl_prompt(), "agent_lightning_rollout")
    validation = project_release(_rl_prompt(DatasetSplit.VALIDATION), "agent_lightning_rollout")
    row = train.external_row()

    assert set(row) == {"input", "is_train", "metadata", "rollout_id"}
    assert row["input"]["schema_version"] == "agentir/hardening/agent-lightning-task/v1"
    assert row["input"]["environment"] == {
        "environment_id": "env-1",
        "environment_revision": "env/v3",
        "verifier_revision": "verifier/v2",
        "tool_registry_revision": "tools/v1",
    }
    assert row["input"]["budget"] == {"max_turns": 12}
    assert row["is_train"] is True
    assert validation.external_row()["is_train"] is False
    assert "reward" not in str(row)
    assert "model_request" not in str(row)
    assert train.model_dump(mode="json") == project_release(
        _rl_prompt(), "agent_lightning_rollout"
    ).model_dump(mode="json")


@pytest.mark.parametrize(
    ("row", "profile"),
    [
        (_tool_sft(), "trl_preference"),
        (_tool_sft(), "trl_prompt"),
        (_rl_prompt(), "trl_sft"),
        (_rl_prompt(), "atif_v1_8"),
        (_rl_prompt(), "trl_preference"),
    ],
)
def test_incompatible_profile_view_pairs_fail_closed(row: object, profile: str) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        project_release(row, profile)  # type: ignore[arg-type]


def test_offline_rl_rollout_cannot_masquerade_as_agent_lightning_data() -> None:
    rollout = RLRolloutRelease(
        metadata=_metadata(),
        prompt=(TrainingMessage(role="user", content="Repair it."),),
        steps=(
            RLStep(
                step_index=0,
                reward=1.0,
                done=True,
                state_hash_before="state-a",
                state_hash_after="state-b",
            ),
        ),
        outcome=VerifierOutcome(
            status=VerifierStatus.SUCCESS,
            reward=1.0,
            components={"tests": 1.0},
            verifier_revision="verifier/v2",
            replay_id="replay-1",
            replayable=True,
        ),
        environment_id="env-1",
        environment_revision="env/v3",
        verifier_revision="verifier/v2",
        tool_registry_revision="tools/v1",
    )

    with pytest.raises(ValueError, match="incompatible"):
        project_release(rollout, "agent_lightning_rollout")
