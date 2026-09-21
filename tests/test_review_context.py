"""Tests for the bounded, non-release review context."""

from __future__ import annotations

from test_hardening import _lineage, _valid_record

from agentir.hardening import AdmissionState, PrivacyPolicy, build_review_context, project
from agentir.hardening.firewall import assess_record, scan_value


def test_review_context_has_redacted_preview_without_quality_decision() -> None:
    hardened = assess_record(
        _valid_record(),
        _lineage(),
        declared_state=AdmissionState.ACCEPTED,
        privacy_policy=PrivacyPolicy.REDACT_PRIVATE_VALUES,
    )
    preview = project(hardened, "sft", preview=True)
    context = build_review_context(
        task_id="task-1",
        mapping={"message_count": 5, "sensitive_provider_payload": "must-not-cross"},
        hardened=hardened,
        projections={"sft": preview},
        registry=None,
        max_string_bytes=8,
        max_context_bytes=16 * 1024,
    )

    assert context["decision"] is None
    assert context["review_only"] is True
    assert context["admission"]["state"] == "candidate"
    assert context["bounds"]["content_truncated"] is True
    assert context["projections"]["sft"]["output"]
    assert not scan_value(context["projections"]["sft"]["output"])
    assert context["mapping"] == {"message_count": 5}
    assert all("message" not in loss for loss in context["admission"]["losses"])
    assert "raw" not in context
