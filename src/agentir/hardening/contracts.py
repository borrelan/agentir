"""Typed release contracts for the AgentIR hardening boundary."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentir.ir.record import AgentIRRecord

QualityDimension = Literal["pass", "fail", "unknown", "not_applicable"]
REQUIRED_QUALITY_DIMENSIONS = frozenset(
    {
        "structural",
        "tool_integrity",
        "tool_correctness",
        "observation_grounding",
        "privacy",
        "reasoning_exclusion",
        "provenance",
        "task_quality",
        "human_review",
        "contamination",
        "deduplication",
    }
)


class SourceClass(StrEnum):
    """Coarse source class used for stable lineage joins."""

    SESSION = "session"
    BACKUP = "backup"
    SIDECAR = "sidecar"
    TOOL_OUTPUT = "tool_output"
    OVERLAY = "overlay"
    DATABASE = "database"
    UNKNOWN = "unknown"


class AdmissionState(StrEnum):
    """Admission state shared by canonical rows and projections."""

    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class LossAction(StrEnum):
    """What a projection did with information it could not safely retain."""

    DROP = "drop"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    TRANSFORM = "transform"


class PrivacyPolicy(StrEnum):
    """Explicit policy for values that can be safely replaced."""

    QUARANTINE = "quarantine"
    REDACT_PRIVATE_VALUES = "redact_private_values"


class ParentSnapshotRef(BaseModel):
    """Immutable identity of the raw object from which a derived row came."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_revision: str
    source_uri: str
    source_path: str
    source_sha256: str
    source_size_bytes: int = Field(ge=0)
    object_id: str
    root_label: str | None = None
    message_start: int | None = Field(default=None, ge=0)
    message_end: int | None = Field(default=None, ge=0)
    manifest_verified: bool = False

    @field_validator(
        "snapshot_revision",
        "source_uri",
        "source_path",
        "object_id",
        mode="before",
    )
    @classmethod
    def non_empty_parent_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("parent snapshot identity fields must be non-empty")

    @field_validator("root_label", mode="before")
    @classmethod
    def optional_root_label(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("parent root_label must be non-empty when supplied")

    @field_validator("source_sha256", mode="before")
    @classmethod
    def normalize_parent_digest(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("parent source_sha256 must be a hexadecimal digest")
        digest = value.strip().lower()
        if digest.startswith("sha256:"):
            digest = digest[7:]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("parent source_sha256 must be a 64-character SHA-256 digest")
        return digest

    @model_validator(mode="after")
    def validate_parent_range(self) -> ParentSnapshotRef:
        if (self.message_start is None) != (self.message_end is None):
            raise ValueError("parent message range must be supplied together")
        if (
            self.message_start is not None
            and self.message_end is not None
            and self.message_end < self.message_start
        ):
            raise ValueError("parent message_end must not precede message_start")
        return self


class SourceLineage(BaseModel):
    """Immutable, joinable identity for one derived unit.

    The upstream ``SourceRef`` is intentionally permissive and does not
    require these fields.  A row entering a trainer release must satisfy this
    stricter contract instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_uri: str
    source_path: str
    source_class: SourceClass
    provider: str
    snapshot_revision: str
    source_sha256: str
    source_size_bytes: int = Field(ge=0)
    source_mtime_ns: int | None = None
    byte_start: int = Field(ge=0)
    byte_end: int = Field(gt=0)
    message_start: int | None = Field(default=None, ge=0)
    message_end: int | None = Field(default=None, ge=0)
    parser_name: str
    parser_revision: str
    unit_id: str
    parent_unit_id: str | None = None
    parent_snapshot: ParentSnapshotRef | None = None

    @field_validator(
        "source_uri",
        "source_path",
        "provider",
        "snapshot_revision",
        "parser_name",
        "parser_revision",
        "unit_id",
        "source_class",
        mode="before",
    )
    @classmethod
    def non_empty_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("lineage identity fields must be non-empty")

    @field_validator("source_sha256", mode="before")
    @classmethod
    def normalize_digest(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("source_sha256 must be a hexadecimal digest")
        digest = value.strip().lower()
        if digest.startswith("sha256:"):
            digest = digest[7:]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("source_sha256 must be a 64-character SHA-256 digest")
        return digest

    @model_validator(mode="after")
    def validate_ranges(self) -> SourceLineage:
        if self.byte_end <= self.byte_start:
            raise ValueError("byte_end must be greater than byte_start")
        if (self.message_start is None) != (self.message_end is None):
            raise ValueError("message_start and message_end must be supplied together")
        if (
            self.message_start is not None
            and self.message_end is not None
            and self.message_end < self.message_start
        ):
            raise ValueError("message_end must not precede message_start")
        return self

    @model_validator(mode="after")
    def validate_parent_snapshot_revision(self) -> SourceLineage:
        if (
            self.parent_snapshot is not None
            and self.parent_snapshot.snapshot_revision != self.snapshot_revision
        ):
            raise ValueError("parent snapshot revision must match lineage snapshot_revision")
        return self


class SourceRange(BaseModel):
    """Range that explains where a loss occurred without retaining content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    byte_start: int = Field(ge=0)
    byte_end: int = Field(gt=0)
    message_start: int | None = Field(default=None, ge=0)
    message_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> SourceRange:
        if self.byte_end <= self.byte_start:
            raise ValueError("byte_end must be greater than byte_start")
        if (self.message_start is None) != (self.message_end is None):
            raise ValueError("message_start and message_end must be supplied together")
        if (
            self.message_start is not None
            and self.message_end is not None
            and self.message_end < self.message_start
        ):
            raise ValueError("message_end must not precede message_start")
        return self


class QualityReview(BaseModel):
    """Explicit, row-bound review evidence required for release admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str
    review_id: str
    manifest_revision: str
    snapshot_revision: str
    reviewer_id: str
    rubric_revision: str
    reviewed_at: str
    decision: Literal["accepted", "quarantined", "rejected"]
    dimensions: dict[str, QualityDimension]
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator(
        "unit_id",
        "review_id",
        "manifest_revision",
        "snapshot_revision",
        "reviewer_id",
        "rubric_revision",
        "reviewed_at",
        mode="before",
    )
    @classmethod
    def non_empty_review_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("quality review identity fields must be non-empty")

    @model_validator(mode="after")
    def validate_acceptance_dimensions(self) -> QualityReview:
        if self.decision != "accepted":
            return self
        if self.unit_id not in self.evidence_ids:
            raise ValueError("accepted quality reviews must cite their unit_id as evidence")
        missing = REQUIRED_QUALITY_DIMENSIONS.difference(self.dimensions)
        if missing:
            raise ValueError("accepted quality reviews must cover: " + ", ".join(sorted(missing)))
        invalid = {
            name
            for name in REQUIRED_QUALITY_DIMENSIONS
            if self.dimensions[name] not in {"pass", "not_applicable"}
        }
        if invalid:
            raise ValueError(
                "accepted quality reviews cannot contain fail/unknown dimensions: "
                + ", ".join(sorted(invalid))
            )
        return self


class QualityReviewManifest(BaseModel):
    """Versioned collection of row-bound review decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_revision: str
    snapshot_revision: str
    rubric_revision: str
    reviews: tuple[QualityReview, ...] = ()

    @field_validator(
        "manifest_revision",
        "snapshot_revision",
        "rubric_revision",
        mode="before",
    )
    @classmethod
    def non_empty_manifest_text(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("review manifest identity fields must be non-empty")

    @model_validator(mode="after")
    def validate_reviews(self) -> QualityReviewManifest:
        review_ids = [review.review_id for review in self.reviews]
        unit_ids = [review.unit_id for review in self.reviews]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("review manifest contains duplicate review_id values")
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("review manifest contains duplicate unit_id values")
        for review in self.reviews:
            if review.manifest_revision != self.manifest_revision:
                raise ValueError("review manifest revision does not match its reviews")
            if review.snapshot_revision != self.snapshot_revision:
                raise ValueError("review snapshot revision does not match its reviews")
            if review.rubric_revision != self.rubric_revision:
                raise ValueError("review rubric revision does not match its reviews")
        return self


class QualityDecision(BaseModel):
    """Provider-independent quality decision and auditable dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: AdmissionState = AdmissionState.CANDIDATE
    dimensions: dict[str, QualityDimension] = Field(default_factory=dict)
    rule_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    review: QualityReview | None = None


class LossDecision(BaseModel):
    """Explicit projection loss; values and protected content are never stored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    action: LossAction
    field: str | None = None
    event_id: str | None = None
    message: str
    source_range: SourceRange | None = None


class HardenedRecord(BaseModel):
    """An AgentIR record admitted to the hardening boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    record: AgentIRRecord
    lineage: SourceLineage
    quality: QualityDecision
    privacy_policy: PrivacyPolicy = PrivacyPolicy.QUARANTINE
    losses: tuple[LossDecision, ...] = ()

    @model_validator(mode="after")
    def identity_must_join(self) -> HardenedRecord:
        if self.record.record_id != self.lineage.unit_id:
            raise ValueError("record_id must equal lineage.unit_id")
        return self


class ProjectionResult(BaseModel):
    """Bounded trainer projection plus its loss/admission report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    state: AdmissionState
    output: list[dict[str, Any]] = Field(default_factory=list)
    losses: tuple[LossDecision, ...] = ()
    emitted_events: int = 0
    omitted_events: int = 0
