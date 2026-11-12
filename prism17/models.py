"""Shared data objects for the Prism17 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Candidate:
    """A passage/target pair proposed by retrieval."""

    passage_id: str
    page: int
    target_id: str
    target_goal: int
    target_text: str
    score: float
    passage_text: str
    section_heading: str | None = None
    is_table: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            passage_id=str(data["passage_id"]),
            page=int(data["page"]),
            target_id=str(data["target_id"]),
            target_goal=int(data["target_goal"]),
            target_text=str(data["target_text"]),
            score=float(data.get("score", 0.0)),
            passage_text=str(data.get("passage_text", "")),
            section_heading=data.get("section_heading"),
            is_table=bool(data.get("is_table", False)),
        )


@dataclass
class JudgedFinding:
    """Structured output from the local LLM judge."""

    candidate: Candidate
    relevant: bool
    sdg_goal: int
    sdg_target: str
    claim: str
    supporting_quote: str
    dimensions: dict[str, bool] = field(default_factory=dict)
    extracted: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_response: str = ""
    scored: bool = True
    review_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate"] = self.candidate.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgedFinding":
        return cls(
            candidate=Candidate.from_dict(data["candidate"]),
            relevant=bool(data.get("relevant", False)),
            sdg_goal=int(data.get("sdg_goal", 0)),
            sdg_target=str(data.get("sdg_target", "")),
            claim=str(data.get("claim", "")),
            supporting_quote=str(data.get("supporting_quote", "")),
            dimensions=dict(data.get("dimensions") or {}),
            extracted=dict(data.get("extracted") or {}),
            confidence=float(data.get("confidence", 0.0)),
            raw_response=str(data.get("raw_response", "")),
            scored=bool(data.get("scored", True)),
            review_reason=data.get("review_reason"),
        )


@dataclass
class VerifiedFinding:
    """A judged finding after quote verification and rubric grading."""

    judged: JudgedFinding
    quote_verified: bool
    quote_method: str | None
    quote_score: float
    source_page: int
    semantic_support: str = "not_checked"
    semantic_score: float | None = None
    level: int | None = None
    level_rationale: str = ""
    disclosure_gaps: list[str] = field(default_factory=list)
    review_reason: str | None = None

    @property
    def is_scored(self) -> bool:
        """Only verified, relevant findings with a rubric level are ever scored.

        Stage 2 semantic support can veto a finding, but ``not_checked`` does
        not: a laptop without the optional NLI model still produces the same
        Stage 1 audit as before.
        """
        return (
            self.judged.relevant
            and self.judged.scored
            and self.quote_verified
            and self.level is not None
            and self.review_reason is None
            and self.semantic_support not in {"not_supported", "uncertain"}
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["judged"] = self.judged.to_dict()
        data["is_scored"] = self.is_scored
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiedFinding":
        return cls(
            judged=JudgedFinding.from_dict(data["judged"]),
            quote_verified=bool(data.get("quote_verified", False)),
            quote_method=data.get("quote_method"),
            quote_score=float(data.get("quote_score", 0.0)),
            source_page=int(data.get("source_page", 0)),
            semantic_support=str(data.get("semantic_support", "not_checked")),
            semantic_score=data.get("semantic_score"),
            level=data.get("level"),
            level_rationale=str(data.get("level_rationale", "")),
            disclosure_gaps=list(data.get("disclosure_gaps") or []),
            review_reason=data.get("review_reason"),
        )


def is_not_relevant(finding: VerifiedFinding) -> bool:
    """True when the judge produced a valid result marking the passage irrelevant."""
    return (
        not finding.is_scored
        and finding.judged.scored
        and not finding.judged.relevant
    )


def needs_human_review(finding: VerifiedFinding) -> bool:
    """True for unscored findings that are not clean 'not relevant' results."""
    return not finding.is_scored and not is_not_relevant(finding)
