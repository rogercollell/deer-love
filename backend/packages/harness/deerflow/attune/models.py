"""Pydantic schemas for the attune runtime and evaluation helpers."""

from enum import Enum

from pydantic import BaseModel, Field


class Domain(str, Enum):
    general = "general"
    coaching = "coaching"
    mental_health = "mental_health"


class SensitivityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"
    unknown = "unknown"


class WisdomFrame(BaseModel):
    """Upstream wisdom guidance computed before the agent responds."""

    emotional_context: str
    sensitivity_level: SensitivityLevel
    is_consequential: bool
    consequential_reason: str | None = None
    wellbeing_risk: bool = False
    affected_parties: list[str] = Field(default_factory=list)
    recommended_posture: str
    guidance: str
    reflection_invitation: str | None = None


class EvaluationStatus(str, Enum):
    ok = "ok"
    fallback_api_error = "fallback_api_error"
    fallback_invalid_json = "fallback_invalid_json"


class Modification(BaseModel):
    type: str
    explanation: str


class EvaluateResponse(BaseModel):
    evaluation_status: EvaluationStatus
    sensitivity_level: SensitivityLevel
    should_refine: bool
    wisdom_score_before: float | None
    wisdom_score_after: float | None
    dimension_scores_before: dict[str, float] | None
    dimension_notes_before: dict[str, str] | None
    refined_response: str
    modifications: list[Modification]
