"""Pydantic schemas for attune wisdom evaluation."""

from enum import Enum

from pydantic import BaseModel


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
