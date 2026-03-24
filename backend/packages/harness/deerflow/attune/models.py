"""Pydantic schemas for attune wisdom evaluation."""

from enum import Enum
from typing import Optional

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
    wisdom_score_before: Optional[float]
    wisdom_score_after: Optional[float]
    dimension_scores_before: Optional[dict[str, float]]
    dimension_notes_before: Optional[dict[str, str]]
    refined_response: str
    modifications: list[Modification]
