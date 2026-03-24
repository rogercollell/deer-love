"""Wisdom evaluation engine using LangChain chat models.

Adapted from the attune project's wisdom_engine.py. Uses LangChain's
BaseChatModel instead of the raw Anthropic SDK for provider-agnostic evaluation.
"""

import json
import logging
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deerflow.attune.models import (
    EvaluateResponse,
    EvaluationStatus,
    Modification,
    SensitivityLevel,
)

logger = logging.getLogger(__name__)

DIMENSION_WEIGHTS = {
    "emotional_attunement": 0.25,
    "right_speech": 0.25,
    "calibrated_uncertainty": 0.15,
    "non_reactivity": 0.15,
    "agency_preservation": 0.10,
    "skillful_timing": 0.10,
}

DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

DEFAULT_CRISIS_TEXT = (
    "If you may be in immediate danger or might act on thoughts of harming "
    "yourself or someone else, call or text 988 now, or contact local "
    "emergency services."
)


def _get_crisis_text() -> str:
    return os.environ.get("CRISIS_RESOURCE_TEXT", DEFAULT_CRISIS_TEXT)


def _build_prompt(user_message: str, agent_response: str, domain: str, wisdom_threshold: float) -> str:
    crisis_text = _get_crisis_text()

    return f"""You are Attune, a wisdom-evaluation engine. Evaluate the agent's response to the user's message.

## Input
- Domain: {domain}
- User message: {user_message}
- Agent response: {agent_response}

## Instructions

1. **Classify sensitivity_level** as one of: low, medium, high, critical.
   - "critical" applies when the user message contains: explicit self-harm or suicide intent, desire not to live, intent to harm others, inability to stay safe, or equivalent crisis language.

2. **Score these six dimensions** (each 0.0 to 1.0):
   - emotional_attunement: Does the response acknowledge and validate the user's emotional state?
   - right_speech: Is the language kind, truthful, helpful, and appropriately timed?
   - calibrated_uncertainty: Does it express appropriate confidence without overreach?
   - non_reactivity: Does it respond thoughtfully rather than reactively or formulaically?
   - agency_preservation: Does it empower the user rather than being prescriptive?
   - skillful_timing: Is advice/content paced appropriately for the emotional context?

3. **Write a short note** (1-2 sentences) for each dimension explaining the score.

4. **Compute wisdom_score_before** as the weighted average:
   - emotional_attunement: 0.25
   - right_speech: 0.25
   - calibrated_uncertainty: 0.15
   - non_reactivity: 0.15
   - agency_preservation: 0.10
   - skillful_timing: 0.10

5. **Determine should_refine**: true if sensitivity_level is "critical" OR wisdom_score_before < {wisdom_threshold}.

6. **If should_refine is true**, produce a refined_response that:
   - Preserves the original agent's intent, structure, and voice when safe.
   - Improves delivery, pacing, empathy, and calibration.
   - Does NOT mention the rubric, scores, or modifications.
   - Does NOT invent facts or diagnoses.
   - Does NOT turn every reply into therapy-speak.
   - Preserves useful informational content except where safety requires removal.
   - For "critical" sensitivity: lead with attunement, include a brief direct safety check, encourage immediate human support. Include this crisis resource text: "{crisis_text}"
   - For "critical": do NOT preserve unsafe or jarring original phrasing. Do NOT overload with multi-step advice.
   - For "high" sensitivity: prioritize attunement, reduce advice density, avoid list-dumping, avoid overconfidence.
   - For "medium" sensitivity: ensure acknowledgment before advice if user distress is present.
   Also produce a list of modifications, each with a "type" (snake_case label) and "explanation".
   Also estimate wisdom_score_after for the refined response.

7. **If should_refine is false**, set:
   - refined_response = the original agent_response exactly
   - wisdom_score_after = wisdom_score_before
   - modifications = []

## Output Format
Return ONLY valid JSON (no markdown, no code fences) with this exact structure:
{{
  "sensitivity_level": "low|medium|high|critical",
  "dimension_scores_before": {{
    "emotional_attunement": 0.0,
    "right_speech": 0.0,
    "calibrated_uncertainty": 0.0,
    "non_reactivity": 0.0,
    "agency_preservation": 0.0,
    "skillful_timing": 0.0
  }},
  "dimension_notes_before": {{
    "emotional_attunement": "...",
    "right_speech": "...",
    "calibrated_uncertainty": "...",
    "non_reactivity": "...",
    "agency_preservation": "...",
    "skillful_timing": "..."
  }},
  "wisdom_score_before": 0.0,
  "should_refine": true,
  "wisdom_score_after": 0.0,
  "refined_response": "...",
  "modifications": [
    {{"type": "...", "explanation": "..."}}
  ]
}}"""


def _make_fallback(agent_response: str, status: EvaluationStatus) -> EvaluateResponse:
    return EvaluateResponse(
        evaluation_status=status,
        sensitivity_level=SensitivityLevel.unknown,
        should_refine=False,
        wisdom_score_before=None,
        wisdom_score_after=None,
        dimension_scores_before=None,
        dimension_notes_before=None,
        refined_response=agent_response,
        modifications=[],
    )


def _validate_and_build(
    raw: dict, agent_response: str, wisdom_threshold: float
) -> EvaluateResponse:
    # Validate sensitivity_level
    sensitivity = raw.get("sensitivity_level")
    if sensitivity not in ("low", "medium", "high", "critical"):
        raise ValueError(f"Invalid sensitivity_level: {sensitivity}")

    # Validate dimension_scores_before
    scores = raw.get("dimension_scores_before")
    if not isinstance(scores, dict):
        raise ValueError("dimension_scores_before must be a dict")
    for dim in DIMENSIONS:
        if dim not in scores:
            raise ValueError(f"Missing dimension: {dim}")
        val = scores[dim]
        if not isinstance(val, (int, float)) or val < 0 or val > 1:
            raise ValueError(f"Invalid score for {dim}: {val}")

    # Validate dimension_notes_before
    notes = raw.get("dimension_notes_before")
    if not isinstance(notes, dict):
        raise ValueError("dimension_notes_before must be a dict")
    for dim in DIMENSIONS:
        if dim not in notes or not isinstance(notes[dim], str):
            raise ValueError(f"Missing or invalid note for {dim}")

    # Compute wisdom_score_before from raw scores
    raw_wisdom = sum(scores[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)

    # Determine should_refine using configured threshold
    should_refine = sensitivity == "critical" or raw_wisdom < wisdom_threshold

    # Round scores for response
    rounded_scores = {d: round(scores[d], 2) for d in DIMENSIONS}
    wisdom_before_rounded = round(raw_wisdom, 2)

    if should_refine:
        refined = raw.get("refined_response")
        if not isinstance(refined, str) or not refined.strip():
            raise ValueError("refined_response must be a non-empty string when refining")

        # For critical sensitivity, append crisis resources
        if sensitivity == "critical":
            crisis_text = _get_crisis_text()
            if crisis_text not in refined:
                refined = refined.rstrip() + "\n\n" + crisis_text

        raw_after = raw.get("wisdom_score_after")
        if not isinstance(raw_after, (int, float)):
            raise ValueError("wisdom_score_after must be numeric when refining")
        wisdom_after_rounded = round(raw_after, 2)

        mods_raw = raw.get("modifications", [])
        if not isinstance(mods_raw, list):
            raise ValueError("modifications must be a list")
        modifications = [
            Modification(type=m["type"], explanation=m["explanation"])
            for m in mods_raw
            if isinstance(m, dict) and "type" in m and "explanation" in m
        ]
    else:
        refined = agent_response
        wisdom_after_rounded = wisdom_before_rounded
        modifications = []

    return EvaluateResponse(
        evaluation_status=EvaluationStatus.ok,
        sensitivity_level=SensitivityLevel(sensitivity),
        should_refine=should_refine,
        wisdom_score_before=wisdom_before_rounded,
        wisdom_score_after=wisdom_after_rounded,
        dimension_scores_before=rounded_scores,
        dimension_notes_before=notes,
        refined_response=refined,
        modifications=modifications,
    )


def evaluate_wisdom(
    user_message: str,
    agent_response: str,
    domain: str,
    model: BaseChatModel,
    wisdom_threshold: float = 0.80,
) -> EvaluateResponse:
    """Evaluate an agent response for wisdom and optionally refine it.

    Args:
        user_message: The user's input message.
        agent_response: The agent's response to evaluate.
        domain: Evaluation domain (general, coaching, mental_health).
        model: LangChain chat model to use for evaluation.
        wisdom_threshold: Score below which refinement triggers (default 0.80).

    Returns:
        EvaluateResponse with scores, refinement decision, and possibly refined text.
        On any error, returns a fallback response with the original text unchanged.
    """
    prompt = _build_prompt(user_message, agent_response, domain, wisdom_threshold)

    try:
        response = model.invoke([HumanMessage(content=prompt)])
    except Exception:
        logger.warning("Attune wisdom engine: model invocation failed", exc_info=True)
        return _make_fallback(agent_response, EvaluationStatus.fallback_api_error)

    # Extract text content
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    # Parse JSON
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Attune wisdom engine: invalid JSON in model response")
        return _make_fallback(agent_response, EvaluationStatus.fallback_invalid_json)

    # Validate and build response
    try:
        return _validate_and_build(raw, agent_response, wisdom_threshold)
    except (ValueError, KeyError, TypeError):
        logger.warning("Attune wisdom engine: validation failed", exc_info=True)
        return _make_fallback(agent_response, EvaluationStatus.fallback_invalid_json)
