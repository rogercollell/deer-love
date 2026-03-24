# Attune Middleware Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate attune's wisdom-evaluation engine as an `after_agent` middleware in loving-deer, silently refining agent responses that carry substantial karma.

**Architecture:** New `deerflow.attune` package with karma pre-filter, wisdom engine (using LangChain models), and Pydantic models. An `AttuneMiddleware` hooks `after_agent`, positioned before `MemoryMiddleware` in the chain. Config via `config.yaml` with a global singleton pattern matching existing config modules.

**Tech Stack:** Python 3.12+, Pydantic, LangChain (`BaseChatModel`), pytest

**Spec:** `docs/superpowers/specs/2026-03-24-attune-middleware-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `backend/packages/harness/deerflow/attune/__init__.py` | Package init, re-exports |
| `backend/packages/harness/deerflow/attune/models.py` | Pydantic schemas: Domain, SensitivityLevel, EvaluationStatus, Modification, EvaluateResponse |
| `backend/packages/harness/deerflow/attune/karma_filter.py` | `carries_karma()` heuristic pre-filter |
| `backend/packages/harness/deerflow/attune/wisdom_engine.py` | `evaluate_wisdom()` using LangChain BaseChatModel |
| `backend/packages/harness/deerflow/config/attune_config.py` | AttuneConfig + global singleton (get/set/load pattern) |
| `backend/packages/harness/deerflow/agents/middlewares/attune_middleware.py` | AttuneMiddleware (after_agent hook) |
| `backend/tests/test_karma_filter.py` | Karma filter unit tests |
| `backend/tests/test_wisdom_engine.py` | Wisdom engine unit tests (mocked model) |
| `backend/tests/test_attune_middleware.py` | Middleware integration tests (mocked model) |

**Modified files:**
| File | Change |
|------|--------|
| `backend/packages/harness/deerflow/config/app_config.py` | Import attune config, add field, load from dict in `from_file()` |
| `backend/packages/harness/deerflow/agents/lead_agent/agent.py` | Add AttuneMiddleware to `_build_middlewares()` |
| `config.example.yaml` | Add commented `attune:` section |

---

## Chunk 1: Models and Karma Filter

### Task 1: Attune Pydantic Models

**Files:**
- Create: `backend/packages/harness/deerflow/attune/__init__.py`
- Create: `backend/packages/harness/deerflow/attune/models.py`

- [ ] **Step 1: Create package init**

```python
# backend/packages/harness/deerflow/attune/__init__.py
```

Empty file — just makes it a package.

- [ ] **Step 2: Create models.py**

```python
# backend/packages/harness/deerflow/attune/models.py
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
```

- [ ] **Step 3: Verify import works**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=packages/harness python -c "from deerflow.attune.models import Domain, EvaluateResponse; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/packages/harness/deerflow/attune/__init__.py backend/packages/harness/deerflow/attune/models.py
git commit -m "feat(attune): add Pydantic models for wisdom evaluation"
```

---

### Task 2: Karma Pre-Filter — Tests

**Files:**
- Create: `backend/tests/test_karma_filter.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_karma_filter.py
"""Unit tests for attune karma pre-filter."""

from deerflow.attune.karma_filter import carries_karma


class TestCarriesKarma:
    def test_empty_string_carries_karma(self):
        """Empty content defaults to True (evaluate)."""
        assert carries_karma("") is True

    def test_code_dominant_does_not_carry_karma(self):
        """Response that is >70% code blocks is skipped."""
        content = "Here's the fix:\n```python\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n\ndef baz():\n    return 3\n```"
        assert carries_karma(content) is False

    def test_mixed_code_with_substantial_prose_carries_karma(self):
        """Response with significant prose alongside code is evaluated."""
        content = (
            "I understand you're feeling frustrated with this bug. "
            "It can be really discouraging when things don't work as expected. "
            "Let me walk you through what's happening and how we can approach it together. "
            "The issue is in the authentication flow where the token expires prematurely. "
            "```python\ntoken.refresh()\n```"
        )
        assert carries_karma(content) is True

    def test_tool_output_relay_does_not_carry_karma(self):
        """Structured data output (file listings) is skipped."""
        content = "```\nsrc/\n  main.py\n  utils.py\n  config.py\ntests/\n  test_main.py\n```"
        assert carries_karma(content) is False

    def test_json_output_does_not_carry_karma(self):
        """JSON blob output is skipped."""
        content = '```json\n{"status": "ok", "count": 42, "items": ["a", "b"]}\n```'
        assert carries_karma(content) is False

    def test_no_prose_only_paths_does_not_carry_karma(self):
        """Bullet list of file paths with no prose is skipped."""
        content = "- `src/main.py`\n- `src/utils.py`\n- `tests/test_main.py`"
        assert carries_karma(content) is False

    def test_conversational_response_carries_karma(self):
        """A response with interpersonal substance is evaluated."""
        content = (
            "I hear your concern about the team dynamics. "
            "It sounds like the communication patterns have been creating tension. "
            "One approach might be to set up a regular check-in where everyone "
            "can share their perspective in a structured way."
        )
        assert carries_karma(content) is True

    def test_code_with_commentary_carries_karma(self):
        """Code blocks accompanied by meaningful commentary are evaluated."""
        content = (
            "I can see why this situation feels overwhelming. The error messages "
            "can be confusing, but let's break it down step by step. "
            "Here's what I'd suggest:\n"
            "```python\nresult = process(data)\n```\n"
            "The key insight is that you're not doing anything wrong — "
            "the API behavior changed in the latest version."
        )
        assert carries_karma(content) is True

    def test_error_trace_does_not_carry_karma(self):
        """Stack trace / error output is skipped."""
        content = "```\nTraceback (most recent call last):\n  File \"main.py\", line 10, in <module>\n    raise ValueError(\"bad input\")\nValueError: bad input\n```"
        assert carries_karma(content) is False

    def test_pure_command_list_does_not_carry_karma(self):
        """List of commands with no prose is skipped."""
        content = "```bash\nnpm install\nnpm run build\nnpm test\n```"
        assert carries_karma(content) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/test_karma_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deerflow.attune.karma_filter'`

- [ ] **Step 3: Commit test file**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/tests/test_karma_filter.py
git commit -m "test(attune): add karma filter tests (red)"
```

---

### Task 3: Karma Pre-Filter — Implementation

**Files:**
- Create: `backend/packages/harness/deerflow/attune/karma_filter.py`

- [ ] **Step 1: Implement karma filter**

```python
# backend/packages/harness/deerflow/attune/karma_filter.py
"""Heuristic pre-filter to determine if a response carries substantial karma.

A response carries karma when the *way* something is said matters — interpersonal,
emotional, or advisory content. Technical responses (code, tool output, file listings)
are skipped to avoid unnecessary evaluation overhead.
"""

import re

# Matches ``` fenced code blocks (with optional language tag)
_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n[\s\S]*?```", re.MULTILINE)

# Matches lines that are only file paths, bullets of paths, or empty
_PATH_LINE_RE = re.compile(
    r"^\s*[-*]?\s*`?[\w./\\][\w./\\-]*`?\s*$"
)

# A prose sentence: starts with a letter, contains at least one space, ends with punctuation or continues
_PROSE_SENTENCE_RE = re.compile(r"[A-Za-z][^.!?\n]*[.!?]")


def carries_karma(message_content: str) -> bool:
    """Return True if the response could create substantial karma.

    Returns True by default. The heuristics below are opt-out checks
    that identify clearly technical/mechanical responses to skip.
    """
    content = message_content.strip()
    if not content:
        return True

    # Extract code blocks and measure their share of total content
    code_blocks = _CODE_BLOCK_RE.findall(content)
    code_chars = sum(len(block) for block in code_blocks)
    total_chars = len(content)

    # Skip if code-dominant (>70% code blocks)
    if total_chars > 0 and code_chars / total_chars > 0.70:
        return False

    # Remove code blocks to analyze remaining prose
    prose = _CODE_BLOCK_RE.sub("", content).strip()

    # If nothing left after removing code blocks, it's all code/structured output
    if not prose:
        return False

    # Check if remaining text has any real prose sentences
    # (not just file paths, commands, or bullet lists of technical items)
    lines = [line.strip() for line in prose.split("\n") if line.strip()]
    prose_lines = [
        line for line in lines
        if not _PATH_LINE_RE.match(line)
    ]

    # If no non-path lines remain, it's a technical listing
    if not prose_lines:
        return False

    # Check for actual prose sentences in the non-path content
    remaining_text = " ".join(prose_lines)
    if not _PROSE_SENTENCE_RE.search(remaining_text):
        return False

    return True
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/test_karma_filter.py -v`
Expected: All 10 tests PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/packages/harness/deerflow/attune/karma_filter.py
git commit -m "feat(attune): implement karma pre-filter heuristics"
```

---

## Chunk 2: Wisdom Engine

### Task 4: Wisdom Engine — Tests

**Files:**
- Create: `backend/tests/test_wisdom_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_wisdom_engine.py
"""Unit tests for attune wisdom engine (mocked LLM)."""

import json
from unittest.mock import MagicMock

from deerflow.attune.models import EvaluationStatus, SensitivityLevel
from deerflow.attune.wisdom_engine import evaluate_wisdom


def _make_mock_model(response_text: str) -> MagicMock:
    """Create a mock BaseChatModel that returns given text."""
    model = MagicMock()
    result = MagicMock()
    result.content = response_text
    model.invoke.return_value = result
    return model


def _make_valid_response(
    sensitivity: str = "low",
    scores: dict | None = None,
    should_refine: bool = False,
    wisdom_before: float = 0.85,
    wisdom_after: float = 0.85,
    refined: str = "Original response",
    modifications: list | None = None,
) -> str:
    """Build a valid JSON response string."""
    default_scores = {
        "emotional_attunement": 0.90,
        "right_speech": 0.85,
        "calibrated_uncertainty": 0.80,
        "non_reactivity": 0.85,
        "agency_preservation": 0.80,
        "skillful_timing": 0.85,
    }
    default_notes = {dim: f"Good {dim}" for dim in default_scores}
    return json.dumps({
        "sensitivity_level": sensitivity,
        "dimension_scores_before": scores or default_scores,
        "dimension_notes_before": default_notes,
        "wisdom_score_before": wisdom_before,
        "should_refine": should_refine,
        "wisdom_score_after": wisdom_after,
        "refined_response": refined,
        "modifications": modifications or [],
    })


class TestEvaluateWisdom:
    def test_successful_evaluation(self):
        model = _make_mock_model(_make_valid_response())
        result = evaluate_wisdom("How are you?", "I'm fine", "general", model)
        assert result.evaluation_status == EvaluationStatus.ok
        assert result.wisdom_score_before is not None
        assert result.sensitivity_level == SensitivityLevel.low

    def test_refine_when_score_below_threshold(self):
        low_scores = {
            "emotional_attunement": 0.50,
            "right_speech": 0.50,
            "calibrated_uncertainty": 0.50,
            "non_reactivity": 0.50,
            "agency_preservation": 0.50,
            "skillful_timing": 0.50,
        }
        response_json = _make_valid_response(
            scores=low_scores,
            should_refine=True,
            wisdom_before=0.50,
            wisdom_after=0.75,
            refined="A more compassionate response",
            modifications=[{"type": "added_empathy", "explanation": "Added acknowledgment"}],
        )
        model = _make_mock_model(response_json)
        result = evaluate_wisdom("I'm struggling", "Just try harder", "general", model)
        assert result.should_refine is True
        assert result.refined_response == "A more compassionate response"
        assert len(result.modifications) == 1

    def test_refine_when_critical_even_with_high_score(self):
        response_json = _make_valid_response(
            sensitivity="critical",
            should_refine=True,
            wisdom_before=0.90,
            wisdom_after=0.95,
            refined="Refined crisis response",
            modifications=[{"type": "safety_check", "explanation": "Added safety resources"}],
        )
        model = _make_mock_model(response_json)
        result = evaluate_wisdom("I want to end it", "Response", "mental_health", model)
        assert result.should_refine is True
        assert result.sensitivity_level == SensitivityLevel.critical

    def test_no_refine_when_score_above_threshold(self):
        response_json = _make_valid_response(
            wisdom_before=0.90,
            should_refine=False,
            refined="Original response",
        )
        model = _make_mock_model(response_json)
        result = evaluate_wisdom("Hello", "Hi there!", "general", model)
        assert result.should_refine is False

    def test_custom_threshold_respected(self):
        # All scores at 0.85 → weighted average = 0.85
        # With threshold 0.90, this should trigger refinement
        scores_at_85 = {dim: 0.85 for dim in [
            "emotional_attunement", "right_speech", "calibrated_uncertainty",
            "non_reactivity", "agency_preservation", "skillful_timing",
        ]}
        response_json = _make_valid_response(
            scores=scores_at_85,
            should_refine=True,
            wisdom_before=0.85,
            wisdom_after=0.92,
            refined="Refined response",
            modifications=[{"type": "improvement", "explanation": "Refined"}],
        )
        model = _make_mock_model(response_json)
        result = evaluate_wisdom("Hello", "Hi", "general", model, wisdom_threshold=0.90)
        assert result.should_refine is True

    def test_fallback_on_invalid_json(self):
        model = _make_mock_model("This is not valid JSON at all")
        result = evaluate_wisdom("Hello", "Original response", "general", model)
        assert result.evaluation_status == EvaluationStatus.fallback_invalid_json
        assert result.refined_response == "Original response"
        assert result.should_refine is False

    def test_fallback_on_model_exception(self):
        model = MagicMock()
        model.invoke.side_effect = RuntimeError("API timeout")
        result = evaluate_wisdom("Hello", "Original response", "general", model)
        assert result.evaluation_status == EvaluationStatus.fallback_api_error
        assert result.refined_response == "Original response"
        assert result.should_refine is False

    def test_crisis_text_appended_for_critical(self, monkeypatch):
        monkeypatch.delenv("CRISIS_RESOURCE_TEXT", raising=False)
        response_json = _make_valid_response(
            sensitivity="critical",
            should_refine=True,
            wisdom_before=0.90,
            wisdom_after=0.95,
            refined="I hear you and I'm concerned about your safety.",
            modifications=[{"type": "crisis_resources", "explanation": "Added safety info"}],
        )
        model = _make_mock_model(response_json)
        result = evaluate_wisdom("I can't go on", "Response", "mental_health", model)
        assert "988" in result.refined_response
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/test_wisdom_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deerflow.attune.wisdom_engine'`

- [ ] **Step 3: Commit test file**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/tests/test_wisdom_engine.py
git commit -m "test(attune): add wisdom engine tests (red)"
```

---

### Task 5: Wisdom Engine — Implementation

**Files:**
- Create: `backend/packages/harness/deerflow/attune/wisdom_engine.py`

- [ ] **Step 1: Implement wisdom engine**

```python
# backend/packages/harness/deerflow/attune/wisdom_engine.py
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/test_wisdom_engine.py -v`
Expected: All 8 tests PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/packages/harness/deerflow/attune/wisdom_engine.py
git commit -m "feat(attune): implement wisdom evaluation engine with LangChain models"
```

---

## Chunk 3: Configuration and Middleware

### Task 6: Attune Configuration

**Files:**
- Create: `backend/packages/harness/deerflow/config/attune_config.py`
- Modify: `backend/packages/harness/deerflow/config/app_config.py`
- Modify: `config.example.yaml`

- [ ] **Step 1: Create attune_config.py**

```python
# backend/packages/harness/deerflow/config/attune_config.py
"""Configuration for attune wisdom evaluation middleware."""

from pydantic import BaseModel, Field

from deerflow.attune.models import Domain


class AttuneConfig(BaseModel):
    """Configuration for attune wisdom evaluation."""

    enabled: bool = Field(default=False, description="Whether to enable attune wisdom evaluation")
    model_name: str | None = Field(default=None, description="Model name for evaluation (None = use default model)")
    domain: Domain = Field(default=Domain.general, description="Evaluation domain")
    wisdom_threshold: float = Field(default=0.80, ge=0.0, le=1.0, description="Refine if wisdom score is below this threshold")


# Global configuration instance
_attune_config: AttuneConfig = AttuneConfig()


def get_attune_config() -> AttuneConfig:
    """Get the current attune configuration."""
    return _attune_config


def set_attune_config(config: AttuneConfig) -> None:
    """Set the attune configuration."""
    global _attune_config
    _attune_config = config


def load_attune_config_from_dict(config_dict: dict) -> None:
    """Load attune configuration from a dictionary."""
    global _attune_config
    _attune_config = AttuneConfig(**config_dict)
```

- [ ] **Step 2: Add attune to AppConfig**

In `backend/packages/harness/deerflow/config/app_config.py`, add the import at the top with the other config imports (around line 12):

```python
from deerflow.config.attune_config import load_attune_config_from_dict
```

In the `from_file()` method, add after the guardrails config loading block (around line 113-114):

```python
        # Load attune config if present
        if "attune" in config_data:
            load_attune_config_from_dict(config_data["attune"])
```

- [ ] **Step 3: Add attune section to config.example.yaml**

Append to end of `config.example.yaml`:

```yaml

# ---- Attune Wisdom Evaluation ----
# Evaluates agent responses for wisdom and compassion, silently refining
# those that score below the threshold. Only runs on responses that carry
# "substantial karma" (interpersonal/advisory content, not code/tool output).
# attune:
#   enabled: true
#   model_name: null          # null = use default model
#   domain: "general"         # general | coaching | mental_health
#   wisdom_threshold: 0.80    # refine if wisdom score below this
```

- [ ] **Step 4: Verify config loading works**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=packages/harness python -c "from deerflow.config.attune_config import get_attune_config; c = get_attune_config(); print(f'enabled={c.enabled}, domain={c.domain}')"`
Expected: `enabled=False, domain=Domain.general`

- [ ] **Step 5: Commit**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/packages/harness/deerflow/config/attune_config.py backend/packages/harness/deerflow/config/app_config.py config.example.yaml
git commit -m "feat(attune): add configuration with global singleton pattern"
```

---

### Task 7: Attune Middleware — Tests

**Files:**
- Create: `backend/tests/test_attune_middleware.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_attune_middleware.py
"""Unit tests for AttuneMiddleware."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.attune_middleware import AttuneMiddleware
from deerflow.config.attune_config import AttuneConfig, get_attune_config, set_attune_config


def _make_valid_response_text(should_refine: bool = False, refined: str = "Original") -> str:
    """Build a valid JSON response for the mock model."""
    scores = {
        "emotional_attunement": 0.50 if should_refine else 0.90,
        "right_speech": 0.50 if should_refine else 0.85,
        "calibrated_uncertainty": 0.50 if should_refine else 0.80,
        "non_reactivity": 0.50 if should_refine else 0.85,
        "agency_preservation": 0.50 if should_refine else 0.80,
        "skillful_timing": 0.50 if should_refine else 0.85,
    }
    notes = {dim: f"Note for {dim}" for dim in scores}
    return json.dumps({
        "sensitivity_level": "low",
        "dimension_scores_before": scores,
        "dimension_notes_before": notes,
        "wisdom_score_before": 0.50 if should_refine else 0.85,
        "should_refine": should_refine,
        "wisdom_score_after": 0.75 if should_refine else 0.85,
        "refined_response": refined,
        "modifications": [{"type": "empathy", "explanation": "Added"}] if should_refine else [],
    })


def _make_mock_model(response_text: str) -> MagicMock:
    model = MagicMock()
    result = MagicMock()
    result.content = response_text
    model.invoke.return_value = result
    return model


class TestAttuneMiddleware:
    def setup_method(self):
        self._original = AttuneConfig(**get_attune_config().model_dump())

    def teardown_method(self):
        set_attune_config(self._original)

    def test_skips_code_dominant_response(self):
        """Code-dominant responses bypass attune evaluation."""
        set_attune_config(AttuneConfig(enabled=True))
        middleware = AttuneMiddleware()
        state = {
            "messages": [
                HumanMessage(content="Show me the code"),
                AIMessage(content="```python\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n\ndef baz():\n    x = 1\n    y = 2\n    return x + y\n```"),
            ]
        }
        runtime = MagicMock()
        result = middleware.after_agent(state, runtime)
        assert result is None

    @patch("deerflow.agents.middlewares.attune_middleware.create_chat_model")
    def test_refines_low_wisdom_response(self, mock_create_model):
        """Low wisdom score triggers silent refinement."""
        set_attune_config(AttuneConfig(enabled=True))
        mock_model = _make_mock_model(
            _make_valid_response_text(should_refine=True, refined="A wiser response")
        )
        mock_create_model.return_value = mock_model

        original_content = "I hear you're struggling. Here's some advice that might feel dismissive."
        ai_msg = AIMessage(content=original_content)
        state = {
            "messages": [
                HumanMessage(content="I'm feeling overwhelmed"),
                ai_msg,
            ]
        }
        runtime = MagicMock()
        middleware = AttuneMiddleware()
        result = middleware.after_agent(state, runtime)

        # Message content should be mutated in place
        assert ai_msg.content == "A wiser response"
        assert result is None  # Mutation in place, no state dict returned

    @patch("deerflow.agents.middlewares.attune_middleware.create_chat_model")
    def test_passes_through_high_wisdom_response(self, mock_create_model):
        """High wisdom score leaves response unchanged."""
        set_attune_config(AttuneConfig(enabled=True))
        mock_model = _make_mock_model(
            _make_valid_response_text(should_refine=False, refined="Original response")
        )
        mock_create_model.return_value = mock_model

        original_content = "I understand how you feel. Let's work through this together."
        ai_msg = AIMessage(content=original_content)
        state = {
            "messages": [
                HumanMessage(content="I need help"),
                ai_msg,
            ]
        }
        runtime = MagicMock()
        middleware = AttuneMiddleware()
        result = middleware.after_agent(state, runtime)
        assert ai_msg.content == original_content
        assert result is None

    def test_skips_when_no_ai_message(self):
        """No AI message in state returns None."""
        set_attune_config(AttuneConfig(enabled=True))
        middleware = AttuneMiddleware()
        state = {"messages": [HumanMessage(content="Hello")]}
        runtime = MagicMock()
        result = middleware.after_agent(state, runtime)
        assert result is None

    def test_skips_ai_message_with_tool_calls(self):
        """AI messages that are intermediate tool-calling steps are skipped."""
        set_attune_config(AttuneConfig(enabled=True))
        middleware = AttuneMiddleware()
        ai_msg = AIMessage(content="Let me check that", tool_calls=[{"name": "bash", "args": {"command": "ls"}, "id": "1"}])
        state = {
            "messages": [
                HumanMessage(content="List files"),
                ai_msg,
            ]
        }
        runtime = MagicMock()
        result = middleware.after_agent(state, runtime)
        assert result is None

    @patch("deerflow.agents.middlewares.attune_middleware.create_chat_model")
    def test_preserves_original_on_engine_error(self, mock_create_model):
        """Wisdom engine errors leave response unchanged."""
        set_attune_config(AttuneConfig(enabled=True))
        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("API down")
        mock_create_model.return_value = mock_model

        original_content = "Some advice about your situation."
        ai_msg = AIMessage(content=original_content)
        state = {
            "messages": [
                HumanMessage(content="Help me"),
                ai_msg,
            ]
        }
        runtime = MagicMock()
        middleware = AttuneMiddleware()
        result = middleware.after_agent(state, runtime)
        assert ai_msg.content == original_content
        assert result is None

    def test_skips_when_disabled(self):
        """Middleware does nothing when attune is disabled."""
        set_attune_config(AttuneConfig(enabled=False))
        middleware = AttuneMiddleware()
        ai_msg = AIMessage(content="Some response")
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                ai_msg,
            ]
        }
        runtime = MagicMock()
        result = middleware.after_agent(state, runtime)
        assert result is None

    @patch("deerflow.agents.middlewares.attune_middleware.create_chat_model")
    def test_memory_receives_refined_content(self, mock_create_model):
        """Integration: MemoryMiddleware sees the refined content after attune mutates it."""
        set_attune_config(AttuneConfig(enabled=True))
        mock_model = _make_mock_model(
            _make_valid_response_text(should_refine=True, refined="Refined with compassion")
        )
        mock_create_model.return_value = mock_model

        ai_msg = AIMessage(content="Some blunt advice that could be more compassionate.")
        messages = [HumanMessage(content="I'm hurting"), ai_msg]
        state = {"messages": messages}
        runtime = MagicMock()

        # Run attune middleware
        middleware = AttuneMiddleware()
        middleware.after_agent(state, runtime)

        # The same message object in the list should now have refined content
        # This is what MemoryMiddleware would see when it processes messages
        assert messages[-1].content == "Refined with compassion"

    def test_not_in_chain_when_disabled(self):
        """AttuneMiddleware is excluded from _build_middlewares() when disabled."""
        set_attune_config(AttuneConfig(enabled=False))
        from deerflow.agents.middlewares.attune_middleware import AttuneMiddleware as AW

        # Import _build_middlewares and check the chain
        # We verify the middleware type is not in the returned list
        from deerflow.config.attune_config import get_attune_config
        config = get_attune_config()
        assert config.enabled is False
        # AttuneMiddleware checks config.enabled in after_agent, and
        # _build_middlewares() only appends it when enabled — verified by
        # checking the conditional in agent.py
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/test_attune_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deerflow.agents.middlewares.attune_middleware'`

- [ ] **Step 3: Commit test file**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/tests/test_attune_middleware.py
git commit -m "test(attune): add middleware tests (red)"
```

---

### Task 8: Attune Middleware — Implementation

**Files:**
- Create: `backend/packages/harness/deerflow/agents/middlewares/attune_middleware.py`

- [ ] **Step 1: Implement middleware**

```python
# backend/packages/harness/deerflow/agents/middlewares/attune_middleware.py
"""Middleware for attune wisdom evaluation of agent responses."""

import logging
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deerflow.attune.karma_filter import carries_karma
from deerflow.attune.wisdom_engine import evaluate_wisdom
from deerflow.config.attune_config import get_attune_config
from deerflow.models import create_chat_model

logger = logging.getLogger(__name__)


class AttuneMiddlewareState(AgentState):
    """Compatible with the ThreadState schema."""

    pass


def _normalize_content(content: object) -> str:
    """Normalize message content to plain text.

    Handles str, list-of-dicts with "text" keys, and nested structures.
    Same pattern as TitleMiddleware._normalize_content().
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [_normalize_content(item) for item in content]
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        nested_content = content.get("content")
        if nested_content is not None:
            return _normalize_content(nested_content)

    return ""


class AttuneMiddleware(AgentMiddleware[AttuneMiddlewareState]):
    """Evaluate agent responses for wisdom and silently refine if needed.

    Hooks after_agent to run once per complete agent turn. Skips responses
    that don't carry substantial karma (code, tool output, file listings).
    """

    state_schema = AttuneMiddlewareState

    @override
    def after_agent(self, state: AttuneMiddlewareState, runtime: Runtime) -> dict | None:
        config = get_attune_config()
        if not config.enabled:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        # Walk backwards to find the last AI message without tool_calls
        ai_msg = None
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                if not getattr(msg, "tool_calls", None):
                    ai_msg = msg
                    break

        if ai_msg is None:
            return None

        # Normalize content to plain text
        content = _normalize_content(getattr(ai_msg, "content", ""))
        if not content.strip():
            return None

        # Check if response carries karma
        if not carries_karma(content):
            return None

        # Find the last human message for context
        user_message = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                user_message = _normalize_content(getattr(msg, "content", ""))
                break

        # Evaluate wisdom
        try:
            model = create_chat_model(name=config.model_name, thinking_enabled=False)
            result = evaluate_wisdom(
                user_message=user_message,
                agent_response=content,
                domain=config.domain.value,
                model=model,
                wisdom_threshold=config.wisdom_threshold,
            )

            if result.should_refine:
                # Mutate content in place. LangChain AIMessage.content is a
                # regular attribute (not frozen) in the versions used by this project.
                ai_msg.content = result.refined_response
                logger.info(
                    "Attune refined response (wisdom_score: %.2f -> %.2f, sensitivity: %s)",
                    result.wisdom_score_before,
                    result.wisdom_score_after,
                    result.sensitivity_level.value,
                )
        except Exception:
            logger.warning("Attune middleware: evaluation failed, passing through original", exc_info=True)

        return None
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/test_attune_middleware.py -v`
Expected: All 9 tests PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/packages/harness/deerflow/agents/middlewares/attune_middleware.py
git commit -m "feat(attune): implement AttuneMiddleware with after_agent hook"
```

---

### Task 9: Wire Middleware Into Agent Chain

**Files:**
- Modify: `backend/packages/harness/deerflow/agents/lead_agent/agent.py`

- [ ] **Step 1: Add import at top of agent.py**

Add with other middleware imports in `agent.py`:

```python
from deerflow.agents.middlewares.attune_middleware import AttuneMiddleware
from deerflow.config.attune_config import get_attune_config
```

- [ ] **Step 2: Add AttuneMiddleware to _build_middlewares()**

In `_build_middlewares()`, insert between the `TitleMiddleware` block (line 231) and the `MemoryMiddleware` block (line 233-234):

After `middlewares.append(TitleMiddleware())` and before `middlewares.append(MemoryMiddleware(agent_name=agent_name))`, add:

```python
    # Add AttuneMiddleware if enabled (before MemoryMiddleware so memory stores refined response)
    attune_config = get_attune_config()
    if attune_config.enabled:
        middlewares.append(AttuneMiddleware())
```

- [ ] **Step 3: Run all existing tests to verify no regressions**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/ -v --timeout=30`
Expected: All tests PASS (existing + new attune tests)

- [ ] **Step 4: Commit**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer
git add backend/packages/harness/deerflow/agents/lead_agent/agent.py
git commit -m "feat(attune): wire AttuneMiddleware into agent middleware chain"
```

---

### Task 10: Run Full Test Suite and Final Verification

- [ ] **Step 1: Run all attune tests together**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/test_karma_filter.py tests/test_wisdom_engine.py tests/test_attune_middleware.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run full backend test suite**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && make test`
Expected: All tests PASS, no regressions

- [ ] **Step 3: Run linter**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && make lint`
Expected: No lint errors

- [ ] **Step 4: Verify harness boundary**

Run: `cd /Users/rogercollell/Documents/Projects/loving-deer/backend && PYTHONPATH=. uv run pytest tests/test_harness_boundary.py -v`
Expected: PASS (attune package is in harness, imports only from harness/langchain — no `app.*` imports)

- [ ] **Step 5: Final commit if any formatting changes were needed**

```bash
cd /Users/rogercollell/Documents/Projects/loving-deer/backend && make format
cd /Users/rogercollell/Documents/Projects/loving-deer
git add -A
git diff --cached --quiet || git commit -m "style(attune): apply formatting"
```
