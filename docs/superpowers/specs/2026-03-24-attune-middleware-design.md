# Attune Middleware Integration Design

## Overview

Integrate attune's wisdom-evaluation engine into loving-deer as an `after_agent` middleware. The middleware evaluates agent responses that carry "substantial karma" — responses where the *way* something is said matters — and silently refines those that score below a wisdom threshold.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Which responses to evaluate | Only those carrying substantial karma | Technical/code responses don't need wisdom evaluation |
| Karma detection | Heuristic pre-filter (no LLM) | Fast, free, covers 80-90% of skip cases |
| Refinement mode | Silent replacement | User experiences a wiser response without friction |
| Model provider | Loving-deer's model factory | Unified config, works with any configured provider |
| Middleware hook | `after_agent` (runs once after full agent loop) | Evaluates only the final response, not intermediate tool-calling steps |
| Middleware list position | Between TitleMiddleware and MemoryMiddleware | Memory (also `after_agent`) stores the refined version; attune runs first in `after_agent` phase |

## Architecture

Note: loving-deer's middleware chain uses different lifecycle hooks. TitleMiddleware hooks `after_model` (fires after each LLM call). AttuneMiddleware and MemoryMiddleware both hook `after_agent` (fires once after the full agent loop completes). Middleware list position controls execution order within each hook phase.

```
Agent loop (multiple LLM calls + tool executions)
    │
    ├─ after_model phase: TitleMiddleware, ViewImageMiddleware, etc.
    │   (fires after each LLM call — attune is NOT here)
    │
    └─ after_agent phase (fires once, after full loop):
         1. AttuneMiddleware  ← evaluates final response
         2. MemoryMiddleware  ← stores the (possibly refined) response

AttuneMiddleware internal flow:
    Extract last AI message
         │
    ┌────┴─────┐
    │ Karma    │
    │ Filter   │
    └────┬─────┘
      skip? → pass through (return None)
      eval? ↓
    ┌──────────┐
    │ Wisdom   │
    │ Engine   │
    └────┬─────┘
      score >= threshold? → pass through
      score < threshold or critical? → replace message content
```

## Components

### 1. Karma Pre-Filter (`attune/karma_filter.py`)

A pure Python function that examines the last AI message and determines whether it carries substantial karma. No LLM call.

```python
def carries_karma(message_content: str) -> bool:
    """Return True if the response could create substantial karma."""
```

A response is **skipped** (no karma) if it matches any of these:

- **Code-dominant**: more than 70% of content is inside code blocks (``` fences)
- **Tool output relay**: consists primarily of structured data (file listings, JSON, error traces)
- **No human-facing substance**: contains only code blocks and/or bullet lists of file paths/commands with no prose sentences

Default is `True` — heuristics are an opt-out list. Errs on the side of wisdom.

### 2. Wisdom Engine (`attune/wisdom_engine.py`)

Refactored from attune's original `wisdom_engine.py` to use LangChain's `BaseChatModel` instead of the raw Anthropic SDK.

```python
async def evaluate_wisdom(
    user_message: str,
    agent_response: str,
    domain: str,
    model: BaseChatModel,
    wisdom_threshold: float = 0.80,
) -> EvaluateResponse:
```

The `wisdom_threshold` parameter controls the `should_refine` decision: the engine sets `should_refine = True` when sensitivity is critical OR wisdom_score < wisdom_threshold. The middleware passes its configured threshold through.

Preserves attune's evaluation prompt, JSON parsing, validation, and fallback logic. Scores responses on six dimensions:

| Dimension | Weight |
|-----------|--------|
| Emotional Attunement | 25% |
| Right Speech | 25% |
| Calibrated Uncertainty | 15% |
| Non-Reactivity | 15% |
| Agency Preservation | 10% |
| Skillful Timing | 10% |

Refinement triggers when:
- Sensitivity is critical, OR
- Wisdom score < configured threshold (default 0.80)

On any error (LLM failure, bad JSON), returns the original response unchanged with fallback status. Never blocks the user.

Crisis resource text (`CRISIS_RESOURCE_TEXT` env var) is appended to critical-sensitivity responses.

### 3. Models (`attune/models.py`)

Pydantic schemas adapted from attune:

- `EvaluateResponse` — full evaluation result with scores, refinement, modifications
- `Modification` — type + explanation of a change
- `SensitivityLevel` enum — low/medium/high/critical/unknown
- `EvaluationStatus` enum — ok/fallback_api_error/fallback_invalid_json
- `Domain` enum — general/coaching/mental_health

### 4. Attune Middleware (`agents/middlewares/attune_middleware.py`)

```python
class AttuneMiddleware(AgentMiddleware[AttuneMiddlewareState]):
```

**`after_agent()` flow:**

1. Get `messages` from state. Walk backwards to find the last AI message (type `"ai"`) that has no `tool_calls` (i.e., a final response, not an intermediate tool-calling step).
2. If no such message found, return `None` (no-op).
3. Normalize content to plain text using `_normalize_content()` (same pattern as TitleMiddleware — handles str, list-of-dicts with `"text"` keys, and nested structures).
4. Call `carries_karma(content)` — if `False`, return `None`.
5. Walk backwards to find the last human message. If none found (edge case), use empty string.
6. Create chat model via `create_chat_model(name=config_model_name, thinking_enabled=False)`.
7. Call `evaluate_wisdom(user_message, agent_response, domain, model, wisdom_threshold)`.
8. If `should_refine` is `True`, replace the AI message's `.content` with `refined_response` (preserving the message object, only changing content).
9. Return `None` — the message is mutated in place within the state list (same pattern as MemoryMiddleware which reads from state without returning modified messages).
10. On any exception, log warning and return `None` (pass through original).

**Sync only.** MemoryMiddleware (the reference `after_agent` middleware) implements only `after_agent()`, not `aafter_agent()`. The wisdom engine call uses synchronous `model.invoke()`. If async is needed later, `aafter_agent()` can be added with `await model.ainvoke()`.

### 5. Configuration (`config/attune_config.py`)

```python
class AttuneConfig(BaseModel):
    enabled: bool = False
    model_name: str | None = None            # None = use default model
    domain: Domain = Domain.general          # Uses Domain enum for validation
    wisdom_threshold: float = 0.80           # refine if score below this
```

The `domain` field uses the `Domain` enum from `attune/models.py` for validation, ensuring only valid values (`general`, `coaching`, `mental_health`) are accepted. In `config.yaml` it's written as a plain string and Pydantic coerces it.

Loaded in `AppConfig.from_file()` via the global singleton pattern (same as title, memory, guardrails configs):
```python
if "attune" in config_data:
    load_attune_config_from_dict(config_data["attune"])
```

Example `config.yaml`:
```yaml
attune:
  enabled: true
  model_name: null
  domain: "general"
  wisdom_threshold: 0.80
```

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `backend/packages/harness/deerflow/attune/karma_filter.py` | Heuristic pre-filter |
| `backend/packages/harness/deerflow/attune/wisdom_engine.py` | Evaluation logic (LangChain) |
| `backend/packages/harness/deerflow/attune/models.py` | Pydantic schemas |
| `backend/packages/harness/deerflow/attune/__init__.py` | Package init |
| `backend/packages/harness/deerflow/agents/middlewares/attune_middleware.py` | After-agent middleware |
| `backend/packages/harness/deerflow/config/attune_config.py` | Config model |

### Modified Files

| File | Change |
|------|--------|
| `backend/packages/harness/deerflow/config/app_config.py` | Add `attune: AttuneConfig` field |
| `backend/packages/harness/deerflow/agents/lead_agent/agent.py` | Add `AttuneMiddleware` to `_build_middlewares()` after TitleMiddleware, before MemoryMiddleware |
| `config.example.yaml` | Add `attune` section with defaults |

### No New Dependencies

Uses loving-deer's existing LangChain model factory. No new packages required.

## Error Handling

- Wisdom engine errors → log warning, pass through original response
- Karma filter errors → log warning, default to `True` (evaluate)
- Config missing → attune disabled by default (`enabled: false`)
- Model creation failure → log error, pass through original response

## Testing

Tests in `backend/tests/`, following the project's TDD mandate.

### `test_karma_filter.py`

- Code-dominant response (>70% code blocks) → returns `False`
- Tool output relay (file listings, JSON blobs) → returns `False`
- No prose (only code + file paths) → returns `False`
- Conversational response with advice → returns `True`
- Mixed response (some code, substantial prose) → returns `True`
- Empty string → returns `True` (default: evaluate)
- Edge case: code blocks with prose commentary → returns `True`

### `test_wisdom_engine.py`

- Successful evaluation returns `EvaluateResponse` with scores and `evaluation_status == "ok"`
- LLM returns invalid JSON → fallback response with original text, `evaluation_status == "fallback_invalid_json"`
- LLM call raises exception → fallback response, `evaluation_status == "fallback_api_error"`
- `should_refine` is `True` when wisdom_score < threshold
- `should_refine` is `True` when sensitivity is critical regardless of score
- `should_refine` is `False` when score >= threshold and sensitivity is low
- Crisis resource text appended when sensitivity is critical
- Wisdom threshold parameter is respected (custom values)

### `test_attune_middleware.py`

- AI message with low karma (code-dominant) → not modified
- AI message with karma, high wisdom score → not modified
- AI message with karma, low wisdom score → content replaced with refined response
- No AI message in state → returns `None`
- AI message with tool_calls (intermediate step) → skipped
- Wisdom engine error → original message preserved, warning logged
- Middleware not added to chain when `attune.enabled` is `False`
- Memory middleware receives refined content (integration test with mock state)

All tests use mocked `BaseChatModel` — no real LLM calls.

## Future Considerations (Not in Scope)

- LLM-based karma detection for ambiguous cases
- Per-thread or per-agent attune settings
- Wisdom score surfaced in frontend UI
- Dharma-agent as a subagent for deep reflection when attune flags critical sensitivity
- Conversation history analysis for karma detection
