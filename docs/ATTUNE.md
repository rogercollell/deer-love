# Attune in deer-love

Attune is the main opinionated addition in `deer-love`.

It is an optional middleware that evaluates the final assistant reply after the agent loop completes and before memory extraction runs. The goal is narrow: improve interpersonal, advisory, or high-stakes responses without interfering with code-heavy or tool-heavy tasks.

## What it does

- Filters out replies that are mostly code, stack traces, file listings, or other structured output.
- Scores eligible replies across six dimensions:
  - emotional attunement
  - right speech
  - calibrated uncertainty
  - non-reactivity
  - agency preservation
  - skillful timing
- Rewrites only the replies that fall below the configured threshold.
- Appends crisis language for critical-sensitivity responses when the evaluator omits it.

## Where it runs

Middleware order matters. In `deer-love`, Attune runs after the main agent turn and before memory extraction so that any refined reply is what the user sees and what memory stores.

## Configuration

Enable it in `config.yaml`:

```yaml
attune:
  enabled: true
  model_name: gpt-5-mini
  domain: general
  wisdom_threshold: 0.80
```

Fields:

- `enabled`: turns the middleware on or off.
- `model_name`: optional evaluator model. If omitted, deer-love uses the default configured model.
- `domain`: one of `general`, `coaching`, or `mental_health`.
- `wisdom_threshold`: replies below this weighted score are refined.

## Practical guidance

- Use a cheaper, fast evaluator model for `model_name` if your main agent model is expensive.
- Start with `domain: general` unless you have a narrow workflow that benefits from a stronger coaching or mental-health framing.
- Keep the threshold conservative at first. `0.80` is a good default if you want Attune to catch clearly blunt or poorly calibrated replies without rewriting everything.
- If you mostly use deer-love for code generation, leave Attune enabled but do not expect it to trigger often. The karma filter is designed to stay out of the way on technical output.

## Current limitations

- The evaluator depends on structured JSON output from a model, so failures degrade to pass-through behavior.
- Attune is scoped to final replies. It does not change tool plans, subagent behavior, or intermediate chain-of-thought.
- The middleware is intentionally conservative about what it rewrites. Technical responses with minimal interpersonal content usually bypass evaluation.
