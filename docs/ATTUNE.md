# Attune in deer-love

Attune is the main opinionated addition in `deer-love`.

It is no longer a post-hoc reply rewriter in the runtime path. Instead, it adds an upstream wisdom layer for turns that may be consequential, relationally sensitive, or risky for the user's wellbeing.

## Runtime behavior

Attune now works in two parts:

- The main lead-agent prompt carries a steadier compassionate baseline on every turn.
- The Attune middleware adds an extra framing pass only when heuristics suggest the turn may involve hard-to-reverse actions or wellbeing risk.

When the middleware triggers, it:

1. Looks at the current user turn and recent conversational context.
2. Builds a short `WisdomFrame` with emotional context, sensitivity, consequentiality, and guidance.
3. Injects that frame ephemerally into model calls through middleware, so it can shape reasoning, tool choices, and phrasing.
4. Uses dialogue-first reflection for consequential moments instead of silently rewriting the final answer.

## What counts as a trigger

The runtime gate is intentionally narrower than a general “human-sensitive reply” detector. It is trying to catch turns such as:

- drafting or sending charged messages to another person
- quitting, breaking up, deleting, escalating, or otherwise making hard-to-reverse moves
- acute distress or signals that the user's wellbeing may be at risk

Ordinary technical questions and low-stakes requests skip the extra model pass.

## Configuration

Enable it in `config.yaml`:

```yaml
attune:
  enabled: true
  model_name: gpt-5-mini
  domain: general
```

Fields:

- `enabled`: turns Attune framing on or off.
- `model_name`: optional framing model. If omitted, deer-love uses the default configured model.
- `domain`: one of `general`, `coaching`, or `mental_health`.

## Practical guidance

- Use a cheaper, fast model for `model_name` if your default assistant model is expensive.
- Start with `domain: general` unless you have a narrow workflow that benefits from a stronger coaching or mental-health posture.
- Expect most purely technical requests to bypass Attune. The baseline prompt should still sound grounded and respectful without the extra pass.
- The older six-dimension wisdom scorer remains in the codebase as an evaluation helper, but it is no longer the runtime middleware path.

## Current limitations

- The framing pass still depends on structured JSON output from a model, so failures degrade to conservative fallback guidance or pass-through behavior.
- The heuristic gate is intentionally simple. It will miss some subtle consequential moments and over-trigger on some edge cases.
- Attune can guide the agent toward reflection, but it does not hard-block tools or actions on its own.
