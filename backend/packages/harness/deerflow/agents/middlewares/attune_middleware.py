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
