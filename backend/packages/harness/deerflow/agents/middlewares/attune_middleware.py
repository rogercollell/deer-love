"""Middleware for upstream attune wisdom framing."""

import logging
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from deerflow.attune.karma_filter import needs_wisdom_frame
from deerflow.attune.models import WisdomFrame
from deerflow.attune.wisdom_frame import build_wisdom_frame
from deerflow.config.attune_config import get_attune_config
from deerflow.models import create_chat_model

logger = logging.getLogger(__name__)

WISDOM_FRAME_TAG = "attune_wisdom"
_DEFAULT_ATTUNE_MODEL_KEY = "__attune_default_model__"
_RECENT_CONTEXT_LIMIT = 6
_RECENT_CONTEXT_CHAR_LIMIT = 2500


class AttuneMiddlewareState(AgentState):
    """Compatible with the ThreadState schema."""

    attune_wisdom_frame: NotRequired[dict[str, object] | None]


def _normalize_content(content: object) -> str:
    """Normalize message content to plain text."""
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


def _resolve_attune_model_name(configured_name: str | None) -> str:
    return configured_name or _DEFAULT_ATTUNE_MODEL_KEY


@lru_cache(maxsize=8)
def _get_attune_model(model_name: str):
    requested_name = None if model_name == _DEFAULT_ATTUNE_MODEL_KEY else model_name
    return create_chat_model(name=requested_name, thinking_enabled=False)


def clear_attune_model_cache() -> None:
    """Reset cached Attune framing models. Intended for tests."""
    _get_attune_model.cache_clear()


def _get_last_human_message(messages: list) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            content = _normalize_content(getattr(msg, "content", ""))
            if content.strip():
                return content.strip()
    return ""


def _build_recent_context(messages: list) -> str:
    items: list[str] = []

    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type not in {"human", "ai"}:
            continue
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            continue

        content = _normalize_content(getattr(msg, "content", "")).strip()
        if not content:
            continue

        speaker = "User" if msg_type == "human" else "Assistant"
        items.append(f"{speaker}: {content}")

    context = "\n".join(items[-_RECENT_CONTEXT_LIMIT:])
    if len(context) <= _RECENT_CONTEXT_CHAR_LIMIT:
        return context
    return context[-_RECENT_CONTEXT_CHAR_LIMIT:]


def _coerce_wisdom_frame(raw: object) -> WisdomFrame | None:
    if raw is None:
        return None
    if isinstance(raw, WisdomFrame):
        return raw
    if isinstance(raw, dict):
        try:
            return WisdomFrame(**raw)
        except Exception:
            logger.warning("Attune middleware: invalid wisdom frame in state", exc_info=True)
            return None
    return None


def _format_wisdom_frame(frame: WisdomFrame) -> str:
    affected = ", ".join(frame.affected_parties) if frame.affected_parties else "none"
    lines = [
        f"<{WISDOM_FRAME_TAG}>",
        f"Sensitivity: {frame.sensitivity_level.value}",
        f"Emotional context: {frame.emotional_context}",
        f"Wellbeing risk: {'yes' if frame.wellbeing_risk else 'no'}",
        f"Consequential turn: {'yes' if frame.is_consequential else 'no'}",
        f"Affected parties: {affected}",
        f"Recommended posture: {frame.recommended_posture}",
        f"Guidance: {frame.guidance}",
    ]

    if frame.consequential_reason:
        lines.append(f"Consequential reason: {frame.consequential_reason}")
    if frame.reflection_invitation:
        lines.append(f"Reflection invitation: {frame.reflection_invitation}")

    lines.extend([
        "Use this guidance as context, not as content to reveal verbatim.",
        "Stay transparent, preserve agency, and avoid paternalistic language.",
        f"</{WISDOM_FRAME_TAG}>",
    ])
    return "\n".join(lines)


def _merge_system_message(system_message: SystemMessage | None, frame: WisdomFrame) -> SystemMessage:
    frame_text = _format_wisdom_frame(frame)
    if system_message is None:
        return SystemMessage(content=frame_text)

    existing = _normalize_content(system_message.content).strip()
    if not existing:
        return SystemMessage(content=frame_text)

    return SystemMessage(content=f"{existing}\n\n{frame_text}")


class AttuneMiddleware(AgentMiddleware[AttuneMiddlewareState]):
    """Compute a wisdom frame once per turn and inject it into model calls."""

    state_schema = AttuneMiddlewareState

    @override
    def before_agent(self, state: AttuneMiddlewareState, runtime: Runtime) -> dict | None:
        config = get_attune_config()
        if not config.enabled:
            return {"attune_wisdom_frame": None}

        messages = state.get("messages", [])
        if not messages:
            return {"attune_wisdom_frame": None}

        user_message = _get_last_human_message(messages)
        if not user_message or not needs_wisdom_frame(user_message):
            return {"attune_wisdom_frame": None}

        try:
            resolved_model_name = _resolve_attune_model_name(config.model_name)
            model = _get_attune_model(resolved_model_name)
            frame = build_wisdom_frame(
                user_message=user_message,
                recent_context=_build_recent_context(messages),
                domain=config.domain.value,
                model=model,
            )
        except Exception:
            logger.warning("Attune middleware: failed to build wisdom frame", exc_info=True)
            return {"attune_wisdom_frame": None}

        logger.info(
            "Attune wisdom frame built (sensitivity=%s, consequential=%s, wellbeing_risk=%s)",
            frame.sensitivity_level.value,
            frame.is_consequential,
            frame.wellbeing_risk,
        )
        return {"attune_wisdom_frame": frame.model_dump()}

    def _inject_frame(self, request: ModelRequest) -> ModelRequest:
        frame = _coerce_wisdom_frame(request.state.get("attune_wisdom_frame"))
        if frame is None:
            return request
        return request.override(system_message=_merge_system_message(request.system_message, frame))

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._inject_frame(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._inject_frame(request))
