"""Unit tests for the upstream AttuneMiddleware."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from deerflow.agents.middlewares.attune_middleware import (
    WISDOM_FRAME_TAG,
    AttuneMiddleware,
    clear_attune_model_cache,
)
from deerflow.attune.models import SensitivityLevel, WisdomFrame
from deerflow.config.attune_config import (
    AttuneConfig,
    get_attune_config,
    set_attune_config,
)


def _make_frame_response(
    *,
    sensitivity: str = "high",
    is_consequential: bool = True,
    wellbeing_risk: bool = False,
) -> str:
    return json.dumps({
        "emotional_context": "The user sounds frustrated and the stakes are interpersonal.",
        "sensitivity_level": sensitivity,
        "is_consequential": is_consequential,
        "consequential_reason": "The action could affect a relationship." if is_consequential else None,
        "wellbeing_risk": wellbeing_risk,
        "affected_parties": ["user", "manager"] if is_consequential else ["user"],
        "recommended_posture": "Be calm, transparent, and non-judgmental.",
        "guidance": "Acknowledge the stakes before helping further.",
        "reflection_invitation": (
            "I notice this could land hard. Do you want to pause and shape it carefully before we proceed?"
            if is_consequential
            else None
        ),
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
        clear_attune_model_cache()

    def teardown_method(self):
        set_attune_config(self._original)
        clear_attune_model_cache()

    def _build_request(self, state: dict) -> ModelRequest:
        return ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Please help me.")],
            system_message=SystemMessage(content="Base system prompt."),
            tool_choice=None,
            tools=[],
            response_format=None,
            state=state,
            runtime=MagicMock(),
            model_settings={},
        )

    def test_before_agent_skips_non_consequential_technical_turns(self):
        set_attune_config(AttuneConfig(enabled=True))
        middleware = AttuneMiddleware()
        state = {"messages": [HumanMessage(content="How do I sort a list in Python?")]}

        result = middleware.before_agent(state, MagicMock())

        assert result == {"attune_wisdom_frame": None}

    @patch("deerflow.agents.middlewares.attune_middleware.create_chat_model")
    def test_before_agent_builds_frame_for_consequential_turn(self, mock_create_model):
        set_attune_config(AttuneConfig(enabled=True))
        mock_create_model.return_value = _make_mock_model(_make_frame_response())
        middleware = AttuneMiddleware()
        state = {
            "messages": [
                HumanMessage(content="Help me draft an email to my manager telling them I'm done covering for the team."),
            ]
        }

        result = middleware.before_agent(state, MagicMock())

        assert result is not None
        raw_frame = result["attune_wisdom_frame"]
        assert isinstance(raw_frame, dict)
        assert raw_frame["is_consequential"] is True
        assert raw_frame["sensitivity_level"] == SensitivityLevel.high.value

    def test_wrap_model_call_injects_frame_into_system_message(self):
        set_attune_config(AttuneConfig(enabled=True))
        middleware = AttuneMiddleware()
        frame = WisdomFrame(
            emotional_context="The user is heated and wants to send a difficult message.",
            sensitivity_level=SensitivityLevel.high,
            is_consequential=True,
            consequential_reason="This message could damage a professional relationship.",
            wellbeing_risk=False,
            affected_parties=["user", "manager"],
            recommended_posture="Be steady and non-judgmental.",
            guidance="Acknowledge the stakes and invite a pause before drafting.",
            reflection_invitation="I notice this could land hard. Do you want to shape it carefully before we send anything?",
        )
        request = self._build_request({"attune_wisdom_frame": frame.model_dump()})

        captured: dict[str, ModelRequest] = {}

        def handler(updated_request: ModelRequest):
            captured["request"] = updated_request
            return "ok"

        result = middleware.wrap_model_call(request, handler)

        assert result == "ok"
        assert captured["request"].system_message is not None
        assert WISDOM_FRAME_TAG in captured["request"].system_message.content
        assert "Base system prompt." in captured["request"].system_message.content
        assert frame.reflection_invitation in captured["request"].system_message.content

    def test_awrap_model_call_injects_frame(self):
        set_attune_config(AttuneConfig(enabled=True))
        middleware = AttuneMiddleware()
        frame = WisdomFrame(
            emotional_context="The user may be unsafe.",
            sensitivity_level=SensitivityLevel.critical,
            is_consequential=False,
            wellbeing_risk=True,
            affected_parties=["user"],
            recommended_posture="Lead with attunement and a safety check.",
            guidance="Acknowledge the distress and keep the response simple.",
            reflection_invitation=None,
        )
        request = self._build_request({"attune_wisdom_frame": frame.model_dump()})
        handler = AsyncMock(return_value="ok")

        result = asyncio.run(middleware.awrap_model_call(request, handler))

        assert result == "ok"
        passed_request = handler.await_args.args[0]
        assert WISDOM_FRAME_TAG in passed_request.system_message.content

    @patch("deerflow.agents.middlewares.attune_middleware.create_chat_model")
    def test_reuses_cached_framing_model(self, mock_create_model):
        set_attune_config(AttuneConfig(enabled=True, model_name="attune-framer"))
        mock_create_model.return_value = _make_mock_model(_make_frame_response())
        middleware = AttuneMiddleware()

        first_state = {
            "messages": [HumanMessage(content="Help me draft a text to my partner ending the relationship.")]
        }
        second_state = {
            "messages": [HumanMessage(content="Help me draft an apology to my manager.")]
        }

        middleware.before_agent(first_state, MagicMock())
        middleware.before_agent(second_state, MagicMock())

        assert mock_create_model.call_count == 1

    @patch("deerflow.agents.lead_agent.agent.get_app_config")
    @patch("deerflow.agents.lead_agent.agent.get_summarization_config")
    @patch("deerflow.agents.lead_agent.agent.build_lead_runtime_middlewares", return_value=[])
    def test_not_in_chain_when_disabled(self, _mock_runtime, mock_summ, mock_app):
        from deerflow.agents.lead_agent.agent import _build_middlewares
        from deerflow.agents.middlewares.attune_middleware import AttuneMiddleware as AW

        mock_summ.return_value = MagicMock(enabled=False)
        mock_app.return_value = MagicMock(
            get_model_config=MagicMock(return_value=None),
            tool_search=MagicMock(enabled=False),
        )

        set_attune_config(AttuneConfig(enabled=False))
        middlewares = _build_middlewares({"configurable": {}}, model_name=None)
        assert not any(isinstance(m, AW) for m in middlewares)

    @patch("deerflow.agents.lead_agent.agent.get_app_config")
    @patch("deerflow.agents.lead_agent.agent.get_summarization_config")
    @patch("deerflow.agents.lead_agent.agent.build_lead_runtime_middlewares", return_value=[])
    def test_in_chain_when_enabled(self, _mock_runtime, mock_summ, mock_app):
        from deerflow.agents.lead_agent.agent import _build_middlewares
        from deerflow.agents.middlewares.attune_middleware import AttuneMiddleware as AW

        mock_summ.return_value = MagicMock(enabled=False)
        mock_app.return_value = MagicMock(
            get_model_config=MagicMock(return_value=None),
            tool_search=MagicMock(enabled=False),
        )

        set_attune_config(AttuneConfig(enabled=True))
        middlewares = _build_middlewares({"configurable": {}}, model_name=None)
        assert any(isinstance(m, AW) for m in middlewares)
