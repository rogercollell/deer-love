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

    @patch("deerflow.agents.lead_agent.agent.get_app_config")
    @patch("deerflow.agents.lead_agent.agent.get_summarization_config")
    @patch("deerflow.agents.lead_agent.agent.build_lead_runtime_middlewares", return_value=[])
    def test_not_in_chain_when_disabled(self, _mock_runtime, mock_summ, mock_app):
        """AttuneMiddleware is excluded from _build_middlewares() when disabled."""
        from deerflow.agents.lead_agent.agent import _build_middlewares
        from deerflow.agents.middlewares.attune_middleware import AttuneMiddleware as AW

        mock_summ.return_value = MagicMock(enabled=False)
        mock_app.return_value = MagicMock(
            get_model_config=MagicMock(return_value=None),
            tool_search=MagicMock(enabled=False),
        )

        set_attune_config(AttuneConfig(enabled=False))
        config = {"configurable": {}}
        middlewares = _build_middlewares(config, model_name=None)
        assert not any(isinstance(m, AW) for m in middlewares)

    @patch("deerflow.agents.lead_agent.agent.get_app_config")
    @patch("deerflow.agents.lead_agent.agent.get_summarization_config")
    @patch("deerflow.agents.lead_agent.agent.build_lead_runtime_middlewares", return_value=[])
    def test_in_chain_when_enabled(self, _mock_runtime, mock_summ, mock_app):
        """AttuneMiddleware is included in _build_middlewares() when enabled."""
        from deerflow.agents.lead_agent.agent import _build_middlewares
        from deerflow.agents.middlewares.attune_middleware import AttuneMiddleware as AW

        mock_summ.return_value = MagicMock(enabled=False)
        mock_app.return_value = MagicMock(
            get_model_config=MagicMock(return_value=None),
            tool_search=MagicMock(enabled=False),
        )

        set_attune_config(AttuneConfig(enabled=True))
        config = {"configurable": {}}
        middlewares = _build_middlewares(config, model_name=None)
        assert any(isinstance(m, AW) for m in middlewares)
