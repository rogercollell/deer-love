"""Unit tests for the upstream attune wisdom frame builder."""

import json
from unittest.mock import MagicMock

from deerflow.attune.models import SensitivityLevel, WisdomFrame
from deerflow.attune.wisdom_frame import build_wisdom_frame


def _make_mock_model(response_text: str) -> MagicMock:
    model = MagicMock()
    result = MagicMock()
    result.content = response_text
    model.invoke.return_value = result
    return model


def _make_valid_frame_response(
    *,
    sensitivity: str = "low",
    is_consequential: bool = False,
    wellbeing_risk: bool = False,
    reflection_invitation: str | None = None,
) -> str:
    return json.dumps({
        "emotional_context": "The user sounds steady and practical.",
        "sensitivity_level": sensitivity,
        "is_consequential": is_consequential,
        "consequential_reason": "This action could affect a relationship." if is_consequential else None,
        "wellbeing_risk": wellbeing_risk,
        "affected_parties": ["user", "manager"] if is_consequential else ["user"],
        "recommended_posture": "Be steady and respectful.",
        "guidance": "Acknowledge the stakes and keep the response non-judgmental.",
        "reflection_invitation": reflection_invitation,
    })


class TestBuildWisdomFrame:
    def test_successful_frame_generation(self):
        model = _make_mock_model(_make_valid_frame_response())
        frame = build_wisdom_frame(
            user_message="How do I sort a list in Python?",
            recent_context="User: How do I sort a list in Python?",
            domain="general",
            model=model,
        )

        assert isinstance(frame, WisdomFrame)
        assert frame.sensitivity_level == SensitivityLevel.low
        assert frame.is_consequential is False

    def test_consequential_frame_backfills_reflection_invitation(self):
        model = _make_mock_model(
            _make_valid_frame_response(is_consequential=True, reflection_invitation=None)
        )
        frame = build_wisdom_frame(
            user_message="Help me draft an email to my manager quitting today.",
            recent_context="User: Help me draft an email to my manager quitting today.",
            domain="general",
            model=model,
        )

        assert frame.is_consequential is True
        assert frame.reflection_invitation is not None

    def test_invalid_json_uses_conservative_fallback(self):
        model = _make_mock_model("not json")
        frame = build_wisdom_frame(
            user_message="Help me draft a text to my partner telling them we're done.",
            recent_context="User: Help me draft a text to my partner telling them we're done.",
            domain="general",
            model=model,
        )

        assert frame.is_consequential is True
        assert frame.recommended_posture

    def test_critical_fallback_includes_crisis_guidance(self):
        model = MagicMock()
        model.invoke.side_effect = RuntimeError("timeout")
        frame = build_wisdom_frame(
            user_message="I can't go on anymore.",
            recent_context="User: I can't go on anymore.",
            domain="mental_health",
            model=model,
        )

        assert frame.sensitivity_level == SensitivityLevel.critical
        assert "988" in frame.guidance
