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
