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

_WELLBEING_RISK_RE = re.compile(
    r"\b("
    r"suicid(?:e|al)|kill myself|end it|end my life|can't go on|cannot go on|"
    r"hurt myself|harm myself|self[- ]harm|panic attack|not safe|unsafe|"
    r"hurt someone|harm someone|spiraling|burned out|burnout"
    r")\b",
    re.IGNORECASE,
)

_RELATIONSHIP_TARGET_RE = re.compile(
    r"\b("
    r"manager|boss|coworker|colleague|client|customer|partner|wife|husband|"
    r"girlfriend|boyfriend|friend|parent|mom|dad|teacher|roommate|landlord|team"
    r")\b",
    re.IGNORECASE,
)

_CONSEQUENTIAL_ACTION_RE = re.compile(
    r"\b("
    r"send|text|email|message|reply|tell|confront|call out|post|publish|announce|"
    r"quit|resign|break up|fire|delete|remove|cancel|confess|apologize|report|escalate"
    r")\b",
    re.IGNORECASE,
)

_DRAFTING_RE = re.compile(
    r"\b(write|draft|compose|help me write|help me draft)\b",
    re.IGNORECASE,
)


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


def needs_wisdom_frame(user_message: str) -> bool:
    """Return True when a turn may warrant upstream wisdom guidance.

    This is intentionally narrower than ``carries_karma()``. Runtime attune
    should only spend an extra model call on turns that may touch another
    person, a hard-to-reverse action, or the user's wellbeing.
    """

    content = user_message.strip()
    if not content:
        return False

    if _WELLBEING_RISK_RE.search(content):
        return True

    if len(content) < 30 and not _PROSE_SENTENCE_RE.search(content):
        return False

    code_blocks = _CODE_BLOCK_RE.findall(content)
    code_chars = sum(len(block) for block in code_blocks)
    total_chars = len(content)
    if total_chars > 0 and code_chars / total_chars > 0.70:
        return False

    if _DRAFTING_RE.search(content) and (
        _CONSEQUENTIAL_ACTION_RE.search(content) or _RELATIONSHIP_TARGET_RE.search(content)
    ):
        return True

    if _CONSEQUENTIAL_ACTION_RE.search(content) and _RELATIONSHIP_TARGET_RE.search(content):
        return True

    return False
