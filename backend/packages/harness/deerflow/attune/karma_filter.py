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
