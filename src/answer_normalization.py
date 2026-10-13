"""Conservative answer normalization for exact-match evaluation.

The previous student scripts mapped any answer starting with ``yes`` or
``no`` to those tokens. That is unsafe: ``not seen here`` starts with
``no`` and would be scored as ``no``. This module only:

1. strips surrounding whitespace
2. lowercases
3. collapses internal whitespace
4. removes trailing sentence punctuation

It never turns ``no`` into ``yes`` or the reverse.
"""

from __future__ import annotations

import re
import string

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = frozenset(string.punctuation)


def normalize_answer(answer: str | None) -> str:
    """Return a conservative normalized answer string."""
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""
    text = text.lower()
    text = text.replace("\t", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    while text and text[-1] in _TRAILING_PUNCT:
        text = text[:-1].rstrip()
    return text


def answers_match(prediction: str | None, ground_truth: str | None) -> bool:
    """True when normalized prediction equals normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def is_yes_no(answer: str | None) -> bool:
    """True only if the fully normalized answer is exactly yes or no."""
    return normalize_answer(answer) in {"yes", "no"}
