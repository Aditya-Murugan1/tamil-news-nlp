"""Lightweight educational Tamil word-form inspection.

Suffix matches are heuristic demonstrations, not a complete Tamil
morphological analysis or a validated lemmatizer.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

# Commonly observed case, number, and derivational endings. Surface forms and
# sandhi can vary; this deliberately small list is only a teaching aid.
_SUFFIXES = (
    ("உடன்", "comitative"),
    ("களுக்கு", "plural_dative"),
    ("களின்", "plural_genitive"),
    ("களில்", "plural_locative"),
    ("களால்", "plural_instrumental"),
    ("களை", "plural_accusative"),
    ("கள்", "plural"),
    ("க்கு", "dative"),
    ("இல்", "locative"),
    ("ஆல்", "instrumental"),
    ("உடைய", "genitive"),
    ("இன்", "genitive"),
    ("ஐ", "accusative"),
    ("ஆக", "adverbial_or_transformative"),
)


def _base_character_count(word: str) -> int:
    """Count Unicode characters that are not combining marks."""
    return sum(
        unicodedata.category(character)[0] != "M"
        for character in word
    )


def _inspect_word(word: str) -> dict[str, Any]:
    for suffix, pattern in _SUFFIXES:
        if not word.endswith(suffix):
            continue
        base = word[: -len(suffix)]
        if _base_character_count(base) >= 2:
            return {
                "word": word,
                "suffix": suffix,
                "pattern": pattern,
                "base_approximation": base,
            }
    return {
        "word": word,
        "suffix": None,
        "pattern": None,
        "base_approximation": None,
    }


def analyze_morphology(tokens: Iterable[str]) -> dict[str, Any]:
    """Summarize word forms, repeated forms, lengths, and suffix patterns.

    Length is measured in Unicode code points (including combining marks),
    which is reproducible but is not equivalent to Tamil syllable count.
    Base approximations are emitted only for recognized endings and should
    not be interpreted as linguistically verified roots.
    """
    words = list(tokens)
    word_forms = [_inspect_word(word) for word in words]
    lengths = [len(word) for word in words]
    repeated_forms = [
        {"word": word, "frequency": frequency}
        for word, frequency in Counter(words).most_common()
        if frequency > 1
    ]
    suffix_patterns = Counter(
        item["pattern"] for item in word_forms if item["pattern"] is not None
    )

    return {
        "word_forms": word_forms,
        "word_length_statistics": {
            "count": len(lengths),
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "average": sum(lengths) / len(lengths) if lengths else 0.0,
        },
        "repeated_word_forms": repeated_forms,
        "suffix_pattern_counts": dict(suffix_patterns),
        "limitation": (
            "Educational suffix heuristics only; Tamil word segmentation, "
            "sandhi, and base forms are not comprehensively analyzed."
        ),
    }
