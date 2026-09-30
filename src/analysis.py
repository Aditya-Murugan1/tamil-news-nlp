"""Reusable entry point for per-article Tamil linguistic analysis."""

from __future__ import annotations

from typing import Any

from src.keywords import extract_keywords
from src.morphology import analyze_morphology
from src.ngrams import get_top_ngrams
from src.preprocessing import clean_text, split_sentences, tokenize_tamil


def _format_ngrams(
    frequencies: list[tuple[tuple[str, ...], int]],
) -> list[dict[str, str | int]]:
    return [
        {"ngram": " ".join(ngram), "frequency": frequency}
        for ngram, frequency in frequencies
    ]


def analyze_text(text: str, top_k: int = 10) -> dict[str, Any]:
    """Return cleaned text and lightweight Tamil linguistic analysis."""
    cleaned_text = clean_text(text)
    tokens = tokenize_tamil(cleaned_text)
    return {
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "sentence_count": len(split_sentences(text)),
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "top_unigrams": _format_ngrams(get_top_ngrams(tokens, 1, top_k)),
        "top_bigrams": _format_ngrams(get_top_ngrams(tokens, 2, top_k)),
        "top_trigrams": _format_ngrams(get_top_ngrams(tokens, 3, top_k)),
        "keywords": extract_keywords(text, top_k=top_k),
        "morphology": analyze_morphology(tokens),
    }
