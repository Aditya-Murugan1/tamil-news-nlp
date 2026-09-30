"""Frequency-based N-gram helpers for tokenized Tamil text."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

NGram = tuple[str, ...]


def get_ngrams(tokens: Iterable[str], n: int) -> list[NGram]:
    """Return consecutive token sequences of length ``n``."""
    if n < 1:
        raise ValueError("n must be at least 1")
    token_list = list(tokens)
    return [tuple(token_list[index : index + n]) for index in range(len(token_list) - n + 1)]


def get_ngram_frequencies(tokens: Iterable[str], n: int) -> Counter[NGram]:
    """Count each consecutive N-gram in the supplied token sequence."""
    return Counter(get_ngrams(tokens, n))


def get_top_ngrams(
    tokens: Iterable[str],
    n: int,
    top_k: int = 10,
) -> list[tuple[NGram, int]]:
    """Return the most frequent N-grams as ``(ngram, frequency)`` pairs."""
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    return get_ngram_frequencies(tokens, n).most_common(top_k)
