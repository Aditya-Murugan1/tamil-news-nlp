"""Unicode-safe preprocessing helpers for Tamil news text."""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from typing import Iterable

_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<[^>]*>")
_ZERO_WIDTH_CHARACTERS = {
    "\u180e",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2060",
    "\ufeff",
}
_SENTENCE_ENDINGS = frozenset(".!?।॥")


def normalize_tamil_text(text: str) -> str:
    """Normalize Unicode to NFC and remove zero-width formatting characters."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    normalized = unicodedata.normalize("NFC", text)
    return "".join(
        character
        for character in normalized
        if character not in _ZERO_WIDTH_CHARACTERS
    )


def remove_noise(text: str) -> str:
    """Remove markup, URLs, control characters, and non-word punctuation."""
    normalized = normalize_tamil_text(text)
    without_html = _HTML_TAG_PATTERN.sub(" ", html.unescape(normalized))
    without_urls = _URL_PATTERN.sub(" ", without_html)

    cleaned_characters: list[str] = []
    for character in without_urls:
        category = unicodedata.category(character)
        if character.isspace():
            cleaned_characters.append(" ")
        elif category[0] in {"L", "M", "N"}:
            cleaned_characters.append(character)
        else:
            cleaned_characters.append(" ")

    return " ".join("".join(cleaned_characters).split())


def clean_text(text: str) -> str:
    """Return normalized text with noise removed and whitespace collapsed."""
    return remove_noise(text)


def tokenize_tamil(text: str, stopwords: Iterable[str] | None = None) -> list[str]:
    """Tokenize Unicode letter/mark/number sequences, preserving Tamil forms."""
    normalized = normalize_tamil_text(text)
    tokens: list[str] = []
    current_token: list[str] = []

    for character in normalized:
        category = unicodedata.category(character)
        if category[0] in {"L", "M", "N"}:
            current_token.append(character)
        elif current_token:
            tokens.append("".join(current_token))
            current_token = []
    if current_token:
        tokens.append("".join(current_token))

    if stopwords is None:
        return tokens

    excluded = {normalize_tamil_text(word).casefold() for word in stopwords}
    return [token for token in tokens if token.casefold() not in excluded]


def split_sentences(text: str) -> list[str]:
    """Split text at common Tamil and Western sentence-ending punctuation."""
    normalized = normalize_tamil_text(text)
    sentences: list[str] = []
    current_sentence: list[str] = []

    for character in normalized:
        if character in _SENTENCE_ENDINGS:
            sentence = "".join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
            current_sentence = []
        else:
            current_sentence.append(character)

    final_sentence = "".join(current_sentence).strip()
    if final_sentence:
        sentences.append(final_sentence)
    return sentences


def token_frequencies(text: str, stopwords: Iterable[str] | None = None) -> Counter[str]:
    """Return token frequencies for a text."""
    return Counter(tokenize_tamil(text, stopwords=stopwords))
