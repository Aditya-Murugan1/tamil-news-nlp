"""Offline unit tests for reusable Tamil linguistic analysis modules."""

from __future__ import annotations

import unittest

from src.analysis import analyze_text
from src.keywords import extract_keywords
from src.ngrams import get_ngram_frequencies, get_ngrams, get_top_ngrams
from src.preprocessing import (
    clean_text,
    normalize_tamil_text,
    split_sentences,
    tokenize_tamil,
)


class TamilPreprocessingTests(unittest.TestCase):
    def test_unicode_normalization_preserves_tamil(self) -> None:
        text = "தமிழ்\u200bமொழி"
        normalized = normalize_tamil_text(text)
        self.assertEqual(normalized, "தமிழ்மொழி")
        self.assertIn("தமிழ்", clean_text("தமிழ் மொழி!"))

    def test_clean_text_removes_html_url_and_noise(self) -> None:
        cleaned = clean_text(
            " <b>சென்னை</b> மழை! https://example.com \u200bசெய்தி "
        )
        self.assertEqual(cleaned, "சென்னை மழை செய்தி")

    def test_tokenization_preserves_tamil_words(self) -> None:
        self.assertEqual(
            tokenize_tamil("சென்னையில் இன்று கனமழை பெய்தது."),
            ["சென்னையில்", "இன்று", "கனமழை", "பெய்தது"],
        )

    def test_sentence_splitting(self) -> None:
        sentences = split_sentences("முதல் செய்தி. அடுத்த செய்தி! இறுதி செய்தி॥")
        self.assertEqual(len(sentences), 3)


class NGramTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = ["தமிழ்", "செய்தி", "தமிழ்", "செய்தி"]

    def test_unigrams(self) -> None:
        self.assertEqual(
            get_ngram_frequencies(self.tokens, 1)[("தமிழ்",)],
            2,
        )

    def test_bigrams(self) -> None:
        self.assertEqual(
            get_ngrams(self.tokens, 2),
            [("தமிழ்", "செய்தி"), ("செய்தி", "தமிழ்"), ("தமிழ்", "செய்தி")],
        )
        self.assertEqual(
            get_top_ngrams(self.tokens, 2, top_k=1),
            [(("தமிழ்", "செய்தி"), 2)],
        )

    def test_trigrams(self) -> None:
        self.assertEqual(
            get_ngram_frequencies(self.tokens, 3)[
                ("தமிழ்", "செய்தி", "தமிழ்")
            ],
            1,
        )

    def test_invalid_ngram_size(self) -> None:
        with self.assertRaises(ValueError):
            get_ngrams(self.tokens, 0)


class KeywordAndAnalysisTests(unittest.TestCase):
    def test_keyword_extraction(self) -> None:
        keywords = extract_keywords("சென்னை மழை சென்னை வெள்ளம்", top_k=2)
        self.assertEqual([item["keyword"] for item in keywords][0], "சென்னை")
        self.assertEqual(len(keywords), 2)

    def test_empty_keyword_input(self) -> None:
        self.assertEqual(extract_keywords("!!!"), [])

    def test_integrated_analysis_shape_and_empty_input(self) -> None:
        result = analyze_text("சென்னையில் மழை பெய்தது. சென்னையில் வெள்ளம் ஏற்பட்டது.")
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["token_count"], 6)
        self.assertEqual(result["unique_token_count"], 5)
        self.assertEqual(result["top_unigrams"][0]["ngram"], "சென்னையில்")
        self.assertIn("suffix_pattern_counts", result["morphology"])

        empty_result = analyze_text("")
        self.assertEqual(empty_result["tokens"], [])
        self.assertEqual(empty_result["sentence_count"], 0)
        self.assertEqual(empty_result["keywords"], [])


if __name__ == "__main__":
    unittest.main()
