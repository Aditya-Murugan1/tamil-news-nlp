"""Lightweight TF-IDF keyword extraction for Tamil text."""

from __future__ import annotations

from collections.abc import Iterable

from src.preprocessing import clean_text, tokenize_tamil


def extract_keywords(
    text: str,
    top_k: int = 10,
    stopwords: Iterable[str] | None = None,
    corpus: Iterable[str] | None = None,
) -> list[dict[str, float | str]]:
    """Return terms ranked by TF-IDF score.

    If a reference ``corpus`` is supplied, IDF is estimated from it and the
    article. Otherwise, scores are derived from term frequency in the single
    input article, so the ranking indicates prominence rather than topic-wide
    rarity.
    """
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if top_k == 0:
        return []

    article_tokens = tokenize_tamil(clean_text(text), stopwords=stopwords)
    if not article_tokens:
        return []

    documents = [article_tokens]
    if corpus is not None:
        for document in corpus:
            tokens = tokenize_tamil(clean_text(document), stopwords=stopwords)
            if tokens:
                documents.append(tokens)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([" ".join(tokens) for tokens in documents])
    terms = vectorizer.get_feature_names_out()
    scores = matrix[0].toarray().ravel()
    ranked_indices = sorted(
        range(len(terms)),
        key=lambda index: (-float(scores[index]), str(terms[index])),
    )[:top_k]
    return [
        {"keyword": str(terms[index]), "score": float(scores[index])}
        for index in ranked_indices
    ]
