"""Optional, lazily loaded Tamil POS tagging with Stanza."""

from __future__ import annotations

import importlib.util
from collections import Counter
from functools import lru_cache
from typing import Any


def is_pos_available() -> bool:
    """Return whether the optional Stanza package is installed."""
    try:
        return importlib.util.find_spec("stanza") is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _load_pipeline() -> Any:
    """Load an already-installed Tamil Stanza POS model; never download it."""
    try:
        import stanza
    except ImportError as error:
        raise RuntimeError(
            "Stanza is not installed. Install the optional requirements and "
            "download the Tamil POS resources as described in the README."
        ) from error

    try:
        return stanza.Pipeline(
            lang="ta",
            processors="tokenize,pos",
            use_gpu=False,
            download_method=None,
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(
            "The Stanza Tamil POS resources are unavailable. Install the "
            "optional requirements, then download them with "
            "`python -c \"import stanza; stanza.download('ta', "
            "processors='tokenize,pos')\"`."
        ) from error


def analyze_pos(text: str) -> dict[str, Any]:
    """Return Stanza's Tamil words and actual model-provided POS tags."""
    if not text.strip():
        return {"available": False, "message": "Enter Tamil text before POS analysis."}
    try:
        pipeline = _load_pipeline()
        document = pipeline(text)
    except Exception as error:
        return {
            "available": False,
            "message": (
                f"{error}\n"
                "POS tagging is optional; classification and other analysis "
                "remain available."
            ),
        }

    tagged_words = [
        {"word": word.text, "pos": word.upos or word.xpos or "UNSPECIFIED"}
        for sentence in document.sentences
        for word in sentence.words
    ]
    counts = Counter(item["pos"] for item in tagged_words)
    return {
        "available": True,
        "words": tagged_words,
        "counts": dict(counts),
        "message": "",
    }
