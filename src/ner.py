"""Lazy Tamil named-entity extraction with the public ExentAI model."""

from __future__ import annotations

import importlib.util
import logging
import unicodedata
from functools import lru_cache
from typing import Any

MODEL_ID = "exentai/indicner-tamil-ExentAI"
_LOGGER = logging.getLogger(__name__)

_READABLE_LABELS = {
    "PER": "Person",
    "PERSON": "Person",
    "LOC": "Location",
    "LOCATION": "Location",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "DATE": "Date",
    "MISC": "Miscellaneous",
}


def is_ner_available() -> bool:
    """Return whether Transformers and PyTorch are installed.

    Model weights are not downloaded here. They are fetched from Hugging Face
    only when ``load_ner_model`` is first called.
    """
    try:
        return all(
            importlib.util.find_spec(module_name) is not None
            for module_name in ("transformers", "torch")
        )
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def load_ner_model() -> Any:
    """Load and cache the public Tamil NER token-classification pipeline."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "Tamil NER requires the optional Transformers and PyTorch "
            "dependencies. Install them with `python -m pip install -r "
            "requirements.txt`."
        ) from error

    try:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "token-classification",
            model=MODEL_ID,
            aggregation_strategy="simple",
            device=device,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not load the Tamil NER model. Check your internet connection "
            "for the first download, available disk space, and the Hugging Face "
            f"model cache. Details: {error}"
        ) from error


def _readable_label(label: str, id2label: dict[Any, Any]) -> str:
    technical_label = label
    if label.startswith("LABEL_"):
        try:
            label_id = int(label.removeprefix("LABEL_"))
        except ValueError:
            pass
        else:
            technical_label = str(
                id2label.get(label_id, id2label.get(str(label_id), label))
            )

    normalized = technical_label.upper()
    if len(normalized) > 2 and normalized[1] == "-" and normalized[0] in "BIUL":
        normalized = normalized[2:]
    readable = _READABLE_LABELS.get(normalized)
    if readable:
        return readable
    return technical_label.removeprefix("B-").removeprefix("I-").replace("_", " ").title()


def _align_to_unicode_marks(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand offsets that split a Tamil base character from its combining mark."""
    while start > 0 and unicodedata.category(text[start])[0] == "M":
        start -= 1
    while end < len(text) and unicodedata.category(text[end])[0] == "M":
        end += 1
    return start, end


def extract_entities(text: str, min_confidence: float = 0.35) -> list[dict[str, Any]]:
    """Extract model-predicted entities meeting the confidence threshold.

    Each result contains ``text``, a readable ``label``, the model ``score``,
    and character offsets when provided by the Transformers pipeline.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    if not text.strip():
        return []

    ner_pipeline = load_ner_model()
    predictions = ner_pipeline(text)
    id2label = getattr(
        getattr(getattr(ner_pipeline, "model", None), "config", None),
        "id2label",
        {},
    )
    entities: list[dict[str, Any]] = []
    for prediction in predictions:
        score = float(prediction.get("score", 0.0))
        if score < min_confidence:
            continue
        label = str(
            prediction.get("entity_group")
            or prediction.get("entity")
            or prediction.get("label")
            or "UNSPECIFIED"
        )
        entity: dict[str, Any] = {
            "text": str(prediction.get("word", prediction.get("text", ""))),
            "label": _readable_label(label, id2label),
            "score": score,
        }
        start = prediction.get("start")
        end = prediction.get("end")
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
        ):
            start, end = _align_to_unicode_marks(text, start, end)
            entity["text"] = text[start:end]
            entity["start"] = start
            entity["end"] = end
        if entity["text"]:
            entities.append(entity)
    return entities


def analyze_ner(text: str, min_confidence: float = 0.35) -> dict[str, Any]:
    """Return entity results and an explicit unavailable state for the UI."""
    if not text.strip():
        return {
            "available": False,
            "entities": [],
            "message": "Enter Tamil text before NER analysis.",
        }
    try:
        entities = extract_entities(text, min_confidence=min_confidence)
    except Exception as error:
        _LOGGER.exception("Tamil NER analysis failed")
        return {
            "available": False,
            "entities": [],
            "message": (
                "Tamil NER could not complete. Classification and other "
                f"analysis remain available. Details: {error}"
            ),
        }
    return {"available": True, "entities": entities, "message": ""}
