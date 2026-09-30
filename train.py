"""Train and evaluate a CPU-friendly Tamil news topic classifier."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.preprocessing import clean_text

SEED = 42
DATASET_NAME = "mteb/TamilNewsClassification"
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

_TEXT_COLUMN_HINTS = (
    "text",
    "content",
    "article",
    "news",
    "document",
    "sentence",
    "headline",
)
_LABEL_COLUMN_HINTS = ("label", "labels", "category", "topic", "class")
_LOGGER = logging.getLogger("tamil_news_nlp")


def _feature_name(feature: Any) -> str:
    return type(feature).__name__.casefold()


def _is_text_feature(feature: Any) -> bool:
    return getattr(feature, "dtype", None) in {"string", "large_string"} or (
        _feature_name(feature) == "value"
        and getattr(feature, "dtype", None) in {"string", "large_string"}
    )


def _is_label_feature(feature: Any) -> bool:
    return _feature_name(feature) == "classlabel" or getattr(feature, "dtype", None) in {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "string",
        "large_string",
    }


def _choose_column(
    columns: Sequence[str],
    features: Mapping[str, Any],
    hints: Sequence[str],
    predicate: Any,
    kind: str,
) -> str:
    candidates = [column for column in columns if predicate(features[column])]
    if not candidates:
        raise ValueError(
            f"Could not identify a {kind} column. "
            f"Available columns: {list(columns)}"
        )

    for hint in hints:
        exact_matches = [column for column in candidates if column.casefold() == hint]
        if exact_matches:
            return exact_matches[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not safely choose a {kind} column from {candidates}. "
        f"Update the schema detection for this dataset."
    )


def detect_columns(dataset_split: Any) -> tuple[str, str]:
    """Detect text and label columns from the loaded dataset's actual schema."""
    columns = list(dataset_split.column_names)
    features = dataset_split.features
    text_column = _choose_column(
        columns, features, _TEXT_COLUMN_HINTS, _is_text_feature, "text"
    )
    label_column = _choose_column(
        columns, features, _LABEL_COLUMN_HINTS, _is_label_feature, "label"
    )
    if text_column == label_column:
        raise ValueError("The detected text and label columns are identical.")
    return text_column, label_column


def _label_to_name(label: Any, label_feature: Any) -> str:
    if _feature_name(label_feature) == "classlabel":
        return str(label_feature.int2str(int(label)))
    return str(label)


def _read_rows(
    dataset_split: Any,
    text_column: str,
    label_column: str,
) -> tuple[list[str], list[str]]:
    label_feature = dataset_split.features[label_column]
    texts: list[str] = []
    labels: list[str] = []
    for row in dataset_split:
        text = row[text_column]
        label = row[label_column]
        if not isinstance(text, str) or not text.strip() or label is None:
            continue
        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue
        texts.append(cleaned_text)
        labels.append(_label_to_name(label, label_feature))
    if not texts:
        raise ValueError("The dataset contains no usable text/label rows.")
    return texts, labels


def load_dataset_rows() -> tuple[dict[str, tuple[list[str], list[str]]], dict[str, Any]]:
    """Load the Hugging Face dataset and inspect its schema at runtime."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from error

    dataset = load_dataset(DATASET_NAME)
    if not isinstance(dataset, Mapping) or not dataset:
        raise ValueError(f"{DATASET_NAME} did not return any dataset splits.")

    split_names = list(dataset.keys())
    first_split = dataset[split_names[0]]
    text_column, label_column = detect_columns(first_split)
    _LOGGER.info("Dataset: %s", DATASET_NAME)
    _LOGGER.info("Available splits: %s", split_names)
    _LOGGER.info("Detected text column: %s", text_column)
    _LOGGER.info("Detected label column: %s", label_column)
    _LOGGER.info("Runtime schema: %s", first_split.features)

    rows = {
        split_name: _read_rows(dataset[split_name], text_column, label_column)
        for split_name in split_names
    }
    return rows, {
        "dataset": DATASET_NAME,
        "splits": split_names,
        "text_column": text_column,
        "label_column": label_column,
        "schema": str(first_split.features),
    }


def _stratified_split(
    texts: list[str],
    labels: list[str],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    from sklearn.model_selection import train_test_split

    try:
        return train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create a stratified train/validation/test split. "
            "Each topic must have enough examples for all three splits."
        ) from error


def create_splits(
    rows: Mapping[str, tuple[list[str], list[str]]],
    seed: int = SEED,
) -> dict[str, tuple[list[str], list[str]]]:
    """Use official train/test splits when available, otherwise split reproducibly."""
    if "train" not in rows:
        if len(rows) != 1:
            raise ValueError(
                f"Expected a 'train' split or one source split; received {list(rows)}."
            )
        source_name = next(iter(rows))
        source_texts, source_labels = rows[source_name]
        train_texts, remaining_texts, train_labels, remaining_labels = _stratified_split(
            source_texts, source_labels, 0.30, seed
        )
        validation_texts, test_texts, validation_labels, test_labels = _stratified_split(
            remaining_texts, remaining_labels, 0.50, seed
        )
        return {
            "train": (train_texts, train_labels),
            "validation": (validation_texts, validation_labels),
            "test": (test_texts, test_labels),
        }

    train_texts, train_labels = rows["train"]
    validation_name = next(
        (name for name in ("validation", "valid", "dev") if name in rows), None
    )
    test_name = "test" if "test" in rows else None
    output = {"train": (train_texts, train_labels)}

    if validation_name is not None:
        output["validation"] = rows[validation_name]
    elif test_name is not None:
        train_part, validation_part, train_part_labels, validation_part_labels = (
            _stratified_split(train_texts, train_labels, 0.15, seed)
        )
        output["train"] = (train_part, train_part_labels)
        output["validation"] = (validation_part, validation_part_labels)
    else:
        train_part, remaining_texts, train_part_labels, remaining_labels = (
            _stratified_split(train_texts, train_labels, 0.30, seed)
        )
        validation_texts, test_texts, validation_labels, test_labels = (
            _stratified_split(remaining_texts, remaining_labels, 0.50, seed)
        )
        return {
            "train": (train_part, train_part_labels),
            "validation": (validation_texts, validation_labels),
            "test": (test_texts, test_labels),
        }

    if test_name is not None:
        output["test"] = rows[test_name]
    else:
        train_texts, train_labels = output["train"]
        train_part, test_part, train_part_labels, test_part_labels = _stratified_split(
            train_texts, train_labels, 0.15, seed
        )
        output["train"] = (train_part, train_part_labels)
        output["test"] = (test_part, test_part_labels)

    return output


def build_vectorizer() -> Any:
    """Create a weighted word/character TF-IDF feature union."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    dtype=np.float32,
                    max_features=300_000,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": 1.0},
    )


def _write_evaluation(
    classifier: Any,
    vectorizer: Any,
    test_texts: list[str],
    test_labels: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = classifier.predict(vectorizer.transform(test_texts))
    label_names = [str(label) for label in classifier.classes_]
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(test_labels, predictions, average="weighted", zero_division=0)
        ),
        "precision": float(
            precision_score(test_labels, predictions, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(test_labels, predictions, average="weighted", zero_division=0)
        ),
        "test_samples": len(test_labels),
        "labels": label_names,
        "dataset": metadata["dataset"],
        "text_column": metadata["text_column"],
        "label_column": metadata["label_column"],
        "seed": SEED,
    }
    report = classification_report(
        test_labels,
        predictions,
        labels=label_names,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, predictions, labels=label_names)

    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (REPORTS_DIR / "confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["actual/predicted", *label_names])
        for label, row in zip(label_names, matrix):
            writer.writerow([label, *row.tolist()])

    figure_size = max(7, min(14, len(label_names) * 1.2))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(label_names)),
        yticks=range(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Tamil News Topic Classification Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics


def train() -> dict[str, Any]:
    """Train, evaluate, and save the classifier and its TF-IDF vectorizer."""
    import joblib
    from sklearn.svm import LinearSVC

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows, metadata = load_dataset_rows()
    splits = create_splits(rows)
    train_texts, train_labels = splits["train"]
    vectorizer = build_vectorizer()
    train_features = vectorizer.fit_transform(train_texts)

    classifier = LinearSVC(random_state=SEED, max_iter=5000)
    classifier.fit(train_features, train_labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODELS_DIR / "classifier.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    label_mapping = {
        "class_to_id": {
            label: index for index, label in enumerate(sorted(classifier.classes_))
        },
        "id_to_class": sorted(classifier.classes_),
    }
    (MODELS_DIR / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = _write_evaluation(
        classifier, vectorizer, *splits["test"], metadata
    )
    _LOGGER.info("Evaluation metrics: %s", json.dumps(metrics, ensure_ascii=False))
    _LOGGER.info("Saved model artifacts under %s", MODELS_DIR)
    _LOGGER.info("Saved evaluation reports under %s", REPORTS_DIR)
    return metrics


if __name__ == "__main__":
    train()
