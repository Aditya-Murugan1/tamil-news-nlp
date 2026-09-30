"""Phase 3 app and saved-model smoke tests; no dataset download is required."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib

import app
from src.analysis import analyze_text
from src.ner import analyze_ner, extract_entities, is_ner_available, load_ner_model
from src.pos_tagger import analyze_pos
from src.preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TEXT = "சென்னையில் இன்று கனமழை பெய்தது. அரசு நிவாரண முகாம்களை திறந்தது."


class ApplicationTests(unittest.TestCase):
    def test_app_imports(self) -> None:
        self.assertTrue(callable(app.main))

    def test_saved_classifier_and_vectorizer_predict(self) -> None:
        classifier_path = PROJECT_ROOT / "models" / "classifier.joblib"
        vectorizer_path = PROJECT_ROOT / "models" / "vectorizer.joblib"
        self.assertTrue(classifier_path.is_file())
        self.assertTrue(vectorizer_path.is_file())
        classifier = joblib.load(classifier_path)
        vectorizer = joblib.load(vectorizer_path)
        prediction = classifier.predict(vectorizer.transform([clean_text(SAMPLE_TEXT)]))
        self.assertEqual(len(prediction), 1)
        self.assertIn(str(prediction[0]), {str(label) for label in classifier.classes_})

    def test_phase2_analysis_remains_functional(self) -> None:
        result = analyze_text(SAMPLE_TEXT)
        self.assertGreater(result["token_count"], 0)
        self.assertGreater(len(result["top_bigrams"]), 0)
        self.assertGreater(len(result["keywords"]), 0)

    def test_optional_modules_return_clear_unavailable_state(self) -> None:
        pos_result = analyze_pos("")
        ner_result = analyze_ner("")
        self.assertFalse(pos_result["available"])
        self.assertFalse(ner_result["available"])
        self.assertTrue(pos_result["message"])
        self.assertTrue(ner_result["message"])

    def test_missing_optional_models_do_not_raise(self) -> None:
        with patch("src.pos_tagger._load_pipeline", side_effect=RuntimeError("model missing")):
            pos_result = analyze_pos(SAMPLE_TEXT)
        with patch("src.ner.load_ner_model", side_effect=RuntimeError("model missing")):
            ner_result = analyze_ner(SAMPLE_TEXT)
        self.assertFalse(pos_result["available"])
        self.assertIn("model missing", pos_result["message"])
        self.assertFalse(ner_result["available"])
        self.assertIn("Tamil NER could not complete.", ner_result["message"])
        self.assertIn("Classification and other analysis remain available.", ner_result["message"])

    def test_ner_dependency_detection(self) -> None:
        with patch("src.ner.importlib.util.find_spec", return_value=object()):
            self.assertTrue(is_ner_available())
        with patch(
            "src.ner.importlib.util.find_spec",
            side_effect=[object(), None],
        ):
            self.assertFalse(is_ner_available())

    def test_ner_module_import_and_empty_text(self) -> None:
        self.assertTrue(callable(load_ner_model))
        with patch("src.ner.load_ner_model") as load_model:
            self.assertEqual(extract_entities(""), [])
        load_model.assert_not_called()

    def test_extract_entities_normalizes_labels_and_filters_confidence(self) -> None:
        fake_pipeline = lambda text: [
            {
                "word": "சென்னை",
                "entity_group": "LOC",
                "score": 0.91,
                "start": 0,
                "end": 6,
            },
            {
                "word": "நபர்",
                "entity_group": "B-PER",
                "score": 0.25,
                "start": 7,
                "end": 11,
            },
        ]
        with patch("src.ner.load_ner_model", return_value=fake_pipeline):
            entities = extract_entities(
                "சென்னை நபர்", min_confidence=0.35
            )
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["label"], "Location")
        self.assertEqual(entities[0]["score"], 0.91)
        self.assertEqual((entities[0]["start"], entities[0]["end"]), (0, 6))

    def test_ner_offsets_do_not_split_tamil_combining_marks(self) -> None:
        fake_pipeline = lambda text: [
            {
                "word": "பிழையான துணைச்சொல்",
                "entity_group": "LOC",
                "score": 0.9,
                "start": 0,
                "end": 4,
            }
        ]
        with patch("src.ner.load_ner_model", return_value=fake_pipeline):
            entities = extract_entities("தமிழ்", min_confidence=0.35)
        self.assertEqual(entities[0]["text"], "தமிழ்")
        self.assertEqual((entities[0]["start"], entities[0]["end"]), (0, 5))

    def test_ner_runtime_failure_is_a_clear_unavailable_result(self) -> None:
        with patch(
            "src.ner.load_ner_model",
            side_effect=RuntimeError("transformers dependency missing"),
        ):
            result = analyze_ner("தமிழ் செய்தி")
        self.assertFalse(result["available"])
        self.assertEqual(result["entities"], [])
        self.assertIn("transformers dependency missing", result["message"])

    def test_invalid_ner_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_entities("செய்தி", min_confidence=1.5)

    def test_model_loader_does_not_retrain(self) -> None:
        classifier, vectorizer = app.load_classifier_artifacts()
        self.assertTrue(hasattr(classifier, "predict"))
        self.assertTrue(hasattr(vectorizer, "transform"))


if __name__ == "__main__":
    unittest.main()
