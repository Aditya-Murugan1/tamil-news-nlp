# Tamil News Linguistic Analysis and Topic Classification System

An academic Tamil-only NLP project for news-topic classification and linguistic analysis. It is a traditional NLP and machine-learning application, not a chatbot, translator, sentiment analyzer, or question-answering system.

## Dataset and classification baseline

- Dataset: [`mteb/TamilNewsClassification`](https://huggingface.co/datasets/mteb/TamilNewsClassification)
- Classifier: TF-IDF with word and character n-grams + LinearSVC
- Existing held-out test accuracy: **83.79%**
- Existing macro F1: **0.827**
- Existing weighted F1: **0.837**

The dataset exposes integer labels. The UI displays `Class N` and does not guess topic names. These metrics are from the saved Phase 1 held-out evaluation and are not recomputed for each article.

The baseline is CPU-friendly, fast to train, interpretable through its feature representation, suitable for a college NLP project, compatible with Tamil text, and provides a stable comparison for future transformer experiments.

## Running locally

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the application:

```powershell
streamlit run app.py
```

The existing files in `models/` are loaded directly. Running the application does **not** retrain the classifier. Run `python train.py` only when intentionally rebuilding it; training downloads the dataset through Hugging Face, and the dataset is not committed to this repository.

The application uses the same `clean_text()` preprocessing function used during training before applying the saved vectorizer and classifier. LinearSVC does not produce probabilities; the UI does not show probability or confidence percentages for classification.

## Tamil NER

Tamil NER uses the public Hugging Face token-classification model [`exentai/indicner-tamil-ExentAI`](https://huggingface.co/exentai/indicner-tamil-ExentAI). Stanza's current pretrained NER coverage does not include Tamil, so Stanza is not used for Tamil NER; the optional Stanza integration is reserved for POS tagging.

Transformers, PyTorch, and Safetensors are included in `requirements.txt`. The NER pipeline is loaded lazily only when NER is explicitly requested, then cached in memory. The first request downloads model files from Hugging Face into its local cache; it can take time and requires an internet connection. CPU inference is supported; CUDA is used only when available.

NER is separate from the news classifier: its model is not trained by this project and does not affect classification. The public model is associated with AI4Bharat IndicNER, but its model card says it was fine-tuned on an unknown dataset and provides little provenance information. Treat its representativeness accordingly. Entity outputs and confidence scores are model predictions, not ground truth. The classifier's evaluation metrics are not NER evaluation results.

## Optional POS tagging

Stanza POS resources remain optional and are not downloaded automatically. To try Tamil POS tagging, install Stanza and explicitly download its Tamil tokenizer/POS resources:

```powershell
python -m pip install -r requirements-optional.txt
python -c "import stanza; stanza.download('ta', processors='tokenize,pos')"
```

Missing POS resources do not disable classification, NER, or other linguistic analysis.

## Architecture

```text
Tamil Text → Normalization → Tokenization → Word/Character N-Grams
          → TF-IDF → LinearSVC → Topic Class
```

```text
Tamil Text → Preprocessing → N-Grams / POS / Morphology / Tamil NER
                         → Linguistic Analysis
```

## Laboratory experiment mapping

| Laboratory Experiment | Project Component |
|---|---|
| Exp 1 | Tamil NLP application and problem statement |
| Exp 2 | Tamil text preprocessing |
| Exp 3 | Lemmatization / word normalization |
| Exp 4 | Morphological / word-form analysis |
| Exp 5 | N-Gram model |
| Exp 6 | POS tagging |
| Exp 7 | POS tagging implementation |
| Exp 8 | Named Entity Recognition |

## Screenshots

Add genuine screenshots from the running application. No screenshots are included yet.

### Classification screen
_Screenshot placeholder._

### Linguistic analysis
_Screenshot placeholder._

### N-Grams
_Screenshot placeholder._

### POS tagging
_Screenshot placeholder._

### Named Entity Recognition
_Screenshot placeholder._

### Model information
_Screenshot placeholder._

## Modules and tests

- `src/preprocessing.py`: Unicode-safe normalization, cleanup, tokenization, and sentence splitting.
- `src/ngrams.py`: unigram, bigram, and trigram frequencies.
- `src/keywords.py`: lightweight article keyword extraction.
- `src/morphology.py`: educational suffix-pattern and word-form statistics, not a complete Tamil parser or lemmatizer.
- `src/analysis.py`: combined per-article analysis.
- `src/pos_tagger.py`: optional lazy-loaded Stanza POS tagging.
- `src/ner.py`: lazy-loaded Hugging Face Tamil NER.

Run the tests without downloading the dataset or NER model:

```powershell
python -m pytest -q
```

NER tests mock the model pipeline so normal tests do not download hundreds of megabytes of model files.

## Limitations

- Tamil morphology, inflection, and sandhi are complex; the current morphology module is an educational heuristic.
- POS tagging needs optional local Stanza Tamil resources.
- NER quality depends on the public model, its training coverage, confidence threshold, and the input domain.
- Dataset label IDs remain class IDs; the project does not claim unsupported topic-name mappings.
- Dataset sources and writing style limit classifier generalization.
- The current news classifier is a TF-IDF + LinearSVC baseline, not a transformer.

## Future work

A future, separate experiment may compare the fixed baseline with IndicBERT, XLM-R, or another transformer. Other potential improvements include validated Tamil morphology and broader Tamil POS/NER evaluation. Such experiments will not replace the current baseline.
