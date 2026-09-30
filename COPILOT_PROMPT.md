# GitHub Copilot Build Prompt

Build this repository into a working academic NLP mini-project titled:
"Tamil News Linguistic Analysis and Topic Classification System Using NLP".

Context:
- This is a college NLP project using ONE natural language: Tamil.
- It must not overlap with these existing class topics: product-review sentiment analysis, intelligent meeting assistant, video caption generator, legal contract analyzer, Hinglish/code-mixed text classifier, plagiarism detector, prompt quality analyzer, cross-lingual answering, multilingual QA.
- The project should explicitly demonstrate the lab concepts: text preprocessing, stemming/lemmatization, morphological/word-form analysis, N-grams, POS tagging, and NER where practical.
- Prioritize a reliable 2-hour MVP over unnecessary sophistication.

Dataset:
- Use Hugging Face dataset `mteb/TamilNewsClassification`.
- Download it programmatically with `datasets.load_dataset`.
- Do not commit the dataset to Git.
- Inspect the actual schema and label feature at runtime instead of hardcoding a label mapping until verified.

Architecture:
1. `train.py`
   - Load the dataset.
   - Detect text and label columns safely.
   - Normalize Tamil Unicode text and remove URLs, HTML, excess whitespace, and unwanted symbols without deleting Tamil characters.
   - Create a reproducible train/validation/test split if only a train split is provided.
   - Use a strong CPU-friendly baseline: TF-IDF with word n-grams (1-2) plus character n-grams (3-5), followed by LinearSVC or LogisticRegression.
   - Save the fitted vectorizer/model and label mapping under `models/`.
   - Print accuracy, macro F1, weighted F1, classification report, and confusion matrix.
   - Save evaluation outputs under `reports/`.
   - Use a fixed random seed.

2. `src/preprocessing.py`
   - Implement Tamil-safe preprocessing.
   - Keep a simple fallback tokenizer using Unicode-aware regex.
   - If Indic NLP Library is available, use it for normalization/tokenization, but do not make the project fail merely because Indic NLP resources are absent.
   - Include helper functions for sentence tokenization and token frequency.

3. `src/linguistic_analysis.py`
   - Implement unigram, bigram, and trigram extraction.
   - Implement keyword extraction using TF-IDF scores/frequency.
   - Add optional POS/lemmatization support using Stanza only when its Tamil model is available; otherwise show a clear "optional module unavailable" state.
   - Add an NER hook that is modular and does not break the main application if a Tamil NER model is unavailable.
   - Never fake NER output. If no supported Tamil NER model is installed, say so in the UI.

4. `app.py`
   - Build a clean Streamlit interface.
   - Let the user paste a Tamil headline/article.
   - Show:
       a. cleaned text,
       b. predicted topic and confidence/probability only if the selected model supports meaningful probabilities,
       c. top keywords,
       d. top unigrams/bigrams/trigrams,
       e. token count,
       f. optional POS/lemma results,
       g. optional NER results.
   - Include a small "Sample Tamil Texts" section with several clearly labeled examples for demo use.
   - Do not use emojis in the academic UI.

5. `README.md`
   - Explain problem statement, objectives, methodology, architecture, NLP techniques, dataset, model, evaluation metrics, limitations, future work, and how each lab experiment maps to the project.
   - Include a short viva explanation.
   - State honestly that this is a Tamil single-language project and that the classifier is a topic-classification model, not translation or question answering.

Coding requirements:
- Keep the code modular and beginner-readable.
- Prefer standard libraries and lightweight dependencies.
- Do not introduce a database, Docker, cloud deployment, FastAPI, React, or a frontend framework.
- Do not use a transformer for the MVP.
- Do not scrape live news sites.
- Do not download huge models unless explicitly requested later.
- Make the project run on a laptop CPU.
- Add clear error messages for missing model files and missing optional NLP resources.
- Verify imports and run a smoke test after implementation.

Important performance decision:
- The laptop CPU is enough for the baseline. Do NOT require the separate PC/GPU for the first version.
- Structure the code so a future transformer model such as IndicBERT/XLM-R can be added as an optional `advanced` module without rewriting the app.

Deliver the project in small, working steps:
1. First implement dataset loading + preprocessing + TF-IDF classifier + saved model.
2. Then implement the Streamlit UI.
3. Then add n-gram/keyword analysis.
4. Finally add optional POS/lemma/NER hooks.
5. Run a smoke test and fix errors.

Do not overengineer. A fully working CPU MVP is more important than adding advanced features.
