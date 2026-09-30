"""Streamlit dashboard for Tamil news classification and linguistic analysis."""

from __future__ import annotations

import html
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st

from src.analysis import analyze_text
from src.ner import MODEL_ID, analyze_ner, is_ner_available
from src.pos_tagger import analyze_pos, is_pos_available
from src.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
_TAMIL_SCRIPT = re.compile(r"[\u0B80-\u0BFF]")
_LOGGER = logging.getLogger("tamil_news_app")

SAMPLE_TEXTS = {
    "Weather and public services (example)": (
        "சென்னையில் கனமழை பெய்ததால் தாழ்வான பகுதிகளில் நீர் தேங்கியது. "
        "மாநகராட்சி பணியாளர்கள் மீட்பு மற்றும் நிவாரண பணிகளில் ஈடுபட்டனர்."
    ),
    "Education (example)": (
        "மாணவர்களின் திறனை மேம்படுத்த புதிய கல்வி திட்டம் அறிமுகப்படுத்தப்பட்டது. "
        "பள்ளிகளில் நூலக வசதிகளை விரிவுபடுத்த அதிகாரிகள் நடவடிக்கை எடுத்தனர்."
    ),
    "Agriculture (example)": (
        "விவசாயிகள் கூட்டுறவு சங்கத்தில் விளைபொருட்களை விற்பனை செய்தனர். "
        "பாசன வசதிகளை மேம்படுத்த அரசு புதிய திட்டத்தை அறிவித்தது."
    ),
}

st.set_page_config(
    page_title="Tamil NLP Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17243a;
        --muted: #64748b;
        --accent: #246b73;
        --accent-soft: #e8f3f2;
        --line: #e3e9ef;
        --surface: #ffffff;
        --canvas: #f5f7fa;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245, 247, 250, 0.94); }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    h1 { font-size: 1.85rem !important; margin-bottom: 0.15rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.04rem !important; }
    p, label, li { color: var(--ink); }
    .brand-kicker {
        color: var(--accent); font-weight: 700; letter-spacing: .12em;
        text-transform: uppercase; font-size: .7rem;
    }
    .subhead { color: var(--muted); margin-top: 0; }
    .status-line { color: var(--muted); font-size: .83rem; margin: .2rem 0 1rem; }
    .panel {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 14px; padding: 1.1rem 1.2rem; margin: .4rem 0 1rem;
    }
    .metric-card {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 12px; padding: .9rem 1rem; min-height: 92px;
    }
    .metric-label { color: var(--muted); font-size: .78rem; margin-bottom: .3rem; }
    .metric-value { color: var(--ink); font-size: 1.35rem; font-weight: 700; }
    .keyword-chip {
        display: inline-block; background: var(--accent-soft); color: #15545a;
        padding: .32rem .62rem; margin: .2rem .25rem .2rem 0;
        border-radius: 999px; border: 1px solid #d2e8e6; font-size: .84rem;
    }
    .entity-highlight {
        background: #d7eeeb; border-bottom: 2px solid var(--accent);
        border-radius: 3px; padding: 1px 2px;
    }
    .entity-label { color: #15545a; font-size: .72rem; font-weight: 700; }
    .stButton > button[kind="primary"] {
        background: var(--accent); border-color: var(--accent);
    }
    .stButton > button { border-radius: 9px; }
    [data-testid="stTabs"] button { font-weight: 600; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 9px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _clear_article() -> None:
    st.session_state["article_input"] = ""
    st.session_state["analysis_result"] = None
    st.session_state["analysis_text"] = ""
    st.session_state["classification_result"] = None
    st.session_state["classification_error"] = ""
    st.session_state["ner_result"] = None
    st.session_state["pos_result"] = None


def _load_selected_example() -> None:
    selected = st.session_state.get("sample_selection", "")
    st.session_state["article_input"] = SAMPLE_TEXTS.get(selected, "")


@st.cache_resource(show_spinner="Loading saved classifier...")
def load_classifier_artifacts() -> tuple[Any, Any]:
    """Load the existing fitted classifier and vectorizer without retraining."""
    import joblib

    classifier_path = MODEL_DIR / "classifier.joblib"
    vectorizer_path = MODEL_DIR / "vectorizer.joblib"
    missing_files = [
        path.name for path in (classifier_path, vectorizer_path) if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing model artifact(s): {', '.join(missing_files)} in {MODEL_DIR}. "
            "Run `python train.py` only if you need to rebuild the model."
        )
    try:
        return joblib.load(classifier_path), joblib.load(vectorizer_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not load the saved classifier artifacts: {error}"
        ) from error


def _classify_text(text: str) -> dict[str, str]:
    """Predict with the existing fitted vectorizer and preprocessing."""
    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("The text is empty after preprocessing.")
    classifier, vectorizer = load_classifier_artifacts()
    predicted_id = str(classifier.predict(vectorizer.transform([cleaned_text]))[0])
    return {"label_id": predicted_id, "display_label": f"Class {predicted_id}"}


def _saved_metrics() -> dict[str, Any]:
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        _LOGGER.exception("Could not read saved evaluation metrics")
        st.warning(f"Saved evaluation metrics could not be read: {error}")
        return {}


def _metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}'
        f'</div><div class="metric-value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_entity_highlight(text: str, entities: list[dict[str, Any]]) -> None:
    valid = [
        entity
        for entity in entities
        if isinstance(entity.get("start"), int)
        and isinstance(entity.get("end"), int)
        and 0 <= entity["start"] < entity["end"] <= len(text)
    ]
    valid.sort(key=lambda entity: (entity["start"], entity["end"]))
    if not valid:
        st.info("This model response did not include usable character offsets.")
        return

    pieces: list[str] = []
    cursor = 0
    for entity in valid:
        if entity["start"] < cursor:
            continue
        pieces.append(html.escape(text[cursor : entity["start"]]))
        label = html.escape(str(entity["label"]))
        entity_text = html.escape(text[entity["start"] : entity["end"]])
        pieces.append(
            f'<span class="entity-highlight">{entity_text}</span>'
            f'<span class="entity-label"> {label}</span>'
        )
        cursor = entity["end"]
    pieces.append(html.escape(text[cursor:]))
    st.markdown(
        '<div class="panel" style="line-height:2.1">'
        + "".join(pieces)
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_sidebar(classifier_status: str, pos_available: bool, ner_available: bool) -> float:
    with st.sidebar:
        st.markdown('<div class="brand-kicker">Project</div>', unsafe_allow_html=True)
        st.markdown("### Tamil NLP Intelligence")
        st.caption("Tamil News Linguistic Analysis & Topic Classification")
        st.divider()

        st.markdown("**Input controls**")
        st.selectbox(
            "Sample text (demonstration only)",
            ["Choose an example...", *SAMPLE_TEXTS.keys()],
            key="sample_selection",
            on_change=_load_selected_example,
        )
        top_n = st.slider(
            "Number of N-grams", 3, 20, 10, key="number_of_ngrams"
        )
        confidence = st.slider(
            "Minimum entity confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Filters model-predicted NER entities; this is not model accuracy.",
        )
        st.caption("POS and NER run only when requested.")
        st.divider()
        st.markdown("**System status**")
        classifier_icon = "●" if classifier_status == "loaded" else "○"
        classifier_text = (
            "Classifier loaded"
            if classifier_status == "loaded"
            else f"Classifier unavailable: {classifier_status}"
        )
        st.caption(f"{classifier_icon} {classifier_text}")
        st.caption("● Linguistic modules loaded")
        st.caption(
            "● POS dependencies available"
            if pos_available
            else "○ Optional POS resources unavailable"
        )
        st.caption(
            "● NER dependencies ready; model loads on request"
            if ner_available
            else "○ Install Transformers and PyTorch for NER"
        )
        st.caption("NER model weights are cached by Hugging Face after first use.")
    return confidence


def _render_input() -> tuple[str, bool]:
    with st.container(border=True):
        st.markdown("#### Article input")
        st.caption("Demonstration samples are illustrative, not real-time news.")
        value = st.text_area(
            "Tamil news article",
            placeholder="தமிழ் செய்தியை இங்கே உள்ளிடவும்...",
            height=170,
            key="article_input",
            label_visibility="collapsed",
        )
    analyze_column, clear_column, _ = st.columns([1.25, 0.75, 5])
    with analyze_column:
        analyze_clicked = st.button(
            "Analyze Article", type="primary", use_container_width=True
        )
    with clear_column:
        st.button("Clear", on_click=_clear_article, use_container_width=True)
    return value, analyze_clicked


def _process_article(text: str, top_n: int) -> None:
    if not text.strip():
        st.session_state["analysis_result"] = None
        st.session_state["analysis_text"] = ""
        st.session_state["classification_result"] = None
        st.session_state["classification_error"] = ""
        st.session_state["ner_result"] = None
        st.session_state["pos_result"] = None
        st.warning("Enter Tamil text before analyzing.")
        return

    analysis = analyze_text(text, top_k=top_n)
    st.session_state["analysis_result"] = analysis
    st.session_state["analysis_text"] = text
    st.session_state["ner_result"] = None
    st.session_state["pos_result"] = None
    try:
        st.session_state["classification_result"] = _classify_text(
            analysis["cleaned_text"]
        )
        st.session_state["classification_error"] = ""
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as error:
        st.session_state["classification_result"] = None
        st.session_state["classification_error"] = str(error)
    if not _TAMIL_SCRIPT.search(text):
        st.warning(
            "No Tamil-script characters were detected. The tools can still "
            "process text, but this project is designed for Tamil."
        )


def _analysis_for_current_top_n(top_n: int) -> dict[str, Any] | None:
    result = st.session_state.get("analysis_result")
    analyzed_text = st.session_state.get("analysis_text", "")
    if result is None:
        return None
    if st.session_state.get("analysis_top_n") != top_n:
        result = analyze_text(analyzed_text, top_k=top_n)
        st.session_state["analysis_result"] = result
        st.session_state["analysis_top_n"] = top_n
    return result


def _render_overview(result: dict[str, Any] | None, metrics: dict[str, Any]) -> None:
    if result is None:
        st.info("Enter an article and select Analyze Article to view results.")
        return
    prediction = st.session_state.get("classification_result")
    entity_result = st.session_state.get("ner_result")
    entities = entity_result.get("entities", []) if entity_result else []
    keywords = result["keywords"]
    average_length = (
        sum(len(token) for token in result["tokens"]) / result["token_count"]
        if result["token_count"]
        else 0.0
    )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        _metric_card("Predicted class", prediction["display_label"] if prediction else "—")
    with metric_columns[1]:
        _metric_card("Tokens", str(result["token_count"]))
    with metric_columns[2]:
        _metric_card("Keywords", str(len(keywords)))
    with metric_columns[3]:
        _metric_card("Entities", str(len(entities)) if entity_result else "Not run")

    st.write("")
    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        st.markdown("### Classification")
        if prediction:
            st.markdown(
                f'<div class="panel"><div class="metric-label">Predicted topic</div>'
                f'<div class="metric-value">{html.escape(prediction["display_label"])}</div>'
                '<div class="metric-label" style="margin-top:.65rem">'
                "TF-IDF + LinearSVC</div></div>",
                unsafe_allow_html=True,
            )
            accuracy = metrics.get("accuracy")
            st.caption(
                f"Saved held-out test accuracy: {accuracy:.2%}"
                if isinstance(accuracy, (int, float))
                else "Saved Phase 1 test accuracy is unavailable."
            )
        else:
            st.error(
                st.session_state.get("classification_error")
                or "The saved classifier could not produce a prediction."
            )
        st.markdown("### Keywords")
        if keywords:
            chips = "".join(
                f'<span class="keyword-chip">{html.escape(str(item["keyword"]))}</span>'
                for item in keywords
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No keywords were found.")
        st.markdown("### Named entities")
        if entity_result is None:
            st.caption("Run NER in the NER section to view model-predicted entities.")
        elif entities:
            for entity in entities[:8]:
                st.markdown(
                    f'<div class="panel" style="padding:.65rem .8rem">'
                    f'<strong>{html.escape(str(entity["text"]))}</strong> '
                    f'<span class="entity-label">{html.escape(str(entity["label"]))}'
                    f' · Model confidence {float(entity["score"]):.2f}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("The model returned no entities above the selected threshold.")

    with right:
        st.markdown("### Article statistics")
        stat_columns = st.columns(2)
        stat_columns[0].metric("Sentences", result["sentence_count"])
        stat_columns[1].metric("Characters", len(st.session_state["analysis_text"]))
        stat_columns[0].metric("Unique tokens", result["unique_token_count"])
        stat_columns[1].metric("Average token length", f"{average_length:.2f}")
        st.markdown("### Cleaned text preview")
        st.markdown(
            f'<div class="panel" lang="ta" style="line-height:1.9">'
            f'{html.escape(result["cleaned_text"] or "No text remains after cleanup.")}'
            "</div>",
            unsafe_allow_html=True,
        )


def _render_linguistics(result: dict[str, Any] | None) -> None:
    if result is None:
        st.info("Analyze an article to see linguistic analysis.")
        return
    st.markdown("### Preprocessed Tamil text")
    st.markdown(
        f'<div class="panel" lang="ta" style="line-height:1.9">'
        f'{html.escape(result["cleaned_text"] or "No text remains after cleanup.")}'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Word statistics")
    stats = result["morphology"]["word_length_statistics"]
    average_length = (
        sum(len(token) for token in result["tokens"]) / result["token_count"]
        if result["token_count"]
        else 0.0
    )
    columns = st.columns(4)
    columns[0].metric("Tokens", result["token_count"])
    columns[1].metric("Unique tokens", result["unique_token_count"])
    columns[2].metric("Average token length", f"{average_length:.2f}")
    columns[3].metric("Average word-form length", f"{stats['average']:.2f}")

    st.markdown("### Morphological / word-form inspection")
    morphology = result["morphology"]
    left, right = st.columns(2)
    with left:
        st.dataframe(morphology["word_forms"], use_container_width=True, hide_index=True)
        st.markdown("**Recognized suffix-pattern counts**")
        if morphology["suffix_pattern_counts"]:
            st.json(morphology["suffix_pattern_counts"])
        else:
            st.caption("No listed suffix pattern was detected.")
    with right:
        st.markdown("**Repeated word forms**")
        st.dataframe(
            morphology["repeated_word_forms"],
            use_container_width=True,
            hide_index=True,
        )
    st.caption(morphology["limitation"])


def _render_ngrams(result: dict[str, Any] | None) -> None:
    if result is None:
        st.info("Analyze an article to view its N-grams.")
        return
    columns = st.columns(3)
    for column, key, title in zip(
        columns,
        ("top_unigrams", "top_bigrams", "top_trigrams"),
        ("Unigrams", "Bigrams", "Trigrams"),
    ):
        with column:
            st.markdown(f"### {title}")
            rows = result[key]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("Not enough tokens.")


def _render_pos_tab(text: str, result: dict[str, Any] | None) -> None:
    st.markdown("### Tamil POS tagging")
    if result is None:
        st.info("Analyze an article before requesting POS tagging.")
        return
    if not is_pos_available():
        st.info(
            "POS resources are optional and Stanza is not installed. Install "
            "`requirements-optional.txt` and download the Tamil POS model using "
            "the README instructions."
        )
    if st.button("Run POS tagging", key="run_pos"):
        with st.spinner("Loading the optional Tamil POS model..."):
            st.session_state["pos_result"] = analyze_pos(text)
    pos_result = st.session_state.get("pos_result")
    if pos_result is None:
        st.caption("POS model resources are loaded only when this button is used.")
    elif not pos_result["available"]:
        st.warning(pos_result["message"])
    else:
        st.dataframe(pos_result["words"], use_container_width=True, hide_index=True)
        st.markdown("**POS tag distribution**")
        st.bar_chart(pos_result["counts"])


def _render_ner_tab(
    text: str,
    result: dict[str, Any] | None,
    confidence_threshold: float,
) -> None:
    st.markdown("### Tamil named-entity recognition")
    st.caption(
        f"Model: `{MODEL_ID}` · Separate from the topic classifier. "
        "Entities are model predictions, not ground truth."
    )
    if result is None:
        st.info("Analyze an article before requesting NER.")
        return
    if not is_ner_available():
        st.warning(
            "Install the NER dependencies with `python -m pip install -r "
            "requirements.txt`. The transformer is not loaded at startup."
        )
    st.caption(
        "The first run downloads the model from Hugging Face and may take "
        "several minutes. Model files are cached locally."
    )
    if st.button("Run Tamil NER", type="primary", key="run_ner"):
        with st.spinner("Loading Tamil NER model for the first time if needed..."):
            st.session_state["ner_result"] = analyze_ner(
                text, min_confidence=confidence_threshold
            )
            st.session_state["ner_threshold"] = confidence_threshold
    ner_result = st.session_state.get("ner_result")
    if ner_result is None:
        return
    if st.session_state.get("ner_threshold") != confidence_threshold:
        st.info("The confidence threshold changed. Run NER again to apply it.")
    if not ner_result["available"]:
        st.warning(ner_result["message"])
        return

    entities = ner_result["entities"]
    st.markdown("### Named entities found")
    st.metric("Entities above threshold", len(entities))
    if entities:
        counts = Counter(entity["label"] for entity in entities)
        summary_columns = st.columns(min(4, max(1, len(counts))))
        for column, (label, count) in zip(summary_columns, counts.items()):
            column.metric(label, count)
        st.dataframe(
            [
                {
                    "Entity": entity["text"],
                    "Type": entity["label"],
                    "Model confidence": f'{entity["score"]:.3f}',
                }
                for entity in entities
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### Entity highlighting")
        _render_entity_highlight(text, entities)
    else:
        st.info("The model returned no entities above the selected threshold.")


def _render_model_tab(metrics: dict[str, Any], ner_available: bool) -> None:
    st.markdown("### Classification model")
    st.write("**TF-IDF + Word/Character N-Grams + LinearSVC**")
    columns = st.columns(3)
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    weighted_f1 = metrics.get("weighted_f1")
    columns[0].metric("Accuracy", f"{accuracy:.2%}" if accuracy is not None else "—")
    columns[1].metric("Macro F1", f"{macro_f1:.3f}" if macro_f1 is not None else "—")
    columns[2].metric(
        "Weighted F1",
        f"{weighted_f1:.3f}" if weighted_f1 is not None else "—",
    )
    st.caption(
        "Metrics are from the saved Phase 1 held-out test evaluation and are not "
        "recalculated for each article."
    )
    st.markdown("### Tamil NER model")
    st.code(MODEL_ID, language=None)
    st.write("Tamil NER model loaded separately from the news classifier.")
    st.info(
        "NER is an auxiliary linguistic-analysis component and does not affect "
        "news classification."
    )
    st.caption(
        "NER dependencies are "
        + ("installed." if ner_available else "not installed in this environment.")
        + " Model weights load lazily only when NER is requested."
    )
    st.markdown("### Why this baseline was selected")
    st.markdown(
        "- CPU-friendly with fast training and inference.\n"
        "- Word and character n-grams support Tamil text directly.\n"
        "- Interpretable and suitable for a college NLP project.\n"
        "- A stable baseline for a future, separate transformer comparison."
    )
    st.markdown("### Classification pipeline")
    st.code(
        "Tamil Text → Normalization → Tokenization → N-Gram Features\n"
        "           → TF-IDF → LinearSVC → Topic Class",
        language=None,
    )
    st.markdown("### Linguistic pipeline")
    st.code(
        "Tamil Text → Preprocessing → N-Grams / POS / Morphology / NER\n"
        "                                → Linguistic Analysis",
        language=None,
    )


def main() -> None:
    st.markdown('<div class="brand-kicker">Tamil NLP Intelligence</div>', unsafe_allow_html=True)
    st.title("Tamil News Linguistic Analysis & Topic Classification")
    st.markdown(
        '<div class="status-line">Traditional NLP · Tamil Language · Machine Learning</div>',
        unsafe_allow_html=True,
    )

    try:
        load_classifier_artifacts()
        classifier_status = "loaded"
    except (FileNotFoundError, RuntimeError, ImportError) as error:
        classifier_status = str(error)
    ner_available = is_ner_available()
    pos_available = is_pos_available()
    confidence_threshold = _render_sidebar(
        classifier_status, pos_available, ner_available
    )

    text, analyze_clicked = _render_input()
    if analyze_clicked:
        _process_article(text, st.session_state.get("number_of_ngrams", 10))
    if "analysis_top_n" not in st.session_state:
        st.session_state["analysis_top_n"] = 10
    # The sidebar slider has no explicit key by default; get its current value
    # through the widget state when available and refresh only presentation data.
    top_n = st.session_state.get("number_of_ngrams", 10)
    result = _analysis_for_current_top_n(top_n)
    metrics = _saved_metrics()

    overview_tab, linguistics_tab, ngrams_tab, pos_tab, ner_tab, model_tab = st.tabs(
        ["Overview", "Linguistics", "N-Grams", "POS", "NER", "Model"]
    )
    with overview_tab:
        _render_overview(result, metrics)
    with linguistics_tab:
        _render_linguistics(result)
    with ngrams_tab:
        _render_ngrams(result)
    with pos_tab:
        _render_pos_tab(st.session_state.get("analysis_text", ""), result)
    with ner_tab:
        _render_ner_tab(
            st.session_state.get("analysis_text", ""),
            result,
            confidence_threshold,
        )
    with model_tab:
        _render_model_tab(metrics, ner_available)


if __name__ == "__main__":
    main()
