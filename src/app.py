import sys
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from .predict import PredictionInputError, predict_url
except ImportError:  # pragma: no cover - script execution fallback
    from predict import PredictionInputError, predict_url


st.set_page_config(
    page_title="Zero-Day Phishing Detection",
    page_icon="🛡️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 900px; padding-top: 2rem; }
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 1rem;
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ Zero-Day Phishing Detection")
st.markdown(
    "Analyze a URL with a **hybrid ANN + BiLSTM** model and inspect "
    "which numerical URL features influenced its score."
)
st.info(
    "Research prototype only. The model score is not a calibrated probability "
    "or a guarantee that a website is safe. Do not visit suspicious URLs."
)

with st.form("url_analysis"):
    submitted_url = st.text_input(
        "URL to analyze",
        placeholder="https://example.com/login",
        help="HTTP and HTTPS URLs are supported. A missing scheme defaults to HTTPS.",
    )
    submitted = st.form_submit_button(
        "Analyze URL", type="primary", use_container_width=True
    )

if submitted:
    try:
        st.session_state["prediction"] = predict_url(submitted_url)
    except PredictionInputError as error:
        st.session_state.pop("prediction", None)
        st.error(str(error))
    except FileNotFoundError as error:
        st.session_state.pop("prediction", None)
        st.error(
            "The trained model files are missing. Place "
            "`phishing_model.keras` and `scaler.pkl` in the project's `model` "
            f"folder. Details: {error}"
        )

result = st.session_state.get("prediction")
if result is not None:
    st.divider()
    st.subheader("Analysis result")
    st.code(result.url, language=None)

    score_column, verdict_column = st.columns(2)
    score_column.metric("Model phishing score", f"{result.phishing_score:.1%}")
    verdict_column.metric(
        "Model classification",
        "Potential phishing" if result.is_phishing else "Not flagged",
    )

    if result.is_phishing:
        st.error(
            "The model classified this URL as potential phishing. Treat it as "
            "suspicious and verify it through a trusted channel."
        )
    else:
        st.success(
            "The model did not flag this URL. This does not confirm that the "
            "website is safe."
        )

    st.markdown("#### What influenced the score?")
    st.caption(
        "For each numerical feature, the app resets that feature to its training "
        "mean while keeping the URL-character input fixed. A positive change "
        "means the original feature raised the score in this model; this is a "
        "local sensitivity check, not a causal explanation."
    )
    impacts = pd.DataFrame(
        [
            {
                "URL feature": impact.name.replace("_", " ").title(),
                "Value": impact.value,
                "Score change": f"{impact.score_change * 100:+.2f} pp",
            }
            for impact in result.feature_impacts
        ]
    )
    st.dataframe(impacts, hide_index=True, use_container_width=True)

st.divider()
st.caption("Academic project demonstration · Hybrid ANN + BiLSTM · Explainable AI")
