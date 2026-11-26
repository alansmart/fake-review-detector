"""Streamlit demo for fake-review classification and genuine-likeness scoring."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "models"

from features import extract_features, FEATURE_COLS
from trust import trust_score, dml_trust_score

st.set_page_config(page_title="Fake Review Detector", page_icon="🤖", layout="wide")

#----------------------------------------------------------------------------------------model loaders
@st.cache_resource
def load_baseline():
    return joblib.load(MODELS / "baseline.joblib")

@st.cache_resource
def load_fusion():
    from fusion import FusionNet
    model = FusionNet(768, len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODELS / "fusion.pt", map_location="cpu"))
    return model.eval(), joblib.load(MODELS / "scaler.joblib")

@st.cache_resource
def load_dml():
    from dml import DMLNet, DMLScorer
    model = DMLNet(768, len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODELS / "dml_model.pt", map_location="cpu"))
    index = torch.load(MODELS / "dml_index.pt", map_location="cpu")
    scorer = DMLScorer(index["Z"], index["y"])
    return model.eval(), scorer, joblib.load(MODELS / "dml_scaler.joblib")

def available_models():
    opts = []
    if (MODELS / "baseline.joblib").exists():
        opts.append("Baseline (TF-IDF + Logistic Regression)")
    if (MODELS / "fusion.pt").exists() and (MODELS / "scaler.joblib").exists():
        opts.append("Fusion (Text + Metadata)")
    if all(
        (MODELS / filename).exists()
        for filename in (
            "dml_model.pt",
            "dml_index.pt",
            "dml_scaler.joblib",
            "dml_novelty_threshold.json",
        )
    ):
        opts.append("DML (Deep Metric Learning)")
    return opts

def text_and_meta(text, feats, scaler):
    from text_model import embed_texts
    t = torch.tensor(embed_texts([text]), dtype=torch.float32).cpu()
    m_raw = pd.DataFrame([feats])[FEATURE_COLS].values
    m = torch.tensor(scaler.transform(m_raw), dtype=torch.float32)
    return t, m
#------------------------------------------------------------------------------------- prediction
def analyze(text, rating, model_name, novelty_threshold):
    feats = extract_features(text, rating)

    if model_name.startswith("Baseline"):
        p_fake = float(load_baseline().predict_proba([text])[0][1])
        return trust_score(p_fake, feats), feats

    if model_name.startswith("Fusion"):
        model, scaler = load_fusion()
        t, m = text_and_meta(text, feats, scaler)
        with torch.no_grad():
            p_fake = float(torch.sigmoid(model(t, m))[0])
        return trust_score(p_fake, feats), feats
    if model_name.startswith("DML"):
        model, scorer, scaler = load_dml()
        t, m = text_and_meta(text, feats, scaler)
        with torch.no_grad():
            p, margin, novelty = scorer.score(model(t, m))
        return (
            dml_trust_score(
                float(p[0]), float(margin[0]), float(novelty[0]), feats,
                novelty_threshold,
            ),
            feats,
        )
    raise ValueError(f"Unsupported model selection: {model_name}")

#-------------------------------------------------------------------------------------------------- UI
st.title("🤖 Fake Review Detector and Genuine-likeness Score")
st.caption(
    "Predictions are estimates, not proof that a review is fake or genuine. "
    "Validate models on representative, labeled data before relying on them."
)

models = available_models()
if not models:
    st.error("No trained models found in `models/`. Run `python src\\baseline.py` first.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", models, index=len(models) - 1)
    if any(name.startswith("DML") for name in models):
        with (MODELS / "dml_novelty_threshold.json").open(
            "r", encoding="utf-8"
        ) as threshold_file:
            threshold_default = float(json.load(threshold_file)["threshold"])
        threshold_default = min(max(threshold_default, 0.01), 2.0)
        novelty_threshold = st.slider(
            "DML novelty threshold",
            0.01,
            2.0,
            threshold_default,
            0.01,
            help="Defaults to the 95th percentile of stratified out-of-fold novelty scores.",
        )
    else:
        novelty_threshold = 0.35
    st.markdown("---")
    st.markdown("**Examples**")
    examples = {
        "✅ Genuine-looking": ("The battery lasts about two days with normal use."
                            "The strap felt stiff at first but softened after a week.", 4),
        "🚩 Suspicious": ("BEST PRODUCT EVER!!! AMAZING!!! BUY IT NOW YOU WONT REGRET!!!", 5),
        "⚠️ Mismatch": ("Terrible quality, broke after one day, total waste of money", 5)
    }

    for label, (txt, rt) in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["text"], st.session_state["rating"] = txt, rt
    text = st.text_area("Paste a review", key="text", height=150)
    rating = st.slider("Star rating", 1, 5, key="rating", value=st.session_state.get("rating", 5))

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.error("Enter a review before analyzing.")
    else:
        with st.spinner(f"Analyzing with {model_name}..."):
            result, feats = analyze(text, rating, model_name, novelty_threshold)

        score, level = result["trust_score"], result["level"]
        color = {"High": "🟢", "Medium": "🟡", "Low": "🔴", "Uncertain": "⚪"}[level]

        st.caption(f"Model used: {model_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Genuine-likeness score", f"{score}/100")
        c2.metric("Assessment", f"{color} {level}")
        c3.metric("Predicted fake probability", f"{result['p_fake']:.1%}")
        st.progress(min(max(score / 100, 0.0), 1.0))
        st.caption(
            "The genuine-likeness score is a weighted indicator, not a probability. "
            "The fake probability is the model's predicted probability for the fake class."
        )

        sentiment = feats["sentiment"]
        sentiment_label = (
            "Positive" if sentiment >= 0.05 else "Negative" if sentiment <= -0.05 else "Neutral"
        )
        st.metric("Text sentiment (VADER)", f"{sentiment_label} ({sentiment:+.2f})")

        st.subheader("Why?")
        if result["reasons"]:
            for reason in result["reasons"]:
                st.warning(reason)
        else:
            st.success("No additional heuristic flags were triggered.")

        st.subheader("Score factors")
        st.bar_chart(pd.Series(result["components"], name="factor (0-1)"))

        with st.expander("Extracted linguistic features"):
            st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "value"}))

        report = {
            "model": model_name,
            "genuine_likeness_score": score,
            "assessment": level,
            "predicted_fake_probability": result["p_fake"],
            "rating": rating,
            "text_sentiment": {
                "label": sentiment_label,
                "vader_compound": round(float(sentiment), 4),
            },
            "reasons": result["reasons"],
            "score_factors": result["components"],
        }
        st.download_button(
            "Download result summary (JSON)",
            data=json.dumps(report, indent=2),
            file_name="review-assessment.json",
            mime="application/json",
        )
        st.caption("The downloaded summary does not contain the review text.")

st.markdown("---")
st.caption("A low score means review manually; it does not prove a review is fake.")

