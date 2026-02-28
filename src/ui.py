import os
import time
import json
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
API_URL       = os.environ.get("API_URL", "http://api:8000")
RESULTS_DIR   = os.environ.get("RESULTS_DIR", "results")
SESSION_KEY   = "prediction_history"

st.set_page_config(
    page_title="Twitter Sentiment Analyser",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
POSITIVE_COLOR = "#2ECC71"
NEGATIVE_COLOR = "#E74C3C"
NEUTRAL_COLOR  = "#95A5A6"

SENTIMENT_COLORS = {"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR}


# ── Helpers ───────────────────────────────────────────────────────────────────
def call_predict(text: str) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API. Make sure the API service is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.json().get('detail', str(e))}")
        return None


def api_healthy() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_history() -> pd.DataFrame:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = []
    return pd.DataFrame(st.session_state[SESSION_KEY])


def add_to_history(text: str, result: dict):
    record = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "text":      text[:120] + ("…" if len(text) > 120 else ""),
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
    }
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = []
    st.session_state[SESSION_KEY].append(record)


def load_run_summary() -> dict | None:
    path = Path(RESULTS_DIR) / "run_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_metrics() -> dict | None:
    path = Path(RESULTS_DIR) / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Logo_of_Twitter.svg/"
             "220px-Logo_of_Twitter.svg.png", width=60)
    st.title("Sentiment Analyser")
    st.markdown("---")

    healthy = api_healthy()
    if healthy:
        st.success("🟢 API is online")
    else:
        st.error("🔴 API is offline")

    st.markdown("---")
    st.subheader("About")
    st.markdown("""
**Model**: DistilBERT (fine-tuned)  
**Dataset**: IMDB reviews  
**Task**: Binary sentiment classification  
**Framework**: PyTorch + Hugging Face  
""")

    if st.button("🗑️ Clear History"):
        st.session_state[SESSION_KEY] = []
        st.rerun()


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_predict, tab_history, tab_model_stats = st.tabs(
    ["🔮 Predict", "📋 History & Charts", "📊 Model Performance"]
)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Sentiment Prediction")
    st.markdown("Enter a tweet or any text to analyse its sentiment.")

    # Example buttons
    st.markdown("**Quick examples:**")
    col1, col2, col3 = st.columns(3)
    examples = {
        "😊 Positive": "I absolutely love this new phone! It's incredibly fast and the camera is stunning.",
        "😢 Negative": "This product is terrible. Broke after two days and customer support was useless.",
        "🤔 Mixed":    "The plot had some good ideas but the execution was disappointing overall.",
    }
    preset_text = ""
    for (label, example), col in zip(examples.items(), [col1, col2, col3]):
        if col.button(label):
            preset_text = example

    user_text = st.text_area(
        "Enter text:",
        value=preset_text,
        height=130,
        placeholder="Type a tweet or review here…",
        max_chars=10_000,
    )

    if st.button("🔍 Analyse Sentiment", type="primary", disabled=not healthy):
        if not user_text.strip():
            st.warning("Please enter some text before analysing.")
        else:
            with st.spinner("Analysing…"):
                result = call_predict(user_text)

            if result:
                add_to_history(user_text, result)
                sentiment  = result["sentiment"]
                confidence = result["confidence"]
                color      = SENTIMENT_COLORS.get(sentiment, NEUTRAL_COLOR)
                emoji      = "😊" if sentiment == "positive" else "😢"

                st.markdown("---")
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.markdown(f"### {emoji} Result")
                    st.markdown(
                        f"<div style='background:{color};padding:20px;border-radius:12px;"
                        f"text-align:center;color:white;font-size:24px;font-weight:bold'>"
                        f"{sentiment.upper()}</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Confidence", f"{confidence*100:.1f}%")

                with c2:
                    # Gauge-style bar chart
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    ax.barh(["Confidence"], [confidence],     color=color,     height=0.4, label=sentiment)
                    ax.barh(["Confidence"], [1 - confidence], left=confidence, color="#ECF0F1", height=0.4)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probability")
                    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
                    ax.set_title(f"Model confidence: {confidence*100:.1f}%")
                    ax.spines[["top", "right", "left"]].set_visible(False)
                    st.pyplot(fig)
                    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY & CHARTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.header("Prediction History & Analytics")
    history_df = get_history()

    if history_df.empty:
        st.info("No predictions yet — run some analyses in the Predict tab!")
    else:
        st.dataframe(history_df, use_container_width=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        # Pie chart — sentiment distribution
        with col_a:
            st.subheader("Sentiment Distribution")
            counts = history_df["sentiment"].value_counts()
            colors = [SENTIMENT_COLORS.get(s, NEUTRAL_COLOR) for s in counts.index]
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index, autopct="%1.0f%%",
                colors=colors, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            for t in autotexts:
                t.set_fontsize(12)
                t.set_color("white")
                t.set_fontweight("bold")
            ax.set_title("Positive vs Negative")
            st.pyplot(fig)
            plt.close(fig)

        # Confidence distribution histogram
        with col_b:
            st.subheader("Confidence Distribution")
            fig, ax = plt.subplots()
            for sentiment, grp in history_df.groupby("sentiment"):
                ax.hist(
                    grp["confidence"], bins=10, alpha=0.7,
                    color=SENTIMENT_COLORS.get(sentiment, NEUTRAL_COLOR),
                    label=sentiment, edgecolor="white",
                )
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Count")
            ax.set_title("Confidence Distribution by Sentiment")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

        # Confidence over time (line chart)
        st.subheader("Confidence Over Time")
        fig, ax = plt.subplots(figsize=(10, 3))
        for sentiment, grp in history_df.groupby("sentiment"):
            ax.scatter(
                grp.index, grp["confidence"],
                color=SENTIMENT_COLORS.get(sentiment, NEUTRAL_COLOR),
                label=sentiment, s=60, zorder=3,
            )
        ax.plot(history_df.index, history_df["confidence"], color="#BDC3C7", linewidth=1, zorder=1)
        ax.set_xlabel("Prediction #")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        # Summary stats
        st.markdown("---")
        st.subheader("Summary Statistics")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Predictions", len(history_df))
        sc2.metric("Positive", int((history_df["sentiment"] == "positive").sum()))
        sc3.metric("Negative", int((history_df["sentiment"] == "negative").sum()))
        sc4.metric("Avg Confidence", f"{history_df['confidence'].mean()*100:.1f}%")

        # Download button
        csv = history_df.to_csv(index=False)
        st.download_button("⬇️ Download History CSV", csv, "predictions.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab_model_stats:
    st.header("Model Performance & Training Metrics")

    metrics     = load_metrics()
    run_summary = load_run_summary()

    if not metrics or not run_summary:
        st.info("No metrics found. Run `python scripts/train.py` first to generate `results/metrics.json`.")
    else:
        # Metric cards
        st.subheader("Evaluation Metrics (Test Set)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%")
        mc2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        mc3.metric("Recall",    f"{metrics['recall']*100:.2f}%")
        mc4.metric("F1 Score",  f"{metrics['f1_score']*100:.2f}%")

        st.markdown("---")

        # Radar / bar chart of metrics
        col_x, col_y = st.columns(2)

        with col_x:
            st.subheader("Metrics Bar Chart")
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
            metric_vals  = [metrics["accuracy"], metrics["precision"],
                            metrics["recall"],   metrics["f1_score"]]
            colors = ["#3498DB", "#9B59B6", "#E67E22", "#2ECC71"]
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(metric_names, metric_vals, color=colors, edgecolor="white", linewidth=1.5)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score")
            ax.set_title("Model Evaluation Metrics")
            for bar, val in zip(bars, metric_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                        f"{val*100:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

        # Training loss curve (if history available)
        with col_y:
            history = run_summary.get("history", [])
            if history:
                st.subheader("Training vs Validation Loss")
                hist_df = pd.DataFrame(history)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(hist_df["epoch"], hist_df["train_loss"], marker="o",
                        color="#3498DB", label="Train Loss", linewidth=2)
                ax.plot(hist_df["epoch"], hist_df["val_loss"],   marker="s",
                        color="#E74C3C", label="Val Loss",   linewidth=2)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Loss per Epoch")
                ax.legend()
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                plt.close(fig)

        # F1 and Accuracy per epoch
        history = run_summary.get("history", [])
        if history:
            st.markdown("---")
            st.subheader("Accuracy & F1 Score per Epoch")
            hist_df = pd.DataFrame(history)
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.plot(hist_df["epoch"], hist_df["accuracy"], marker="o",
                    color="#2ECC71", label="Accuracy", linewidth=2)
            ax.plot(hist_df["epoch"], hist_df["f1"],       marker="s",
                    color="#9B59B6", label="F1 Score", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.05)
            ax.set_title("Validation Metrics per Epoch")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

        # Hyperparameters
        st.markdown("---")
        st.subheader("Training Hyperparameters")
        hp = run_summary.get("hyperparameters", {})
        hp_df = pd.DataFrame(hp.items(), columns=["Parameter", "Value"])
        st.table(hp_df)

        if "training_time_seconds" in run_summary:
            secs = run_summary["training_time_seconds"]
            st.caption(f"Total training time: {secs/60:.1f} minutes ({secs:.0f} s)")