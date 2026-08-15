import os
import re
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="AI Message Security Dashboard",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CLEANED_DATA_PATH = PROJECT_ROOT / "sms_cleaned.csv"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

MAX_LENGTH = 128
LSTM_MAX_LEN = 100

# ----------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --primary-blue: #1B4F91;
    --accent-blue: #2E86AB;
    --danger-red: #E63946;
    --safe-green: #2A9D8F;
    --warn-amber: #F4A261;
}
.main { background-color: #F7F9FC; }
.kpi-card {
    background: white; border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06); border-left: 5px solid var(--accent-blue);
    text-align: center;
}
.kpi-value { font-size: 28px; font-weight: 700; color: var(--primary-blue); }
.kpi-label { font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.model-card {
    background: white; border-radius: 12px; padding: 16px 20px; margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 4px solid var(--accent-blue);
}
.status-spam { background:#FDECEC; color:var(--danger-red); padding:14px; border-radius:10px; font-weight:700; text-align:center; font-size:22px;}
.status-safe { background:#E8F6F3; color:var(--safe-green); padding:14px; border-radius:10px; font-weight:700; text-align:center; font-size:22px;}
.status-suspicious { background:#FEF3E4; color:var(--warn-amber); padding:14px; border-radius:10px; font-weight:700; text-align:center; font-size:22px;}
h1, h2, h3 { color: var(--primary-blue); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TEXT CLEANING (mirrors the notebook pipeline)
# ----------------------------------------------------------------------
def clean_message(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

SPAM_KEYWORDS = {
    "prize": ["win", "won", "winner", "prize", "lottery", "reward", "jackpot", "claim"],
    "free": ["free", "gift", "bonus", "voucher"],
    "urgent": ["urgent", "immediately", "now", "asap", "act now", "hurry", "limited time", "expire"],
    "financial": ["bank", "account", "credit", "loan", "cash", "money", "payment", "refund", "debit"],
    "otp": ["otp", "verification code", "password", "pin", "verify your account"],
    "promotional": ["offer", "discount", "deal", "sale", "subscribe", "buy now", "click here"],
}

def detect_spam_indicators(message):
    text = str(message)
    lower = text.lower()
    indicators = []
    labels = {
        "prize": "Prize/lottery-related keyword detected",
        "free": "Free-offer keyword detected",
        "urgent": "Urgency detected",
        "financial": "Financial/bank keyword detected",
        "otp": "OTP or account-verification request detected",
        "promotional": "Promotional language detected",
    }
    for category, words in SPAM_KEYWORDS.items():
        if any(w in lower for w in words):
            indicators.append(labels[category])
    if re.search(r"https?://|www\.", lower):
        indicators.append("Suspicious URL detected")
    if re.search(r"\b\d{10,}\b", text) or re.search(r"\b0\d{9,}\b", text):
        indicators.append("Phone number detected")
    if re.search(r"[\u00a3$\u20ac\u20b9]\s?\d", text):
        indicators.append("Currency amount detected")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.4 and len(letters) > 8:
        indicators.append("Excessive capital letters detected")
    if text.count("!") >= 2 or text.count("?") >= 2:
        indicators.append("Excessive punctuation detected")
    return indicators

def calculate_risk_score(model_results, indicators):
    spam_confidences = [r["confidence"] for r in model_results.values() if r["label"] == "spam"]
    ham_confidences = [r["confidence"] for r in model_results.values() if r["label"] == "ham"]
    n_models = len(model_results)
    spam_votes = len(spam_confidences)
    if spam_votes == 0:
        base_score = (1 - np.mean(ham_confidences)) * 40
    else:
        base_score = (spam_votes / n_models) * 60 + np.mean(spam_confidences) * 30
    indicator_bonus = min(len(indicators) * 4, 15)
    return round(float(min(base_score + indicator_bonus, 100)), 2)

def interpret_risk(score):
    if score <= 30:
        return "LOW RISK"
    elif score <= 60:
        return "MEDIUM RISK"
    elif score <= 80:
        return "HIGH RISK"
    return "VERY HIGH RISK"

def filter_message(final_label, overall_confidence, safe_threshold=0.75, spam_threshold=0.75):
    if final_label == "ham" and overall_confidence >= safe_threshold:
        return "SAFE"
    if final_label == "spam" and overall_confidence >= spam_threshold:
        return "SPAM"
    return "SUSPICIOUS"

def generate_recommendation(status):
    if status == "SPAM":
        return ("Avoid clicking any links in this message. Do not provide OTP, passwords, "
                "bank details, or personal information. Consider blocking the sender.")
    elif status == "SUSPICIOUS":
        return ("This message shows some suspicious characteristics. Verify the sender through "
                "an official channel before responding or clicking any links.")
    return "This message appears safe based on the trained models."

# ----------------------------------------------------------------------
# MODEL LOADING (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    bundle = {"error": None}
    try:
        import torch
        from transformers import (
            DistilBertTokenizerFast, DistilBertForSequenceClassification,
            BertTokenizerFast, BertForSequenceClassification,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        db_dir = MODELS_DIR / "distilbert"
        bert_dir = MODELS_DIR / "bert"
        bilstm_dir = MODELS_DIR / "bilstm"

        if db_dir.exists():
            bundle["distilbert_tokenizer"] = DistilBertTokenizerFast.from_pretrained(db_dir)
            bundle["distilbert_model"] = DistilBertForSequenceClassification.from_pretrained(db_dir).to(device)
            bundle["distilbert_model"].eval()
        if bert_dir.exists():
            bundle["bert_tokenizer"] = BertTokenizerFast.from_pretrained(bert_dir)
            bundle["bert_model"] = BertForSequenceClassification.from_pretrained(bert_dir).to(device)
            bundle["bert_model"].eval()
        if bilstm_dir.exists():
            import tensorflow as tf
            bundle["bilstm_model"] = tf.keras.models.load_model(bilstm_dir / "bilstm_model.h5")
            with open(bilstm_dir / "tokenizer.pkl", "rb") as f:
                bundle["bilstm_tokenizer"] = pickle.load(f)
            with open(bilstm_dir / "config.json") as f:
                bundle["bilstm_max_len"] = json.load(f).get("max_len", LSTM_MAX_LEN)

        bundle["device"] = device
    except Exception as e:
        bundle["error"] = str(e)
    return bundle

def predict_transformer(message, model, tokenizer, device):
    import torch
    inputs = tokenizer(message, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))
    return {"label": "spam" if pred_class == 1 else "ham", "confidence": float(probs[pred_class])}

def predict_bilstm(message, model, tokenizer, max_len):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0).flatten()[0])
    label = "spam" if prob >= 0.5 else "ham"
    confidence = prob if label == "spam" else 1 - prob
    return {"label": label, "confidence": confidence}

def analyze_message(message_text, bundle):
    if not message_text or not str(message_text).strip():
        return {"error": "Please enter a valid, non-empty SMS message."}

    clean_text = clean_message(message_text)
    model_results = {}

    if bundle.get("distilbert_model"):
        model_results["DistilBERT"] = predict_transformer(
            clean_text, bundle["distilbert_model"], bundle["distilbert_tokenizer"], bundle["device"])
    if bundle.get("bert_model"):
        model_results["BERT"] = predict_transformer(
            clean_text, bundle["bert_model"], bundle["bert_tokenizer"], bundle["device"])
    if bundle.get("bilstm_model"):
        model_results["BiLSTM"] = predict_bilstm(
            clean_text, bundle["bilstm_model"], bundle["bilstm_tokenizer"], bundle["bilstm_max_len"])

    if not model_results:
        return {"error": "No trained models were found. Please run the training notebook first."}

    spam_votes = sum(1 for r in model_results.values() if r["label"] == "spam")
    final_label = "spam" if spam_votes >= (len(model_results) / 2) else "ham"
    matching = [r["confidence"] for r in model_results.values() if r["label"] == final_label]
    overall_confidence = float(np.mean(matching)) if matching else float(np.mean([r["confidence"] for r in model_results.values()]))

    indicators = detect_spam_indicators(message_text)
    risk_score = calculate_risk_score(model_results, indicators)
    risk_level = interpret_risk(risk_score)
    status = filter_message(final_label, overall_confidence)
    recommendation = generate_recommendation(status)

    return {
        "model_results": model_results,
        "final_prediction": final_label.upper(),
        "overall_confidence": round(overall_confidence * 100, 2),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "status": status,
        "indicators": indicators,
        "recommendation": recommendation,
    }

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
@st.cache_data
def load_cleaned_data():
    if CLEANED_DATA_PATH.exists():
        return pd.read_csv(CLEANED_DATA_PATH)
    return pd.DataFrame(columns=["label", "message", "clean_message"])

@st.cache_data
def load_comparison():
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

data_df = load_cleaned_data()
comparison_df = load_comparison()
best_model_name = comparison_df.loc[comparison_df["F1-Score"].idxmax(), "Model"] if not comparison_df.empty else "N/A"

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.markdown("## \U0001F6E1\uFE0F AI Message Security")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Analyze SMS", "Bulk SMS Analysis", "Spam Dashboard", "Model Comparison", "Dataset Explorer", "About Project"],
)
st.sidebar.markdown("---")
if st.sidebar.button("\U0001F50D Analyze New Message"):
    page = "Analyze SMS"

# ----------------------------------------------------------------------
# HOME PAGE
# ----------------------------------------------------------------------
if page == "Home":
    st.title("AI SMS Spam Detection & Intelligent Message Filtering")
    st.subheader("Protect Your Messages with AI-Powered Spam Detection")

    total_msgs = len(data_df)
    spam_msgs = (data_df["label"] == "spam").sum() if "label" in data_df else 0
    spam_pct = round(spam_msgs / total_msgs * 100, 1) if total_msgs else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in zip(
        [c1, c2, c3, c4, c5],
        ["Total Messages", "Spam Messages", "Safe Messages", "Spam %", "Best Model"],
        [total_msgs, spam_msgs, total_msgs - spam_msgs, f"{spam_pct}%", best_model_name],
    ):
        col.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("### Project Overview")
    st.write(
        "This platform analyzes SMS messages using three AI models — **DistilBERT**, **BERT**, and "
        "**BiLSTM** — to detect spam, calculate a risk score, identify suspicious patterns, and "
        "provide intelligent filtering with clear security recommendations."
    )
    st.markdown("### Models Used")
    st.write("- DistilBERT (`distilbert-base-uncased`)\n- BERT (`bert-base-uncased`)\n- Bidirectional LSTM (TensorFlow/Keras)")

    if st.button("\U0001F680 Analyze SMS Message", type="primary"):
        st.info("Use the **Analyze SMS** page from the sidebar to analyze a message.")

# ----------------------------------------------------------------------
# ANALYZE SMS PAGE
# ----------------------------------------------------------------------
elif page == "Analyze SMS":
    st.title("\U0001F50D SMS Analysis")
    bundle = load_models()
    if bundle.get("error"):
        st.error(f"Model loading error: {bundle['error']}")

    message = st.text_area(
        "Enter SMS Message",
        placeholder="Congratulations! You have won a free prize. Click the link now.",
        height=120,
    )

    if st.button("Analyze Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message to analyze.")
        else:
            result = analyze_message(message, bundle)
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("### Model Predictions")
                cols = st.columns(len(result["model_results"]))
                for col, (name, r) in zip(cols, result["model_results"].items()):
                    col.markdown(
                        f'<div class="model-card"><b>{name}</b><br>'
                        f'Prediction: <b>{r["label"].upper()}</b><br>'
                        f'Confidence: <b>{r["confidence"]*100:.1f}%</b></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("### Final Prediction")
                st.write(f"**Final Prediction:** {result['final_prediction']}  |  **Overall Confidence:** {result['overall_confidence']}%")

                st.markdown("### Spam Risk Score")
                st.progress(int(result["risk_score"]))
                st.write(f"**{result['risk_score']}% — {result['risk_level']}**")

                st.markdown("### Intelligent Message Filtering")
                css_class = {"SPAM": "status-spam", "SAFE": "status-safe", "SUSPICIOUS": "status-suspicious"}[result["status"]]
                st.markdown(f'<div class="{css_class}">Message Status: {result["status"]}</div>', unsafe_allow_html=True)

                if result["indicators"]:
                    st.markdown("**Detected Indicators:**")
                    for ind in result["indicators"]:
                        st.write(f"- {ind}")
                else:
                    st.write("No suspicious indicators detected.")

                st.markdown("### Security Recommendation")
                st.info(result["recommendation"])

# ----------------------------------------------------------------------
# BULK SMS ANALYSIS PAGE
# ----------------------------------------------------------------------
elif page == "Bulk SMS Analysis":
    st.title("\U0001F4CB Bulk SMS Analysis")
    bundle = load_models()
    st.write("Upload a CSV file with a `message` column to analyze multiple messages at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the uploaded CSV: {e}")
            bulk_df = None

        if bulk_df is not None:
            msg_col = None
            for c in bulk_df.columns:
                if c.strip().lower() in ("message", "sms", "text"):
                    msg_col = c
                    break
            if msg_col is None:
                st.error("The uploaded CSV must contain a 'message' column.")
            else:
                with st.spinner("Analyzing messages..."):
                    rows = []
                    for msg in bulk_df[msg_col].astype(str):
                        res = analyze_message(msg, bundle)
                        if "error" in res:
                            rows.append({"Message": msg, "Prediction": "N/A", "Confidence": "N/A", "Risk": "N/A"})
                        else:
                            rows.append({
                                "Message": msg,
                                "Prediction": res["final_prediction"],
                                "Confidence": f"{res['overall_confidence']}%",
                                "Risk": f"{res['risk_score']}%",
                            })
                    results_table = pd.DataFrame(rows)

                st.dataframe(results_table, use_container_width=True)
                csv_bytes = results_table.to_csv(index=False).encode("utf-8")
                st.download_button("\U0001F4E5 Download Filtered Results", data=csv_bytes, file_name="filtered_results.csv", mime="text/csv")

# ----------------------------------------------------------------------
# SPAM DASHBOARD PAGE
# ----------------------------------------------------------------------
elif page == "Spam Dashboard":
    st.title("\U0001F4CA Spam Dashboard")

    if data_df.empty:
        st.warning("No dataset found. Please run the training notebook first.")
    else:
        total_msgs = len(data_df)
        spam_msgs = (data_df["label"] == "spam").sum()
        ham_msgs = total_msgs - spam_msgs
        spam_pct = round(spam_msgs / total_msgs * 100, 1)

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, value in zip(
            [c1, c2, c3, c4, c5],
            ["Total Messages", "Spam Messages", "Safe Messages", "Spam %", "Avg Risk Score"],
            [total_msgs, spam_msgs, ham_msgs, f"{spam_pct}%", "See Analyze SMS"],
        ):
            col.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots()
            sns.countplot(x=data_df["label"], ax=ax, palette=["#2E86AB", "#E63946"])
            ax.set_title("Spam vs Ham Distribution")
            st.pyplot(fig)
        with colB:
            fig2, ax2 = plt.subplots()
            lengths = data_df["message"].astype(str).apply(len)
            sns.histplot(lengths, bins=30, ax=ax2, color="#2E86AB")
            ax2.set_title("Message Length Distribution")
            st.pyplot(fig2)

# ----------------------------------------------------------------------
# MODEL COMPARISON PAGE
# ----------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("\U0001F9E0 Model Comparison")
    st.subheader("DistilBERT vs BERT vs BiLSTM")

    if comparison_df.empty:
        st.warning("No model comparison results found. Please run the training notebook first.")
    else:
        st.dataframe(comparison_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax)
        plt.xticks(rotation=0)
        st.pyplot(fig)
        st.success(f"\U0001F3C6 Best Performing Model: **{best_model_name}**")

# ----------------------------------------------------------------------
# DATASET EXPLORER PAGE
# ----------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.title("\U0001F4C2 Dataset Explorer")

    if data_df.empty:
        st.warning("No dataset found. Please run the training notebook first.")
    else:
        search = st.text_input("Search messages")
        label_filter = st.selectbox("Filter by label", ["All", "spam", "ham"])

        filtered = data_df.copy()
        if search:
            filtered = filtered[filtered["message"].astype(str).str.contains(search, case=False, na=False)]
        if label_filter != "All":
            filtered = filtered[filtered["label"] == label_filter]

        filtered = filtered.copy()
        filtered["message_length"] = filtered["message"].astype(str).apply(len)
        st.dataframe(filtered[["label", "message", "message_length"]], use_container_width=True)

# ----------------------------------------------------------------------
# ABOUT PROJECT PAGE
# ----------------------------------------------------------------------
elif page == "About Project":
    st.title("\u2139\uFE0F About This Project")
    st.markdown("### Project Title")
    st.write("AI Agent for SMS Spam Detection and Intelligent Message Filtering")

    st.markdown("### Technologies")
    st.write("Python, NLP, BERT, DistilBERT, BiLSTM, Transformers, TensorFlow/PyTorch, Streamlit, Google Colab, ngrok")

    st.markdown("### Dataset")
    st.write("UCI SMS Spam Collection / Kaggle SMS Spam Collection Dataset (`uciml/sms-spam-collection-dataset`)")

    report_path = PROJECT_ROOT / "project_report.md"
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button("\U0001F4C4 Download AI Spam Detection & Security Report", data=f, file_name="project_report.md")
