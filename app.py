import os
import pickle

import streamlit as st

from preprocess import clean_text

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error("model.pkl / vectorizer.pkl not found in the repo.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


model, vectorizer = load_artifacts()

st.title("📩 SMS Spam Detector")
st.write("AI Agent for SMS Spam Detection and Intelligent Message Filtering")

message = st.text_area("Enter an SMS message to check:", height=120,
                        placeholder="e.g. Congratulations! You've won a free prize, click here to claim now.")

col1, col2 = st.columns([1, 1])
with col1:
    check_clicked = st.button("Check Message", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if check_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        spam_idx = classes.index("spam")
        spam_prob = proba[spam_idx]

        if prediction == "spam":
            st.error(f"🚨 This message looks like **SPAM** ({spam_prob*100:.1f}% confidence)")
        else:
            st.success(f"✅ This message looks like **HAM (not spam)** ({(1-spam_prob)*100:.1f}% confidence)")

        with st.expander("See details"):
            st.write(f"Cleaned text used for prediction: `{cleaned}`")
            st.write({cls: f"{p*100:.2f}%" for cls, p in zip(classes, proba)})

st.divider()
st.caption("Model: TF-IDF + Logistic Regression, trained on the UCI SMS Spam Collection dataset.")
