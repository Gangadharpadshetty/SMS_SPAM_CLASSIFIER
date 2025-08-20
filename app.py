import json
import os

import streamlit as st

from src.config import MODEL_PATH, VECTORIZER_PATH, ensure_directories_exist
from src.preprocess import transform_text
from src.utils import ensure_nltk_resources, load_pickle


@st.cache_resource(show_spinner=False)
def load_artifacts():
    # Prefer artifacts directory, fall back to project root for backward compatibility
    vectorizer_path_candidates = [VECTORIZER_PATH, os.path.join(os.getcwd(), "vectorizer.pkl")]
    model_path_candidates = [MODEL_PATH, os.path.join(os.getcwd(), "model.pkl")]

    vectorizer = None
    model = None
    for p in vectorizer_path_candidates:
        if os.path.exists(p):
            vectorizer = load_pickle(p)
            break
    for p in model_path_candidates:
        if os.path.exists(p):
            model = load_pickle(p)
            break
    return vectorizer, model


def main() -> None:
    ensure_directories_exist()
    ensure_nltk_resources()

    st.title("Email/SMS Spam Classifier")
    st.write("Type a message and click Predict.")

    # Sidebar actions
    with st.sidebar:
        st.header("Model")
        if st.button("Train / Re-train model", use_container_width=True):
            with st.spinner("Training model. This may take a minute the first time (downloads data)..."):
                from src.train import train_and_evaluate

                metrics = train_and_evaluate()
                load_artifacts.clear()
            st.success("Training complete.")
            st.json(metrics)

        # Show metrics if present
        metrics_path = os.path.join(os.path.dirname(MODEL_PATH), "metrics.json")
        if os.path.exists(metrics_path):
            st.caption("Latest evaluation metrics")
            with open(metrics_path, "r", encoding="utf-8") as f:
                st.json(json.load(f))

    vectorizer, model = load_artifacts()
    if vectorizer is None or model is None:
        st.warning("Model artifacts not found. Click 'Train / Re-train model' in the sidebar or run 'python cli_train.py' first.")
        return

    input_sms = st.text_area("Enter the message", height=140, placeholder="Free entry in 2 days...", help="Your SMS or email text")

    if st.button("Predict"):
        if not input_sms.strip():
            st.info("Please enter a message.")
            return
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        label = "Spam" if int(result) == 1 else "Not Spam"
        st.subheader(label)


if __name__ == "__main__":
    main()