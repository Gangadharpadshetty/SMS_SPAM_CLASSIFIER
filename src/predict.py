from typing import Dict, Tuple

from .config import MODEL_PATH, VECTORIZER_PATH
from .preprocess import transform_text
from .utils import load_pickle


def load_model_and_vectorizer():
    vectorizer = load_pickle(VECTORIZER_PATH)
    model = load_pickle(MODEL_PATH)
    return vectorizer, model


def predict_label(text: str) -> int:
    vectorizer, model = load_model_and_vectorizer()
    transformed = transform_text(text)
    X = vectorizer.transform([transformed])
    return int(model.predict(X)[0])


def predict_proba(text: str) -> Tuple[int, float]:
    vectorizer, model = load_model_and_vectorizer()
    transformed = transform_text(text)
    X = vectorizer.transform([transformed])
    label = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][label])
    else:
        proba = 1.0
    return label, proba


