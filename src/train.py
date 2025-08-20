import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB

from .config import (
    MODEL_PATH,
    TRAIN_CSV,
    TEST_CSV,
    VECTORIZER_PATH,
    ensure_directories_exist,
)
from .data import download_dataset, load_processed, load_raw_dataframe, split_and_save
from .preprocess import transform_text
from .utils import save_pickle


def _prepare_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories_exist()
    download_dataset()
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
        df = load_raw_dataframe()
        split_and_save(df)
    train_df, test_df = load_processed()
    return train_df, test_df


def _preprocess_column(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
    return df[text_col].astype(str).apply(transform_text)


def train_and_evaluate() -> dict:
    train_df, test_df = _prepare_datasets()

    # Map labels to binary
    label_map = {"ham": 0, "spam": 1}
    y_train = train_df["label"].map(label_map).values
    y_test = test_df["label"].map(label_map).values

    # Preprocess
    x_train_clean = _preprocess_column(train_df)
    x_test_clean = _preprocess_column(test_df)

    # Vectorize
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\\b\\w+\\b",
        min_df=2,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(x_train_clean)
    X_test = vectorizer.transform(x_test_clean)

    # Model
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "num_train": int(X_train.shape[0]),
        "num_test": int(X_test.shape[0]),
        "vocab_size": int(len(vectorizer.vocabulary_)),
    }

    # Persist artifacts
    save_pickle(vectorizer, VECTORIZER_PATH)
    save_pickle(model, MODEL_PATH)

    # Save metrics
    artifacts_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    results = train_and_evaluate()
    print(json.dumps(results, indent=2))


