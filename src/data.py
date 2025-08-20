import io
import os
import zipfile
from typing import Tuple

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .config import DATA_URL, RAW_TXT, RAW_ZIP, TRAIN_CSV, TEST_CSV


def download_dataset() -> None:
    if os.path.exists(RAW_TXT):
        return
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    with open(RAW_ZIP, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(os.path.dirname(RAW_ZIP))


def load_raw_dataframe() -> pd.DataFrame:
    # The raw file is tab-separated with columns: label<TAB>text
    df = pd.read_csv(RAW_TXT, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    return df


def split_and_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[str, str]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    return TRAIN_CSV, TEST_CSV


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, test_df


