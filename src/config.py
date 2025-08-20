import os


ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# UCI SMS Spam Collection dataset
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)

RAW_ZIP = os.path.join(RAW_DIR, "smsspamcollection.zip")
RAW_TXT = os.path.join(RAW_DIR, "SMSSpamCollection")

TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")

VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")


def ensure_directories_exist() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


