import string
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from .utils import ensure_nltk_resources


_stemmer = PorterStemmer()
_stop_words = None


def _get_stopwords() -> Iterable[str]:
    global _stop_words
    if _stop_words is None:
        ensure_nltk_resources()
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def transform_text(text: str) -> str:
    ensure_nltk_resources()
    text = text.lower().strip()
    tokens = nltk.word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        if token.isalnum():
            cleaned_tokens.append(token)

    tokens = []
    for token in cleaned_tokens:
        if token not in _get_stopwords() and token not in string.punctuation:
            tokens.append(token)

    stemmed_tokens = [_stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


