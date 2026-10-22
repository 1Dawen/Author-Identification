"""Leakage-safe engineered features for an XGBoost authorship classifier."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from . import LABELS


_STOPS = set(ENGLISH_STOP_WORDS)
_TOKENS = re.compile(r"\b\w+\b", re.UNICODE)


def style_features(texts) -> np.ndarray:
    """Unsupervised per-document descriptors (do not learn from a dataset)."""
    rows = []
    for value in texts:
        text = str(value)
        tokens = _TOKENS.findall(text.lower())
        words = max(1, len(tokens))
        chars = max(1, len(text))
        rows.append([
            float(len(text)),
            float(len(tokens)),
            len(set(tokens)) / words,
            sum(len(t) for t in tokens) / words,
            sum(c in string.punctuation for c in text) / chars,
            sum(c.isupper() for c in text) / chars,
            sum(t in _STOPS for t in tokens) / words,
            float(text.count("!") + text.count("?")),
        ])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 8)


def _count_nb():
    return make_pipeline(CountVectorizer(ngram_range=(1, 2), max_features=15000),
                         MultinomialNB(alpha=0.7))


def _aligned_predictions(fitted, texts) -> np.ndarray:
    """Enforce fixed competition class order independent of sklearn's fitted order."""
    p = fitted.predict_proba(list(texts))
    classes = fitted.classes_ if hasattr(fitted, "classes_") else fitted.steps[-1][1].classes_
    return np.column_stack([p[:, list(classes).index(label)] for label in LABELS])


@dataclass
class TextFeatureBuilder:
    """Fit a text representation on training samples only.

    Label-derived Naive Bayes features for *training* are out-of-fold;
    held-out rows are predicted with a separate full-training NB model.
    """

    word_components: int = 60
    char_components: int = 30
    inner_folds: int = 5
    seed: int = 40

    def _fit_svd(self, texts, *, kind: str, components: int):
        vectorizer = (TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                      max_features=25000, sublinear_tf=True)
                      if kind == "word" else
                      TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                      min_df=1, max_features=30000, sublinear_tf=True))
        matrix = vectorizer.fit_transform(texts)
        n = min(components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if n < 1:
            return vectorizer, None, np.empty((len(texts), 0), dtype=np.float32)
        svd = TruncatedSVD(n_components=n, random_state=self.seed)
        values = svd.fit_transform(matrix).astype(np.float32)
        return vectorizer, svd, values

    @staticmethod
    def _project(texts, vectorizer, svd):
        if svd is None:
            return np.empty((len(texts), 0), dtype=np.float32)
        return svd.transform(vectorizer.transform(texts)).astype(np.float32)

    def fit_transform(self, texts, labels):
        """Train-time matrix with honest OOF class-probability features."""
        texts, labels = list(texts), np.asarray(list(labels))
        if len(texts) != len(labels):
            raise ValueError("Text and label lengths differ")
        if not set(labels).issubset(LABELS) or set(labels) != set(LABELS):
            raise ValueError("Training data must include all three authors")
        counts = [int(np.sum(labels == name)) for name in LABELS]
        if min(counts) < 2:
            raise ValueError("At least two training observations per class required")
        self.word_vec, self.word_svd, word = self._fit_svd(texts, kind="word", components=self.word_components)
        self.char_vec, self.char_svd, char = self._fit_svd(texts, kind="char", components=self.char_components)
        self.nb_model = _count_nb()
        folds = min(self.inner_folds, min(counts))
        if folds < 2:
            raise ValueError("At least two inner folds required")
        oof = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        for train_idx, validation_idx in cv.split(texts, labels):
            model = _count_nb()
            model.fit([texts[i] for i in train_idx], labels[train_idx])
            oof[validation_idx] = _aligned_predictions(model, [texts[i] for i in validation_idx])
        self.nb_model.fit(texts, labels)
        return np.column_stack([style_features(texts), word, char, oof]).astype(np.float32)

    def transform(self, texts):
        """Use exclusively fitted objects; do not learn from validation/test texts."""
        texts = list(texts)
        if not hasattr(self, "nb_model"):
            raise RuntimeError("TextFeatureBuilder must be fitted on training data")
        return np.column_stack([
            style_features(texts),
            self._project(texts, self.word_vec, self.word_svd),
            self._project(texts, self.char_vec, self.char_svd),
            _aligned_predictions(self.nb_model, texts),
        ]).astype(np.float32)
