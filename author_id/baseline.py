"""Leakage-safe TF-IDF + Logistic Regression comparison baseline."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import LABELS


class LinearAuthorClassifier:
    def fit(self, texts, labels):
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            LogisticRegression(max_iter=1000, random_state=40),
        )
        self.model.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts):
        from numpy import column_stack
        p = self.model.predict_proba(list(texts))
        return column_stack([p[:, list(self.model.classes_).index(label)] for label in LABELS])
