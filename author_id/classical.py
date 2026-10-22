"""XGBoost with stylometric, SVD and out-of-fold Naive Bayes features."""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from . import LABELS
from .features import TextFeatureBuilder


class XGBAuthorClassifier:
    def __init__(self, *, estimators=350, word_components=60, char_components=30,
                 inner_folds=5, seed=40):
        self.estimators = estimators
        self.word_components = word_components
        self.char_components = char_components
        self.inner_folds = inner_folds
        self.seed = seed

    def fit(self, texts, labels):
        labels = np.asarray(list(labels))
        self.features = TextFeatureBuilder(
            word_components=self.word_components,
            char_components=self.char_components,
            inner_folds=self.inner_folds, seed=self.seed,
        )
        x = self.features.fit_transform(list(texts), labels)
        y = np.asarray([LABELS.index(label) for label in labels], dtype=int)
        self.model = XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            n_estimators=self.estimators, max_depth=3, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            tree_method="hist", n_jobs=2, random_state=self.seed,
        )
        self.model.fit(x, y)
        return self

    def predict_proba(self, texts):
        if not hasattr(self, "model"):
            raise RuntimeError("Fit the classifier before prediction")
        p = self.model.predict_proba(self.features.transform(list(texts)))
        return np.asarray(p, dtype=float)
