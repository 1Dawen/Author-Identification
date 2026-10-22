import numpy as np

from author_id.classical import XGBAuthorClassifier
from author_id.data import metrics_for


def test_xgboost_minimal_fit(demo_train):
    model = XGBAuthorClassifier(estimators=8, word_components=5,
                                char_components=3, inner_folds=2).fit(
                                    demo_train.text, demo_train.author)
    p = model.predict_proba(demo_train.text.iloc[:6])
    assert p.shape == (6, 3)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert np.isfinite(metrics_for(demo_train.author.iloc[:6], p)["log_loss"])
