import numpy as np

from author_id.features import TextFeatureBuilder, style_features
from author_id.baseline import LinearAuthorClassifier


def test_style_features_handle_empty_strings():
    x = style_features(["", "Hello, world!", "This one has words"])
    assert x.shape == (3, 8)
    assert np.isfinite(x).all()
    assert x[0].sum() == 0


def test_feature_builder_training_only_and_oof(demo_train):
    sample = demo_train.iloc[:90]
    builder = TextFeatureBuilder(word_components=5, char_components=3, inner_folds=2)
    training = builder.fit_transform(sample.text, sample.author)
    original_vocab = dict(builder.word_vec.vocabulary_)
    heldout = builder.transform(["notseenbefore unique invented tokens"])
    assert training.shape[0] == 90
    assert heldout.shape == (1, training.shape[1])
    assert np.isfinite(training).all() and np.isfinite(heldout).all()
    assert builder.word_vec.vocabulary_ == original_vocab
    assert "notseenbefore" not in original_vocab
    # Inner held-out OOF probabilities are normalized.
    assert np.allclose(training[:, -3:].sum(axis=1), 1, atol=1e-5)


def test_baseline_order_and_probabilities(demo_train):
    clf = LinearAuthorClassifier().fit(demo_train.text, demo_train.author)
    p = clf.predict_proba(demo_train.text.iloc[:3])
    assert p.shape == (3, 3)
    assert np.allclose(p.sum(axis=1), 1)
