import numpy as np
import pytest

torch = pytest.importorskip("torch")

from author_id.neural import LSTMAuthorClassifier, WordIndexer, glove_matrix


def test_indexer_ignores_validation_vocabulary():
    indexer = WordIndexer(max_words=12, max_length=4).fit(["cat dog", "cat mouse"])
    tokens, lengths = indexer.transform(["unseen cat", ""])
    assert tokens.shape == (2, 4)
    assert tokens[0, 0].item() == 1
    assert lengths.tolist() == [2, 1]
    assert "unseen" not in indexer.vocab


def test_glove_reads_only_known_words(tmp_path):
    file = tmp_path / "glove.txt"
    file.write_text("cat 0.1 0.2\ndog 0.3 0.4\nother 0.8 0.8\n")
    weights, count = glove_matrix(file, {"<pad>": 0, "<unk>": 1, "cat": 2}, dimension=2)
    assert weights.shape == (3, 2)
    assert count == 1
    assert np.allclose(weights[2], [0.1, 0.2])


def test_lstm_training_and_safe_reload(demo_train, tmp_path):
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    clf = LSTMAuthorClassifier(epochs=1, batch_size=32, max_words=400,
                               max_length=24, embedding_dim=12, hidden_size=12)
    clf.fit(demo_train.text, demo_train.author)
    p = clf.predict_proba(demo_train.text.iloc[:5])
    assert p.shape == (5, 3)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    checkpoint = tmp_path / "test.pt"
    clf.save(checkpoint)
    reloaded = LSTMAuthorClassifier.load(checkpoint)
    assert np.allclose(reloaded.predict_proba(demo_train.text.iloc[:5]), p, atol=1e-6)
