"""PyTorch LSTM text classifier with optional local GloVe initialization.

This retains the original research's LSTM/word-embedding approach, not
its TensorFlow training implementation or historical parameter checkpoint.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from . import LABELS

_TOKEN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(sentence):
    return _TOKEN.findall(str(sentence).lower())


class WordIndexer:
    def __init__(self, *, max_words=20000, max_length=80):
        self.max_words, self.max_length = int(max_words), int(max_length)

    def fit(self, texts):
        counts = Counter(token for text in texts for token in tokenize(text))
        self.vocab = {"<pad>": 0, "<unk>": 1}
        for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:self.max_words - 2]:
            self.vocab[word] = len(self.vocab)
        return self

    def transform(self, texts):
        if not hasattr(self, "vocab"):
            raise RuntimeError("Fit tokenizer on training text first")
        rows, lengths = [], []
        for text in texts:
            ids = [self.vocab.get(word, 1) for word in tokenize(text)][:self.max_length]
            if not ids:
                ids = [1]
            lengths.append(len(ids))
            rows.append(ids + [0] * (self.max_length - len(ids)))
        return (torch.tensor(rows, dtype=torch.long), torch.tensor(lengths, dtype=torch.long))


def glove_matrix(path: str | Path, vocab: dict, dimension: int = 100):
    """Stream embeddings and keep only the small fitted training vocabulary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GloVe vectors not found: {path}")
    rng = np.random.default_rng(40)
    matrix = rng.normal(0, 0.02, size=(len(vocab), dimension)).astype(np.float32)
    matrix[0] = 0
    found = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            word, sep, remainder = line.partition(" ")
            if not sep or word not in vocab:
                continue
            vector = np.fromstring(remainder, sep=" ", dtype=np.float32)
            if len(vector) != dimension:
                raise ValueError(f"GloVe dimensionality mismatch for '{word}'")
            matrix[vocab[word]] = vector
            found += 1
    return matrix, found


class LSTMNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim=100, hidden_size=96, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 64),
                                    nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, len(LABELS)))

    def forward(self, token_ids, lengths):
        embeddings = self.embedding(token_ids)
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(hidden[-1])


class LSTMAuthorClassifier:
    def __init__(self, *, epochs=10, embedding_dim=100, hidden_size=96,
                 batch_size=64, max_words=20000, max_length=80, seed=40,
                 glove_path=None, learning_rate=0.001, device="cpu"):
        self.config = dict(epochs=int(epochs), embedding_dim=int(embedding_dim),
                           hidden_size=int(hidden_size), batch_size=int(batch_size),
                           max_words=int(max_words), max_length=int(max_length),
                           seed=int(seed), learning_rate=float(learning_rate), device=device)
        self.glove_path = str(glove_path) if glove_path else None

    def fit(self, texts, labels):
        texts, labels = list(texts), list(labels)
        if set(labels) != set(LABELS):
            raise ValueError("Training data must include all three authors")
        if len(texts) != len(labels):
            raise ValueError("Text/label lengths differ")
        cfg = self.config
        torch.manual_seed(cfg["seed"])
        self.indexer = WordIndexer(max_words=cfg["max_words"], max_length=cfg["max_length"]).fit(texts)
        self.network = LSTMNetwork(len(self.indexer.vocab), cfg["embedding_dim"], cfg["hidden_size"])
        self.glove_matches = 0
        if self.glove_path:
            weights, self.glove_matches = glove_matrix(self.glove_path, self.indexer.vocab, cfg["embedding_dim"])
            with torch.no_grad():
                self.network.embedding.weight.copy_(torch.from_numpy(weights))
            self.network.embedding.weight.requires_grad_(False)  # Original model froze GloVe.
        self.network.to(cfg["device"])
        tensor, lengths = self.indexer.transform(texts)
        target = torch.tensor([LABELS.index(y) for y in labels], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(tensor, lengths, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                                             generator=torch.Generator().manual_seed(cfg["seed"]))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=cfg["learning_rate"])
        loss_fn = nn.CrossEntropyLoss()
        self.training_losses = []
        for _ in range(cfg["epochs"]):
            self.network.train()
            total = 0.0
            for tokens, length, labels_batch in loader:
                optimizer.zero_grad()
                prediction = self.network(tokens.to(cfg["device"]), length.to(cfg["device"]))
                loss = loss_fn(prediction, labels_batch.to(cfg["device"]))
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                total += loss.item() * len(labels_batch)
            self.training_losses.append(total / len(dataset))
        return self

    def predict_proba(self, texts):
        if not hasattr(self, "network"):
            raise RuntimeError("Fit the classifier before prediction")
        rows, lengths = self.indexer.transform(list(texts))
        self.network.eval()
        values = []
        bs = self.config["batch_size"]
        with torch.inference_mode():
            for i in range(0, len(rows), bs):
                result = self.network(rows[i:i+bs].to(self.config["device"]),
                                      lengths[i:i+bs].to(self.config["device"]))
                values.append(torch.softmax(result, dim=1).cpu().numpy())
        return np.concatenate(values, axis=0) if values else np.empty((0, len(LABELS)))

    def save(self, path):
        if not hasattr(self, "network"):
            raise RuntimeError("Cannot save an unfitted model")
        torch.save({"state_dict": self.network.state_dict(), "vocab": self.indexer.vocab,
                    "config": self.config, "glove_matches": self.glove_matches}, path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        # Only load local trusted checkpoints; weights_only limits pickle loading.
        state = torch.load(path, map_location=device, weights_only=True)
        config = dict(state["config"])
        config["device"] = device
        result = cls(**config)
        result.indexer = WordIndexer(max_words=config["max_words"], max_length=config["max_length"])
        result.indexer.vocab = state["vocab"]
        result.network = LSTMNetwork(len(result.indexer.vocab), config["embedding_dim"], config["hidden_size"])
        result.network.load_state_dict(state["state_dict"])
        result.network.to(device).eval()
        result.glove_matches = state.get("glove_matches", 0)
        return result
