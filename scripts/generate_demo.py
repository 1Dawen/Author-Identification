"""Create fictional labeled excerpts for validating the ML code offline.

Author IDs here are arbitrary teaching labels, not authentic author attribution.
"""
from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

SEEDS = {
    "EAP": ["The candle flickered beside the quiet corridor", "A shadow crossed the empty chamber", "The clock echoed through the silent hall"],
    "HPL": ["Beyond the valley stood an unfamiliar stone", "The distant horizon concealed a strange formation", "An ancient map revealed the northern mountain"],
    "MWS": ["The traveller wrote a letter beside the window", "A curious visitor arrived before the rain", "The experiment continued near the garden"],
}
SUFFIXES = ["before dawn.", "in the evening.", "without warning.", "during the storm.",
            "as the river flowed.", "while everyone listened.", "near the old road.",
            "beneath the dark sky.", "after a long silence.", "that morning."]


def generate(path: Path = Path("data/demo")):
    rng = random.Random(40)
    rows = []
    for label, starts in SEEDS.items():
        for i in range(45):
            sentence = starts[i % len(starts)] + " " + SUFFIXES[i % len(SUFFIXES)]
            rows.append({"id": f"demo_{label}_{i:03d}", "text": sentence, "author": label})
    rng.shuffle(rows)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "train.csv", index=False)
    examples = []
    for label, starts in SEEDS.items():
        for i in range(4):
            examples.append({"id": f"unseen_{label}_{i:03d}",
                             "text": starts[(i+1) % len(starts)] + " " + SUFFIXES[(i+3) % len(SUFFIXES)]})
    pd.DataFrame(examples).to_csv(path / "test.csv", index=False)
    print(f"Wrote fictional demo CSVs to {path}")


if __name__ == "__main__":
    generate()
