from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def demo_train():
    return pd.read_csv(Path(__file__).resolve().parents[1] / "data/demo/train.csv")
