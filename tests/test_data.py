import numpy as np
import pandas as pd
import pytest

from author_id.data import holdout_split, metrics_for, submission_frame, validate_frame
from author_id import LABELS


def test_input_schema_and_invalid_text():
    frame = pd.DataFrame({"id": [1], "text": ["hello"], "author": ["EAP"]})
    assert len(validate_frame(frame, labeled=True)) == 1
    with pytest.raises(ValueError, match="Missing columns"):
        validate_frame(frame.drop(columns="text"), labeled=True)
    with pytest.raises(ValueError, match="nonempty"):
        validate_frame(frame.assign(text="  "), labeled=True)
    with pytest.raises(ValueError, match="unique"):
        validate_frame(pd.concat([frame, frame]), labeled=True)
    with pytest.raises(ValueError, match="EAP"):
        validate_frame(frame.assign(author="BAD"), labeled=True)


def test_holdout_stratification(demo_train):
    train, val = holdout_split(demo_train)
    assert len(train) + len(val) == len(demo_train)
    assert not set(train["id"]) & set(val["id"])
    assert set(train.author) == set(val.author) == set(LABELS)


def test_metrics_and_submission_order():
    p = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
    d = metrics_for(LABELS, p)
    assert d["accuracy"] == 1.0
    assert d["log_loss"] > 0
    out = submission_frame(["x", "y", "z"], p)
    assert out.columns.tolist() == ["id", *LABELS]
    assert np.allclose(out[list(LABELS)].sum(axis=1), 1)
    with pytest.raises(ValueError, match="sum to one"):
        metrics_for(LABELS, p * 0.4)
