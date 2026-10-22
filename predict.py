"""Load a locally trained model and make Kaggle-format probability predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from author_id.data import submission_frame, validate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument("--model", type=Path, required=True, help="Local .joblib or .pt model created with train.py")
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    args = parser.parse_args()
    raw = validate_frame(pd.read_csv(args.test), labeled=False)
    if args.model.suffix == ".pt":
        from author_id.neural import LSTMAuthorClassifier
        model = LSTMAuthorClassifier.load(args.model)
    elif args.model.suffix == ".joblib":
        # joblib uses pickle; load ONLY artifacts you created and trust.
        model = joblib.load(args.model)
    else:
        parser.error("Model file must end in .pt or .joblib")
    result = submission_frame(raw["id"], model.predict_proba(raw["text"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
