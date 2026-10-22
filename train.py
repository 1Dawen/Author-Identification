"""Compare models on one untouched stratified holdout; optionally produce submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from author_id.baseline import LinearAuthorClassifier
from author_id.classical import XGBAuthorClassifier
from author_id.data import holdout_split, metrics_for, submission_frame, validate_frame


def build_model(name, args):
    if name == "baseline":
        return LinearAuthorClassifier()
    if name == "xgboost":
        return XGBAuthorClassifier(estimators=args.estimators,
                                   word_components=args.word_components,
                                   char_components=args.char_components,
                                   inner_folds=args.inner_folds, seed=args.seed)
    if name == "lstm":
        try:
            from author_id.neural import LSTMAuthorClassifier
        except ImportError as exc:
            raise RuntimeError("Install requirements-lstm.txt for the optional PyTorch model") from exc
        return LSTMAuthorClassifier(epochs=args.epochs, glove_path=args.glove,
                                    batch_size=args.batch_size, max_words=args.max_words,
                                    max_length=args.max_length, seed=args.seed)
    raise ValueError(f"Unknown model: {name}")


def run(args):
    source = validate_frame(pd.read_csv(args.train), labeled=True)
    training, validation = holdout_split(source, seed=args.seed)
    choices = list(dict.fromkeys(name.strip() for name in args.models.split(",") if name.strip()))
    if not choices or any(n not in {"baseline", "xgboost", "lstm"} for n in choices):
        raise ValueError("--models must contain baseline, xgboost and/or lstm")
    results = []
    for name in choices:
        print(f"Evaluating {name} on {len(training)} train / {len(validation)} validation rows", flush=True)
        model = build_model(name, args).fit(training["text"], training["author"])
        summary = metrics_for(validation["author"], model.predict_proba(validation["text"]))
        results.append({"model": name, **summary})
        print(f"  accuracy={summary['accuracy']:.4f}; log_loss={summary['log_loss']:.4f}", flush=True)

    report = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output / "validation_metrics.csv", index=False)
    metadata = {"split_seed": args.seed, "validation_fraction": 0.2,
                "selection_metric": "multiclass_log_loss", "rows": len(source),
                "dataset": "synthetic_demo" if "demo" in args.train.parts else "supplied_train_csv",
                "best_model": report.iloc[0]["model"]}
    (args.output / "run_info.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\n", report.to_string(index=False))
    print("Saved validation metrics to", args.output / "validation_metrics.csv")

    if args.test:
        unseen = validate_frame(pd.read_csv(args.test), labeled=False)
        chosen = report.iloc[0]["model"]
        print(f"Retraining {chosen} on all {len(source)} labeled rows", flush=True)
        model = build_model(chosen, args).fit(source["text"], source["author"])
        probs = model.predict_proba(unseen["text"])
        submission_frame(unseen["id"], probs).to_csv(args.output / "submission.csv", index=False)
        if chosen == "lstm":
            model.save(args.output / "model_lstm.pt")
        else:
            joblib.dump(model, args.output / "model.joblib")
        (args.output / "selected_model.txt").write_text(chosen + "\n", encoding="utf-8")
        print(f"Saved submission and fitted {chosen} model under {args.output}")
    if "demo" in args.train.parts:
        print("NOTE: Results from synthetic texts are software tests, not author recognition benchmarks.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--test", type=Path, default=None, help="Optional Kaggle test.csv; triggers full-data refit and submission")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--models", default="baseline,xgboost", help="Comma-separated: baseline,xgboost,lstm")
    parser.add_argument("--estimators", type=int, default=350)
    parser.add_argument("--word-components", type=int, default=60)
    parser.add_argument("--char-components", type=int, default=30)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-words", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--glove", type=Path, default=None, help="Optional local glove.6B.100d.txt (not redistributed)")
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()
    if not args.train.is_file():
        parser.error(f"Could not find {args.train}. Download the Spooky Author Identification Kaggle training CSV.")
    if args.test and not args.test.is_file():
        parser.error(f"Could not find {args.test}")
    run(args)


if __name__ == "__main__":
    main()
