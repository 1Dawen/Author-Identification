# Technical notes and limitations

## Source and reconstruction

The original project used two separate notebooks, `SAI_LSTM.ipynb` and `SAI_XGBOOST.ipynb`, from an academic comparison of methods for the Kaggle **Spooky Author Identification** challenge. It was a collaborative academic project. This repository is a **cleaned, standalone implementation** informed by those notebooks, not a byte-for-byte reproduction of the original runs. The original notebooks, reports and other contributors' files are deliberately not republished.

Original notebook outputs recorded **0.835 holdout accuracy** for an LSTM and approximately **0.35352 validation multiclass log loss** for the first XGBoost model. **Neither result has been independently reproduced with this implementation.** They are not directly comparable because they are different metrics, and the original XGBoost preprocessing is not an unbiased held-out evaluation.

## Refactoring decisions

- **One held-out set:** Stratified 80/20 split, seed 40; no corpus-derived preprocessing is fitted using validation text.
- **Metric:** Standard *per-example* multiclass log loss (sklearn `log_loss`) and accuracy, with fixed label order EAP, HPL, MWS. The original custom LSTM loss divided by all one-hot elements (three per sample), understating standard log loss by a factor of three.
- **XGBoost:** Text statistics plus word/character TF-IDF reduced with TruncatedSVD. The optional Naive Bayes class-probability features are generated **out-of-fold** on the training partition and inferred from a full-training NB pipeline on held-out text. The original notebooks had fitted text vectorizers on train + Kaggle test and reused full-dataset representations in validation.
- **LSTM:** A small PyTorch LSTM with train-only vocabulary, cross-entropy training, dropout, and optional GloVe 100-dimensional initialization. Original Keras implementation (including its extended training schedule) is replaced with a maintainable alternative, not a faithful checkpoint/architecture reproduction. When GloVe is not supplied, embeddings are trained from scratch.
- **Second-stage Markov/XGBoost stacking:** Omitted because the archived experiment fitted a second-stage model on the first-stage validation data without a separate holdout for its own evaluation. This would not substantiate a generalization claim.
- **Selection and refit:** Training compares models on the same holdout, ranks by log loss, and, if `--test` is provided, refits the selected model on all competition training data before generating Kaggle predictions. Selection on a holdout makes the reported best value somewhat optimistic, especially if many configurations are compared; it is not an external-test guarantee.

## Practical limitations

- The original Kaggle competition CSVs and GloVe 6B weights were not included in the provided archive; live competition benchmark scores cannot be reproduced here.
- The demonstration sentences are invented and deliberately class-separable. Demo validation is **only a software smoke test**.
- Pretrained GloVe is optional. To use it, obtain `glove.6B.100d.txt` separately from an authorized source and supply `--glove path/to/glove.6B.100d.txt` with `--models lstm`.
- This compares different model families under one consistent holdout, but training budgets, regularization and hyperparameters are not necessarily optimal or matched. Do not infer a universal model ranking from one split.
- Model files serialized with joblib must only be loaded from a trusted source; they use Python pickle. The PyTorch model loads a state dictionary with `weights_only=True`.
