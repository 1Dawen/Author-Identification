# Spooky Author Identification

**Natural Language Processing | TF-IDF | XGBoost | PyTorch LSTM | Model Evaluation**

Classifying short literary excerpts as written by **Edgar Allan Poe (EAP)**, **H. P. Lovecraft (HPL)**, or **Mary Wollstonecraft Shelley (MWS)**. This project revisits an academic author-identification comparison and rebuilds the experiments as reusable Python pipelines with consistent validation, saved models and tests.

The implementation compares three approaches:

| Approach | Text representation | Model |
| --- | --- | --- |
| Linear baseline | Word TF-IDF (1–2 grams) | Logistic Regression |
| Feature-based classifier | Word/character TF-IDF + SVD, stylometric statistics, out-of-fold Naive Bayes probabilities | XGBoost |
| Sequence classifier | Train-only token vocabulary, learned embeddings or optional pretrained GloVe | PyTorch LSTM |

All three models are evaluated on the **same stratified 80/20 holdout**, using **multiclass log loss and accuracy**. Vectorizers, vocabulary, embeddings and Naive Bayes feature models are fitted only using the training partition. The evaluation does not use Kaggle test data to fit features.

> **Results status:** The original competition data and GloVe files were not available in the archived project. No new official-dataset accuracy or leaderboard result is claimed here. The included fictional texts verify that the software runs, not that it can identify authors in real literature. See [technical notes](docs/technical_notes.md).

## Run on the competition dataset

1. Install **Python 3.10+** and download `train.csv` and `test.csv` from the [Kaggle Spooky Author Identification competition](https://www.kaggle.com/competitions/spooky-author-identification/data).
2. Place the files at `data/raw/train.csv` and `data/raw/test.csv`.
3. Install the dependencies and run an evaluation.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Compare a linear baseline with XGBoost.
python train.py --train data/raw/train.csv --test data/raw/test.csv
```

To include the **LSTM**, install the optional PyTorch dependency and pass its model name:

```bash
pip install -r requirements-lstm.txt
python train.py --train data/raw/train.csv --test data/raw/test.csv \
  --models baseline,xgboost,lstm --epochs 10
```

The LSTM can initialize embeddings from local **GloVe 6B 100d** vectors, matching the broad approach used in the original experiment. Download that file separately and add `--glove /path/to/glove.6B.100d.txt` when enabling the LSTM. Without it, embeddings are learned from scratch. GloVe weights are not included or automatically downloaded.

On Windows PowerShell, enter each command on a single line rather than using `\` for continuation.

After validation, `train.py` picks the **lowest-log-loss** model. With `--test`, it retrains that model on all labeled rows, creates `outputs/submission.csv` (columns `id`, `EAP`, `HPL`, `MWS`), and saves a fitted model. Files under `outputs/` are excluded from Git.

To make predictions later with a previously fitted model:

```bash
python predict.py --test data/raw/test.csv --model outputs/model.joblib
# Or, if the LSTM was selected:
python predict.py --test data/raw/test.csv --model outputs/model_lstm.pt
```

**Only load `.joblib` files that you created or trust:** the serialization format can execute code when unpickled.

## Quick demonstration without competition data

The repository contains a small set of **generated, fictional, class-labeled phrases**. The labels are artificial. This example checks data preparation, model fitting, inference and submission formatting without downloading anything.

```bash
python scripts/generate_demo.py
python train.py --train data/demo/train.csv --test data/demo/test.csv \
  --models baseline,xgboost --estimators 25 \
  --word-components 8 --char-components 4 --inner-folds 2
```

To try the demo LSTM, first install `requirements-lstm.txt`, then add `lstm` to `--models` and `--epochs 1`. Synthetic demo metrics are **not** benchmarks and should never be quoted as evidence of authorship prediction quality.

## Notebook and testing

The [cleaned walkthrough notebook](notebooks/spooky_author_identification.ipynb) covers input checks, class distribution, handcrafted text statistics, model evaluation, and Kaggle-format predictions. It automatically uses the fictional demo if the official files are not available, clearly indicating the source.

```bash
pip install -r requirements-notebook.txt
jupyter lab
```

Automated tests use the fictional demo and exercise the two model families, data validation, class order, leakage checks, and checkpoint loading:

```bash
pip install -r requirements-dev.txt
pip install -r requirements-lstm.txt  # Optional, to test the PyTorch LSTM too
python -m pytest -q
```

## Project structure

```text
spooky-author-identification/
├── README.md
├── train.py                     # Compare models, select by validation log loss, refit
├── predict.py                   # Inference from saved models
├── author_id/
│   ├── data.py                  # Validate schema and compute shared metrics
│   ├── features.py              # Stylometry, TF-IDF/SVD, honest NB features
│   ├── baseline.py              # Logistic Regression baseline
│   ├── classical.py             # XGBoost
│   └── neural.py                # PyTorch LSTM, optional GloVe initialization
├── notebooks/
│   └── spooky_author_identification.ipynb
├── scripts/generate_demo.py     # Reproducible fictional training/test data
├── data/demo/                   # Generated demo only
├── data/raw/                    # Place official Kaggle CSVs here (ignored by Git)
├── tests/
├── docs/technical_notes.md
├── results/README.md
├── requirements*.txt
├── .gitignore
└── .github/workflows/tests.yml
```

## Reproducibility and limitations

- **Dataset:** The public repository does not redistribute competition texts, labels for the competition test set, or third-party embeddings. Users must obtain the official data themselves.
- **Preprocessing:** The original notebooks fitted some text preprocessing on material outside the training fold and did not independently validate the later stacked model. Those experimental practices have been replaced or omitted.
- **Architecture:** The LSTM has been reimplemented in PyTorch rather than reproducing the original Keras checkpoint, model schedule or saved metrics. Original saved results **cannot be assumed to carry over**.
- **Evaluation:** All models use the same single held-out split and standard metrics, but choosing the best model on that holdout can make its score optimistic. The competition leaderboard would provide a separate measure.

This is a cleaned reimplementation of an **academic collaborative project**, based on available experimental notebooks. It is not a claim that the original notebooks or experiments were authored or conducted by one person. Historical outputs and methodological differences are detailed in the [technical notes](docs/technical_notes.md).
