# Dataset

Download the original **Spooky Author Identification** `train.csv` and `test.csv` from:

https://www.kaggle.com/competitions/spooky-author-identification/data

Place them in `data/raw/`. The required schemas are:

- `train.csv`: `id`, `text`, `author` (EAP, HPL, MWS)
- `test.csv`: `id`, `text`

No competition data, test labels, or third-party embedding files are redistributed.

`data/demo/` contains fictional, generated text labeled with the same three codes **only to check that the software runs**. It has no meaning as a literary authorship benchmark. Rebuild it with `python scripts/generate_demo.py`.
