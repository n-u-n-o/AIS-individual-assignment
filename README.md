# When Filling the Gaps Isn't Neutral: Auditing the Fairness Impact of Missing-Data Imputation

Individual assignment for *Artificial Intelligence and Society* (Master in Artificial Intelligence, Faculty of
Sciences, University of Porto), 2026/2027 — Nuno Dos Santos Moreira.

Does filling in missing values change how fair a classifier is? This project injects missing values into five
fairness benchmarks under four mechanisms (MCAR, MAR, MNAR and group-dependent missingness), all calibrated to remove
the same share of values (5%, 15% or 30%), fills them in with mean, KNN and MICE imputation, and compares every
condition with a complete-data baseline on the same train/test split, over 20 repetitions.

The report is in [`report/reportfinalIAS_revised.pdf`](report/reportfinalIAS_revised.pdf).

## Repository contents

| Path | Contents |
|---|---|
| `new_version4.ipynb` | The complete code: data loading, missingness generation, experiments, statistics, and every table and figure of the report (saved with its outputs, so it can be read without running it) |
| `requirements.txt` | Package versions used (Python 3.11 or newer; tested with Python 3.13) |
| `data/` | The five raw datasets (the notebook verifies their MD5 checksums) |
| `results/cache/` | Raw results of both experimental stages, one CSV per (stage, dataset, seed) |
| `report/` | The report (PDF) |

## How to reproduce

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace new_version4.ipynb
```

Or open it with `jupyter notebook new_version4.ipynb` and run all cells. The notebook writes every table of the report
(LaTeX and CSV) to `tables/` and every figure to `figures/`; `tables/numbers.tex` contains every number quoted in the
report's text.

- **With `results/cache/`** (as in this repository), the notebook reuses the cached experiment results and recomputes
  all statistics, tables and figures in under a minute.
- **From scratch:** delete `results/cache/` first. This repeats all 5,000 model fits: about 4.3 CPU-hours in total
  (mostly KNN imputation on Adult), parallelised over all CPU cores. All seeds are fixed, so a rerun in the same
  environment reproduces the cached results.
- No internet connection is needed: the data are read from `data/`. If a file is missing, the notebook downloads it
  from the mirror listed in its Section 1 and checks its MD5 checksum.

## Data sources

| File | Dataset | Source |
|---|---|---|
| `adult-all.csv` | Adult income (train + test, 48,842 rows) | UCI Machine Learning Repository (Kohavi, 1996) |
| `compas-scores-two-years.csv` | COMPAS two-year recidivism | ProPublica (Angwin et al., 2016) |
| `german.csv` | German Credit | UCI Machine Learning Repository, Statlog (Hofmann, 1994) |
| `credit-card-clients.csv` | Taiwanese credit-card default | UCI Machine Learning Repository (Yeh and Lien, 2009) |
| `law_school_clean.csv` | LSAC Law School bar-passage study | Wightman (1998), cleaned version from Le Quy et al. (2022) |

## Design in brief

- Five datasets, two sensitive attributes each; sensitive attributes are never model inputs. Data are split 70/30,
  stratified by label × primary group.
- Missing values are injected into four numeric features per dataset under MCAR, MAR, MNAR and group-dependent
  missingness, calibrated so that every mechanism removes exactly the target share of values.
- Mean, KNN and MICE imputers, the scaler and the encoder are fit on the training split only. Logistic regression is
  the audited model, histogram gradient boosting a robustness check.
- Stage 1 runs all rates with 10 seeds; Stage 2 adds 10 seeds at the selected rate (30%), giving 20 repetitions, each
  paired with a complete-data baseline on the same split.
- Paired changes are tested with Wilcoxon signed-rank and Friedman tests, with Benjamini–Hochberg correction.
