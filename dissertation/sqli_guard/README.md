# SQL Injection Detection Models (Baselines + Transformer)

This folder contains a clean pipeline to train and evaluate multiple models on the `rbsqli_dataset.csv` dataset.
The goal is to compare classic ML baselines against a transformer model and plug the best one into FastAPI as middleware.

## Project Layout

- `config.py` central settings and column names
- `data_utils.py` streaming utilities for the 2GB CSV
- `split_dataset.py` creates `data/train.csv`, `data/val.csv`, `data/test.csv`
- `train_baselines.py` trains streaming baselines (SGD + HashingVectorizer)
- `train_transformer.py` trains a transformer model (HuggingFace)
- `evaluate.py` evaluates models on the test set
- `middleware.py` FastAPI middleware for inference
- `example_fastapi_app.py` demo app

## Environment

The installed `python3` in this machine is 3.14. Many ML libraries are not yet stable on 3.14.
Use Python 3.10 or 3.11 for this project.

Example (macOS):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Split the dataset (streaming)

```bash
python split_dataset.py --csv ../rbsqli_dataset.csv --out_dir data
```

For a quick test on a small subset:

```bash
python split_dataset.py --csv ../rbsqli_dataset.csv --out_dir data --limit_rows 200000
```

## 2) Train baseline models

Streaming Logistic Regression (SGD):

```bash
python train_baselines.py --model sgd_log
```

Streaming Linear SVM:

```bash
python train_baselines.py --model sgd_svm
```

Small/medium TF-IDF Logistic Regression:

```bash
python train_baselines.py --model tfidf_logreg --max_rows 300000
```

## 3) Train transformer model

```bash
python train_transformer.py --model_name distilbert-base-uncased --epochs 2 --batch_size 8
```

You can swap `distilbert-base-uncased` for a code/SQL model if you have one cached locally.

## 4) Evaluate

Sklearn:

```bash
python evaluate.py --model_type sklearn --model_path models/sgd_log.joblib
```

Transformer:

```bash
python evaluate.py --model_type transformer --model_path models/transformer
```

## 5) FastAPI middleware usage

Set environment variables and run:

```bash
export SQLI_MODEL_TYPE=sklearn
export SQLI_MODEL_PATH=/Users/oeze/Documents/wlv/dissertation/sqli_guard/models/sgd_log.joblib
export SQLI_THRESHOLD=0.5

uvicorn example_fastapi_app:app --reload
```

If you want the middleware to block suspicious requests instead of only flagging:

```python
app.add_middleware(SQLiDetectionMiddleware, block=True)
```

## Research Guidance (prove transformer advantage)

- Compare **precision/recall/F1** and **PR-AUC** across baselines vs transformer.
- Use a **held-out test set** that contains unseen tables, columns, and query patterns.
- Run **adversarial tests**: mutate queries (whitespace, case, comment symbols, UNION tricks).
- Add a **rule-based baseline** (regex or keyword rules) and compare false positives.
- Measure **latency** in middleware (p50, p95) and show tradeoffs in a table.
- Perform **ablation**: train transformer with and without extra columns (command/table/op fields) to show impact.

## Notes

- The pipeline uses `HashingVectorizer` for streaming to handle the 2GB CSV without loading all data.
- For transformer training, GPU helps a lot. On CPU, reduce batch size and epochs.
