# SQLi Detection (Clean, Commented, Single-File Core)

This version rewrites the pipeline with heavy comments and a single main file, following the style patterns from:
- `/Users/oeze/Documents/wlv/7CS028/workshop1/task_1.py`
- `/Users/oeze/Documents/wlv/7CS028/class_2/GD_Adam.py`
- `/Users/oeze/Documents/wlv/7CS033/Okechukwu Eze - 2504607 - 7CS033/mergeJupyterFiles.py`

The main file is `sqli_pipeline.py` and the FastAPI app is `fastapi_app.py`.

## Files

- `sqli_pipeline.py` main pipeline (split, train, eval, predict)
- `fastapi_app.py` FastAPI middleware app
- `requirements.txt` dependencies

## Environment

Your system `python3` is 3.14. Most ML libraries are not stable on 3.14 yet.
Use Python 3.10 or 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Split the dataset

```bash
python sqli_pipeline.py --task split
```

Quick test on a smaller subset:

```bash
python sqli_pipeline.py --task split --limit_rows 200000
```

## 2) Train baseline models

Logistic (streaming):

```bash
python sqli_pipeline.py --task train_sgd_log
```

Linear SVM (streaming):

```bash
python sqli_pipeline.py --task train_sgd_svm
```

## 3) Train transformer model

```bash
python sqli_pipeline.py --task train_transformer --epochs 2 --batch_size 8
```

## 4) Evaluate

Sklearn model:

```bash
python sqli_pipeline.py --task eval_sklearn --model_path models/sgd_log.joblib
```

Transformer model:

```bash
python sqli_pipeline.py --task eval_transformer --model_path models/transformer
```

## 5) Run FastAPI middleware

```bash
export SQLI_MODEL_PATH=/Users/oeze/Documents/wlv/dissertation/sqli_guard_v2/models/sgd_log.joblib
export SQLI_THRESHOLD=0.5
export SQLI_BLOCK=false

uvicorn fastapi_app:app --reload
```

Test request:

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"q":"SELECT * FROM users WHERE id=1 OR 1=1"}'
```

Headers will show the model flag:
- `X-SQLi-Detected`
- `X-SQLi-Score`

## Notes

- All code is in `sqli_pipeline.py` for easier reading and referencing.
- The FastAPI middleware uses the baseline model for speed.
- If you want to switch to transformer in middleware, I can add that.
