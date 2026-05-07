"""
SQL Injection Detection - Single File Pipeline

Style references:
- /Users/oeze/Documents/wlv/7CS028/workshop1/task_1.py
- /Users/oeze/Documents/wlv/7CS028/class_2/GD_Adam.py
- /Users/oeze/Documents/wlv/7CS033/Okechukwu Eze - 2504607 - 7CS033/mergeJupyterFiles.py

What this file does
- Split the big CSV into train/val/test (streaming)
- Train baseline models (SGD + HashingVectorizer)
- Train transformer model (HuggingFace)
- Evaluate models on test set
- Provide a lightweight predict() helper for middleware
"""

# ---- standard library ----
import argparse
import csv
import json
import math
import os
import random

# ---- third party ----
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# transformer (optional)
try:
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    import torch
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ------------------------------------------------------------
# GLOBAL SETTINGS (keep simple like your class scripts)
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "rbsqli_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# dataset columns (from your screenshot)
TEXT_COLUMNS = [
    "sql_query",
    "sql_command",
    "target_table",
    "selected_columns",
    "comparison_operator",
    "logical_operator",
    "sql_comment_syntax",
    "injection_type",
]
LABEL_COLUMN = "vulnerability_status"
POS_LABEL = "Yes"

# baseline vectorizer settings
NGRAM_RANGE = (1, 2)
HASH_DIM = 2 ** 20

# transformer settings
TRANSFORMER_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256


# ------------------------------------------------------------
# DATA UTILITIES
# ------------------------------------------------------------

def normalize_text(value):
    """Clean a single field to a compact string."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()


def row_to_text(row):
    """Combine multiple SQL related fields into one training text."""
    parts = []
    for col in TEXT_COLUMNS:
        if col in row and row[col] is not None:
            parts.append(normalize_text(row[col]))
    return " | ".join([p for p in parts if p])


def label_to_int(value):
    """Binary label mapping (Yes -> 1, everything else -> 0)."""
    return 1 if str(value).strip() == POS_LABEL else 0


def iter_csv_chunks(csv_path, chunksize=50_000, usecols=None):
    """Yield dataset in chunks to keep memory low for the 2GB CSV."""
    usecols = usecols or (TEXT_COLUMNS + [LABEL_COLUMN])
    return pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols)


def split_dataset(csv_path, out_dir, train_ratio=0.8, val_ratio=0.1, limit_rows=None):
    """Split the CSV into train/val/test with streaming writes."""
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")

    header = TEXT_COLUMNS + [LABEL_COLUMN]

    # write headers once
    for p in [train_path, val_path, test_path]:
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    seen = 0
    for chunk in iter_csv_chunks(csv_path):
        for _, row in chunk.iterrows():
            if limit_rows is not None and seen >= limit_rows:
                return train_path, val_path, test_path

            r = random.random()
            if r < train_ratio:
                target_path = train_path
            elif r < train_ratio + val_ratio:
                target_path = val_path
            else:
                target_path = test_path

            with open(target_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([row.get(c, "") for c in header])

            seen += 1

    return train_path, val_path, test_path


# ------------------------------------------------------------
# BASELINE MODEL (SGD + HashingVectorizer)
# ------------------------------------------------------------

def iter_text_label_from_csv(csv_path, chunksize=25_000):
    """Stream texts and labels in chunks (baseline training)."""
    for chunk in iter_csv_chunks(csv_path, chunksize=chunksize):
        texts = []
        labels = []
        for _, row in chunk.iterrows():
            texts.append(row_to_text(row))
            labels.append(label_to_int(row.get(LABEL_COLUMN)))
        yield texts, np.array(labels)


def train_baseline_sgd(train_csv, val_csv, model_name="sgd_log"):
    """Train a streaming SGD model (logistic or linear SVM)."""
    if model_name == "sgd_svm":
        clf = SGDClassifier(loss="hinge", alpha=1e-5, max_iter=1, tol=None)
    else:
        clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1, tol=None)

    vectorizer = HashingVectorizer(
        n_features=HASH_DIM,
        ngram_range=NGRAM_RANGE,
        alternate_sign=False,
        norm="l2",
    )

    classes = np.array([0, 1])

    for i, (texts, labels) in enumerate(iter_text_label_from_csv(train_csv)):
        X = vectorizer.transform(texts)
        if i == 0:
            clf.partial_fit(X, labels, classes=classes)
        else:
            clf.partial_fit(X, labels)
        if i % 10 == 0:
            print(f"Trained {i} chunks")

    # quick validation
    y_true = []
    y_score = []
    for texts, labels in iter_text_label_from_csv(val_csv):
        Xv = vectorizer.transform(texts)
        scores = clf.decision_function(Xv)
        y_true.extend(labels.tolist())
        y_score.extend(scores.tolist())

    auc = roc_auc_score(y_true, y_score)
    print("Validation AUC:", round(auc, 4))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump({"vectorizer": vectorizer, "model": clf}, model_path)
    print("Saved model to", model_path)


# ------------------------------------------------------------
# TRANSFORMER MODEL (HuggingFace)
# ------------------------------------------------------------

def build_text_from_batch(batch, index):
    """Build a single text string from batched HF dataset."""
    parts = []
    for col in TEXT_COLUMNS:
        col_values = batch.get(col, [])
        if not col_values:
            continue
        val = col_values[index]
        if val is None:
            continue
        val = " ".join(str(val).split())
        if val:
            parts.append(val)
    return " | ".join(parts)


def label_to_int_local(value):
    return 1 if str(value).strip() == POS_LABEL else 0


def compute_metrics(eval_pred):
    """Simple metrics for transformer training."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    report = classification_report(labels, preds, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
    }


def train_transformer(train_csv, val_csv, model_name=TRANSFORMER_MODEL, epochs=2, batch_size=8, lr=2e-5):
    """Train a transformer on the split CSV files."""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers not installed. Install dependencies first.")

    output_dir = os.path.join(MODEL_DIR, "transformer")
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(
        "csv",
        data_files={"train": train_csv, "validation": val_csv},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        size = len(batch[LABEL_COLUMN])
        texts = [build_text_from_batch(batch, i) for i in range(size)]
        tokens = tokenizer(texts, padding=False, truncation=True, max_length=MAX_LENGTH)
        tokens["labels"] = [label_to_int_local(v) for v in batch[LABEL_COLUMN]]
        return tokens

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved transformer model to", output_dir)


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_text_and_labels(csv_path, max_rows=None):
    texts = []
    labels = []
    for chunk in iter_csv_chunks(csv_path):
        for _, row in chunk.iterrows():
            texts.append(row_to_text(row))
            labels.append(label_to_int(row.get(LABEL_COLUMN)))
            if max_rows is not None and len(texts) >= max_rows:
                return texts, labels
    return texts, labels


def evaluate_sklearn(model_path, test_csv, max_rows=None):
    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    texts, labels = load_text_and_labels(test_csv, max_rows=max_rows)
    X = vectorizer.transform(texts)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        proba = np.array([sigmoid(s) for s in scores])

    preds = (proba >= 0.5).astype(int)

    print(classification_report(labels, preds))
    print("ROC-AUC:", round(roc_auc_score(labels, proba), 4))
    print("PR-AUC:", round(average_precision_score(labels, proba), 4))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))


def evaluate_transformer(model_dir, test_csv, max_rows=None):
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers not installed. Install dependencies first.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    texts, labels = load_text_and_labels(test_csv, max_rows=max_rows)

    # batch inference (CPU safe)
    probs = []
    model.eval()
    for i in range(0, len(texts), 16):
        batch = texts[i : i + 16]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())

    proba = np.array(probs)
    preds = (proba >= 0.5).astype(int)

    print(classification_report(labels, preds))
    print("ROC-AUC:", round(roc_auc_score(labels, proba), 4))
    print("PR-AUC:", round(average_precision_score(labels, proba), 4))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))


# ------------------------------------------------------------
# SIMPLE PREDICT FUNCTION (for middleware)
# ------------------------------------------------------------

def predict_sklearn(model_path, text):
    """Load once per process in production; this is fine for demo."""
    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    X = vectorizer.transform([text])

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[:, 1][0])
    else:
        score = float(model.decision_function(X)[0])
        proba = sigmoid(score)

    return {"probability": proba, "is_sqli": proba >= 0.5}


# ------------------------------------------------------------
# MAIN CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["split", "train_sgd_log", "train_sgd_svm", "train_transformer", "eval_sklearn", "eval_transformer"])
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()

    train_csv = os.path.join(OUT_DIR, "train.csv")
    val_csv = os.path.join(OUT_DIR, "val.csv")
    test_csv = os.path.join(OUT_DIR, "test.csv")

    if args.task == "split":
        split_dataset(DATASET_PATH, OUT_DIR, limit_rows=args.limit_rows)
        print("Done splitting dataset")
        print("Train:", train_csv)
        print("Val:", val_csv)
        print("Test:", test_csv)

    elif args.task == "train_sgd_log":
        train_baseline_sgd(train_csv, val_csv, model_name="sgd_log")

    elif args.task == "train_sgd_svm":
        train_baseline_sgd(train_csv, val_csv, model_name="sgd_svm")

    elif args.task == "train_transformer":
        train_transformer(train_csv, val_csv, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    elif args.task == "eval_sklearn":
        if not args.model_path:
            raise ValueError("--model_path is required for eval_sklearn")
        evaluate_sklearn(args.model_path, test_csv)

    elif args.task == "eval_transformer":
        if not args.model_path:
            raise ValueError("--model_path is required for eval_transformer")
        evaluate_transformer(args.model_path, test_csv)


if __name__ == "__main__":
    main()
