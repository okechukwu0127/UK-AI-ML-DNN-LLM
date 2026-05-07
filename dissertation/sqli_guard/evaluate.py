"""
Evaluate trained models on the test set.
Supports sklearn (joblib) and transformer directories.
"""

import argparse
import math
import os

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import DATA_DIR, MAX_LENGTH
from data_utils import load_text_and_labels


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


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


def batch_predict_transformer(texts, tokenizer, model, max_length=MAX_LENGTH, batch_size=16):
    probs = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def evaluate_transformer(model_dir, test_csv, max_rows=None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    texts, labels = load_text_and_labels(test_csv, max_rows=max_rows)
    proba = batch_predict_transformer(texts, tokenizer, model)
    preds = (proba >= 0.5).astype(int)

    print(classification_report(labels, preds))
    print("ROC-AUC:", round(roc_auc_score(labels, proba), 4))
    print("PR-AUC:", round(average_precision_score(labels, proba), 4))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default=os.path.join(DATA_DIR, "test.csv"))
    parser.add_argument("--model_type", choices=["sklearn", "transformer"], default="sklearn")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    if args.model_type == "sklearn":
        evaluate_sklearn(args.model_path, args.test_csv, max_rows=args.max_rows)
    else:
        evaluate_transformer(args.model_path, args.test_csv, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
