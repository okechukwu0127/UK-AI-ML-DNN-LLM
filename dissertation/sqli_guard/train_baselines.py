"""
Train classical ML baselines for SQLi detection.
Uses HashingVectorizer + SGD for streaming big data.
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from config import DATA_DIR, MODEL_DIR, MAX_FEATURES, NGRAM_RANGE
from data_utils import iter_csv_chunks, row_to_text, label_to_int


def iter_text_label_from_csv(csv_path, chunksize=25_000):
    for chunk in iter_csv_chunks(csv_path, chunksize=chunksize):
        texts = []
        labels = []
        for _, row in chunk.iterrows():
            texts.append(row_to_text(row))
            labels.append(label_to_int(row.get("vulnerability_status")))
        yield texts, np.array(labels)


def train_sgd_stream(train_csv, val_csv, model_name="sgd_log", max_features=2**20):
    if model_name == "sgd_svm":
        clf = SGDClassifier(loss="hinge", alpha=1e-5, max_iter=1, tol=None)
    else:
        clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1, tol=None)

    vectorizer = HashingVectorizer(
        n_features=max_features,
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


def train_tfidf_logreg(train_csv, val_csv, max_rows=None):
    # small/medium dataset training (not for full 2GB)
    texts = []
    labels = []
    seen = 0
    for chunk in iter_csv_chunks(train_csv):
        for _, row in chunk.iterrows():
            texts.append(row_to_text(row))
            labels.append(label_to_int(row.get("vulnerability_status")))
            seen += 1
            if max_rows is not None and seen >= max_rows:
                break
        if max_rows is not None and seen >= max_rows:
            break

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X, labels)

    # validation
    val_texts = []
    val_labels = []
    for chunk in iter_csv_chunks(val_csv):
        for _, row in chunk.iterrows():
            val_texts.append(row_to_text(row))
            val_labels.append(label_to_int(row.get("vulnerability_status")))
            if max_rows is not None and len(val_texts) >= max_rows:
                break
        if max_rows is not None and len(val_texts) >= max_rows:
            break

    Xv = vectorizer.transform(val_texts)
    preds = clf.predict(Xv)
    print(classification_report(val_labels, preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "tfidf_logreg.joblib")
    joblib.dump({"vectorizer": vectorizer, "model": clf}, model_path)
    print("Saved model to", model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=os.path.join(DATA_DIR, "train.csv"))
    parser.add_argument("--val_csv", default=os.path.join(DATA_DIR, "val.csv"))
    parser.add_argument("--model", default="sgd_log", choices=["sgd_log", "sgd_svm", "tfidf_logreg"])
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    if args.model in ["sgd_log", "sgd_svm"]:
        train_sgd_stream(args.train_csv, args.val_csv, model_name=args.model)
    else:
        train_tfidf_logreg(args.train_csv, args.val_csv, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
