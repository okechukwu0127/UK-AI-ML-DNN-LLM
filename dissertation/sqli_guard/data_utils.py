"""
Utility helpers for reading and preparing SQLi dataset.
Designed to handle large CSV with chunked reading.
"""

import csv
import random
from os import path

import pandas as pd

from config import TEXT_COLUMNS, LABEL_COLUMN, POS_LABEL, SEED


def normalize_text(value):
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()


def row_to_text(row):
    parts = []
    for col in TEXT_COLUMNS:
        if col in row and row[col] is not None:
            parts.append(normalize_text(row[col]))
    return " | ".join([p for p in parts if p])


def label_to_int(value):
    return 1 if str(value).strip() == POS_LABEL else 0


def iter_csv_chunks(csv_path, chunksize=50_000, usecols=None):
    usecols = usecols or (TEXT_COLUMNS + [LABEL_COLUMN])
    return pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols)


def split_dataset(csv_path, out_dir, train_ratio=0.8, val_ratio=0.1, seed=SEED, limit_rows=None):
    random.seed(seed)

    train_path = path.join(out_dir, "train.csv")
    val_path = path.join(out_dir, "val.csv")
    test_path = path.join(out_dir, "test.csv")

    # write headers once
    header = TEXT_COLUMNS + [LABEL_COLUMN]
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


def load_text_and_labels_stream(csv_path, max_rows=None):
    count = 0
    for chunk in iter_csv_chunks(csv_path):
        for _, row in chunk.iterrows():
            yield row_to_text(row), label_to_int(row.get(LABEL_COLUMN))
            count += 1
            if max_rows is not None and count >= max_rows:
                return
