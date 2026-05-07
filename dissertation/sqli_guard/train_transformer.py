"""
Train a transformer model for SQL injection detection.
Uses HuggingFace transformers + datasets.
"""

import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import DATA_DIR, MODEL_DIR, TRANSFORMER_MODEL, MAX_LENGTH, TEXT_COLUMNS, LABEL_COLUMN, POS_LABEL


def build_text_from_batch(batch, index):
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


def label_to_int(value):
    return 1 if str(value).strip() == POS_LABEL else 0


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=os.path.join(DATA_DIR, "train.csv"))
    parser.add_argument("--val_csv", default=os.path.join(DATA_DIR, "val.csv"))
    parser.add_argument("--model_name", default=TRANSFORMER_MODEL)
    parser.add_argument("--output_dir", default=os.path.join(MODEL_DIR, "transformer"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(
        "csv",
        data_files={"train": args.train_csv, "validation": args.val_csv},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(batch):
        size = len(batch[LABEL_COLUMN])
        texts = [build_text_from_batch(batch, i) for i in range(size)]
        tokens = tokenizer(texts, padding=False, truncation=True, max_length=args.max_length)
        tokens["labels"] = [label_to_int(v) for v in batch[LABEL_COLUMN]]
        return tokens

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Saved transformer model to", args.output_dir)


if __name__ == "__main__":
    main()
