"""
Unified prediction for SQLi models (sklearn or transformer).
"""

import json
import math

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MAX_LENGTH


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class SQLiDetector:
    def __init__(self, model_type, model_path, threshold=0.5):
        self.model_type = model_type
        self.model_path = model_path
        self.threshold = threshold
        self.vectorizer = None
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        if self.model_type == "sklearn":
            bundle = joblib.load(self.model_path)
            self.vectorizer = bundle["vectorizer"]
            self.model = bundle["model"]
        elif self.model_type == "transformer":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
        else:
            raise ValueError("Unsupported model_type")

    def predict_proba(self, texts):
        if self.model_type == "sklearn":
            X = self.vectorizer.transform(texts)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[:, 1]
            else:
                scores = self.model.decision_function(X)
                proba = np.array([sigmoid(s) for s in scores])
            return proba

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            return proba

    def predict(self, text):
        proba = float(self.predict_proba([text])[0])
        return {"probability": proba, "is_sqli": proba >= self.threshold}


def extract_text_from_request(body_bytes, query_params):
    # try json
    text_parts = []
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
        if isinstance(payload, dict):
            for k, v in payload.items():
                text_parts.append(f"{k}={v}")
        elif isinstance(payload, list):
            for item in payload:
                text_parts.append(str(item))
        else:
            text_parts.append(str(payload))
    except Exception:
        # not json, best effort decode
        try:
            text_parts.append(body_bytes.decode("utf-8", errors="ignore"))
        except Exception:
            pass

    # query params
    for k, v in query_params.items():
        text_parts.append(f"{k}={v}")

    text = " | ".join([p for p in text_parts if p])
    return text
