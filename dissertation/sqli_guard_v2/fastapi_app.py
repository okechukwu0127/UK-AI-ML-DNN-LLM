"""
FastAPI middleware example for SQLi detection.

Style references:
- /Users/oeze/Documents/wlv/7CS028/workshop1/task_1.py
- /Users/oeze/Documents/wlv/7CS028/class_2/GD_Adam.py
"""

import os
import json
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from sqli_pipeline import predict_sklearn

app = FastAPI()

# model path must point to your trained baseline model
MODEL_PATH = os.getenv(
    "SQLI_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "sgd_log.joblib"),
)

THRESHOLD = float(os.getenv("SQLI_THRESHOLD", "0.5"))
BLOCK_REQUEST = os.getenv("SQLI_BLOCK", "false").lower() == "true"


def extract_text_from_request(body_bytes, query_params):
    """Extract text from JSON body and query params."""
    text_parts = []

    # try JSON
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
        # fallback: raw body
        try:
            text_parts.append(body_bytes.decode("utf-8", errors="ignore"))
        except Exception:
            pass

    # query params
    for k, v in query_params.items():
        text_parts.append(f"{k}={v}")

    return " | ".join([p for p in text_parts if p])


class SQLiDetectionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        query_params = dict(request.query_params)
        text = extract_text_from_request(body, query_params)

        if text:
            result = predict_sklearn(MODEL_PATH, text)
            request.state.sqli_result = result

            if BLOCK_REQUEST and result["is_sqli"] and result["probability"] >= THRESHOLD:
                return JSONResponse(
                    {"detail": "SQL injection suspected", "score": result["probability"]},
                    status_code=403,
                )

        response = await call_next(request)

        if hasattr(request.state, "sqli_result"):
            response.headers["X-SQLi-Detected"] = str(request.state.sqli_result["is_sqli"])
            response.headers["X-SQLi-Score"] = str(round(request.state.sqli_result["probability"], 6))

        return response


# attach middleware
app.add_middleware(SQLiDetectionMiddleware)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(payload: dict):
    return {"received": payload}
