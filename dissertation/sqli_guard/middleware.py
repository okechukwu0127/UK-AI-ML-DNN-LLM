"""
FastAPI middleware for SQL injection detection.
"""

import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from predict import SQLiDetector, extract_text_from_request


def build_detector_from_env():
    model_type = os.getenv("SQLI_MODEL_TYPE", "sklearn")
    model_path = os.getenv("SQLI_MODEL_PATH", "")
    threshold = float(os.getenv("SQLI_THRESHOLD", "0.5"))

    if not model_path:
        raise ValueError("SQLI_MODEL_PATH is required")

    return SQLiDetector(model_type=model_type, model_path=model_path, threshold=threshold)


class SQLiDetectionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, detector=None, block=False):
        super().__init__(app)
        self.detector = detector or build_detector_from_env()
        self.block = block

    async def dispatch(self, request, call_next):
        body = await request.body()
        query_params = dict(request.query_params)
        text = extract_text_from_request(body, query_params)

        if text:
            result = self.detector.predict(text)
            request.state.sqli_result = result

            if self.block and result["is_sqli"]:
                return JSONResponse(
                    {"detail": "SQL injection suspected", "score": result["probability"]},
                    status_code=403,
                )

        response = await call_next(request)
        if hasattr(request.state, "sqli_result"):
            response.headers["X-SQLi-Detected"] = str(request.state.sqli_result["is_sqli"])
            response.headers["X-SQLi-Score"] = str(round(request.state.sqli_result["probability"], 6))
        return response
