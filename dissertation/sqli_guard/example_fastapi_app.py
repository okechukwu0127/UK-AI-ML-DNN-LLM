"""
Example FastAPI app with SQLi middleware.
"""

from fastapi import FastAPI
from middleware import SQLiDetectionMiddleware

app = FastAPI()

# add middleware (block=False means just flag, not block)
app.add_middleware(SQLiDetectionMiddleware, block=False)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(payload: dict):
    # your normal logic here
    return {"received": payload}
