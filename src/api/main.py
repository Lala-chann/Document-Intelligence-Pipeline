"""
main.py - fast api tetbiqinin giris noqtesi
"""


import time

from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from src.api.inference import predict
from src.api.schemas import TicketRequest, TicketResponse
from src.api.rate_limiter import SlidingWindowRateLimiter
from src.api.lru_cache import LRUCache, make_key



limiters = {
    "svm" : SlidingWindowRateLimiter(limit=20,window_seconds=60),
    "claude" : SlidingWindowRateLimiter(limit=5, window_seconds=60),
}

cache = LRUCache(capacity=500)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predict(subject="warmup",body="warmup",queue="General Inquiry", model="svm")
        print("Model is loaded")
    except FileNotFoundError:
        print("Warning: Model could not find filles. Execute train_final_model.py first")
    yield





app = FastAPI(
    title = 'Ticket Triage API',
    description = "controls priority of custumer's ticket",
    version = "0.1.0",
    lifespan=lifespan
)

@app.get("/status")
def status():
    """check if system status is okay"""
    return {"status": "ok", "claude_cache": cache.stats()}


@app.post("/predict", response_model=TicketResponse)
def predict_priority(request: TicketRequest, http_request: Request):

    start = time.perf_counter()
    cache_key = None

    if request.model == "claude":
        cache_key = make_key(request.model, request.queue,
                             request.subject.strip(), request.body.strip())
        cached_priority = cache.get(cache_key)
        if cached_priority is not None:
            return TicketResponse(
                priority=cached_priority,
                model_used=request.model,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                cached= True
            )



    client_ip =http_request.client.host if http_request.client else "unknown"
    allowed, retry_after = limiters[request.model].allow(client_ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many request. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    try:
        priority = predict(
            subject=request.subject,
            body=request.body,
            queue=request.queue,
            model=request.model,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model file can not open, Execute train_final_model.py beforehand",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if cache_key is not None:
        cache.put(cache_key, priority)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return TicketResponse(
        priority=priority,
        model_used=request.model,
        latency_ms=latency_ms,
    )