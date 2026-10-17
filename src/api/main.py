"""
main.py - fast api tetbiqinin giris noqtesi
"""


import time

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.api.inference import predict
from src.api.schemas import TicketRequest, TicketResponse

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
    return {"status": "ok"}


@app.post("/predict", response_model=TicketResponse)
def predict_priority(request: TicketRequest):
    start = time.perf_counter()
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

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return TicketResponse(
        priority=priority,
        model_used=request.model,
        latency_ms=latency_ms,
    )