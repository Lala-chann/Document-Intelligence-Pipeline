"""
schemas.py API==nin giris/cixis formatlarini teyin edit (Pydantic file)
"""

from typing import Literal
from pydantic import BaseModel, Field

ModelChoice = Literal["svm", "claude"]
Priority = Literal["low", "medium", "high"]

class TicketRequest(BaseModel):
    subject : str = Field(..., min_length=1, max_length=500)
    body : str = Field(..., min_length=1, max_length=5000)
    queue: str = Field(..., min_length=1, max_length=100)
    model: ModelChoice = Field(
        default="svm",
        description="'svm' is fastest one / 'claude' is zero-shot LLM"
    )


class TicketResponse(BaseModel):
    priority : Priority
    model_used : ModelChoice
    latency_ms : float