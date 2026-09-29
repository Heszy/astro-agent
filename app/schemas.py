from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class ToolTrace(BaseModel):
    step: int
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    status: str
    elapsed_seconds: float


class ChatResponse(BaseModel):
    answer: str
    model: str
    trace: list[ToolTrace]
    steps: int
    termination_reason: str
    api_usage: dict[str, int]
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    data_path: str
    rows: int | None = None
    api_key_configured: bool
    dataset_id: str
    version_id: str | None = None
