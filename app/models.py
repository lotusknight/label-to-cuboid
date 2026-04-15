from typing import Any

from pydantic import BaseModel, Field


class ImageResult(BaseModel):
    image_index: int
    filename: str
    count: int = 0
    cuboids: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class InferenceResponse(BaseModel):
    prompt: str
    total_cuboids: int
    results: list[ImageResult]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    gpu_batch_size: int

