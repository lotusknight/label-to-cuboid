from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.logging_config import setup_logging
from app.models import HealthResponse, InferenceResponse
from app.pipeline import CuboidPipeline, decode_uploads

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

pipeline = CuboidPipeline(settings=settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("service_starting")
    try:
        await run_in_threadpool(pipeline.load_models)
    except Exception:
        logger.exception("model_loading_failed")
        raise
    yield
    pipeline.unload_models()
    logger.info("service_stopped")


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=pipeline.is_ready,
        gpu_batch_size=settings.gpu_batch_size,
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    images: list[UploadFile] = File(...),
    prompt: str = Form(...),
    confidence_threshold: float | None = Form(default=None),
) -> InferenceResponse:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if not images:
        raise HTTPException(status_code=400, detail="at least one image is required")
    if len(images) > settings.max_images_per_request:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too many images: {len(images)} > "
                f"MAX_IMAGES_PER_REQUEST={settings.max_images_per_request}"
            ),
        )
    if confidence_threshold is not None and not (0.0 <= confidence_threshold <= 1.0):
        raise HTTPException(
            status_code=400, detail="confidence_threshold must be between 0 and 1"
        )

    request_start = time.perf_counter()
    raw_uploads: list[tuple[str, bytes]] = []
    for image in images:
        payload = await image.read()
        if not payload:
            raise HTTPException(
                status_code=400,
                detail=f"uploaded file '{image.filename or 'unknown'}' is empty",
            )
        raw_uploads.append((image.filename or "unknown", payload))

    logger.info(
        "request_received",
        extra={
            "images_count": len(raw_uploads),
            "prompt": prompt,
            "threshold": (
                confidence_threshold
                if confidence_threshold is not None
                else settings.confidence_threshold
            ),
        },
    )

    try:
        decoded = await run_in_threadpool(decode_uploads, raw_uploads)
        results = await run_in_threadpool(
            pipeline.run_batch_inference, decoded, prompt, confidence_threshold
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("request_failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    total_cuboids = sum(int(item["count"]) for item in results)
    elapsed = time.perf_counter() - request_start
    logger.info(
        "request_completed",
        extra={
            "images_count": len(raw_uploads),
            "total_cuboids": total_cuboids,
            "seconds": round(elapsed, 3),
        },
    )
    return InferenceResponse(prompt=prompt, total_cuboids=total_cuboids, results=results)

