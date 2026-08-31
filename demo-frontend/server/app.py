from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .runtime import (
    DemoInputError,
    apply_robustness,
    batch_manager,
    decode_image_bytes,
    image_cache,
    model_registry,
    prepare_clean_image,
    source_preview_data_url,
    validate_relative_path,
)
from .schemas import ModelName, RobustnessRequest
from .settings import DIST_ROOT, MAX_BATCH_FILES, MAX_FILE_BYTES


def create_app() -> FastAPI:
    app = FastAPI(title="AIGI Detect Demo API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        return model_registry.health()

    @app.post("/api/infer")
    async def infer(
        image: Annotated[UploadFile, File(...)],
        model: Annotated[ModelName, Form()] = "m2",
    ) -> dict:
        payload = await image.read(MAX_FILE_BYTES + 1)
        try:
            decoded, metadata = decode_image_bytes(payload)
            clean = prepare_clean_image(decoded)
            image_id = image_cache.put(clean, metadata)
            result = await asyncio.to_thread(model_registry.predict, clean, model)
        except DemoInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"{type(error).__name__}: {error}") from error
        return {
            "image_id": image_id,
            "file": {"name": image.filename or "image", **metadata},
            "source_preview": source_preview_data_url(decoded),
            **result,
        }

    @app.post("/api/robustness/{image_id}")
    async def robustness(image_id: str, request: RobustnessRequest) -> dict:
        try:
            clean, _ = image_cache.get(image_id)
            transformed, applied = apply_robustness(clean, request)
            result = await asyncio.to_thread(model_registry.predict, transformed, request.model)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except DemoInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"{type(error).__name__}: {error}") from error
        return {"image_id": image_id, "applied_transforms": applied, **result}

    @app.post("/api/batches")
    async def create_batch(
        files: Annotated[list[UploadFile], File(...)],
        paths: Annotated[list[str], Form(...)],
        model: Annotated[ModelName, Form()] = "m2",
    ) -> dict:
        if not files or len(files) > MAX_BATCH_FILES:
            raise HTTPException(status_code=422, detail=f"Batch size must be between 1 and {MAX_BATCH_FILES}")
        if len(files) != len(paths):
            raise HTTPException(status_code=422, detail="Batch file and path counts differ")
        try:
            normalized = [validate_relative_path(path) for path in paths]
            if len(set(normalized)) != len(normalized):
                raise DemoInputError("Duplicate image_path values are not allowed")
        except DemoInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        temp_dir = batch_manager.new_temp_dir()
        stored_paths: list[Path] = []
        try:
            for index, upload in enumerate(files):
                target = temp_dir / f"{index:04d}.upload"
                size = 0
                with target.open("wb") as handle:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_FILE_BYTES:
                            raise DemoInputError(
                                f"{normalized[index]} exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB"
                            )
                        handle.write(chunk)
                stored_paths.append(target)
            job = batch_manager.create(model, normalized, stored_paths, temp_dir)
        except Exception as error:
            shutil.rmtree(temp_dir, ignore_errors=True)
            status = 422 if isinstance(error, DemoInputError) else 503
            raise HTTPException(status_code=status, detail=str(error)) from error
        return job.public()

    @app.get("/api/batches/{job_id}")
    async def get_batch(job_id: str) -> dict:
        try:
            return batch_manager.get(job_id).public()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/batches/{job_id}/download")
    async def download_batch(job_id: str) -> JSONResponse:
        try:
            job = batch_manager.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if job.status != "complete":
            raise HTTPException(status_code=409, detail="Batch job is not complete")
        response = JSONResponse(content=job.results)
        response.headers["Content-Disposition"] = (
            f'attachment; filename="aigi-detect-{job.model}-{job.id[:8]}.json"'
        )
        return response

    if DIST_ROOT.is_dir():
        app.mount("/", StaticFiles(directory=DIST_ROOT, html=True), name="frontend")
    return app


app = create_app()
