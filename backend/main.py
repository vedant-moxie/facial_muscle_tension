"""
Creator Signal Lab — FastAPI entrypoint.

This file is intentionally thin. Each analysis modality lives in its own
package and exposes a FastAPI router:

    backend/face/      — facial-tension AUs   (routes.py + pipeline/ + models.py)
    backend/audio/     — Sonic Rebellion       (routes.py + pipeline/ + config.py)
    backend/presence/  — pose / presence       (routes.py + pipeline/runner.py)

To add a fourth modality, drop a sibling package with a `routes.py`
exposing `router = APIRouter()` and call `include_router` below.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load backend/.env into os.environ on startup if python-dotenv is available.
# Soft dependency — falls back to OS env vars if dotenv isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:                                  # noqa: BLE001
    pass

from face import routes as face_routes
from audio import routes as audio_routes
from presence import routes as presence_routes

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("effortlessness")

app = FastAPI(
    title="Creator Signal Lab",
    version="1.2.0",
    description="Face tension (FACS AUs) · Sonic Rebellion audio · pose presence.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(face_routes.router)
app.include_router(audio_routes.router)
app.include_router(presence_routes.router)


@app.on_event("startup")
async def _startup() -> None:
    await face_routes.warm_models()


@app.on_event("shutdown")
async def _shutdown() -> None:
    face_routes.shutdown_executor()
    audio_routes.shutdown_executor()
    presence_routes.shutdown_executor()
