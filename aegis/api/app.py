"""Aplicacao FastAPI do AEGIS — conecta tudo."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aegis.core.config import get_settings
from aegis.storage.database import Database
from aegis.pipeline.camera_pipeline import CameraPipeline
from aegis.api.routers import alerts, cameras, heatmap, zones

logger = logging.getLogger(__name__)

# Registro global — compartilhado com os routers via app.state
_db: Database | None = None
_pipelines: dict[str, CameraPipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    cfg = get_settings()

    # Inicializa BD
    _db = Database()
    await _db.init()
    app.state.db = _db
    app.state.pipelines = _pipelines

    logger.info("AEGIS iniciado — API em %s:%s", cfg.host, cfg.port)
    yield

    # Encerramento
    for pid, pipe in _pipelines.items():
        pipe.stop()
    if _db:
        await _db.close()
    logger.info("AEGIS encerrado.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AEGIS",
        description="Sistema de Deteccao de Ameacas Urbanas — vigilancia multi-camera em tempo real com deteccao de anomalias comportamentais.",
        version="1.0.0",
        lifespan=lifespan,
    )

    cfg = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(cameras.router)
    app.include_router(alerts.router)
    app.include_router(zones.router)
    app.include_router(heatmap.router)

    @app.get("/health")
    async def health():
        active = {cid: p.running for cid, p in _pipelines.items()}
        return {"status": "ok", "cameras": active}

    return app
