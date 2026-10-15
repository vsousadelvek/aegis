"""Geracao de heatmap — retorna grid de densidade a partir de pontos de deteccao."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from aegis.core.schemas import HeatmapRequest

router = APIRouter(prefix="/heatmap", tags=["heatmap"])


@router.post("/grid")
async def heatmap_grid(body: HeatmapRequest, request: Request):
    """Retorna o grid de densidade como JSON (para renderizacao customizada)."""
    db = request.app.state.db
    since = datetime.now(timezone.utc) - timedelta(hours=body.hours)
    points = await db.get_heatmap_points(body.camera_id, since)

    if not points:
        return {"camera_id": body.camera_id, "grid": [], "total_points": 0}

    res = body.resolution
    grid = np.zeros((res, res), dtype=np.float32)
    for x_norm, y_norm in points:
        xi = min(int(x_norm * res), res - 1)
        yi = min(int(y_norm * res), res - 1)
        grid[yi, xi] += 1.0

    # Blur gaussiano para suavizar o heatmap
    grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=2.0)
    if grid.max() > 0:
        grid = grid / grid.max()

    return {
        "camera_id": body.camera_id,
        "grid": grid.tolist(),
        "total_points": len(points),
        "resolution": res,
    }


@router.post("/image")
async def heatmap_image(body: HeatmapRequest, request: Request):
    """Retorna o heatmap como imagem PNG com colormap."""
    db = request.app.state.db
    since = datetime.now(timezone.utc) - timedelta(hours=body.hours)
    points = await db.get_heatmap_points(body.camera_id, since)

    res = body.resolution
    grid = np.zeros((res, res), dtype=np.float32)
    for x_norm, y_norm in points:
        xi = min(int(x_norm * res), res - 1)
        yi = min(int(y_norm * res), res - 1)
        grid[yi, xi] += 1.0

    grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=2.0)
    if grid.max() > 0:
        grid = (grid / grid.max() * 255).astype(np.uint8)
    else:
        grid = grid.astype(np.uint8)

    colored = cv2.applyColorMap(grid, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".png", colored)

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
