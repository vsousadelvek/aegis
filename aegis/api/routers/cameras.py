"""Gerenciamento de cameras — adicionar/remover, stream MJPEG ao vivo."""

from __future__ import annotations

import asyncio
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aegis.core.schemas import CameraMetrics
from aegis.pipeline.camera_pipeline import CameraPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])


class CameraAdd(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=64)
    source: str = Field(..., description="URL RTSP, caminho de arquivo, ou indice do dispositivo (ex: '0')")


class CameraInfo(BaseModel):
    camera_id: str
    source: str
    running: bool


def _get_pipelines(request: Request) -> dict[str, CameraPipeline]:
    return request.app.state.pipelines


def _get_db(request: Request):
    return request.app.state.db


@router.post("", status_code=201)
async def add_camera(body: CameraAdd, request: Request) -> CameraInfo:
    pipelines = _get_pipelines(request)
    db = _get_db(request)

    if body.camera_id in pipelines:
        raise HTTPException(409, f"Camera '{body.camera_id}' ja existe")

    # Resolve fonte: string numerica -> int (indice de webcam)
    source: str | int = body.source
    if body.source.isdigit():
        source = int(body.source)

    loop = asyncio.get_running_loop()

    # Carrega zonas existentes para esta camera
    zones = await db.list_zones(body.camera_id)

    pipe = CameraPipeline(
        camera_id=body.camera_id,
        source=source,
        db=db,
        loop=loop,
        zones=zones,
    )
    pipe.start()
    pipelines[body.camera_id] = pipe

    return CameraInfo(camera_id=body.camera_id, source=body.source, running=pipe.running)


@router.delete("/{camera_id}")
async def remove_camera(camera_id: str, request: Request):
    pipelines = _get_pipelines(request)
    pipe = pipelines.pop(camera_id, None)
    if pipe is None:
        raise HTTPException(404, f"Camera '{camera_id}' nao encontrada")
    pipe.stop()
    return {"removed": camera_id}


@router.get("", response_model=list[CameraInfo])
async def list_cameras(request: Request):
    pipelines = _get_pipelines(request)
    return [
        CameraInfo(camera_id=cid, source=str(p._source), running=p.running)
        for cid, p in pipelines.items()
    ]


@router.get("/{camera_id}/metrics", response_model=CameraMetrics)
async def camera_metrics(camera_id: str, request: Request):
    pipelines = _get_pipelines(request)
    pipe = pipelines.get(camera_id)
    if pipe is None:
        raise HTTPException(404, f"Camera '{camera_id}' nao encontrada")

    with pipe.state.lock:
        return CameraMetrics(
            camera_id=camera_id,
            fps=round(pipe.state.fps, 1),
            inference_ms=round(pipe.state.inference_ms, 1),
            person_count=pipe.state.person_count,
            vehicle_count=pipe.state.vehicle_count,
            active_tracks=pipe.state.active_tracks,
            alerts_total=pipe.state.alerts_total,
        )


@router.get("/{camera_id}/stream")
async def stream_mjpeg(camera_id: str, request: Request):
    """Stream MJPEG ao vivo de frames anotados."""
    pipelines = _get_pipelines(request)
    pipe = pipelines.get(camera_id)
    if pipe is None:
        raise HTTPException(404, f"Camera '{camera_id}' nao encontrada")

    async def generate():
        while True:
            with pipe.state.lock:
                frame = pipe.state.annotated
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            await asyncio.sleep(0.033)  # ~30 fps cap

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{camera_id}/snapshot")
async def snapshot(camera_id: str, request: Request):
    """Snapshot JPEG unico do ultimo frame anotado."""
    pipelines = _get_pipelines(request)
    pipe = pipelines.get(camera_id)
    if pipe is None:
        raise HTTPException(404, f"Camera '{camera_id}' nao encontrada")

    with pipe.state.lock:
        frame = pipe.state.annotated
    if frame is None:
        raise HTTPException(503, "Nenhum frame disponivel ainda")

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(
        iter([buf.tobytes()]),
        media_type="image/jpeg",
    )
