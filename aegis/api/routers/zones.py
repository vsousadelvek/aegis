"""CRUD de zonas — perimetros, areas proibidas, limites de lotacao."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from aegis.core.schemas import Zone, ZoneCreate
from aegis.pipeline.camera_pipeline import CameraPipeline

router = APIRouter(prefix="/zones", tags=["zonas"])


@router.post("", response_model=Zone, status_code=201)
async def create_zone(body: ZoneCreate, request: Request):
    db = request.app.state.db
    zone = await db.insert_zone(body)

    # Hot-reload das zonas no pipeline em execucao
    pipelines: dict[str, CameraPipeline] = request.app.state.pipelines
    pipe = pipelines.get(body.camera_id)
    if pipe:
        all_zones = await db.list_zones(body.camera_id)
        pipe.set_zones(all_zones)

    return zone


@router.get("", response_model=list[Zone])
async def list_zones(request: Request, camera_id: str | None = None):
    db = request.app.state.db
    return await db.list_zones(camera_id)


@router.delete("/{zone_id}")
async def delete_zone(zone_id: int, request: Request):
    db = request.app.state.db
    ok = await db.delete_zone(zone_id)
    if not ok:
        raise HTTPException(404, "Zona nao encontrada")

    # Recarrega zonas em todos os pipelines (zona pode pertencer a qualquer camera)
    pipelines: dict[str, CameraPipeline] = request.app.state.pipelines
    for cid, pipe in pipelines.items():
        zones = await db.list_zones(cid)
        pipe.set_zones(zones)

    return {"deleted": zone_id}
