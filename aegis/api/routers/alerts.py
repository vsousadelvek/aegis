"""Consulta e listagem de alertas."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aegis.core.schemas import Alert, AlertQuery

router = APIRouter(prefix="/alerts", tags=["alertas"])


@router.post("/query", response_model=list[Alert])
async def query_alerts(body: AlertQuery, request: Request):
    db = request.app.state.db
    return await db.query_alerts(body)


@router.get("", response_model=list[Alert])
async def list_alerts(
    request: Request,
    camera_id: str | None = None,
    threat_type: str | None = None,
    limit: int = 50,
):
    from aegis.core.schemas import ThreatType as TT
    q = AlertQuery(
        camera_id=camera_id,
        threat_type=TT(threat_type) if threat_type else None,
        limit=min(limit, 500),
    )
    db = request.app.state.db
    return await db.query_alerts(q)
