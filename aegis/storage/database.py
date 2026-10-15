"""Camada de banco de dados async — SQLAlchemy 2.0 + aiosqlite."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from aegis.core.config import get_settings
from aegis.core.schemas import Alert, AlertCreate, AlertQuery, Zone, ZoneCreate


class Base(DeclarativeBase):
    pass


class AlertRow(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, nullable=False, index=True)
    threat_type = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    bbox_json = Column(Text, nullable=True)
    clip_path = Column(String, nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class ZoneRow(Base):
    __tablename__ = "zones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    zone_type = Column(String, nullable=False)
    polygon_json = Column(Text, nullable=False)
    max_crowd = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class HeatmapPointRow(Base):
    __tablename__ = "heatmap_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, nullable=False, index=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class Database:
    def __init__(self) -> None:
        settings = get_settings()
        self._engine = create_async_engine(settings.db_url, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    def _session(self) -> AsyncSession:
        return self._session_factory()

    # --- Alertas ---

    async def insert_alert(self, data: AlertCreate) -> Alert:
        row = AlertRow(
            camera_id=data.camera_id,
            threat_type=data.threat_type.value,
            severity=data.severity.value,
            message=data.message,
            bbox_json=data.bbox.model_dump_json() if data.bbox else None,
            clip_path=data.clip_path,
            metadata_json=json.dumps(data.metadata) if data.metadata else None,
        )
        async with self._session() as s:
            s.add(row)
            await s.commit()
            await s.refresh(row)
        return self._alert_from_row(row)

    async def query_alerts(self, q: AlertQuery) -> list[Alert]:
        stmt = select(AlertRow).order_by(AlertRow.created_at.desc()).limit(q.limit)
        if q.camera_id:
            stmt = stmt.where(AlertRow.camera_id == q.camera_id)
        if q.threat_type:
            stmt = stmt.where(AlertRow.threat_type == q.threat_type.value)
        if q.severity:
            stmt = stmt.where(AlertRow.severity == q.severity.value)
        if q.since:
            stmt = stmt.where(AlertRow.created_at >= q.since)
        async with self._session() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return [self._alert_from_row(r) for r in rows]

    async def count_alerts(self, camera_id: str) -> int:
        async with self._session() as s:
            result = await s.execute(
                select(func.count()).select_from(AlertRow).where(AlertRow.camera_id == camera_id)
            )
            return result.scalar_one()

    @staticmethod
    def _alert_from_row(row: AlertRow) -> Alert:
        from aegis.core.schemas import BBox
        bbox = BBox.model_validate_json(row.bbox_json) if row.bbox_json else None
        meta = json.loads(row.metadata_json) if row.metadata_json else None
        return Alert(
            id=row.id,
            camera_id=row.camera_id,
            threat_type=row.threat_type,
            severity=row.severity,
            message=row.message,
            bbox=bbox,
            clip_path=row.clip_path,
            metadata=meta,
            created_at=row.created_at,
        )

    # --- Zonas ---

    async def insert_zone(self, data: ZoneCreate) -> Zone:
        row = ZoneRow(
            camera_id=data.camera_id,
            name=data.name,
            zone_type=data.zone_type,
            polygon_json=json.dumps([p.model_dump() for p in data.polygon]),
            max_crowd=data.max_crowd,
        )
        async with self._session() as s:
            s.add(row)
            await s.commit()
            await s.refresh(row)
        return self._zone_from_row(row)

    async def list_zones(self, camera_id: str | None = None) -> list[Zone]:
        stmt = select(ZoneRow).order_by(ZoneRow.created_at.desc())
        if camera_id:
            stmt = stmt.where(ZoneRow.camera_id == camera_id)
        async with self._session() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return [self._zone_from_row(r) for r in rows]

    async def delete_zone(self, zone_id: int) -> bool:
        async with self._session() as s:
            row = await s.get(ZoneRow, zone_id)
            if not row:
                return False
            await s.delete(row)
            await s.commit()
            return True

    @staticmethod
    def _zone_from_row(row: ZoneRow) -> Zone:
        from aegis.core.schemas import Point
        polygon = [Point(**p) for p in json.loads(row.polygon_json)]
        return Zone(
            id=row.id,
            camera_id=row.camera_id,
            name=row.name,
            zone_type=row.zone_type,
            polygon=polygon,
            max_crowd=row.max_crowd,
            created_at=row.created_at,
        )

    # --- Heatmap ---

    async def record_heatmap_points(self, camera_id: str, points: list[tuple[float, float]]) -> None:
        if not points:
            return
        rows = [HeatmapPointRow(camera_id=camera_id, x=x, y=y) for x, y in points]
        async with self._session() as s:
            s.add_all(rows)
            await s.commit()

    async def get_heatmap_points(
        self, camera_id: str, since: datetime
    ) -> list[tuple[float, float]]:
        stmt = (
            select(HeatmapPointRow.x, HeatmapPointRow.y)
            .where(HeatmapPointRow.camera_id == camera_id)
            .where(HeatmapPointRow.created_at >= since)
        )
        async with self._session() as s:
            rows = (await s.execute(stmt)).all()
        return [(r.x, r.y) for r in rows]
