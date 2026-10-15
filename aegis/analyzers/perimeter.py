"""Deteccao de invasao de perimetro — pessoa entra em zona restrita poligonal."""

from __future__ import annotations

from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon

from aegis.core.schemas import (
    AlertCreate, BBox, Detection, PoseDetection, Severity, ThreatType, Zone,
)
from aegis.analyzers.base import BaseAnalyzer

# Cooldown por (camera, zona, track) para evitar spam do mesmo evento
_COOLDOWN_S = 10.0


class PerimeterGuard(BaseAnalyzer):
    def __init__(self) -> None:
        # chave: (camera_id, zone_id, track_id) -> timestamp do ultimo alerta
        self._last_alert: dict[tuple[str, int, int | None], float] = {}

    def update(
        self,
        camera_id: str,
        detections: list[Detection],
        poses: list[PoseDetection],
        zones: list[Zone],
        timestamp: float,
        frame_w: int,
        frame_h: int,
    ) -> list[AlertCreate]:
        perimeter_zones = [z for z in zones if z.zone_type == "perimeter"]
        if not perimeter_zones:
            return []

        persons = [d for d in detections if d.class_name == "person"]
        alerts: list[AlertCreate] = []

        for zone in perimeter_zones:
            poly = ShapelyPolygon([(p.x * frame_w, p.y * frame_h) for p in zone.polygon])
            for det in persons:
                center = det.bbox.center
                if poly.contains(ShapelyPoint(center.x, center.y)):
                    key = (camera_id, zone.id, det.track_id)
                    last = self._last_alert.get(key, -_COOLDOWN_S - 1)
                    if timestamp - last < _COOLDOWN_S:
                        continue
                    self._last_alert[key] = timestamp
                    alerts.append(
                        AlertCreate(
                            camera_id=camera_id,
                            threat_type=ThreatType.PERIMETER_BREACH,
                            severity=Severity.HIGH,
                            message=f"Pessoa (track {det.track_id}) invadiu zona '{zone.name}'",
                            bbox=det.bbox,
                            metadata={"zone_id": zone.id, "zone_name": zone.name},
                        )
                    )
        return alerts

    def reset(self) -> None:
        self._last_alert.clear()
