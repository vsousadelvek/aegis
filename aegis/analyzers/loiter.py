"""Deteccao de permanencia de veiculo — veiculo parado em zona proibida por tempo excessivo."""

from __future__ import annotations

from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon

from aegis.core.config import get_settings
from aegis.core.schemas import (
    AlertCreate, Detection, PoseDetection, Severity, ThreatType, Zone,
)
from aegis.analyzers.base import BaseAnalyzer

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


class LoiterDetector(BaseAnalyzer):
    def __init__(self) -> None:
        cfg = get_settings()
        self._timeout = cfg.loiter_timeout_s
        # chave: (camera_id, zone_id, track_id) -> timestamp da primeira deteccao
        self._first_seen: dict[tuple[str, int, int], float] = {}
        self._alerted: set[tuple[str, int, int]] = set()

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
        no_park_zones = [z for z in zones if z.zone_type == "no_parking"]
        if not no_park_zones:
            return []

        vehicles = [d for d in detections if d.class_name in VEHICLE_CLASSES and d.track_id is not None]
        alerts: list[AlertCreate] = []

        active_keys: set[tuple[str, int, int]] = set()

        for zone in no_park_zones:
            poly = ShapelyPolygon([(p.x * frame_w, p.y * frame_h) for p in zone.polygon])
            for det in vehicles:
                center = det.bbox.center
                if not poly.contains(ShapelyPoint(center.x, center.y)):
                    continue
                key = (camera_id, zone.id, det.track_id)  # type: ignore[arg-type]
                active_keys.add(key)
                if key not in self._first_seen:
                    self._first_seen[key] = timestamp
                    continue
                elapsed = timestamp - self._first_seen[key]
                if elapsed >= self._timeout and key not in self._alerted:
                    self._alerted.add(key)
                    alerts.append(
                        AlertCreate(
                            camera_id=camera_id,
                            threat_type=ThreatType.LOITERING_VEHICLE,
                            severity=Severity.MEDIUM,
                            message=(
                                f"Veiculo (track {det.track_id}) parado em "
                                f"'{zone.name}' por {elapsed:.0f}s"
                            ),
                            bbox=det.bbox,
                            metadata={
                                "zone_id": zone.id,
                                "zone_name": zone.name,
                                "elapsed_s": round(elapsed, 1),
                            },
                        )
                    )

        # Limpa entradas de tracks que sairam da zona
        stale = [k for k in self._first_seen if k not in active_keys]
        for k in stale:
            del self._first_seen[k]
            self._alerted.discard(k)

        return alerts

    def reset(self) -> None:
        self._first_seen.clear()
        self._alerted.clear()
