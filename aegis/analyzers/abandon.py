"""Deteccao de objeto abandonado — objeto estatico sem dono proximo."""

from __future__ import annotations

import math
from aegis.core.config import get_settings
from aegis.core.schemas import (
    AlertCreate, Detection, PoseDetection, Severity, ThreatType, Zone,
)
from aegis.analyzers.base import BaseAnalyzer

PORTABLE_CLASSES = {"backpack", "umbrella", "handbag", "suitcase"}
_OWNER_RADIUS_PX = 150  # pixels — se nenhuma pessoa esta dentro desse raio, objeto esta "sem dono"


class AbandonDetector(BaseAnalyzer):
    def __init__(self) -> None:
        cfg = get_settings()
        self._timeout = cfg.abandon_timeout_s
        # chave: track_id -> (primeiro_timestamp_sem_dono, cx, cy)
        self._candidates: dict[int, tuple[float, float, float]] = {}
        self._alerted: set[int] = set()

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
        objects = [d for d in detections if d.class_name in PORTABLE_CLASSES and d.track_id is not None]
        persons = [d for d in detections if d.class_name == "person"]
        person_centers = [(d.bbox.center.x, d.bbox.center.y) for d in persons]

        alerts: list[AlertCreate] = []
        seen_tracks: set[int] = set()

        for obj in objects:
            tid = obj.track_id
            assert tid is not None
            seen_tracks.add(tid)
            cx, cy = obj.bbox.center.x, obj.bbox.center.y

            # Verifica se alguma pessoa esta proxima
            has_owner = any(
                math.hypot(cx - px, cy - py) < _OWNER_RADIUS_PX
                for px, py in person_centers
            )

            if has_owner:
                # Reseta se o dono voltou
                self._candidates.pop(tid, None)
                self._alerted.discard(tid)
                continue

            if tid not in self._candidates:
                self._candidates[tid] = (timestamp, cx, cy)
                continue

            first_ts, _, _ = self._candidates[tid]
            elapsed = timestamp - first_ts
            if elapsed >= self._timeout and tid not in self._alerted:
                self._alerted.add(tid)
                alerts.append(
                    AlertCreate(
                        camera_id=camera_id,
                        threat_type=ThreatType.ABANDONED_OBJECT,
                        severity=Severity.HIGH,
                        message=(
                            f"{obj.class_name} abandonado(a) (track {tid}) — "
                            f"sem dono por {elapsed:.0f}s"
                        ),
                        bbox=obj.bbox,
                        metadata={
                            "object_class": obj.class_name,
                            "elapsed_s": round(elapsed, 1),
                        },
                    )
                )

        # Limpa tracks que nao estao mais visiveis
        stale = [t for t in self._candidates if t not in seen_tracks]
        for t in stale:
            del self._candidates[t]
            self._alerted.discard(t)

        return alerts

    def reset(self) -> None:
        self._candidates.clear()
        self._alerted.clear()
