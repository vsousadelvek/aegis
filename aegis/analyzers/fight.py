"""Deteccao de briga / agressao por analise temporal de keypoints de pose.

Em vez de treinar um modelo completo de reconhecimento de acao (que requer um
dataset rotulado de brigas), usamos um classificador heuristico baseado em
velocidade sobre keypoints COCO:

1. Para cada pessoa rastreada, mantemos uma janela deslizante de posicoes de keypoints.
2. Calculamos a velocidade media dos membros (pulsos + tornozelos) na janela.
3. Calculamos a proximidade entre pessoas (distancia centro-a-centro).
4. Um par de pessoas com:
   - alta velocidade de membros (acima do limiar), E
   - proximidade (dentro de ~1.5 alturas corporais)
   eh marcado como possivel briga.

Versao futura pode substituir por um 1D-Conv ou LSTM treinado sobre clips de briga,
consumindo o mesmo buffer de keypoints.
"""

from __future__ import annotations

import collections
import math
from aegis.core.config import get_settings
from aegis.core.schemas import (
    AlertCreate, Detection, PoseDetection, Severity, ThreatType, Zone,
)
from aegis.analyzers.base import BaseAnalyzer

# Indices de keypoints COCO para extremidades dos membros
_WRIST_L, _WRIST_R = 9, 10
_ANKLE_L, _ANKLE_R = 15, 16
_LIMB_INDICES = [_WRIST_L, _WRIST_R, _ANKLE_L, _ANKLE_R]

_VELOCITY_THRESHOLD = 15.0  # pixels/frame — calibrado para inferencia em 640px
_PROXIMITY_RATIO = 2.0  # distancia maxima do centro como multiplo da altura media da bbox
_COOLDOWN_S = 8.0


class FightDetector(BaseAnalyzer):
    def __init__(self) -> None:
        cfg = get_settings()
        self._seq_len = cfg.fight_seq_len
        # track_id -> deque de snapshots de keypoints [(x, y), ...]
        self._buffers: dict[int, collections.deque[list[tuple[float, float]]]] = {}
        self._last_alert: dict[str, float] = {}  # camera -> timestamp do ultimo alerta

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
        # Atualiza buffers com poses atuais
        seen_tracks: set[int] = set()
        track_meta: dict[int, PoseDetection] = {}

        for pose in poses:
            tid = pose.track_id
            if tid is None:
                continue
            seen_tracks.add(tid)
            track_meta[tid] = pose

            limb_positions = [
                (pose.keypoints.xy[i][0], pose.keypoints.xy[i][1])
                for i in _LIMB_INDICES
                if pose.keypoints.xy[i][2] > 0.3  # limiar de confianca
            ]
            if len(limb_positions) < 2:
                continue

            if tid not in self._buffers:
                self._buffers[tid] = collections.deque(maxlen=self._seq_len)
            self._buffers[tid].append(limb_positions)

        # Limpa tracks obsoletos
        stale = [t for t in self._buffers if t not in seen_tracks]
        for t in stale:
            del self._buffers[t]

        # Calcula velocidade media de membros por track
        track_velocity: dict[int, float] = {}
        for tid, buf in self._buffers.items():
            if len(buf) < 4:
                continue
            velocities: list[float] = []
            for i in range(1, len(buf)):
                prev, curr = buf[i - 1], buf[i]
                n = min(len(prev), len(curr))
                for j in range(n):
                    dx = curr[j][0] - prev[j][0]
                    dy = curr[j][1] - prev[j][1]
                    velocities.append(math.hypot(dx, dy))
            if velocities:
                track_velocity[tid] = sum(velocities) / len(velocities)

        # Busca pares com alta velocidade + proximidade
        alerts: list[AlertCreate] = []
        fast_tracks = [t for t, v in track_velocity.items() if v > _VELOCITY_THRESHOLD]

        for i, t1 in enumerate(fast_tracks):
            for t2 in fast_tracks[i + 1:]:
                p1 = track_meta.get(t1)
                p2 = track_meta.get(t2)
                if p1 is None or p2 is None:
                    continue

                c1 = p1.bbox.center
                c2 = p2.bbox.center
                dist = math.hypot(c1.x - c2.x, c1.y - c2.y)

                avg_h = ((p1.bbox.y2 - p1.bbox.y1) + (p2.bbox.y2 - p2.bbox.y1)) / 2
                if avg_h < 1:
                    continue

                if dist < avg_h * _PROXIMITY_RATIO:
                    last = self._last_alert.get(camera_id, -_COOLDOWN_S - 1)
                    if timestamp - last < _COOLDOWN_S:
                        continue
                    self._last_alert[camera_id] = timestamp

                    # Combina bboxes para o alerta
                    merged = _merge_bboxes(p1.bbox, p2.bbox)
                    v1 = track_velocity.get(t1, 0.0)
                    v2 = track_velocity.get(t2, 0.0)
                    alerts.append(
                        AlertCreate(
                            camera_id=camera_id,
                            threat_type=ThreatType.FIGHT,
                            severity=Severity.CRITICAL,
                            message=(
                                f"Possivel briga entre tracks {t1} e {t2} — "
                                f"velocidade de membros {max(v1, v2):.1f} px/frame"
                            ),
                            bbox=merged,
                            metadata={
                                "track_ids": [t1, t2],
                                "velocities": [round(v1, 2), round(v2, 2)],
                                "distance_px": round(dist, 1),
                            },
                        )
                    )
        return alerts

    def reset(self) -> None:
        self._buffers.clear()
        self._last_alert.clear()


def _merge_bboxes(a, b):
    from aegis.core.schemas import BBox
    return BBox(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
    )
