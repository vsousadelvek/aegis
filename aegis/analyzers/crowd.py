"""Deteccao de anomalia de multidao com baseline adaptativo."""

from __future__ import annotations

import collections
from aegis.core.config import get_settings
from aegis.core.schemas import (
    AlertCreate, Detection, PoseDetection, Severity, ThreatType, Zone,
)
from aegis.analyzers.base import BaseAnalyzer

_HISTORY_LEN = 300  # frames de historico de contagem para estatisticas
_COOLDOWN_FRAMES = 90  # nao re-alertar dentro dessa janela de frames


class CrowdDetector(BaseAnalyzer):
    def __init__(self) -> None:
        cfg = get_settings()
        self._sigma = cfg.crowd_density_sigma
        # historico de contagem de pessoas por camera
        self._history: dict[str, collections.deque[int]] = {}
        self._cooldown: dict[str, int] = {}  # camera -> frames desde ultimo alerta
        self._frame_counter: dict[str, int] = {}

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
        person_count = sum(1 for d in detections if d.class_name == "person")

        if camera_id not in self._history:
            self._history[camera_id] = collections.deque(maxlen=_HISTORY_LEN)
            self._cooldown[camera_id] = 0
            self._frame_counter[camera_id] = 0

        history = self._history[camera_id]
        self._frame_counter[camera_id] += 1

        # Verifica zonas com limite explicito de lotacao
        alerts: list[AlertCreate] = []
        for zone in zones:
            if zone.zone_type == "crowd_limit" and zone.max_crowd is not None:
                if person_count > zone.max_crowd:
                    if self._cooldown[camera_id] <= 0:
                        alerts.append(
                            AlertCreate(
                                camera_id=camera_id,
                                threat_type=ThreatType.CROWD_ANOMALY,
                                severity=Severity.HIGH,
                                message=(
                                    f"Limite de lotacao excedido em '{zone.name}': "
                                    f"{person_count} > {zone.max_crowd}"
                                ),
                                metadata={
                                    "zone_id": zone.id,
                                    "person_count": person_count,
                                    "limit": zone.max_crowd,
                                },
                            )
                        )
                        self._cooldown[camera_id] = _COOLDOWN_FRAMES

        # Anomalia adaptativa: so apos ter dados suficientes de baseline
        if len(history) >= 30:
            import numpy as np
            mean = float(np.mean(history))
            std = float(np.std(history)) + 1e-6
            if person_count > mean + self._sigma * std:
                if self._cooldown[camera_id] <= 0:
                    alerts.append(
                        AlertCreate(
                            camera_id=camera_id,
                            threat_type=ThreatType.CROWD_ANOMALY,
                            severity=Severity.MEDIUM,
                            message=(
                                f"Densidade anomala de pessoas: {person_count} pessoas "
                                f"(baseline {mean:.1f} +/- {std:.1f})"
                            ),
                            metadata={
                                "person_count": person_count,
                                "baseline_mean": round(mean, 2),
                                "baseline_std": round(std, 2),
                            },
                        )
                    )
                    self._cooldown[camera_id] = _COOLDOWN_FRAMES

        history.append(person_count)
        if self._cooldown[camera_id] > 0:
            self._cooldown[camera_id] -= 1

        return alerts

    def reset(self) -> None:
        self._history.clear()
        self._cooldown.clear()
        self._frame_counter.clear()
