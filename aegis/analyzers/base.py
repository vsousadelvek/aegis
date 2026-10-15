"""Classe base para todos os analisadores comportamentais."""

from __future__ import annotations

import abc
from aegis.core.schemas import AlertCreate, Detection, PoseDetection, Zone


class BaseAnalyzer(abc.ABC):
    """Todo analisador recebe dados por frame e retorna zero ou mais alertas."""

    @abc.abstractmethod
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
        ...

    def reset(self) -> None:
        """Reseta estado interno (ex: ao trocar de camera)."""
