"""Buffer circular de frames com extracao de clips de incidentes."""

from __future__ import annotations

import collections
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from aegis.core.config import get_settings

logger = logging.getLogger(__name__)


class ClipBuffer:
    """Mantem os ultimos N segundos de frames anotados em memoria.

    Quando um alerta acontece, extrai [evento - pre_s, evento + post_s]
    e salva como MP4.
    """

    def __init__(self, camera_id: str, fps: float = 25.0) -> None:
        cfg = get_settings()
        self._camera_id = camera_id
        self._fps = max(fps, 1.0)
        self._pre_s = cfg.clip_pre_s
        self._post_s = cfg.clip_post_s
        self._clips_dir = cfg.clips_dir
        self._clips_dir.mkdir(parents=True, exist_ok=True)

        max_frames = int(cfg.clip_buffer_s * self._fps)
        self._buffer: collections.deque[tuple[float, np.ndarray]] = collections.deque(maxlen=max_frames)

    def push(self, frame: np.ndarray) -> None:
        """Adiciona um frame com timestamp ao buffer circular."""
        self._buffer.append((time.monotonic(), frame))

    def save_clip(self, event_time: float) -> str | None:
        """Recorta [evento - pre_s, evento + post_s] do buffer e grava MP4.

        Retorna o caminho do arquivo ou None se nao ha frames suficientes.
        """
        if not self._buffer:
            return None

        start = event_time - self._pre_s
        end = event_time + self._post_s
        frames = [f for ts, f in self._buffer if start <= ts <= end]

        if not frames:
            return None

        ts_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self._camera_id}_{ts_str}.mp4"
        path = self._clips_dir / filename

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()

        logger.info("Clip salvo: %s (%d frames)", path, len(frames))
        return str(path)
