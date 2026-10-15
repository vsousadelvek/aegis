"""Pipeline de processamento por camera — roda em thread dedicada."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from aegis.core.config import get_settings
from aegis.core.schemas import Alert, AlertCreate, Detection, PoseDetection, Zone
from aegis.detection.tracker import TrackedDetector
from aegis.analyzers.base import BaseAnalyzer
from aegis.analyzers.perimeter import PerimeterGuard
from aegis.analyzers.loiter import LoiterDetector
from aegis.analyzers.crowd import CrowdDetector
from aegis.analyzers.abandon import AbandonDetector
from aegis.analyzers.fight import FightDetector
from aegis.pipeline.clip_buffer import ClipBuffer

if TYPE_CHECKING:
    import asyncio
    from aegis.storage.database import Database

logger = logging.getLogger(__name__)


class FrameState:
    """Snapshot do ultimo frame processado e seus resultados — lido pela API."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.annotated: np.ndarray | None = None
        self.detections: list[Detection] = []
        self.poses: list[PoseDetection] = []
        self.person_count: int = 0
        self.vehicle_count: int = 0
        self.active_tracks: int = 0
        self.fps: float = 0.0
        self.inference_ms: float = 0.0
        self.alerts_total: int = 0


class CameraPipeline:
    """Le uma fonte de video, roda deteccao+rastreamento+analisadores e expoe estado."""

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        db: Database,
        loop: asyncio.AbstractEventLoop,
        zones: list[Zone] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self._source = source
        self._db = db
        self._loop = loop
        self._zones = zones or []
        self._zones_lock = threading.Lock()

        self.state = FrameState()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        cfg = get_settings()
        self._frame_skip = cfg.frame_skip

    # --- Gerenciamento de zonas (chamado pelo contexto async da API) ---

    def set_zones(self, zones: list[Zone]) -> None:
        with self._zones_lock:
            self._zones = list(zones)

    def get_zones(self) -> list[Zone]:
        with self._zones_lock:
            return list(self._zones)

    # --- Ciclo de vida ---

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.camera_id}")
        self._thread.start()
        logger.info("Pipeline iniciado para camera %s (fonte=%s)", self.camera_id, self._source)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Pipeline parado para camera %s", self.camera_id)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- Loop principal (roda em thread) ---

    def _run(self) -> None:
        detector = TrackedDetector()
        analyzers: list[BaseAnalyzer] = [
            PerimeterGuard(),
            LoiterDetector(),
            CrowdDetector(),
            AbandonDetector(),
            FightDetector(),
        ]
        clip_buf: ClipBuffer | None = None
        cap = cv2.VideoCapture(self._source if isinstance(self._source, str) else int(self._source))

        if not cap.isOpened():
            logger.error("Nao foi possivel abrir a fonte: %s", self._source)
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        clip_buf = ClipBuffer(self.camera_id, fps=src_fps)
        frame_idx = 0
        alerts_total = 0

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # Loop para arquivos de video; para para streams
                    if isinstance(self._source, str) and not self._source.startswith("rtsp"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                frame_idx += 1
                if frame_idx % self._frame_skip != 0:
                    continue

                t0 = time.perf_counter()
                now = time.monotonic()

                # Deteccao + rastreamento
                detections = detector.detect_and_track(frame)
                poses = detector.detect_poses_tracked(frame)

                inference_ms = (time.perf_counter() - t0) * 1000

                # Anotar frame
                annotated = self._annotate(frame.copy(), detections, poses)

                # Enviar ao buffer de clip
                clip_buf.push(annotated)

                # Snapshot das zonas
                with self._zones_lock:
                    zones_snap = list(self._zones)

                h, w = frame.shape[:2]

                # Rodar analisadores
                new_alerts: list[AlertCreate] = []
                for analyzer in analyzers:
                    new_alerts.extend(
                        analyzer.update(self.camera_id, detections, poses, zones_snap, now, w, h)
                    )

                # Persistir alertas e salvar clips
                for alert_data in new_alerts:
                    clip_path = clip_buf.save_clip(event_time=now)
                    alert_data.clip_path = clip_path
                    self._persist_alert(alert_data)
                    alerts_total += 1

                # Registrar pontos de heatmap (centros de pessoas)
                heatmap_pts = [
                    (d.bbox.center.x / w, d.bbox.center.y / h)
                    for d in detections
                    if d.class_name == "person"
                ]
                if heatmap_pts:
                    self._record_heatmap(heatmap_pts)

                # Atualizar estado compartilhado
                person_count = sum(1 for d in detections if d.class_name == "person")
                vehicle_count = sum(1 for d in detections if d.class_name in {"car", "motorcycle", "bus", "truck"})
                track_ids = {d.track_id for d in detections if d.track_id is not None}

                elapsed = time.perf_counter() - t0
                with self.state.lock:
                    self.state.frame = frame
                    self.state.annotated = annotated
                    self.state.detections = detections
                    self.state.poses = poses
                    self.state.person_count = person_count
                    self.state.vehicle_count = vehicle_count
                    self.state.active_tracks = len(track_ids)
                    self.state.fps = 1.0 / max(elapsed, 1e-6)
                    self.state.inference_ms = inference_ms
                    self.state.alerts_total = alerts_total

        except Exception:
            logger.exception("Pipeline %s crashou", self.camera_id)
        finally:
            cap.release()
            logger.info("Camera %s liberada", self.camera_id)

    # --- Helpers ---

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        detections: list[Detection],
        poses: list[PoseDetection],
    ) -> np.ndarray:
        for det in detections:
            b = det.bbox
            color = (0, 255, 0) if det.class_name == "person" else (255, 165, 0)
            cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            label = f"{det.class_name}"
            if det.track_id is not None:
                label += f" #{det.track_id}"
            cv2.putText(
                frame, label, (int(b.x1), int(b.y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        # Desenhar esqueletos de pose
        _SKELETON = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        ]
        for pose in poses:
            kps = pose.keypoints.xy
            for i, j in _SKELETON:
                if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                    pt1 = (int(kps[i][0]), int(kps[i][1]))
                    pt2 = (int(kps[j][0]), int(kps[j][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
            for kp in kps:
                if kp[2] > 0.3:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

        return frame

    def _persist_alert(self, alert_data: AlertCreate) -> None:
        """Agenda insercao no BD no event loop asyncio (thread-safe)."""
        import asyncio
        asyncio.run_coroutine_threadsafe(self._db.insert_alert(alert_data), self._loop)

    def _record_heatmap(self, points: list[tuple[float, float]]) -> None:
        import asyncio
        asyncio.run_coroutine_threadsafe(
            self._db.record_heatmap_points(self.camera_id, points), self._loop
        )
