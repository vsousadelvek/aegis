"""Rastreamento ByteTrack via Ultralytics .track() — deteccao + pose com IDs persistentes."""

from __future__ import annotations

import logging
from ultralytics import YOLO
from aegis.core.config import get_settings
from aegis.core.schemas import BBox, Detection, Keypoints, PoseDetection

logger = logging.getLogger(__name__)


class TrackedDetector:
    """Combina YOLO .track() para atribuir IDs persistentes via ByteTrack."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._det_model = YOLO(cfg.det_model)
        self._pose_model = YOLO(cfg.pose_model)
        self._conf = cfg.det_conf
        self._iou = cfg.det_iou
        self._device = cfg.device
        self._track_buffer = cfg.track_buffer
        logger.info(
            "TrackedDetector inicializado: det=%s  pose=%s  device=%s",
            cfg.det_model, cfg.pose_model, cfg.device,
        )

    def detect_and_track(self, frame) -> list[Detection]:
        """Deteccao + rastreamento ByteTrack: retorna deteccoes com track_id."""
        results = self._det_model.track(
            frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                tid = int(box.id[0]) if box.id is not None else None
                detections.append(
                    Detection(
                        bbox=BBox(
                            x1=float(box.xyxy[0][0]),
                            y1=float(box.xyxy[0][1]),
                            x2=float(box.xyxy[0][2]),
                            y2=float(box.xyxy[0][3]),
                        ),
                        class_id=cls_id,
                        class_name=r.names[cls_id],
                        confidence=float(box.conf[0]),
                        track_id=tid,
                    )
                )
        return detections

    def detect_poses_tracked(self, frame) -> list[PoseDetection]:
        """Estimacao de pose + rastreamento: retorna poses com track_id."""
        results = self._pose_model.track(
            frame,
            conf=self._conf,
            device=self._device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        poses: list[PoseDetection] = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            for i, box in enumerate(r.boxes):
                tid = int(box.id[0]) if box.id is not None else None
                kps_data = r.keypoints[i].data[0]
                kps = [
                    (float(kps_data[j][0]), float(kps_data[j][1]), float(kps_data[j][2]))
                    for j in range(kps_data.shape[0])
                ]
                poses.append(
                    PoseDetection(
                        bbox=BBox(
                            x1=float(box.xyxy[0][0]),
                            y1=float(box.xyxy[0][1]),
                            x2=float(box.xyxy[0][2]),
                            y2=float(box.xyxy[0][3]),
                        ),
                        confidence=float(box.conf[0]),
                        track_id=tid,
                        keypoints=Keypoints(xy=kps),
                    )
                )
        return poses
