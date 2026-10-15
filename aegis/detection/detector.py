"""Wrapper de deteccao YOLOv8 — carrega modelos de deteccao e pose."""

from __future__ import annotations

import logging
from ultralytics import YOLO
from aegis.core.config import get_settings
from aegis.core.schemas import BBox, Detection, Keypoints, PoseDetection

logger = logging.getLogger(__name__)


class Detector:
    """Wrapper fino sobre ultralytics.YOLO para deteccao + pose."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._det_model = YOLO(cfg.det_model)
        self._pose_model = YOLO(cfg.pose_model)
        self._conf = cfg.det_conf
        self._iou = cfg.det_iou
        self._device = cfg.device
        logger.info(
            "Modelos carregados: det=%s  pose=%s  device=%s",
            cfg.det_model, cfg.pose_model, cfg.device,
        )

    def detect(self, frame) -> list[Detection]:
        """Roda deteccao de objetos e retorna lista de Detection."""
        results = self._det_model(
            frame, conf=self._conf, iou=self._iou, device=self._device, verbose=False
        )
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
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
                    )
                )
        return detections

    def detect_poses(self, frame) -> list[PoseDetection]:
        """Roda estimacao de pose e retorna lista de PoseDetection."""
        results = self._pose_model(
            frame, conf=self._conf, device=self._device, verbose=False
        )
        poses: list[PoseDetection] = []
        for r in results:
            if r.keypoints is None:
                continue
            for i, box in enumerate(r.boxes):
                kps_data = r.keypoints[i].data[0]  # (17, 3)
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
                        keypoints=Keypoints(xy=kps),
                    )
                )
        return poses
