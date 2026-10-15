"""Script de demonstracao do AEGIS.

Gera um video de demonstracao mostrando o sistema processando um feed de camera
com deteccao, rastreamento, anotacoes e alertas sobrepostos.

Uso:
    python demo.py                          # usa webcam (dispositivo 0)
    python demo.py --source videos/rua.mp4  # usa arquivo de video
    python demo.py --source rtsp://...      # usa stream RTSP

O video de saida e salvo em demo_output.mp4 por padrao.
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from aegis.core.config import get_settings
from aegis.core.schemas import AlertCreate, Detection, PoseDetection, Zone, Point
from aegis.detection.tracker import TrackedDetector
from aegis.analyzers.perimeter import PerimeterGuard
from aegis.analyzers.loiter import LoiterDetector
from aegis.analyzers.crowd import CrowdDetector
from aegis.analyzers.abandon import AbandonDetector
from aegis.analyzers.fight import FightDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aegis.demo")


def draw_overlay(frame: np.ndarray, fps: float, alerts: list[AlertCreate],
                 person_count: int, vehicle_count: int, track_count: int,
                 total_alerts: int) -> np.ndarray:
    """Desenha painel de informacoes e alertas sobre o frame."""
    h, w = frame.shape[:2]

    # Painel superior semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Info line
    info = (
        f"AEGIS | FPS: {fps:.1f} | "
        f"Pessoas: {person_count} | Veiculos: {vehicle_count} | "
        f"Tracks: {track_count} | Alertas: {total_alerts}"
    )
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Painel de alertas recentes (canto inferior)
    if alerts:
        y_start = h - 10 - len(alerts) * 35
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, y_start - 10), (w, h), (0, 0, 80), -1)
        frame = cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0)
        for i, alert in enumerate(alerts[-3:]):
            color = (0, 0, 255) if alert.severity.value == "critical" else (0, 165, 255)
            text = f"[{alert.severity.value.upper()}] {alert.message}"
            cv2.putText(frame, text, (10, y_start + i * 35 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame


def annotate_frame(frame: np.ndarray, detections: list[Detection],
                   poses: list[PoseDetection]) -> np.ndarray:
    """Desenha bounding boxes, labels e esqueletos de pose."""
    for det in detections:
        b = det.bbox
        if det.class_name == "person":
            color = (0, 255, 0)
        elif det.class_name in {"car", "motorcycle", "bus", "truck"}:
            color = (255, 165, 0)
        else:
            color = (255, 255, 0)
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        label = det.class_name
        if det.track_id is not None:
            label += f" #{det.track_id}"
        label += f" {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (int(b.x1), int(b.y1) - th - 6), (int(b.x1) + tw + 4, int(b.y1)), color, -1)
        cv2.putText(frame, label, (int(b.x1) + 2, int(b.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Esqueletos de pose
    SKELETON = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    for pose in poses:
        kps = pose.keypoints.xy
        for i, j in SKELETON:
            if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
        for kp in kps:
            if kp[2] > 0.3:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="AEGIS - Demo de deteccao de ameacas urbanas")
    parser.add_argument("--source", default="0", help="Fonte: caminho do video, URL RTSP, ou indice de webcam (padrao: 0)")
    parser.add_argument("--output", default="demo_output.mp4", help="Caminho do video de saida")
    parser.add_argument("--max-frames", type=int, default=0, help="Limite de frames (0 = sem limite)")
    parser.add_argument("--show", action="store_true", help="Mostrar janela ao vivo (requer display)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Nao foi possivel abrir a fonte: %s", args.source)
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    logger.info("Fonte: %s (%dx%d @ %.1f FPS, %d frames)", args.source, w, h, src_fps, total)

    # Inicializa detector e analisadores
    logger.info("Carregando modelos...")
    detector = TrackedDetector()
    analyzers = [
        PerimeterGuard(),
        LoiterDetector(),
        CrowdDetector(),
        AbandonDetector(),
        FightDetector(),
    ]

    # Writer de saida
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, src_fps, (w, h))

    cfg = get_settings()
    frame_skip = cfg.frame_skip
    frame_idx = 0
    total_alerts = 0
    recent_alerts: list[AlertCreate] = []

    logger.info("Processando...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            if frame_idx % frame_skip != 0:
                out.write(frame)
                continue

            t0 = time.perf_counter()
            now = time.monotonic()

            # Deteccao + rastreamento
            detections = detector.detect_and_track(frame)
            poses = detector.detect_poses_tracked(frame)

            elapsed = time.perf_counter() - t0
            fps = 1.0 / max(elapsed, 1e-6)

            # Rodar analisadores
            new_alerts: list[AlertCreate] = []
            for analyzer in analyzers:
                new_alerts.extend(
                    analyzer.update("demo", detections, poses, [], now, w, h)
                )
            total_alerts += len(new_alerts)
            recent_alerts.extend(new_alerts)
            recent_alerts = recent_alerts[-5:]  # manter ultimos 5

            for a in new_alerts:
                logger.info("ALERTA: [%s] %s", a.severity.value.upper(), a.message)

            # Anotar frame
            annotated = annotate_frame(frame.copy(), detections, poses)

            person_count = sum(1 for d in detections if d.class_name == "person")
            vehicle_count = sum(1 for d in detections if d.class_name in {"car", "motorcycle", "bus", "truck"})
            track_ids = {d.track_id for d in detections if d.track_id is not None}

            annotated = draw_overlay(
                annotated, fps, recent_alerts,
                person_count, vehicle_count, len(track_ids), total_alerts,
            )

            out.write(annotated)

            if args.show:
                cv2.imshow("AEGIS Demo", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 100 == 0:
                logger.info("Frame %d/%d | FPS: %.1f | Alertas: %d", frame_idx, total, fps, total_alerts)

    finally:
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()

    logger.info("Demo concluida: %d frames processados, %d alertas gerados", frame_idx, total_alerts)
    logger.info("Video salvo em: %s", args.output)


if __name__ == "__main__":
    main()
