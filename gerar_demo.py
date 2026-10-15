"""Gera videos de demonstracao do AEGIS para publicacao.

Produz dois formatos:
  - demo_linkedin.mp4  (16:9, 1280x720, ~60s) para LinkedIn
  - demo_reels.mp4     (9:16, 1080x1920, ~30s) para Instagram Reels

Uso:
    python gerar_demo.py --source videos/paulista_domingo.mp4
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from aegis.core.config import get_settings
from aegis.core.schemas import AlertCreate, Detection, PoseDetection
from aegis.detection.tracker import TrackedDetector
from aegis.analyzers.perimeter import PerimeterGuard
from aegis.analyzers.loiter import LoiterDetector
from aegis.analyzers.crowd import CrowdDetector
from aegis.analyzers.abandon import AbandonDetector
from aegis.analyzers.fight import FightDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("aegis.demo")

# Cores do tema
COR_HEADER = (20, 20, 20)
COR_ALERTA_BG = (0, 0, 60)
COR_VERDE = (0, 230, 118)
COR_AMARELO = (0, 200, 255)
COR_VERMELHO = (60, 60, 255)
COR_CIANO = (255, 200, 0)
COR_TEXTO = (255, 255, 255)


def _put_text(frame, text, pos, scale=0.5, color=COR_TEXTO, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_header(frame, fps, person_count, vehicle_count, track_count, total_alerts, frame_w):
    """Painel superior com metricas."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 50), COR_HEADER, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Logo / titulo
    _put_text(frame, "AEGIS", (12, 35), 0.8, COR_VERDE, 2)

    # Metricas
    x = 120
    metricas = [
        (f"FPS: {fps:.0f}", COR_VERDE),
        (f"Pessoas: {person_count}", COR_CIANO),
        (f"Veiculos: {vehicle_count}", COR_AMARELO),
        (f"Tracks: {track_count}", COR_TEXTO),
        (f"Alertas: {total_alerts}", COR_VERMELHO if total_alerts > 0 else COR_TEXTO),
    ]
    for text, color in metricas:
        _put_text(frame, text, (x, 33), 0.45, color)
        x += len(text) * 12 + 20

    return frame


def draw_alerts_panel(frame, alerts, frame_h, frame_w):
    """Painel lateral/inferior com alertas recentes."""
    if not alerts:
        return frame
    panel_h = min(len(alerts) * 30 + 10, 100)
    y_start = frame_h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (frame_w, frame_h), COR_ALERTA_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, a in enumerate(alerts[-3:]):
        color = COR_VERMELHO if a.severity.value in ("critical", "high") else COR_AMARELO
        sev = a.severity.value.upper()
        text = f"[{sev}] {a.message}"
        if len(text) > 80:
            text = text[:77] + "..."
        _put_text(frame, text, (10, y_start + 22 + i * 28), 0.4, color)

    return frame


def annotate_detections(frame, detections, poses):
    """Bboxes, labels, esqueletos."""
    for det in detections:
        b = det.bbox
        if det.class_name == "person":
            color = COR_VERDE
        elif det.class_name in {"car", "motorcycle", "bus", "truck"}:
            color = COR_AMARELO
        else:
            color = (200, 200, 200)

        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        label = det.class_name
        if det.track_id is not None:
            label += f" #{det.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (int(b.x1), int(b.y1) - th - 6), (int(b.x1) + tw + 4, int(b.y1)), color, -1)
        cv2.putText(frame, label, (int(b.x1) + 2, int(b.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

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


def crop_to_reels(frame, target_w=1080, target_h=1920):
    """Recorta frame 16:9 para 9:16 (centro) com bordas preenchidas se necessario."""
    h, w = frame.shape[:2]
    # Escala para que a largura caiba em target_w
    scale = target_w / w
    new_w = target_w
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    if new_h >= target_h:
        # Corta verticalmente (topo e baixo)
        y_start = (new_h - target_h) // 2
        return resized[y_start:y_start + target_h, :]
    else:
        # Preenche com preto em cima e embaixo
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, :] = resized
        return canvas


def process_video(source_path, out_linkedin, out_reels, max_frames_linkedin=0, max_frames_reels=900):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.error("Nao foi possivel abrir: %s", source_path)
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    logger.info("Fonte: %s (%dx%d @ %.1f FPS, %d frames)", source_path, w, h, src_fps, total)

    logger.info("Carregando modelos YOLOv8...")
    detector = TrackedDetector()
    analyzers = [PerimeterGuard(), LoiterDetector(), CrowdDetector(), AbandonDetector(), FightDetector()]

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer_lk = cv2.VideoWriter(out_linkedin, fourcc, src_fps, (w, h))
    writer_rl = cv2.VideoWriter(out_reels, fourcc, src_fps, (1080, 1920))

    cfg = get_settings()
    frame_skip = cfg.frame_skip
    frame_idx = 0
    total_alerts = 0
    recent_alerts: list[AlertCreate] = []

    # Cache do ultimo frame anotado (pra frames pulados)
    last_annotated = None
    last_reels = None

    logger.info("Processando frames...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames_linkedin and frame_idx > max_frames_linkedin:
                break

            if frame_idx % frame_skip != 0:
                if last_annotated is not None:
                    writer_lk.write(last_annotated)
                    if frame_idx <= max_frames_reels and last_reels is not None:
                        writer_rl.write(last_reels)
                continue

            t0 = time.perf_counter()
            now = time.monotonic()

            detections = detector.detect_and_track(frame)
            poses = detector.detect_poses_tracked(frame)

            elapsed = time.perf_counter() - t0
            fps_val = 1.0 / max(elapsed, 1e-6)

            # Analisadores
            new_alerts: list[AlertCreate] = []
            for analyzer in analyzers:
                new_alerts.extend(analyzer.update("demo", detections, poses, [], now, w, h))
            total_alerts += len(new_alerts)
            recent_alerts.extend(new_alerts)
            recent_alerts = recent_alerts[-5:]

            for a in new_alerts:
                logger.info("ALERTA: [%s] %s", a.severity.value.upper(), a.message)

            person_count = sum(1 for d in detections if d.class_name == "person")
            vehicle_count = sum(1 for d in detections if d.class_name in {"car", "motorcycle", "bus", "truck"})
            track_ids = {d.track_id for d in detections if d.track_id is not None}

            annotated = annotate_detections(frame.copy(), detections, poses)
            annotated = draw_header(annotated, fps_val, person_count, vehicle_count, len(track_ids), total_alerts, w)
            annotated = draw_alerts_panel(annotated, recent_alerts, h, w)

            last_annotated = annotated
            writer_lk.write(annotated)

            # Reels
            if frame_idx <= max_frames_reels:
                reels_frame = crop_to_reels(annotated)
                last_reels = reels_frame
                writer_rl.write(reels_frame)

            if frame_idx % 50 == 0:
                pct = (frame_idx / total * 100) if total else 0
                logger.info("Frame %d/%d (%.0f%%) | FPS inf: %.1f | Alertas: %d",
                            frame_idx, total, pct, fps_val, total_alerts)

    finally:
        cap.release()
        writer_lk.release()
        writer_rl.release()

    logger.info("Concluido! %d frames processados, %d alertas gerados", frame_idx, total_alerts)
    logger.info("LinkedIn: %s", out_linkedin)
    logger.info("Reels:    %s", out_reels)


def main():
    parser = argparse.ArgumentParser(description="Gera videos de demo do AEGIS para LinkedIn e Instagram Reels")
    parser.add_argument("--source", required=True, help="Video de entrada")
    parser.add_argument("--linkedin", default="demo_linkedin.mp4", help="Saida LinkedIn (16:9)")
    parser.add_argument("--reels", default="demo_reels.mp4", help="Saida Reels (9:16)")
    parser.add_argument("--max-frames", type=int, default=0, help="Limite de frames (0 = tudo)")
    parser.add_argument("--reels-seconds", type=int, default=30, help="Duracao max do Reels em segundos")
    args = parser.parse_args()

    cap_check = cv2.VideoCapture(args.source)
    fps = cap_check.get(cv2.CAP_PROP_FPS) or 30
    cap_check.release()
    reels_max = int(fps * args.reels_seconds)

    process_video(args.source, args.linkedin, args.reels, args.max_frames, reels_max)


if __name__ == "__main__":
    main()
