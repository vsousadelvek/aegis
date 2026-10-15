"""Monta o video final de demonstracao do AEGIS combinando multiplas fontes.

Estrutura do video:
  1. Tela de abertura (3s) — logo AEGIS + titulo
  2. Clip 1 — video processado (centro SP)
  3. Transicao fade
  4. Clip 2 — video processado (Paulista)
  5. Tela de encerramento (3s) — GitHub + stack

Gera LinkedIn (16:9) e Reels (9:16).

Uso:
    python montar_demo_final.py
"""

from __future__ import annotations
import argparse
import logging
import os
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
COR_FUNDO = (15, 15, 15)
COR_VERDE = (0, 230, 118)
COR_VERDE_ESCURO = (0, 160, 80)
COR_AMARELO = (0, 200, 255)
COR_VERMELHO = (60, 60, 255)
COR_CIANO = (255, 200, 0)
COR_CINZA = (180, 180, 180)
COR_TEXTO = (255, 255, 255)
COR_HEADER_BG = (20, 20, 20)
COR_ALERTA_BG = (0, 0, 50)

TARGET_W = 1280
TARGET_H = 720
TARGET_FPS = 30


def _put_text(frame, text, pos, scale=0.5, color=COR_TEXTO, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _put_text_centered(frame, text, y, scale=0.8, color=COR_TEXTO, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    _put_text(frame, text, (x, y), scale, color, thickness)


def gerar_tela_intro(writer, n_frames, w, h):
    """Tela de abertura com logo e titulo."""
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = COR_FUNDO

        # Fade in
        alpha = min(1.0, i / (n_frames * 0.3))

        # Linhas decorativas
        cv2.line(frame, (w // 4, h // 2 - 80), (3 * w // 4, h // 2 - 80), COR_VERDE_ESCURO, 1, cv2.LINE_AA)
        cv2.line(frame, (w // 4, h // 2 + 80), (3 * w // 4, h // 2 + 80), COR_VERDE_ESCURO, 1, cv2.LINE_AA)

        # Titulo
        _put_text_centered(frame, "AEGIS", h // 2 - 20, 2.0, tuple(int(c * alpha) for c in COR_VERDE), 3)
        _put_text_centered(frame, "Sistema de Deteccao de Ameacas Urbanas", h // 2 + 30, 0.6,
                           tuple(int(c * alpha) for c in COR_CINZA), 1)
        _put_text_centered(frame, "Vigilancia inteligente em tempo real", h // 2 + 60, 0.5,
                           tuple(int(c * alpha) for c in COR_CINZA), 1)

        writer.write(frame)


def gerar_tela_stats(writer, n_frames, w, h, total_detections, total_tracks, total_alerts, fps_media):
    """Tela com estatisticas finais."""
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = COR_FUNDO

        alpha = min(1.0, i / (n_frames * 0.3))

        cv2.line(frame, (w // 4, 120), (3 * w // 4, 120), COR_VERDE_ESCURO, 1, cv2.LINE_AA)

        _put_text_centered(frame, "AEGIS", 100, 1.2, tuple(int(c * alpha) for c in COR_VERDE), 2)

        y = 200
        stats = [
            ("Deteccoes processadas", f"{total_detections:,}".replace(",", "."), COR_CIANO),
            ("Objetos rastreados", f"{total_tracks}", COR_AMARELO),
            ("Alertas gerados", f"{total_alerts}", COR_VERMELHO),
            ("FPS medio (CPU)", f"{fps_media:.1f}", COR_VERDE),
        ]
        for label, value, color in stats:
            color_faded = tuple(int(c * alpha) for c in color)
            _put_text(frame, f"{label}:", (w // 4, y), 0.55, tuple(int(c * alpha) for c in COR_CINZA), 1)
            _put_text(frame, value, (w // 4 + 320, y), 0.7, color_faded, 2)
            y += 50

        y += 30
        cv2.line(frame, (w // 4, y - 10), (3 * w // 4, y - 10), COR_VERDE_ESCURO, 1, cv2.LINE_AA)

        _put_text_centered(frame, "YOLOv8 | ByteTrack | FastAPI | Docker", y + 30, 0.5,
                           tuple(int(c * alpha) for c in COR_CINZA), 1)
        _put_text_centered(frame, "github.com/vsousadelvek/aegis", y + 70, 0.6,
                           tuple(int(c * alpha) for c in COR_VERDE), 1)

        writer.write(frame)


def draw_header(frame, fps, person_count, vehicle_count, track_count, total_alerts, frame_w):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 48), COR_HEADER_BG, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    _put_text(frame, "AEGIS", (12, 33), 0.75, COR_VERDE, 2)

    x = 110
    metricas = [
        (f"FPS: {fps:.0f}", COR_VERDE),
        (f"Pessoas: {person_count}", COR_CIANO),
        (f"Veiculos: {vehicle_count}", COR_AMARELO),
        (f"Tracks: {track_count}", COR_TEXTO),
        (f"Alertas: {total_alerts}", COR_VERMELHO if total_alerts > 0 else COR_TEXTO),
    ]
    for text, color in metricas:
        _put_text(frame, text, (x, 31), 0.42, color)
        x += len(text) * 11 + 18

    # Barra verde fina embaixo do header
    cv2.line(frame, (0, 48), (frame_w, 48), COR_VERDE_ESCURO, 1)

    return frame


def draw_alerts_panel(frame, alerts, frame_h, frame_w):
    if not alerts:
        return frame
    n = min(len(alerts), 3)
    panel_h = n * 28 + 12
    y_start = frame_h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (frame_w, frame_h), COR_ALERTA_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.line(frame, (0, y_start), (frame_w, y_start), COR_VERMELHO, 1)

    for i, a in enumerate(alerts[-3:]):
        color = COR_VERMELHO if a.severity.value in ("critical", "high") else COR_AMARELO
        sev = a.severity.value.upper()[:4]
        text = f"[{sev}] {a.message}"
        if len(text) > 90:
            text = text[:87] + "..."
        _put_text(frame, text, (10, y_start + 20 + i * 28), 0.38, color)

    return frame


def annotate_detections(frame, detections, poses):
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
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(frame, (int(b.x1), int(b.y1) - th - 5), (int(b.x1) + tw + 4, int(b.y1)), color, -1)
        cv2.putText(frame, label, (int(b.x1) + 2, int(b.y1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

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


def resize_to_target(frame, tw, th):
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y_off = (th - nh) // 2
    x_off = (tw - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def crop_to_reels(frame, target_w=1080, target_h=1920):
    h, w = frame.shape[:2]
    scale = target_w / w
    new_w = target_w
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    if new_h >= target_h:
        y_start = (new_h - target_h) // 2
        return resized[y_start:y_start + target_h, :]
    else:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, :] = resized
        return canvas


def fade_transition(writer_lk, writer_rl, last_frame, n_frames=15):
    """Fade to black entre clips."""
    for i in range(n_frames):
        alpha = 1.0 - (i / n_frames)
        faded = (last_frame * alpha).astype(np.uint8)
        resized = resize_to_target(faded, TARGET_W, TARGET_H)
        writer_lk.write(resized)
        if writer_rl:
            writer_rl.write(crop_to_reels(resized))


def processar_clip(source_path, writer_lk, writer_rl, detector, analyzers, max_frames, frame_skip,
                   write_reels=True):
    """Processa um video e escreve nos writers."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.error("Nao foi possivel abrir: %s", source_path)
        return 0, 0, 0, 0.0, None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    logger.info("Clip: %s (%dx%d, %d frames)", source_path, w, h, total_src)

    frame_idx = 0
    total_alerts = 0
    total_detections = 0
    all_tracks = set()
    recent_alerts: list[AlertCreate] = []
    fps_values = []
    last_annotated = None

    # Resetar estado dos analisadores
    for a in analyzers:
        if hasattr(a, '_history'):
            a._history.clear()
        if hasattr(a, '_tracks'):
            a._tracks.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        if frame_idx % frame_skip != 0:
            if last_annotated is not None:
                out = resize_to_target(last_annotated, TARGET_W, TARGET_H)
                writer_lk.write(out)
                if write_reels and writer_rl:
                    writer_rl.write(crop_to_reels(out))
            continue

        t0 = time.perf_counter()
        now = time.monotonic()

        detections = detector.detect_and_track(frame)
        poses = detector.detect_poses_tracked(frame)

        elapsed = time.perf_counter() - t0
        fps_val = 1.0 / max(elapsed, 1e-6)
        fps_values.append(fps_val)

        total_detections += len(detections)
        for d in detections:
            if d.track_id is not None:
                all_tracks.add(d.track_id)

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
        out = resize_to_target(annotated, TARGET_W, TARGET_H)
        writer_lk.write(out)
        if write_reels and writer_rl:
            writer_rl.write(crop_to_reels(out))

        if frame_idx % 100 == 0:
            logger.info("  Frame %d/%d | FPS: %.1f | Alertas: %d", frame_idx, total_src, fps_val, total_alerts)

    cap.release()
    fps_media = sum(fps_values) / len(fps_values) if fps_values else 0
    logger.info("  Clip concluido: %d frames, %d deteccoes, %d tracks, %d alertas",
                frame_idx, total_detections, len(all_tracks), total_alerts)

    return total_detections, len(all_tracks), total_alerts, fps_media, last_annotated


def main():
    parser = argparse.ArgumentParser(description="Monta demo final do AEGIS")
    parser.add_argument("--clips", nargs="+", default=[
        "videos/sp_centro_walkng.mp4",
        "videos/paulista_domingo.mp4",
    ], help="Videos de entrada")
    parser.add_argument("--linkedin", default="demo_final_linkedin.mp4")
    parser.add_argument("--reels", default="demo_final_reels.mp4")
    parser.add_argument("--max-per-clip", type=int, default=0, help="Max frames por clip (0 = tudo)")
    parser.add_argument("--reels-total-seconds", type=int, default=30, help="Duracao total do Reels")
    args = parser.parse_args()

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer_lk = cv2.VideoWriter(args.linkedin, fourcc, TARGET_FPS, (TARGET_W, TARGET_H))
    writer_rl = cv2.VideoWriter(args.reels, fourcc, TARGET_FPS, (1080, 1920))

    logger.info("Carregando modelos...")
    detector = TrackedDetector()
    analyzers = [PerimeterGuard(), LoiterDetector(), CrowdDetector(), AbandonDetector(), FightDetector()]

    cfg = get_settings()
    frame_skip = cfg.frame_skip

    # Intro (3s)
    logger.info("Gerando intro...")
    gerar_tela_intro(writer_lk, TARGET_FPS * 3, TARGET_W, TARGET_H)
    # Intro Reels
    for _ in range(TARGET_FPS * 3):
        intro_frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        gerar_tela_intro.__code__  # dummy
        break
    # Gerar intro separado pro reels
    intro_frames = []
    for i in range(TARGET_FPS * 3):
        frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        frame[:] = COR_FUNDO
        alpha = min(1.0, i / (TARGET_FPS * 0.9))
        cv2.line(frame, (TARGET_W // 4, TARGET_H // 2 - 80), (3 * TARGET_W // 4, TARGET_H // 2 - 80), COR_VERDE_ESCURO, 1)
        cv2.line(frame, (TARGET_W // 4, TARGET_H // 2 + 80), (3 * TARGET_W // 4, TARGET_H // 2 + 80), COR_VERDE_ESCURO, 1)
        _put_text_centered(frame, "AEGIS", TARGET_H // 2 - 20, 2.0, tuple(int(c * alpha) for c in COR_VERDE), 3)
        _put_text_centered(frame, "Deteccao de Ameacas Urbanas", TARGET_H // 2 + 30, 0.6,
                           tuple(int(c * alpha) for c in COR_CINZA), 1)
        writer_rl.write(crop_to_reels(frame))

    # Processar clips
    total_det = 0
    total_trk = 0
    total_alr = 0
    fps_all = []
    reels_frame_count = TARGET_FPS * 3  # ja escrevemos a intro
    reels_max = TARGET_FPS * args.reels_total_seconds

    for idx, clip_path in enumerate(args.clips):
        if not os.path.exists(clip_path):
            logger.warning("Clip nao encontrado: %s", clip_path)
            continue

        logger.info("Processando clip %d/%d: %s", idx + 1, len(args.clips), clip_path)
        write_reels = reels_frame_count < reels_max

        det, trk, alr, fps_m, last_frame = processar_clip(
            clip_path, writer_lk, writer_rl, detector, analyzers,
            args.max_per_clip, frame_skip, write_reels,
        )
        total_det += det
        total_trk += trk
        total_alr += alr
        fps_all.append(fps_m)

        # Transicao entre clips
        if idx < len(args.clips) - 1 and last_frame is not None:
            fade_transition(writer_lk, writer_rl if write_reels else None, last_frame)

    # Tela final (4s)
    logger.info("Gerando tela final...")
    fps_media = sum(fps_all) / len(fps_all) if fps_all else 0
    n_final = TARGET_FPS * 4
    for i in range(n_final):
        frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        frame[:] = COR_FUNDO
        alpha = min(1.0, i / (n_final * 0.3))

        cv2.line(frame, (TARGET_W // 4, 120), (3 * TARGET_W // 4, 120), COR_VERDE_ESCURO, 1)
        _put_text_centered(frame, "AEGIS", 100, 1.2, tuple(int(c * alpha) for c in COR_VERDE), 2)

        y = 190
        stats = [
            ("Deteccoes", f"{total_det:,}".replace(",", "."), COR_CIANO),
            ("Objetos rastreados", f"{total_trk}", COR_AMARELO),
            ("Alertas gerados", f"{total_alr}", COR_VERMELHO),
            ("FPS medio (CPU)", f"{fps_media:.1f}", COR_VERDE),
        ]
        for label, value, color in stats:
            cf = tuple(int(c * alpha) for c in color)
            _put_text(frame, f"{label}:", (TARGET_W // 4, y), 0.55, tuple(int(c * alpha) for c in COR_CINZA), 1)
            _put_text(frame, value, (TARGET_W // 4 + 300, y), 0.65, cf, 2)
            y += 45

        y += 20
        cv2.line(frame, (TARGET_W // 4, y), (3 * TARGET_W // 4, y), COR_VERDE_ESCURO, 1)
        _put_text_centered(frame, "YOLOv8 | ByteTrack | FastAPI | Docker", y + 35, 0.5,
                           tuple(int(c * alpha) for c in COR_CINZA), 1)
        _put_text_centered(frame, "github.com/vsousadelvek/aegis", y + 70, 0.6,
                           tuple(int(c * alpha) for c in COR_VERDE), 1)

        writer_lk.write(frame)
        writer_rl.write(crop_to_reels(frame))

    writer_lk.release()
    writer_rl.release()

    for f in [args.linkedin, args.reels]:
        cap = cv2.VideoCapture(f)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = frames / TARGET_FPS
        size = os.path.getsize(f) / 1024 / 1024
        w_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        logger.info("%s: %dx%d | %d frames | %.1fs | %.1fMB", f, w_, h_, frames, dur, size)

    logger.info("Pronto!")


if __name__ == "__main__":
    main()
