"""Configuracao do AEGIS — fonte unica de verdade."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", extra="ignore")

    # --- Deteccao ---
    det_model: str = Field("yolov8n.pt", description="Caminho ou nome do modelo YOLO de deteccao")
    pose_model: str = Field("yolov8n-pose.pt", description="Caminho ou nome do modelo YOLO de pose")
    det_conf: float = Field(0.35, ge=0.0, le=1.0)
    det_iou: float = Field(0.45, ge=0.0, le=1.0)
    device: str = Field("cuda:0", description="Dispositivo torch (cuda:0, cpu, etc.)")

    # --- Rastreamento ---
    track_buffer: int = Field(30, description="Buffer de perda do ByteTrack (frames)")

    # --- Analisadores ---
    crowd_baseline_window_h: int = Field(1, description="Horas para baseline adaptativo de multidao")
    crowd_density_sigma: float = Field(2.5, description="Multiplicador de desvio para anomalia de multidao")
    abandon_timeout_s: float = Field(30.0, description="Segundos ate considerar objeto abandonado")
    loiter_timeout_s: float = Field(60.0, description="Segundos de veiculo na zona antes do alerta")
    fight_seq_len: int = Field(16, description="Frames de historico de pose para classificacao de briga")
    fight_threshold: float = Field(0.6, ge=0.0, le=1.0)

    # --- Pipeline ---
    frame_skip: int = Field(2, description="Processar a cada N frames (1=todos)")
    clip_buffer_s: float = Field(10.0, description="Segundos do buffer circular de frames para clips")
    clip_pre_s: float = Field(5.0, description="Segundos antes do evento no clip salvo")
    clip_post_s: float = Field(5.0, description="Segundos apos o evento no clip salvo")

    # --- Armazenamento ---
    db_url: str = Field("sqlite+aiosqlite:///aegis.db")
    clips_dir: Path = Field(Path("clips"))

    # --- API ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
