"""Schemas de dominio compartilhados pelo sistema."""

from __future__ import annotations

import enum
from datetime import datetime
from pydantic import BaseModel, Field


# --- Enums ---

class ThreatType(str, enum.Enum):
    FIGHT = "fight"
    ABANDONED_OBJECT = "abandoned_object"
    CROWD_ANOMALY = "crowd_anomaly"
    LOITERING_VEHICLE = "loitering_vehicle"
    PERIMETER_BREACH = "perimeter_breach"


class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# --- Geometria ---

class Point(BaseModel):
    x: float
    y: float


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Point:
        return Point(x=(self.x1 + self.x2) / 2, y=(self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


# --- Deteccao ---

class Detection(BaseModel):
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None


class Keypoints(BaseModel):
    """17 keypoints COCO: cada um eh (x, y, confianca)."""
    xy: list[tuple[float, float, float]]  # len=17


class PoseDetection(BaseModel):
    bbox: BBox
    confidence: float
    track_id: int | None = None
    keypoints: Keypoints


# --- Zonas ---

class ZoneCreate(BaseModel):
    camera_id: str
    name: str
    zone_type: str = Field(..., description="perimeter | no_parking | crowd_limit")
    polygon: list[Point] = Field(..., min_length=3)
    max_crowd: int | None = Field(None, description="Apenas para zonas crowd_limit")


class Zone(ZoneCreate):
    id: int
    created_at: datetime


# --- Alertas ---

class AlertCreate(BaseModel):
    camera_id: str
    threat_type: ThreatType
    severity: Severity
    message: str
    bbox: BBox | None = None
    clip_path: str | None = None
    metadata: dict | None = None


class Alert(AlertCreate):
    id: int
    created_at: datetime


class AlertQuery(BaseModel):
    camera_id: str | None = None
    threat_type: ThreatType | None = None
    severity: Severity | None = None
    since: datetime | None = None
    limit: int = Field(50, le=500)


# --- Metricas ---

class CameraMetrics(BaseModel):
    camera_id: str
    fps: float
    inference_ms: float
    person_count: int
    vehicle_count: int
    active_tracks: int
    alerts_total: int


# --- Heatmap ---

class HeatmapRequest(BaseModel):
    camera_id: str
    hours: int = Field(1, ge=1, le=168)
    resolution: int = Field(50, ge=10, le=200, description="Celulas do grid por eixo")
