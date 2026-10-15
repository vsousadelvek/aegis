"""Testes de validacao dos schemas de dominio."""

from aegis.core.schemas import (
    Alert,
    AlertCreate,
    BBox,
    CameraMetrics,
    Detection,
    HeatmapRequest,
    Keypoints,
    Point,
    PoseDetection,
    Severity,
    ThreatType,
    Zone,
    ZoneCreate,
)


def test_bbox_centro():
    bb = BBox(x1=10, y1=20, x2=50, y2=60)
    c = bb.center
    assert c.x == 30.0
    assert c.y == 40.0


def test_bbox_area():
    bb = BBox(x1=0, y1=0, x2=100, y2=50)
    assert bb.area == 5000.0


def test_detection_ida_e_volta():
    det = Detection(
        bbox=BBox(x1=1, y1=2, x2=3, y2=4),
        class_id=0,
        class_name="person",
        confidence=0.95,
        track_id=7,
    )
    data = det.model_dump()
    det2 = Detection.model_validate(data)
    assert det2.track_id == 7
    assert det2.confidence == 0.95


def test_criacao_alerta():
    a = AlertCreate(
        camera_id="cam-01",
        threat_type=ThreatType.FIGHT,
        severity=Severity.CRITICAL,
        message="teste de briga",
        bbox=BBox(x1=10, y1=20, x2=100, y2=200),
    )
    assert a.threat_type == ThreatType.FIGHT
    assert a.severity == Severity.CRITICAL


def test_criacao_zona_poligono():
    z = ZoneCreate(
        camera_id="cam-01",
        name="zona teste",
        zone_type="perimeter",
        polygon=[Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1)],
    )
    assert len(z.polygon) == 3


def test_keypoints():
    kps = Keypoints(xy=[(float(i), float(i), 0.9) for i in range(17)])
    assert len(kps.xy) == 17


def test_heatmap_request_padroes():
    h = HeatmapRequest(camera_id="cam-01")
    assert h.hours == 1
    assert h.resolution == 50


def test_metricas_camera():
    m = CameraMetrics(
        camera_id="cam-01",
        fps=25.0,
        inference_ms=18.5,
        person_count=3,
        vehicle_count=1,
        active_tracks=4,
        alerts_total=0,
    )
    assert m.fps == 25.0
