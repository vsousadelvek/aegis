"""Testes unitarios dos analisadores comportamentais."""

from aegis.core.schemas import (
    AlertCreate,
    BBox,
    Detection,
    Keypoints,
    Point,
    PoseDetection,
    Severity,
    ThreatType,
    Zone,
)
from aegis.analyzers.perimeter import PerimeterGuard
from aegis.analyzers.crowd import CrowdDetector
from aegis.analyzers.abandon import AbandonDetector


def _make_zone(zone_id: int = 1, zone_type: str = "perimeter") -> Zone:
    from datetime import datetime, timezone

    return Zone(
        id=zone_id,
        camera_id="cam-01",
        name="zona-teste",
        zone_type=zone_type,
        polygon=[
            Point(x=0.0, y=0.0),
            Point(x=0.5, y=0.0),
            Point(x=0.5, y=0.5),
            Point(x=0.0, y=0.5),
        ],
        created_at=datetime.now(timezone.utc),
    )


def _make_person(x: float, y: float, track_id: int = 1) -> Detection:
    return Detection(
        bbox=BBox(x1=x - 20, y1=y - 40, x2=x + 20, y2=y + 40),
        class_id=0,
        class_name="person",
        confidence=0.9,
        track_id=track_id,
    )


class TestGuardaPerimetro:
    def test_sem_zonas_sem_alertas(self):
        guard = PerimeterGuard()
        alerts = guard.update("cam-01", [_make_person(100, 100)], [], [], 1.0, 640, 480)
        assert alerts == []

    def test_pessoa_dentro_da_zona_dispara(self):
        guard = PerimeterGuard()
        zone = _make_zone()
        person = _make_person(100, 100)
        alerts = guard.update("cam-01", [person], [], [zone], 1.0, 640, 480)
        assert len(alerts) == 1
        assert alerts[0].threat_type == ThreatType.PERIMETER_BREACH

    def test_pessoa_fora_da_zona_nao_alerta(self):
        guard = PerimeterGuard()
        zone = _make_zone()
        person = _make_person(500, 400)
        alerts = guard.update("cam-01", [person], [], [zone], 1.0, 640, 480)
        assert alerts == []

    def test_cooldown_evita_spam(self):
        guard = PerimeterGuard()
        zone = _make_zone()
        person = _make_person(100, 100)
        alerts1 = guard.update("cam-01", [person], [], [zone], 1.0, 640, 480)
        assert len(alerts1) == 1
        # Mesma pessoa, 2 segundos depois — ainda dentro do cooldown
        alerts2 = guard.update("cam-01", [person], [], [zone], 3.0, 640, 480)
        assert alerts2 == []
        # Apos cooldown (>10s)
        alerts3 = guard.update("cam-01", [person], [], [zone], 12.0, 640, 480)
        assert len(alerts3) == 1


class TestDetectorMultidao:
    def test_sem_anomalia_com_contagem_baixa(self):
        detector = CrowdDetector()
        persons = [_make_person(100 + i * 50, 200, track_id=i) for i in range(3)]
        alerts = detector.update("cam-01", persons, [], [], 1.0, 640, 480)
        assert alerts == []

    def test_limite_lotacao_zona_dispara(self):
        from datetime import datetime, timezone

        detector = CrowdDetector()
        zone = Zone(
            id=1,
            camera_id="cam-01",
            name="praca",
            zone_type="crowd_limit",
            polygon=[Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1)],
            max_crowd=2,
            created_at=datetime.now(timezone.utc),
        )
        persons = [_make_person(100 + i * 50, 200, track_id=i) for i in range(5)]
        alerts = detector.update("cam-01", persons, [], [zone], 1.0, 640, 480)
        assert len(alerts) == 1
        assert alerts[0].threat_type == ThreatType.CROWD_ANOMALY


class TestDetectorAbandono:
    def test_objeto_com_dono_sem_alerta(self):
        detector = AbandonDetector()
        bag = Detection(
            bbox=BBox(x1=90, y1=190, x2=130, y2=230),
            class_id=24,
            class_name="backpack",
            confidence=0.8,
            track_id=10,
        )
        person = _make_person(100, 200, track_id=1)
        alerts = detector.update("cam-01", [bag, person], [], [], 1.0, 640, 480)
        assert alerts == []

    def test_objeto_sem_dono_dispara_apos_timeout(self):
        detector = AbandonDetector()
        bag = Detection(
            bbox=BBox(x1=90, y1=190, x2=130, y2=230),
            class_id=24,
            class_name="backpack",
            confidence=0.8,
            track_id=10,
        )
        # Sem pessoa por perto — dono ausente
        alerts1 = detector.update("cam-01", [bag], [], [], 0.0, 640, 480)
        assert alerts1 == []  # Comecou a rastrear agora

        # 15 segundos depois — ainda abaixo do timeout (padrao 30s)
        alerts2 = detector.update("cam-01", [bag], [], [], 15.0, 640, 480)
        assert alerts2 == []

        # 31 segundos depois — deve disparar
        alerts3 = detector.update("cam-01", [bag], [], [], 31.0, 640, 480)
        assert len(alerts3) == 1
        assert alerts3[0].threat_type == ThreatType.ABANDONED_OBJECT
