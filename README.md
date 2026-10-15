# AEGIS — Sistema de Deteccao de Ameacas Urbanas

Vigilancia multi-camera em tempo real com deteccao de anomalias comportamentais.
Do pixel ao alerta em menos de 200 ms numa unica GPU.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![License](https://img.shields.io/badge/Licenca-MIT-green)

---

## O que faz

Aponte N feeds de camera (RTSP, arquivo, webcam) para o AEGIS. Ele roda
deteccao, rastreamento e cinco analisadores comportamentais **por camera,
em paralelo**, e dispara alertas estruturados com um clip de 10 segundos
do incidente:

| Analisador | Detecta | Como |
|---|---|---|
| **DetectorBriga** | Agressao / briga | Velocidade temporal de membros sobre keypoints de pose — identifica pares de pessoas com movimento rapido de pulsos/tornozelos em proximidade |
| **DetectorAbandono** | Objetos abandonados | Rastreia objetos portateis; se nenhuma pessoa permanece dentro do raio de dono por N segundos, dispara |
| **DetectorMultidao** | Densidade anomala de pessoas | Baseline adaptativo (media/desvio movel da contagem); tambem respeita zonas com limite explicito de lotacao |
| **DetectorPermanencia** | Veiculos estacionados ilegalmente | Rastreia veiculos dentro de poligonos de zona proibida; alerta apos timeout configuravel |
| **GuardaPerimetro** | Invasao de area restrita | Ponto-em-poligono (Shapely) no centro da pessoa vs zonas de perimetro definidas pelo operador |

Cada alerta inclui: tipo de ameaca, severidade, bounding box, ID da camera,
timestamp, e um clip MP4 salvo automaticamente da janela do incidente.

---

## Arquitetura

```
Cameras (RTSP / mp4 / webcam)
    |
    v
+------------------------------------------------------------+
|  PipelineCamera  (uma thread por camera)                   |
|                                                            |
|  Frame --> YOLOv8n (det) --+--> ByteTrack ----+            |
|            YOLOv8n-pose ---+                  |            |
|                                               v            |
|                                        CadeiaAnalisadores  |
|                                        +- DetectorBriga    |
|                                        +- DetectorAbandono |
|                                        +- DetectorMultidao |
|                                        +- DetectorPerman.  |
|                                        +- GuardaPerimetro  |
|                                               |            |
|                                        Alerta + Clip       |
+----------------------------+--------------------------+----+
                             |                          |
                             v                          v
                    +-----------------+        +------------------+
                    |  Servidor FastAPI|        |  SQLite + clips/ |
                    |  /stream/{cam}  |        |  (persistencia)  |
                    |  /alertas       |        +------------------+
                    |  /heatmap       |
                    |  /zonas         |
                    |  /cameras       |
                    |  /health        |
                    +-----------------+
```

### Decisoes de projeto

- **Thread-por-camera, nao processo**: mantem memoria GPU compartilhada (uma
  instancia YOLO por thread do pipeline, mas os pesos sao carregados uma vez
  pelo CUDA). Uma unica RTX 3090 roda confortavelmente 4-6 cameras a 15+ FPS
  cada com `yolov8n`.
- **Analise comportamental baseada em regras, nao classificador caixa-preta**: o
  detector de briga usa velocidade temporal de membros nos keypoints COCO —
  nenhum dataset rotulado de briga necessario. Substitua por um 1D-Conv ou LSTM
  futuramente implementando a mesma interface `AnalisadorBase`.
- **Zonas com hot-reload**: crie/delete uma zona pela API e o pipeline em
  execucao ja reconhece na hora, sem restart.
- **Buffer de clip circular**: cada camera mantem os ultimos N segundos em
  memoria; quando um alerta dispara, recorta `[evento - 5s, evento + 5s]` e
  grava o MP4. Zero I/O em disco ate algo acontecer.
- **100% local**: nenhuma dependencia de nuvem. Compativel com LGPD.

---

## Inicio rapido

### Local (recomendado para dev)

```bash
# 1. Criar venv
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux

# 2. Instalar
pip install -e ".[dev]"

# 3. Rodar
python -m aegis
# API em http://localhost:8000
# Docs em http://localhost:8000/docs
```

### Docker (producao)

```bash
cp .env.example .env
docker compose up --build
```

Requer NVIDIA Container Toolkit para acesso a GPU.

---

## Uso

### 1. Adicionar uma camera

```bash
# A partir de um arquivo de video
curl -X POST http://localhost:8000/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "saguao", "source": "videos/saguao.mp4"}'

# A partir de RTSP
curl -X POST http://localhost:8000/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "entrada", "source": "rtsp://admin:senha@192.168.1.100/stream"}'

# A partir de webcam (dispositivo 0)
curl -X POST http://localhost:8000/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "webcam", "source": "0"}'
```

### 2. Assistir o stream ao vivo

Abra no navegador:
```
http://localhost:8000/cameras/saguao/stream
```

### 3. Definir zonas

```bash
# Zona proibida para estacionamento (coordenadas normalizadas 0-1)
curl -X POST http://localhost:8000/zonas \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "saguao",
    "name": "Faixa de bombeiros",
    "zone_type": "no_parking",
    "polygon": [
      {"x": 0.1, "y": 0.8},
      {"x": 0.5, "y": 0.8},
      {"x": 0.5, "y": 1.0},
      {"x": 0.1, "y": 1.0}
    ]
  }'

# Perimetro restrito
curl -X POST http://localhost:8000/zonas \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "entrada",
    "name": "Sala de servidores",
    "zone_type": "perimeter",
    "polygon": [
      {"x": 0.6, "y": 0.2},
      {"x": 0.9, "y": 0.2},
      {"x": 0.9, "y": 0.6},
      {"x": 0.6, "y": 0.6}
    ]
  }'
```

### 4. Consultar alertas

```bash
# Todos os alertas
curl http://localhost:8000/alertas

# Filtrado
curl "http://localhost:8000/alertas?tipo_ameaca=briga&limit=10"
```

### 5. Gerar heatmap

```bash
# Como grid JSON
curl -X POST http://localhost:8000/heatmap/grid \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "saguao", "hours": 2, "resolution": 80}'

# Como imagem PNG com colormap
curl -X POST http://localhost:8000/heatmap/imagem \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "saguao", "hours": 24}' --output heatmap.png
```

### 6. Metricas

```bash
curl http://localhost:8000/cameras/saguao/metricas
# {"camera_id":"saguao","fps":22.3,"inference_ms":18.7,"contagem_pessoas":5,...}
```

---

## Referencia da API

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| `GET` | `/health` | Saude do sistema + status das cameras |
| `POST` | `/cameras` | Adicionar uma fonte de camera |
| `GET` | `/cameras` | Listar todas as cameras |
| `DELETE` | `/cameras/{id}` | Parar e remover uma camera |
| `GET` | `/cameras/{id}/stream` | Stream MJPEG ao vivo anotado |
| `GET` | `/cameras/{id}/snapshot` | Frame JPEG unico |
| `GET` | `/cameras/{id}/metricas` | FPS, latencia, contagens |
| `GET` | `/alertas` | Listar alertas (com filtros) |
| `POST` | `/alertas/consulta` | Consulta avancada de alertas |
| `POST` | `/zonas` | Criar uma zona |
| `GET` | `/zonas` | Listar zonas |
| `DELETE` | `/zonas/{id}` | Deletar uma zona |
| `POST` | `/heatmap/grid` | Grid de densidade (JSON) |
| `POST` | `/heatmap/imagem` | Heatmap como PNG |

Documentacao interativa em `/docs` (Swagger UI) e `/redoc`.

---

## Configuracao

Todas as configuracoes sao variaveis de ambiente prefixadas com `AEGIS_`.
Veja `.env.example` ou `aegis/core/config.py` para a lista completa.

| Variavel | Padrao | Descricao |
|----------|--------|-----------|
| `AEGIS_DEVICE` | `cuda:0` | Dispositivo Torch |
| `AEGIS_DET_MODEL` | `yolov8n.pt` | Modelo de deteccao |
| `AEGIS_POSE_MODEL` | `yolov8n-pose.pt` | Modelo de pose |
| `AEGIS_DET_CONF` | `0.35` | Limiar de confianca |
| `AEGIS_FRAME_SKIP` | `2` | Processar a cada N frames |
| `AEGIS_ABANDON_TIMEOUT_S` | `30` | Segundos antes do alerta de abandono |
| `AEGIS_LOITER_TIMEOUT_S` | `60` | Segundos antes do alerta de permanencia |
| `AEGIS_CROWD_DENSITY_SIGMA` | `2.5` | Multiplicador de desvio para anomalia de multidao |
| `AEGIS_FIGHT_SEQ_LEN` | `16` | Frames de historico de pose para deteccao de briga |
| `AEGIS_CLIP_BUFFER_S` | `10` | Duracao do buffer circular |

---

## Estrutura do projeto

```
aegis/
  core/
    config.py          configuracao via pydantic-settings
    schemas.py         modelos de dominio (Detection, Alert, Zone, etc.)
  detection/
    detector.py        wrapper YOLOv8 det + pose
    tracker.py         integracao ByteTrack via Ultralytics .track()
  analyzers/
    base.py            ABC AnalisadorBase
    perimeter.py       invasao de perimetro (ponto-em-poligono)
    loiter.py          permanencia de veiculo em zona proibida
    crowd.py           anomalia adaptativa de densidade de multidao
    abandon.py         deteccao de objeto abandonado
    fight.py           analise temporal de velocidade de keypoints de pose
  pipeline/
    clip_buffer.py     buffer circular de frames + extracao de clip
    camera_pipeline.py thread por camera: detectar -> rastrear -> analisar -> alertar
  api/
    app.py             fabrica da aplicacao FastAPI + lifespan
    routers/
      cameras.py       CRUD de cameras + stream MJPEG + snapshot
      alerts.py        consulta de alertas
      zones.py         CRUD de zonas com hot-reload
      heatmap.py       grid de densidade + imagem com colormap
  storage/
    database.py        SQLAlchemy async (SQLite) para alertas, zonas, heatmap
  __main__.py          ponto de entrada: python -m aegis
tests/
clips/                 clips de incidentes salvos automaticamente
```

---

## Stack tecnologico

| Camada | Tecnologia |
|--------|-----------|
| Deteccao | YOLOv8n + YOLOv8n-pose (Ultralytics) |
| Rastreamento | ByteTrack (integrado) |
| Analise comportamental | Analisadores customizados (Shapely, NumPy, heuristicas temporais) |
| Backend | FastAPI + Uvicorn |
| Banco de dados | SQLAlchemy 2.0 async + aiosqlite |
| Video I/O | OpenCV |
| Exportacao de clip | OpenCV VideoWriter (mp4v) |
| Container | Docker + NVIDIA Container Toolkit |

---

## Licenca

MIT
