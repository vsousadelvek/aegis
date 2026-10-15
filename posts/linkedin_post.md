# Post LinkedIn

Construi do zero um sistema de vigilancia urbana com IA que roda 100% local.

AEGIS detecta, rastreia e analisa comportamentos em tempo real:

- Deteccao de pessoas, veiculos e objetos (YOLOv8)
- Rastreamento persistente com ByteTrack
- Estimacao de pose (17 keypoints por pessoa)
- 5 analisadores comportamentais:
  - Invasao de perimetro
  - Veiculo parado em zona proibida
  - Anomalia de multidao
  - Objeto abandonado
  - Deteccao de briga (via analise temporal de pose)

Do pixel ao alerta em menos de 200ms numa unica GPU.

Stack:
Python | YOLOv8 | ByteTrack | FastAPI | OpenCV | SQLAlchemy | Docker

Diferenciais tecnicos:
- Pipeline por camera em thread dedicada com buffer circular de frames
- Analisadores baseados em regras temporais, nao caixas-pretas
- Clips de evidencia gerados automaticamente no momento do alerta
- Heatmap de movimentacao com baseline adaptativo
- Hot-reload de zonas sem reiniciar o sistema
- API completa com stream MJPEG, alertas, zonas e metricas
- 100% on-premise — sem depender de nuvem, compativel com LGPD

Nao eh um tutorial de YOLO. Eh um sistema completo, da captura de video ao alerta estruturado.

Codigo aberto: https://github.com/vsousadelvek/aegis

#ComputerVision #Python #YOLOv8 #IA #Vigilancia #FastAPI #DeepLearning #OpenCV #Docker #LGPD
