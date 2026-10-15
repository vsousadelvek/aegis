# Imagem base com suporte a CUDA para inferencia GPU
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Dependencias de sistema para OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY aegis/ aegis/

RUN pip install --no-cache-dir .

# Porta da API
EXPOSE 8000

# Diretorio de clips persistente
VOLUME ["/app/clips"]

CMD ["python", "-m", "aegis"]
