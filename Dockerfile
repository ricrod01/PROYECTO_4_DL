# Imagen Linux con Python compatible con tensorflow==2.19.1.
FROM python:3.13.7-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Las dependencias se instalan dentro del contenedor, NO en Python de Windows.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install -r /tmp/requirements.txt \
 && python -m pip install "streamlit>=1.40,<2.0" "plotly>=5.24,<7.0"

# El código y datos se montan con Docker Compose para persistir los resultados.
CMD ["python", "scripts/entrenar_modelos.py", "--config", "scripts/config.json"]
