# Multi-stage Dockerfile for AiZynthFinder ONNX FastAPI service
# Stage 1: build wheels for Python deps (faster cold start)
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps needed to build native wheels (rdkit, tables, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --no-cache-dir "poetry==1.8.3" "poetry-plugin-export==1.8.0"

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Add service deps not in pyproject.toml
RUN echo "fastapi>=0.110.0" >> requirements.txt \
 && echo "uvicorn[standard]>=0.27.0" >> requirements.txt \
 && echo "google-cloud-storage>=2.0.0" >> requirements.txt

RUN pip wheel --wheel-dir=/build/wheels -r requirements.txt


# Stage 2: runtime image
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    AIZYNTH_CONFIG=/app/config_production.yml \
    GCS_BIGDATA_BUCKET=biochem-db-by-hobs \
    GCS_BIGDATA_PREFIX=aizynthfinder/bigdata \
    BIGDATA_DIR=/app/bigdata

# Runtime libs only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt /app/requirements.txt
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

COPY aizynthfinder ./aizynthfinder
COPY plugins ./plugins
COPY service.py config_production.yml ./
COPY download_bigdata.py start.sh ./
RUN chmod +x /app/start.sh

COPY pyproject.toml poetry.lock README.md ./
RUN pip install --no-deps -e .

EXPOSE 8080

# --workers 1: AiZynthFinder is not designed for concurrent access.
# Cloud Run scales by spawning more container instances, not more workers.
CMD ["/app/start.sh"]
