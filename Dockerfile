# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 3: Testing & CI/CD
#
# Multi-stage build:
#   builder  - compiles wheels (scikit-surprise needs a C compiler)
#   trainer  - trains the SVD model on MovieLens 100K (seeded, reproducible)
#   runtime  - slim image: wheels + app + trained model, non-root user
# =============================================================================

# ---------- 1. builder --------------------------------------------------------
FROM python:3.10-slim AS builder
RUN apt-get -o Acquire::Retries=5 update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---------- 2. trainer --------------------------------------------------------
FROM python:3.10-slim AS trainer
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
WORKDIR /app
COPY scripts/ ./scripts/
RUN python scripts/train_model.py --skip-cv

# ---------- 3. runtime --------------------------------------------------------
FROM python:3.10-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

WORKDIR /app
COPY app/ ./app/
COPY --from=trainer /app/models/ ./models/

RUN useradd --create-home --uid 1000 api && chown -R api /app
USER api

EXPOSE 8000

# Health check (python instead of curl: curl is not in the slim image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
