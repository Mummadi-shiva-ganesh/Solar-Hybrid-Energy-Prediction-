# ─────────────────────────────────────────────────────────────────────────────
# Solar Energy Prediction - Production Dockerfile
# Section 10.4: Containerization with Docker
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Base image ──────────────────────────────────────────────────────
# python:3.9-slim is a minimal Debian-based image (~150 MB vs ~900 MB for full)
FROM python:3.9-slim

# ── Metadata labels ───────────────────────────────────────────────────────────
LABEL maintainer="Solar Yield Team"
LABEL version="1.0"
LABEL description="Solar Energy Prediction API — Flask + Random Forest"

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1 is required by XGBoost for OpenMP parallel processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first (Docker layer caching: deps only rebuild on change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application files ────────────────────────────────────────────────────
# Copy only what the API needs; excludes .git, .venv, __pycache__, etc.
COPY src/       ./src/
COPY models/    ./models/
COPY web/       ./web/
COPY docs/      ./docs/

# ── Environment variables ─────────────────────────────────────────────────────
ENV FLASK_APP=src/api.py
ENV FLASK_ENV=production
# Security: override the API key at deploy time via:
#   docker run -e API_KEY=your-secret ...
ENV API_KEY=solar-yield-secret-2026
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will mark container as 'unhealthy' if /api/health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# ── Run the API ───────────────────────────────────────────────────────────────
# Use python directly (Flask dev + watchdog for hot-reload in container)
CMD ["python", "src/api.py"]
