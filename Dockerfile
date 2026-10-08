# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/arc-evaluation-api

RUN groupadd --system arc-api \
 && useradd --system --gid arc-api --home-dir /opt/arc-evaluation-api \
      --no-create-home --shell /usr/sbin/nologin arc-api \
 && install -d -o arc-api -g arc-api -m 0750 /var/lib/arc-evaluation-api

COPY requirements.lock .
RUN python3 -m venv .venv \
 && .venv/bin/pip install --no-cache-dir -r requirements.lock

COPY arc_evaluation_api/ arc_evaluation_api/
COPY deployment/ deployment/
COPY --chown=root:arc-api data/ data/

ENV ARC_DATASET_ROOT=/opt/arc-evaluation-api/data \
    ARC_DATABASE_PATH=/var/lib/arc-evaluation-api/submissions.sqlite3 \
    ARC_HOST=0.0.0.0 \
    ARC_PORT=8000

EXPOSE 8000
USER arc-api
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()"
ENTRYPOINT [".venv/bin/python", "-m", "arc_evaluation_api"]
