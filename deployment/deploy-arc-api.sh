#!/usr/bin/env bash
# Deploy ARC evaluation API on omega-experimental.
# Reads ARC_API_KEY from /etc/arc-evaluation-api.env (mode 0640, root:arc-api).
set -euo pipefail

cd /opt/arc-evaluation-api

# Create dataset directory with synthetic placeholder if empty
mkdir -p /opt/arc-evaluation-api/data/training
mkdir -p /opt/arc-evaluation-api/data/evaluation

# Start the service
exec .venv/bin/python -m arc_evaluation_api
