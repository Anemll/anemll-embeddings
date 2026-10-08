#!/usr/bin/env bash
# Thin wrapper around scripts/download_models.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${PYTHON:-python3}" "$ROOT/scripts/download_models.py" "$@"
