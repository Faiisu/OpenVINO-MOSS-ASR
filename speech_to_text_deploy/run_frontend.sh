#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${DEPLOY_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
UV_BIN="$(command -v uv || true)"
MONITOR_HOST="${MONITOR_HOST:-0.0.0.0}"
MONITOR_PORT="${MONITOR_PORT:-18000}"

if [[ -z "${UV_BIN}" && -x "${PROJECT_ROOT}/.tools/bin/uv" ]]; then
  UV_BIN="${PROJECT_ROOT}/.tools/bin/uv"
fi

if [[ -z "${UV_BIN}" ]]; then
  printf 'Error: uv was not found in PATH or %s/.tools/bin/uv\n' "${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Error: project-local Python is missing: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
exec "${UV_BIN}" run --no-project --python "${PYTHON_BIN}" \
  -m speech_to_text_deploy.monitor.server \
  --host "${MONITOR_HOST}" --port "${MONITOR_PORT}" "$@"
