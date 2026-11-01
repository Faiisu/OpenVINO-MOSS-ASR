#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${DEPLOY_DIR}/.." && pwd)"
SERVICE_USER="${MOSS_SERVICE_USER:-${SUDO_USER:-${USER:-}}}"
UNIT_DIR="/etc/systemd/system"
MODEL_UNIT="moss-model.service"
FRONTEND_UNIT="moss-frontend.service"

usage() {
  cat <<'EOF'
Usage: sudo ./speech_to_text_deploy/systemctl_deploy.sh <command>

Commands:
  install    Install and enable the model pipeline and frontend services
  uninstall  Stop and remove both services
  start      Start both services
  stop       Stop both services
  restart    Restart both services
  status     Show both service statuses
  logs       Follow logs from both services

Set MOSS_SERVICE_USER if the services should run as a user other than SUDO_USER.
EOF
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    printf 'Error: this command requires root. Run it with sudo.\n' >&2
    exit 1
  fi
}

require_service_user() {
  if [[ -z "${SERVICE_USER}" ]] || ! id "${SERVICE_USER}" >/dev/null 2>&1; then
    printf 'Error: set MOSS_SERVICE_USER or run this command with sudo from the target user.\n' >&2
    exit 1
  fi
}

install_units() {
  require_root
  require_service_user

  local python_bin="${PROJECT_ROOT}/.venv/bin/python"
  local model_dir="${PROJECT_ROOT}/.tools/moss-openvino-fp16"
  local uv_path="$(command -v uv || true)"
  local user_home="$(getent passwd "${SERVICE_USER}" | cut -d: -f6)"
  local service_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

  if [[ -x "${user_home}/.local/bin/uv" ]]; then
    uv_path="${user_home}/.local/bin/uv"
  elif [[ -x "${PROJECT_ROOT}/.tools/bin/uv" ]]; then
    uv_path="${PROJECT_ROOT}/.tools/bin/uv"
  fi
  if [[ -n "${uv_path}" ]]; then
    service_path="$(dirname -- "${uv_path}"):${service_path}"
  fi

  if [[ ! -x "${python_bin}" ]]; then
    printf 'Error: project-local Python is missing: %s\n' "${python_bin}" >&2
    exit 1
  fi
  if [[ ! -d "${model_dir}" ]]; then
    printf 'Error: prepared OpenVINO model directory is missing: %s\n' "${model_dir}" >&2
    exit 1
  fi

  local supplementary_groups=""
  for group in audio render; do
    if getent group "${group}" >/dev/null; then
      supplementary_groups+=" ${group}"
    fi
  done

  install -d -m 0755 "${UNIT_DIR}"
  cat >"${UNIT_DIR}/${MODEL_UNIT}" <<EOF
[Unit]
Description=MOSS continuous audio collection and transcription pipeline
Wants=network-online.target
After=network-online.target sound.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment=PATH=${service_path}
Environment=PYTHONUNBUFFERED=1
SupplementaryGroups=${supplementary_groups# }
ExecStart=${DEPLOY_DIR}/run_model.sh
Restart=always
RestartSec=5
TimeoutStopSec=660

[Install]
WantedBy=multi-user.target
EOF

  cat >"${UNIT_DIR}/${FRONTEND_UNIT}" <<EOF
[Unit]
Description=MOSS live output frontend
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment=PATH=${service_path}
Environment=PYTHONUNBUFFERED=1
ExecStart=${DEPLOY_DIR}/run_frontend.sh
Restart=always
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

  chmod 0644 "${UNIT_DIR}/${MODEL_UNIT}" "${UNIT_DIR}/${FRONTEND_UNIT}"
  systemctl daemon-reload
  systemctl enable --now "${MODEL_UNIT}" "${FRONTEND_UNIT}"
  printf 'Installed and started %s and %s for user %s.\n' \
    "${MODEL_UNIT}" "${FRONTEND_UNIT}" "${SERVICE_USER}"
}

manage_services() {
  local action="$1"
  systemctl "${action}" "${MODEL_UNIT}" "${FRONTEND_UNIT}"
}

uninstall_units() {
  require_root
  systemctl disable --now "${MODEL_UNIT}" "${FRONTEND_UNIT}" || true
  rm -f "${UNIT_DIR}/${MODEL_UNIT}" "${UNIT_DIR}/${FRONTEND_UNIT}"
  systemctl daemon-reload
  systemctl reset-failed "${MODEL_UNIT}" "${FRONTEND_UNIT}" || true
  printf 'Removed %s and %s.\n' "${MODEL_UNIT}" "${FRONTEND_UNIT}"
}

command_name="${1:-}"
case "${command_name}" in
  install) install_units ;;
  uninstall) uninstall_units ;;
  start|stop|restart|status) manage_services "${command_name}" ;;
  logs) journalctl --no-pager --full -f -u "${MODEL_UNIT}" -u "${FRONTEND_UNIT}" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
