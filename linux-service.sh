#!/usr/bin/env bash
# Install and start docparse-vietocr as a systemd service. Run from anywhere, as a normal user with sudo:
#   bash linux-service.sh             install (uv, dependencies, models, .env, service) and start
#   bash linux-service.sh uninstall   stop and remove the service (files stay)
# Re-running is safe: it updates dependencies and restarts the service.
# Logs: journalctl -u docparse-vietocr -f
set -euo pipefail

NAME=docparse-vietocr
UNIT=/etc/systemd/system/$NAME.service
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
cd "$DIR"

if [[ "${1:-}" == "uninstall" ]]; then
    sudo systemctl disable --now "$NAME" 2>/dev/null || true
    sudo rm -f "$UNIT"
    sudo systemctl daemon-reload
    echo "Service $NAME removed."
    exit 0
fi

if [[ $EUID -eq 0 && -z "${SUDO_USER:-}" ]]; then
    echo "Run this as the user that should own the service (it uses sudo itself), not as root." >&2
    exit 1
fi

# Everything below except the systemd part runs as RUN_USER, so uv, the venv and the models belong to it.
as_user() { if [[ "$(id -un)" == "$RUN_USER" ]]; then "$@"; else sudo -u "$RUN_USER" -H "$@"; fi; }
USER_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

UV="$(as_user bash -lc 'command -v uv' 2>/dev/null || true)"
[[ -z "$UV" && -x "$USER_HOME/.local/bin/uv" ]] && UV="$USER_HOME/.local/bin/uv"
if [[ -z "$UV" ]]; then
    echo "Installing uv..."
    as_user bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    UV="$USER_HOME/.local/bin/uv"
fi
echo "uv: $UV"

echo "Installing dependencies..."
as_user "$UV" sync --frozen --no-dev

# Models go to DOCPARSE_MODEL_DIR (default ./models); skipped when already there.
as_user "$UV" run --frozen --no-dev python scripts/download_models.py

if [[ ! -f .env ]]; then
    as_user cp .env.example .env
    echo "Created .env from .env.example - edit it to change port, token, device."
fi

echo "Installing systemd service $NAME (user $RUN_USER)..."
sudo tee "$UNIT" > /dev/null <<UNITFILE
[Unit]
Description=docparse-vietocr (scanned page -> layout + text lines)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$DIR
ExecStart=$UV run --frozen --no-dev python -m app.main
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNITFILE
sudo systemctl daemon-reload
sudo systemctl enable "$NAME" > /dev/null
sudo systemctl restart "$NAME"

PORT="$(grep -E '^DOCPARSE_PORT=' .env | cut -d= -f2 | tr -d '[:space:]"'"'" || true)"
PORT="${PORT:-8126}"
echo "Waiting for the server on port $PORT (model loading takes a while)..."
for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "Up: http://$(hostname -I | awk '{print $1}'):$PORT  (logs: journalctl -u $NAME -f)"
        exit 0
    fi
    if ! systemctl is-active --quiet "$NAME"; then break; fi
    sleep 2
done
echo "Server did not answer /health yet. Check: journalctl -u $NAME -n 50" >&2
exit 1
