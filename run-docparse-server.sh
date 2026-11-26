#!/usr/bin/env bash
# Start docparse-vietocr in the background (no service; it stops at reboot). Run from anywhere:
#   bash run-docparse-server.sh          install deps + models (first run), start, return the terminal
#   bash run-docparse-server.sh stop     stop it
#   bash run-docparse-server.sh status   is it running?
# Log: server.log in this folder (tail -f server.log). For start-at-boot use linux-service.sh instead.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$DIR/server.pid"
LOG="$DIR/server.log"
cd "$DIR"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-start}" in
    stop)
        if running; then
            pid="$(cat "$PIDFILE")"
            kill "$pid"
            for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
            kill -9 "$pid" 2>/dev/null || true
            echo "Stopped (pid $pid)."
        else
            echo "Not running."
        fi
        rm -f "$PIDFILE"
        exit 0 ;;
    status)
        if running; then echo "Running (pid $(cat "$PIDFILE")). Log: $LOG"; else echo "Not running."; exit 1; fi
        exit 0 ;;
    start) ;;
    *) echo "Usage: bash run-docparse-server.sh [start|stop|status]" >&2; exit 2 ;;
esac

if running; then
    echo "Already running (pid $(cat "$PIDFILE")). Stop it with: bash run-docparse-server.sh stop"
    exit 0
fi

if ! command -v uv > /dev/null; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

echo "Installing dependencies..."
uv sync --frozen --no-dev

# Models go to DOCPARSE_MODEL_DIR (default ./models); skipped when already there.
uv run --frozen --no-dev python scripts/download_models.py

# Settings (DOCPARSE_HOST, DOCPARSE_PORT, DOCPARSE_TOKEN, ...) come from .env if present; see .env.example.
# Run the venv's python directly so the PID is the server itself (stop kills the right process), through a
# symlink named docparse-vietocr so nvidia-smi / ps / top show that instead of "python".
ln -sfn python .venv/bin/docparse-vietocr
echo "Starting server..."
PYTHONUNBUFFERED=1 nohup "$DIR/.venv/bin/docparse-vietocr" -m app.main >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"

PORT="$(grep -E '^DOCPARSE_PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d '[:space:]"'"'" || true)"
PORT="${PORT:-8126}"
echo "Waiting for the server on port $PORT (model loading takes a while)..."
for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "Up: http://$(hostname -I | awk '{print $1}'):$PORT  (pid $(cat "$PIDFILE"), log: $LOG)"
        echo "Stop with: bash run-docparse-server.sh stop"
        exit 0
    fi
    if ! running; then
        echo "Server exited during startup. Last log lines:" >&2
        tail -n 20 "$LOG" >&2
        rm -f "$PIDFILE"
        exit 1
    fi
    sleep 2
done
echo "Started (pid $(cat "$PIDFILE")) but /health is not answering yet. Watch: tail -f $LOG" >&2
