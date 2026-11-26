# Install docparse-vietocr as a systemd service on 172.16.20.76 (run once, as a sudoer).
# Before: uv sync --frozen --no-dev && uv run --frozen --no-dev python scripts/download_models.py
#         and a .env with DOCPARSE_HOST=0.0.0.0 (see .env.example).
# GPU 0 = RTX 5060 Ti (shared with the TEI reranker); GPU 1 is Ollama's - do not use it.
sudo tee /etc/systemd/system/docparse-vietocr.service > /dev/null <<'UNIT'
[Unit]
Description=docparse-vietocr (scanned page -> layout + text lines)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=atoma
WorkingDirectory=/home/atoma/Atoma/docparse-vietocr
ExecStart=/home/atoma/.local/bin/uv run --frozen --no-dev python -m app.main
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now docparse-vietocr
