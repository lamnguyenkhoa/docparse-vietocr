@echo off
setlocal
rem Double-click to install dependencies (first run), fetch the models (first run) and start the server.
title docparse-vietocr
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    ) else (
        echo Installing uv...
        powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    )
)

echo Installing dependencies...
uv sync --frozen --no-dev
if errorlevel 1 goto :error

rem Models go to DOCPARSE_MODEL_DIR (default .\models); skipped when already there.
uv run --frozen --no-dev python scripts\download_models.py
if errorlevel 1 goto :error

rem Settings (DOCPARSE_HOST, DOCPARSE_PORT, DOCPARSE_TOKEN, ...) come from .env if present; see .env.example.
echo Starting server...
uv run --frozen --no-dev python -m app.main
if errorlevel 1 goto :error
goto :eof

:error
echo.
echo Something went wrong. See the messages above.
pause
exit /b 1
