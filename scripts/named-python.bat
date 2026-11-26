@echo off
rem Makes .venv\Scripts\docparse-vietocr.exe: a copy of the venv's base python.exe (plus its DLLs), so Task Manager
rem and nvidia-smi show "docparse-vietocr.exe" instead of "python.exe". .venv\Scripts\python.exe is only a launcher
rem that starts the base python.exe as a child; the copy sits next to ..\pyvenv.cfg, so it runs as the venv itself.
rem Re-copied every time (keeps up with a Python upgrade); a copy locked by a running server is left as is.
setlocal
cd /d "%~dp0.."
set "PYHOME="
for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"home" .venv\pyvenv.cfg') do set "PYHOME=%%b"
if not defined PYHOME (
    echo No .venv\pyvenv.cfg - run uv sync first.
    exit /b 1
)
set "PYHOME=%PYHOME:~1%"
copy /y "%PYHOME%\python.exe" .venv\Scripts\docparse-vietocr.exe >nul 2>nul
for %%f in ("%PYHOME%\python3*.dll" "%PYHOME%\vcruntime140*.dll") do copy /y "%%~f" .venv\Scripts\ >nul 2>nul
if not exist .venv\Scripts\docparse-vietocr.exe (
    echo Could not create .venv\Scripts\docparse-vietocr.exe from %PYHOME%
    exit /b 1
)
exit /b 0
