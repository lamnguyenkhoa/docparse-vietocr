@echo off
setlocal
rem Install and start docparse-vietocr as a background service (Task Scheduler, starts at boot, no window).
rem   Double-click                         install (uv, dependencies, models, .env, task) and start
rem   windows-service.bat uninstall        stop and remove the task (files stay)
rem Re-running is safe: it updates dependencies and restarts the server. Log: server.log in this folder.
title docparse-vietocr service setup
cd /d "%~dp0"
set "TASK=docparse-vietocr"

rem Registering a boot task needs admin: re-launch elevated if we are not.
net session >nul 2>nul
if errorlevel 1 (
    echo Asking for administrator rights...
    if "%~1"=="" (
        powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    ) else (
        powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '%~1' -Verb RunAs"
    )
    exit /b
)

if /i "%~1"=="uninstall" (
    powershell -NoProfile -Command "Stop-ScheduledTask -TaskName '%TASK%' -EA SilentlyContinue; Unregister-ScheduledTask -TaskName '%TASK%' -Confirm:$false -EA SilentlyContinue"
    call :killserver
    echo Service %TASK% removed.
    pause
    exit /b 0
)

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
for /f "delims=" %%u in ('where uv') do if not defined UV set "UV=%%u"
echo uv: %UV%

rem Stop a running copy first so uv sync can replace files it has open.
powershell -NoProfile -Command "Stop-ScheduledTask -TaskName '%TASK%' -EA SilentlyContinue"
call :killserver

echo Installing dependencies...
"%UV%" sync --frozen --no-dev
if errorlevel 1 goto :error

rem Models go to DOCPARSE_MODEL_DIR (default .\models); skipped when already there.
"%UV%" run --frozen --no-dev python scripts\download_models.py
if errorlevel 1 goto :error

if not exist .env (
    copy .env.example .env >nul
    echo Created .env from .env.example - edit it to change port, token, device.
)

echo Registering scheduled task %TASK% (user %USERNAME%)...
powershell -NoProfile -ExecutionPolicy ByPass -Command ^
  "$q = [char]34; $a = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument ('/c ' + $q + $q + $env:UV + $q + ' run --frozen --no-dev python -m app.main >> server.log 2>&1' + $q) -WorkingDirectory '%CD%';" ^
  "$t = New-ScheduledTaskTrigger -AtStartup;" ^
  "$p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest;" ^
  "$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries;" ^
  "Register-ScheduledTask -TaskName '%TASK%' -Action $a -Trigger $t -Principal $p -Settings $s -Force | Out-Null;" ^
  "Start-ScheduledTask -TaskName '%TASK%'"
if errorlevel 1 goto :error

set "PORT=8126"
for /f "tokens=2 delims==" %%p in ('findstr /b "DOCPARSE_PORT=" .env') do set "PORT=%%p"
echo Waiting for the server on port %PORT% (model loading takes a while)...
powershell -NoProfile -Command ^
  "for ($i=0; $i -lt 90; $i++) { try { Invoke-RestMethod http://127.0.0.1:%PORT%/health -TimeoutSec 2 | Out-Null; exit 0 } catch { Start-Sleep 2 } }; exit 1"
if errorlevel 1 (
    echo Server did not answer /health yet. Check server.log in this folder.
    pause
    exit /b 1
)
echo.
echo Up on port %PORT%. It starts automatically at boot. Log: %CD%\server.log
echo Remove with: windows-service.bat uninstall
pause
exit /b 0

:killserver
rem Ends the running server: uv / python processes whose command line is "... -m app.main".
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'uv.exe','python.exe' -and $_.CommandLine -like '*-m app.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }"
exit /b 0

:error
echo.
echo Something went wrong. See the messages above.
pause
exit /b 1
