@echo off
cd /d %~dp0
title MiniMind Agent - Start All

set "PYTHON=C:\Users\64987\anaconda3\python.exe"

echo ========================================
echo  MiniMind Agent - Start All
echo ========================================
echo.

%PYTHON% --version
if errorlevel 1 (
    set "PYTHON=python"
    python --version
    if errorlevel 1 (
        echo [ERROR] Python not found.
        pause
        exit /b 1
    )
)

netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Port 8000 already in use, skip server start
    goto :start_webui
)

echo [1/2] Starting server...
start "MiniMind-Server" %PYTHON% serve.py

echo [2/2] Waiting for server (max 30s)...
set /a count=0
:wait
timeout /t 1 >nul
set /a count+=1
if %count% geq 30 (
    echo [ERROR] Server timeout
    pause
    exit /b 1
)
curl -s http://localhost:8000/health >nul 2>nul
if errorlevel 1 goto wait

echo.
echo Server is ready!

:start_webui
echo.
echo Starting WebUI...
echo WebUI: http://localhost:7860
echo Press Ctrl+C to stop
echo.
%PYTHON% main.py --webui
if errorlevel 1 (
    echo.
    echo [ERROR] WebUI failed to start
)
echo.
pause
