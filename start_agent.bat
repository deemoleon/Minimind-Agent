@echo off
cd /d %~dp0
title MiniMind Agent - WebUI

set "PYTHON=C:\Users\64987\anaconda3\python.exe"

echo ========================================
echo  Start WebUI
echo ========================================
echo  WebUI: http://localhost:7860
echo  Press Ctrl+C to stop
echo.

%PYTHON% --version >nul 2>&1
if errorlevel 1 set "PYTHON=python"

%PYTHON% main.py --webui
if errorlevel 1 (
    echo.
    echo [ERROR] WebUI failed to start. Make sure server is running.
)
echo.
pause
