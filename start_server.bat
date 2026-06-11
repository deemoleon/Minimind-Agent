@echo off
cd /d %~dp0
title MiniMind Agent - Server

set "PYTHON=C:\Users\64987\anaconda3\python.exe"

echo ========================================
echo  Start Server
echo ========================================
echo  Server: http://localhost:8000
echo  Press Ctrl+C to stop
echo.

%PYTHON% --version >nul 2>&1
if errorlevel 1 set "PYTHON=python"

%PYTHON% serve.py
if errorlevel 1 (
    echo.
    echo [ERROR] Server failed to start
)
echo.
pause
