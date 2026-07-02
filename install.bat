@echo off
chcp 65001 >nul
cd /d %~dp0
title MiniMind Agent - Install

echo ========================================
echo  Install Dependencies
echo ========================================
echo.

if not exist venv\Scripts\python.exe (
    echo [1/2] Creating virtual environment...
    py -3.12 -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Check Python installation.
        pause
        exit /b 1
    )
) else (
    echo [INFO] venv already exists, skip creation
)

echo [2/2] Installing dependencies...
.\venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Install failed. Check your network.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Install Complete!
echo ========================================
pause