@echo off
cd /d %~dp0
title MiniMind Agent - Install

set "PYTHON=C:\Users\64987\anaconda3\python.exe"

echo ========================================
echo  Install Dependencies
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

%PYTHON% -m pip install -r requirements.txt
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
