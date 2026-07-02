@echo off
chcp 65001 >nul
cd /d E:\Vibecoding\Agent
title MiniMind Agent - WebUI
echo WebUI: http://localhost:7860
echo Press Ctrl+C to stop
echo.
.\venv\Scripts\python.exe main.py --webui
pause