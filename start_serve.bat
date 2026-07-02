@echo off
chcp 65001 >nul
cd /d E:\Vibecoding\Agent
.\venv\Scripts\python.exe serve.py > tmp\serve.log 2> tmp\serve_err.log
