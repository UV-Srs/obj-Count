@echo off
title Insulation Pipe Detection - FastAPI Backend
echo ==============================================================================
echo             Insulation Pipe Detection - FastAPI Backend (Port 8000)
echo ==============================================================================
cd /d "%~dp0backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
