@echo off
title Launch Pipe Detection System
echo ==============================================================================
echo             Launching Insulation Pipe Detection Studio
echo ==============================================================================
echo Starting Backend (FastAPI on Port 8000)...
start "PipeVision Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 2 /nobreak >nul
echo Starting Frontend (React on Port 5173)...
start "PipeVision Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
timeout /t 2 /nobreak >nul
echo Opening browser at http://localhost:5173...
start http://localhost:5173
pause
