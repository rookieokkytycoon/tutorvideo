@echo off
REM ============================================================
REM  Agentic Video Tutor - double-click launcher for Windows
REM  Checks Python, installs dependencies, asks for your API key,
REM  starts the server and opens your browser.
REM ============================================================
cd /d "%~dp0"
title Agentic Video Tutor

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python was not found.
  echo Install it from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" during install, then run me again.
  echo.
  pause
  exit /b 1
)

echo Installing dependencies (first run only, ~30s)...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo pip install failed - check your internet connection and try again.
  pause
  exit /b 1
)

if "%ANTHROPIC_API_KEY%"=="" (
  echo.
  set /p ANTHROPIC_API_KEY="Paste your Anthropic API key (starts with sk-ant-) and press Enter: "
)

echo.
echo Starting server... your browser will open at http://localhost:8000
echo Keep this window open while you use the app. Press Ctrl+C here to stop.
echo.
start "" http://localhost:8000
python server.py
pause
