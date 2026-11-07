@echo off
rem Start the Gahan AI Validation Viewer. Pass --cli ... for a headless run.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  exit /b 1
)
if "%~1"=="" (
  start "" .venv\Scripts\pythonw.exe app.py
) else (
  .venv\Scripts\python.exe app.py %*
)
