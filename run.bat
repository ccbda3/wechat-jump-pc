@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYW=%~dp0.venv\Scripts\pythonw.exe"

if not exist "%PY%" (
  echo [ERROR] Virtual environment not found. Please run setup.bat first.
  pause
  exit /b 1
)

if /i "%~1"=="debug" (
  "%PY%" gui_app.py
  echo.
  echo [GUI exited with code %errorlevel%]
  pause
  exit /b 0
)
if /i "%~1"=="cli" (
  "%PY%" wechat_jump_auto_pc.py
  pause
  exit /b 0
)
if /i "%~1"=="window" (
  "%PY%" select_region.py
  pause
  exit /b 0
)

start "" "%PYW%" gui_app.py
