@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "native\build\jump_rl.exe" (
  echo [ERROR] jump_rl.exe not found. Please rebuild first.
  pause
  exit /b 1
)
start "" "native\build\jump_rl.exe"
