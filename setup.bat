@echo off
cd /d "%~dp0"
echo Creating virtual environment with Python 3.13 ...
py -3.13 -m venv .venv
if errorlevel 1 goto FAIL
echo Installing dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto FAIL
echo.
echo [OK] Setup complete. Double-click run.bat to start the GUI.
pause
exit /b 0

:FAIL
echo [ERROR] Setup failed. Make sure Python 3.13 with tkinter is installed.
pause
exit /b 1
