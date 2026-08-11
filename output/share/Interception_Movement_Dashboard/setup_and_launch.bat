@echo off
cd /d "%~dp0"
python -m pip install -r requirements_dashboard.txt
if errorlevel 1 pause & exit /b 1
call launch_dashboard.bat
