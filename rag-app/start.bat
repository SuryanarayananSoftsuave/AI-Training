@echo off
REM One-click start: double-click this file in File Explorer (run setup.bat first).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
echo.
pause
