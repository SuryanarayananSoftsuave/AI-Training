@echo off
REM One-click setup: double-click this file in File Explorer.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
echo.
pause
