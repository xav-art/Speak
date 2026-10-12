@echo off
rem Starts Speak. Uses Speak's private Python environment when setup has created one.
title Speak
cd /d "%~dp0"
set "SPEAK_PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "SPEAK_PY=%~dp0.venv\Scripts\python.exe"
"%SPEAK_PY%" -X utf8 speak.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Speak stopped because of a problem. Read the message above, then press any key to close this window.
    pause >nul
)
