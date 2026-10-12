@echo off
title Speak
set "SPEAK_PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "SPEAK_PY=%~dp0.venv\Scripts\python.exe"
"%SPEAK_PY%" -X utf8 "%~dp0speak.py" %*
