@echo off
rem Speak setup: double-click this file to install or update Speak in this folder.
rem No admin rights are needed. See install.ps1 for what it does.
title Speak setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
pause
