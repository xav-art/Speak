@echo off
rem Removes Speak's shortcuts, PATH entry and files. Asks before deleting
rem your personal dictionary and settings.
title Uninstall Speak
set "SPEAK_INSTALL_DIR=%~dp0."
cd /d "%TEMP%"
rem "(goto)" ends the batch context first, so cmd does not look for this file
rem again after the uninstaller has deleted it.
(goto) 2>nul & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SPEAK_INSTALL_DIR%\install.ps1" -Uninstall -InstallDir "%SPEAK_INSTALL_DIR%" & echo. & pause
