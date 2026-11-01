@echo off
title Dubbing Control Center
mode con: cols=110 lines=35
cd /d "%~dp0"
"%~dp0env\Scripts\python.exe" "%~dp0dubbing_tui.py"
pause
