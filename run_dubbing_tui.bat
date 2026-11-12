@echo off
setlocal
set "COLS=%~1"
set "LINES=%~2"
if "%COLS%"=="" set "COLS=110"
if "%LINES%"=="" set "LINES=35"
title Dubbing Control Center
mode con: cols=%COLS% lines=%LINES%
cd /d "%~dp0"
"%~dp0env\Scripts\python.exe" "%~dp0dubbing_tui.py"
pause
endlocal
