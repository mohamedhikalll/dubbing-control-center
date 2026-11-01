@echo off
setlocal
if /I "%~1"=="--open" goto OPEN
if /I "%~1"=="--tui" goto OPEN
if not exist "%~dp0env\Scripts\python.exe" (
    echo ERROR: Python environment not found: "%~dp0env\Scripts\python.exe"
    echo Keep or restore the existing env folder beside dub.bat.
    exit /b 2
)
"%~dp0env\Scripts\python.exe" "%~dp0dub_cli.py" %*
exit /b %ERRORLEVEL%
:OPEN
if not exist "%~dp0env\Scripts\python.exe" (
    echo ERROR: Python environment not found: "%~dp0env\Scripts\python.exe"
    echo Keep or restore the existing env folder beside dub.bat.
    exit /b 2
)
where wt.exe >nul 2>&1
if not errorlevel 1 (
    start "" wt.exe -w new --title "Dubbing Control Center" cmd.exe /d /c ""%~dp0run_dubbing_tui.bat""
    exit /b 0
)
start "Dubbing Control Center" cmd.exe /d /k ""%~dp0run_dubbing_tui.bat""
exit /b 0
