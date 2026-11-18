@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8:backslashreplace"
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ============================================================
echo   Dubbing Control Center - Install and Run
echo ============================================================
echo.
if not exist "%ROOT%requirements.txt" (
  echo ERROR: requirements.txt was not found beside this file.
  goto :fail
)
if not exist "%ROOT%dubbing_tui.py" (
  echo ERROR: dubbing_tui.py was not found. Extract the complete project first.
  goto :fail
)

rem Refresh common WinGet install paths for this process.
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%ProgramFiles%\WinGet\Links;%LOCALAPPDATA%\Programs\Git\cmd;%ProgramFiles%\Git\cmd;%LOCALAPPDATA%\Programs\Python\Launcher;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"

rem Git is needed because some requirements are installed directly from GitHub.
where.exe git >nul 2>&1
if errorlevel 1 (
  call :need_winget
  if errorlevel 1 goto :fail
  echo Installing Git for Windows...
  winget install --id Git.Git -e --no-upgrade --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo ERROR: Git installation failed.
    goto :fail
  )
  set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%ProgramFiles%\WinGet\Links;%LOCALAPPDATA%\Programs\Git\cmd;%ProgramFiles%\Git\cmd;%PATH%"
  where.exe git >nul 2>&1
  if errorlevel 1 (
    echo ERROR: Git was installed but is not available in PATH. Restart Windows Terminal and run this file again.
    goto :fail
  )
)

rem Install Python 3.12 only when env is absent and Python 3.12 is unavailable.
set "PYTHON=%ROOT%env\Scripts\python.exe"
if exist "%PYTHON%" goto :check_env_python
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 goto :create_env
call :need_winget
if errorlevel 1 goto :fail
echo Installing Python 3.12...
winget install --id Python.Python.3.12 -e --no-upgrade --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
  echo ERROR: Python 3.12 installation failed.
  goto :fail
)
set "PATH=%LOCALAPPDATA%\Programs\Python\Launcher;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 goto :create_env
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "BASE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined BASE_PYTHON if exist "%ProgramFiles%\Python312\python.exe" set "BASE_PYTHON=%ProgramFiles%\Python312\python.exe"
if not defined BASE_PYTHON (
  echo ERROR: Python 3.12 was installed but is not available in this terminal.
  echo Restart this file after the installer updates PATH.
  goto :fail
)

:create_env
if not defined BASE_PYTHON set "BASE_PYTHON=py -3.12"
echo Creating the project Python environment...
if "%BASE_PYTHON%"=="py -3.12" (
  py -3.12 -m venv "%ROOT%env"
) else (
  "%BASE_PYTHON%" -m venv "%ROOT%env"
)
if errorlevel 1 (
  echo ERROR: Could not create the Python 3.12 environment.
  goto :fail
)

:check_env_python
if not exist "%PYTHON%" (
  echo ERROR: Python environment was not created correctly.
  goto :fail
)
"%PYTHON%" -c "import sys; print('Using Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 (
  echo ERROR: The existing env uses a Python version other than 3.12. It was not replaced.
  goto :fail
)

rem FFmpeg and FFprobe are required for audio/video processing.
where.exe ffmpeg >nul 2>&1
if errorlevel 1 goto :install_ffmpeg
where.exe ffprobe >nul 2>&1
if errorlevel 1 goto :install_ffmpeg
goto :install_python_packages

:install_ffmpeg
call :need_winget
if errorlevel 1 goto :fail
echo Installing FFmpeg and FFprobe...
winget install --id Gyan.FFmpeg -e --no-upgrade --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
  echo ERROR: FFmpeg installation failed.
  goto :fail
)
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%ProgramFiles%\WinGet\Links;%PATH%"
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "$root=Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'; Get-ChildItem $root -Directory -Filter 'Gyan.FFmpeg*' -ErrorAction SilentlyContinue | ForEach-Object { Get-ChildItem $_.FullName -File -Filter ffmpeg.exe -Recurse -ErrorAction SilentlyContinue } | Select-Object -First 1 -ExpandProperty DirectoryName"`) do set "PATH=%%D;%PATH%"
where.exe ffmpeg >nul 2>&1
if errorlevel 1 (
  echo ERROR: FFmpeg was installed but is not available in PATH. Restart this file after PATH updates.
  goto :fail
)
where.exe ffprobe >nul 2>&1
if errorlevel 1 (
  echo ERROR: FFprobe was not found after FFmpeg installation.
  goto :fail
)

:install_python_packages
echo.
echo Installing or verifying all packages from requirements.txt...
"%PYTHON%" -m pip install --disable-pip-version-check -r "%ROOT%requirements.txt"
if errorlevel 1 (
  echo.
  echo ERROR: Package installation failed. Check Internet access and the error above.
  goto :fail
)

echo.
echo Installation finished. Starting Dubbing Control Center...
call "%ROOT%run_dubbing_tui.bat"
exit /b %ERRORLEVEL%

:need_winget
where.exe winget >nul 2>&1
if errorlevel 1 (
  echo ERROR: WinGet is unavailable. Install App Installer from Microsoft Store, then retry.
  exit /b 1
)
exit /b 0

:fail
echo.
pause
exit /b 1
