@echo off
setlocal
REM 啟動服務（CUDA / XPU 通用）

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
set "VENV_PYTHON=%PROJECT_ROOT%\service\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [run_service] Python environment not found at "%PROJECT_ROOT%\service\.venv"
    echo Please run setup_env.ps1 first.
    pause
    exit /b 1
)

cd /d "%PROJECT_ROOT%"
set "PYTHONPATH=%PROJECT_ROOT%"
echo Starting the service...
"%VENV_PYTHON%" -m service.app

set "EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %EXIT_CODE%
