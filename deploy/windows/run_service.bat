@echo off
setlocal
REM 啟動服務（CUDA / 一般用途）
REM 若使用 Intel XPU，請改執行 run_service_xpu.bat

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
set "VENV_PYTHON=%PROJECT_ROOT%\src\service\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [run_service] Python environment not found at "%PROJECT_ROOT%\src\service\.venv"
    echo Please run setup_env.ps1 first.
    pause
    exit /b 1
)

cd /d "%PROJECT_ROOT%\src"
set "PYTHONPATH=%PROJECT_ROOT%\src"
echo Starting the service...
"%VENV_PYTHON%" -m service.app

set "EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %EXIT_CODE%
