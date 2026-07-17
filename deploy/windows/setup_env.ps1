# 偵測 GPU 種類，以對應的 cmake 旗標與 uv extra 建立 Python 環境。
# 用法：
#   .\setup_env.ps1              # 自動偵測
#   .\setup_env.ps1 -Accel xpu  # 手動指定 cuda | xpu
param(
    [ValidateSet("cuda", "xpu")]
    [string]$Accel = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..\").Path
$ServiceDir  = Join-Path $ProjectRoot "src\service"

# 自動偵測：有 nvidia-smi 就 CUDA，否則預設 XPU（Intel iGPU / Arc）
if (-not $Accel) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $Accel = "cuda"
    } else {
        $Accel = "xpu"
    }
}
Write-Host "[setup_env] accelerator=$Accel"

switch ($Accel) {
    "cuda" { $CmakeFlag = "-DGGML_CUDA=on" }
    "xpu"  { $CmakeFlag = "-DGGML_VULKAN=on" }
}

# XPU：載入 Intel oneAPI 環境變數
if ($Accel -eq "xpu") {
    $SetvarsPath = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    if (Test-Path $SetvarsPath) {
        Write-Host "[setup_env] 初始化 oneAPI 環境..."
        $envLines = cmd /c "call `"$SetvarsPath`" --force && set" 2>&1
        foreach ($line in $envLines) {
            if ($line -match "^([^=]+)=(.*)$") {
                [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
            }
        }
        Write-Host "[setup_env] oneAPI 環境載入完成"
    } else {
        Write-Warning "[setup_env] 未找到 oneAPI setvars.bat，跳過 oneAPI 初始化"
    }
}

Set-Location $ServiceDir
$env:CMAKE_ARGS = $CmakeFlag
Write-Host "[setup_env] CMAKE_ARGS=$CmakeFlag  uv sync --extra $Accel"
uv sync --extra $Accel

Write-Host ""
Write-Host "=========================================="
Write-Host "  環境設定完成"
Write-Host "  Accelerator : $Accel"
Write-Host "  CMAKE_ARGS  : $CmakeFlag"
Write-Host "  Service Dir : $ServiceDir"
Write-Host "=========================================="
