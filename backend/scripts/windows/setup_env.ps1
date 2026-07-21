# 偵測 GPU 種類，以對應的 cmake 旗標與 uv extra 建立 Python 環境。
# 用法：
#   .\setup_env.ps1              # 自動偵測
#   .\setup_env.ps1 -Accel xpu  # 手動指定 cuda | xpu
#   .\setup_env.ps1 -SetupLlama # 一併取得 llama.cpp 原始碼（選配，僅 llama-server 需要）
param(
    [ValidateSet("cuda", "xpu")]
    [string]$Accel = "",
    [switch]$SetupLlama
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..\").Path
$ServiceDir  = Join-Path $ProjectRoot "service"

# llama.cpp（選配，僅 llama-server 需要）：改由安裝期抓取，取代原本的 git submodule。
# 版本 pin 在此手動維護；要 bump 版本就改 $LlamaCppRef。
$LlamaCppDir = Join-Path $ServiceDir "utils\llama.cpp"
$LlamaCppUrl = "https://github.com/ggml-org/llama.cpp"
$LlamaCppRef = "50494a28003d15bb0b9a7a848fd5b6b713f39835"

function Setup-LlamaCpp {
    if (Test-Path (Join-Path $LlamaCppDir ".git")) {
        Write-Host "[setup_env] 更新既有 llama.cpp：$LlamaCppDir"
        git -C $LlamaCppDir fetch origin
    } else {
        Write-Host "[setup_env] clone llama.cpp：$LlamaCppUrl"
        if (Test-Path $LlamaCppDir) { Remove-Item -Recurse -Force $LlamaCppDir }
        git clone $LlamaCppUrl $LlamaCppDir
    }
    Write-Host "[setup_env] checkout 釘死版本 $LlamaCppRef"
    git -C $LlamaCppDir checkout --detach $LlamaCppRef
    Write-Host "[setup_env] llama.cpp 就緒；如需 llama-server 請自行 build（見 README）"
}

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

if ($SetupLlama -or $env:TRUSTA_SETUP_LLAMA -eq "1") {
    Setup-LlamaCpp
} else {
    Write-Host "[setup_env] 跳過 llama.cpp 取得（選配；如需 llama-server 加 -SetupLlama）"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  環境設定完成"
Write-Host "  Accelerator : $Accel"
Write-Host "  CMAKE_ARGS  : $CmakeFlag"
Write-Host "  Service Dir : $ServiceDir"
Write-Host "  llama.cpp   : $(if ($SetupLlama -or $env:TRUSTA_SETUP_LLAMA -eq '1') { $LlamaCppRef } else { 'skipped' })"
Write-Host "=========================================="
