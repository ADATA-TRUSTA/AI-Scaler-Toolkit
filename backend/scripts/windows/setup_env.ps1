# Detect the GPU type and build the Python environment with the matching uv extra
# (torch cuda / xpu variant).
# llama (prebuilt binary + GGUF convert tooling) is installed by default, but only
# what is actually missing — an existing binary or convert checkout is left alone.
# Usage:
#   .\setup_env.ps1                # auto-detect the accelerator; install llama only if missing
#   .\setup_env.ps1 -Accel xpu     # force cuda | xpu
#   .\setup_env.ps1 -Llama skip    # skip llama entirely
#   .\setup_env.ps1 -Llama force   # reinstall even if already present
#   .\setup_env.ps1 -LlamaBackend vulkan  # force the generic Vulkan build (sees every Intel/AMD/NVIDIA card)
param(
    [ValidateSet("cuda", "xpu")]
    [string]$Accel = "",
    # auto (default) = install only what is missing; force = reinstall; skip = do nothing.
    # env TRUSTA_INSTALL_LLAMA (auto / 1 / 0) overrides it, matching setup_env.sh.
    [ValidateSet("auto", "force", "skip")]
    [string]$Llama = "auto",
    [switch]$InstallLlama,             # Deprecated alias for -Llama force; kept so existing invocations keep working
    [string]$LlamaVersion = "",        # Pinned llama version; defaults to b10107, or env TRUSTA_LLAMA_VERSION
    # llama inference backend, decoupled from the torch accel: auto = cuda when NVIDIA is present,
    # else vulkan (env TRUSTA_LLAMA_BACKEND also overrides it)
    [ValidateSet("auto", "cuda", "vulkan")]
    [string]$LlamaBackend = "auto"
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..\").Path
$ServiceDir  = Join-Path $ProjectRoot "service"

# GGUF convert tooling (convert_hf_to_gguf.py / convert_lora_to_gguf.py + conversion/ + gguf-py):
# only shipped with the llama.cpp sources, not with the prebuilt binary. A sparse + blobless
# fetch pulls just these paths (~1.7MB) — no C++, nothing to compile. The pinned revision is
# maintained by hand here.
$LlamaConvertDir = Join-Path $ServiceDir "utils\llama.cpp"
$LlamaCppUrl     = "https://github.com/ggml-org/llama.cpp"
$LlamaCppRef     = "c0bc8591e8815c63cb01dd3f051a8b0df02501c9"  # = tag b10107 HEAD
$ConvertPaths    = @("convert_hf_to_gguf.py", "convert_lora_to_gguf.py", "conversion", "gguf-py")

# Official prebuilt llama (ggml-org/llama-install.sh): no MSVC / CUDA toolkit / Vulkan SDK / CMake.
# $LlamaBackend picks the llama build (decoupled from the torch accel): vulkan needs SKIP_CUDA=1
# to actually take the Vulkan path (otherwise the official installer grabs CUDA whenever an
# NVIDIA card is present).
function Install-LlamaPrebuilt {
    $installerUrl  = "https://raw.githubusercontent.com/ggml-org/llama-install.sh/master/install.ps1"
    $installerPath = Join-Path $env:TEMP "llama-install.ps1"
    Write-Host "[setup_env] downloading the official install.ps1: $installerUrl"
    Invoke-WebRequest -UseBasicParsing -Uri $installerUrl -OutFile $installerPath

    if ($script:LlamaBackendResolved -eq "vulkan") {
        $env:SKIP_CUDA = "1"
        Write-Host "[setup_env] llama backend=vulkan -> SKIP_CUDA=1 (generic Vulkan, sees Intel/AMD/NVIDIA)"
    } else {
        Write-Host "[setup_env] llama backend=cuda (native CUDA, NVIDIA only)"
    }
    if ($LlamaVersion) {
        $env:LLAMA_VERSION = $LlamaVersion
        Write-Host "[setup_env] pinning LLAMA_VERSION=$LlamaVersion"
    }
    try {
        # Run the installer in a child process so our $ErrorActionPreference="Stop" is not
        # inherited, which would turn errors the installer deliberately swallows with 2>$null
        # (e.g. cleaning a temp dir that does not exist) into fatal ones.
        # $env:SKIP_CUDA / $env:LLAMA_VERSION are passed down through the process environment.
        & powershell -NoProfile -ExecutionPolicy Bypass -File $installerPath
        if ($LASTEXITCODE -ne 0) {
            # Install only when a compatible prebuilt exists: a failure (no build for this
            # platform) warns and skips instead of aborting the whole setup
            Write-Warning "[setup_env] llama install failed (exit $LASTEXITCODE) — this platform may have no compatible prebuilt. Skipped the llama binary; point LLAMA_SERVER_BINARY in .env at your own build, or re-run later."
            $script:LlamaInstalled = $false
            return
        }
    } finally {
        Remove-Item Env:\SKIP_CUDA -ErrorAction SilentlyContinue
        Remove-Item Env:\LLAMA_VERSION -ErrorAction SilentlyContinue
        Remove-Item $installerPath -ErrorAction SilentlyContinue
    }
    $script:LlamaInstalled = $true
    Write-Host "[setup_env] llama installed: $env:LOCALAPPDATA\Microsoft\WindowsApps\llama.exe"
}

# Where the llama binary is expected. An explicit LLAMA_SERVER_BINARY wins; otherwise
# the location the official installer writes to. .env is consulted too, since that is
# where the service itself reads the override from.
function Get-LlamaBinaryPath {
    if ($env:LLAMA_SERVER_BINARY) { return $env:LLAMA_SERVER_BINARY }
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        $line = Select-String -Path $envFile -Pattern '^\s*LLAMA_SERVER_BINARY\s*=\s*(.+)$' |
            Select-Object -Last 1
        if ($line) {
            $value = $line.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
            if ($value) { return $value }
        }
    }
    return (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\llama.exe")
}

function Test-LlamaBinary {
    if (Test-Path (Get-LlamaBinaryPath)) { return $true }
    if (Get-Command llama -ErrorAction SilentlyContinue) { return $true }
    return $false
}

# The sparse checkout is only useful if the scripts the conversion code calls are actually there.
function Test-LlamaConvertTooling {
    return (Test-Path (Join-Path $LlamaConvertDir "convert_hf_to_gguf.py")) -and
           (Test-Path (Join-Path $LlamaConvertDir "convert_lora_to_gguf.py")) -and
           (Test-Path (Join-Path $LlamaConvertDir "gguf-py"))
}

function Get-LlamaConvertTooling {
    if (-not (Test-Path (Join-Path $LlamaConvertDir ".git"))) {
        if (Test-Path $LlamaConvertDir) { Remove-Item -Recurse -Force $LlamaConvertDir }
        Write-Host "[setup_env] sparse checkout of the convert tooling: $LlamaCppUrl"
        git init $LlamaConvertDir
        git -C $LlamaConvertDir remote add origin $LlamaCppUrl
        git -C $LlamaConvertDir sparse-checkout set --no-cone @ConvertPaths
    }
    Write-Host "[setup_env] fetching the convert scripts (pinned to $LlamaCppRef, only $($ConvertPaths -join ', '))"
    git -C $LlamaConvertDir fetch --depth 1 --filter=blob:none origin $LlamaCppRef
    git -C $LlamaConvertDir checkout --detach FETCH_HEAD
    Write-Host "[setup_env] convert tooling ready (pure Python, no build step)"
}

# Auto-detect: CUDA when nvidia-smi exists, otherwise default to XPU (Intel iGPU / Arc)
if (-not $Accel) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $Accel = "cuda"
    } else {
        $Accel = "xpu"
    }
}
Write-Host "[setup_env] accelerator=$Accel"

# XPU: the torch xpu SYCL runtime ships with pip (uv sync --extra xpu), so oneAPI is not needed;
# it only needs a recent enough Intel GPU driver (an old one hangs torch xpu GEMM kernel compilation).

Set-Location $ProjectRoot
Write-Host "[setup_env] uv sync --extra $Accel"
uv sync --extra $Accel

# llama build selection (decoupled from the torch accel): env TRUSTA_LLAMA_BACKEND overrides the auto default
if ($LlamaBackend -eq "auto" -and $env:TRUSTA_LLAMA_BACKEND) { $LlamaBackend = $env:TRUSTA_LLAMA_BACKEND }
if ($LlamaBackend -notin @("auto", "cuda", "vulkan")) {
    throw "[setup_env] unsupported LlamaBackend: $LlamaBackend (use auto | cuda | vulkan)"
}
$script:LlamaBackendResolved = $LlamaBackend
if ($LlamaBackendResolved -eq "auto") {
    # auto: native CUDA build when an NVIDIA card is present (fastest), otherwise the generic Vulkan build
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $script:LlamaBackendResolved = "cuda"
    } else {
        $script:LlamaBackendResolved = "vulkan"
    }
}

# Pinned version: -LlamaVersion wins, then env TRUSTA_LLAMA_VERSION, then the default
if (-not $LlamaVersion) {
    if ($env:TRUSTA_LLAMA_VERSION) { $LlamaVersion = $env:TRUSTA_LLAMA_VERSION } else { $LlamaVersion = "b10107" }
}

# Resolve the mode: env overrides the parameter, and the deprecated switch means force
$LlamaMode = $Llama
if ($InstallLlama) { $LlamaMode = "force" }
if ($env:TRUSTA_INSTALL_LLAMA) {
    switch ($env:TRUSTA_INSTALL_LLAMA) {
        "auto"  { $LlamaMode = "auto" }
        "1"     { $LlamaMode = "force" }
        "0"     { $LlamaMode = "skip" }
        default { throw "[setup_env] unsupported TRUSTA_INSTALL_LLAMA value: $($env:TRUSTA_INSTALL_LLAMA) (use auto / 1 / 0)" }
    }
}

$script:LlamaInstalled = $false
if ($LlamaMode -eq "skip") {
    Write-Host "[setup_env] skipping llama (-Llama skip)"
    $LlamaStatus = "skipped"
} else {
    # Binary
    if ($LlamaMode -ne "force" -and (Test-LlamaBinary)) {
        Write-Host "[setup_env] llama binary already present at $(Get-LlamaBinaryPath) — leaving it alone (-Llama force to reinstall)"
        $binStatus = "already present"
    } else {
        Install-LlamaPrebuilt
        if ($LlamaInstalled) {
            $binStatus = "prebuilt $LlamaVersion ($LlamaBackendResolved)"
        } else {
            $binStatus = "binary skipped (no compatible prebuilt / install failed)"
        }
    }

    # Convert tooling
    if ($LlamaMode -ne "force" -and (Test-LlamaConvertTooling)) {
        Write-Host "[setup_env] convert tooling already present at $LlamaConvertDir — leaving it alone"
        $convertStatus = "already present"
    } else {
        Get-LlamaConvertTooling
        $convertStatus = "fetched"
    }

    $LlamaStatus = "binary: $binStatus / convert tooling: $convertStatus"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Environment setup complete"
Write-Host "  Accelerator : $Accel"
Write-Host "  Service Dir : $ServiceDir"
Write-Host "  llama       : $LlamaStatus"
Write-Host "=========================================="
