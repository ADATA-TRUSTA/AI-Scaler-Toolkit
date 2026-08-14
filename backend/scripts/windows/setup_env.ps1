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
#   .\setup_env.ps1 -SkipXpuCheck  # skip the post-sync XPU smoke check
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
    [string]$LlamaBackend = "auto",
    [switch]$SkipXpuCheck,             # Skip the post-sync XPU smoke check (env TRUSTA_SKIP_XPU_CHECK=1 also skips it)
    # The check is killed after this long: an old driver can hang, not raise. Env
    # TRUSTA_XPU_CHECK_TIMEOUT sets the same thing (matching setup_env.sh); this parameter wins.
    [int]$XpuCheckTimeoutSec = 300
)

$ErrorActionPreference = "Stop"

# Fail with a single readable line, the way setup_env.sh does. `throw` would print the
# exception, the offending source line and the CategoryInfo/FullyQualifiedErrorId block
# on top of the message, which buries the part the user needs to act on.
function Die {
    param([string]$Message)
    Write-Host "[setup_env] $Message"
    exit 1
}

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

# Record the binary setup_env actually resolved, so the service reads a decision instead of
# repeating the search. PATH is per-process: setup runs in your shell, the service may run
# under a service manager with a different one, and then the two disagree. Only ever adds the
# key when it is absent — an explicit value is the user's, and is never touched.
function Save-LlamaBinaryToEnv {
    param([string]$Resolved)
    if (-not $Resolved) { return }
    $envFile = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $envFile)) {
        # settings.py falls back to .env.example when .env is missing; creating one here would
        # change which file the service loads, so only say what could not be recorded.
        Write-Host "[setup_env] no .env, so LLAMA_SERVER_BINARY was not recorded (resolved: $Resolved)"
        return
    }
    if (Select-String -Path $envFile -Pattern '^\s*LLAMA_SERVER_BINARY\s*=' -Quiet) {
        Write-Host "[setup_env] .env already sets LLAMA_SERVER_BINARY, leaving it alone"
        return
    }
    $lines = @(
        "",
        "# Recorded by setup_env: the llama binary it resolved, so the service does not",
        "# have to re-resolve it from a possibly different PATH.",
        "LLAMA_SERVER_BINARY=$Resolved"
    )
    Add-Content -Path $envFile -Value $lines -Encoding utf8
    Write-Host "[setup_env] recorded LLAMA_SERVER_BINARY=$Resolved in .env"
}

# Official prebuilt llama (ggml-org/llama-install.sh): no MSVC / CUDA toolkit / Vulkan SDK / CMake.
# $LlamaBackend picks the llama build (decoupled from the torch accel): the installer probes
# CUDA -> Vulkan -> CPU and keeps the first hit, so vulkan needs SKIP_CUDA=1 (otherwise it grabs
# CUDA whenever an NVIDIA card is present). Unlike the Linux installer there is no ROCm probe
# here, so SKIP_CUDA alone is enough to reach Vulkan.
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
# where the service itself reads the override from — including service/settings.py's
# fallback to .env.example when .env is absent, so setup and the service agree.
function Get-LlamaBinaryPath {
    if ($env:LLAMA_SERVER_BINARY) { return $env:LLAMA_SERVER_BINARY }
    foreach ($name in @(".env", ".env.example")) {
        $envFile = Join-Path $ProjectRoot $name
        if (-not (Test-Path $envFile)) { continue }
        $line = Select-String -Path $envFile -Pattern '^\s*LLAMA_SERVER_BINARY\s*=\s*(.+)$' |
            Select-Object -Last 1
        if ($line) {
            $value = $line.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
            if ($value) { return $value }
        }
        break
    }
    return (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\llama.exe")
}

# Returns the path that actually holds a usable binary — the configured/default one, or
# whatever `llama` resolves to on PATH — and $null when there is none. Callers report the
# path that was really found instead of the one that was merely expected.
function Resolve-LlamaBinary {
    $configured = Get-LlamaBinaryPath
    if (Test-Path $configured -PathType Leaf) { return $configured }
    $onPath = Get-Command llama -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

# The sparse checkout is only useful if the scripts the conversion code calls are actually there.
function Test-LlamaConvertTooling {
    return (Test-Path (Join-Path $LlamaConvertDir "convert_hf_to_gguf.py")) -and
           (Test-Path (Join-Path $LlamaConvertDir "convert_lora_to_gguf.py")) -and
           (Test-Path (Join-Path $LlamaConvertDir "gguf-py"))
}

# XPU only: prove the install can actually compute. torch.xpu.is_available() can report True
# and still fail (or hang) on the first kernel when the wheel's oneAPI runtime does not match
# the installed Intel driver — see scripts/xpu_smoke.py. Run under a timeout to cover the hang.
# uv installs into UV_PROJECT_ENVIRONMENT when it is set, so the venv is not always .\.venv
function Get-VenvPython {
    $venv = $env:UV_PROJECT_ENVIRONMENT
    if (-not $venv) { $venv = Join-Path $ProjectRoot ".venv" }
    if (-not [System.IO.Path]::IsPathRooted($venv)) { $venv = Join-Path $ProjectRoot $venv }
    return (Join-Path $venv "Scripts\python.exe")
}

# Returns $true when the check actually ran and passed, $false when it could not run at all —
# the caller must not report a check that never ran as a pass.
function Test-XpuUsable {
    $script:XpuSkipReason = $null
    $checkScript = Join-Path $ProjectRoot "scripts\xpu_smoke.py"
    if (-not (Test-Path $checkScript)) {
        # Keep this non-fatal and identical to setup_env.sh: a missing prerequisite means an
        # incomplete checkout, not a broken driver.
        Write-Warning "[setup_env] scripts\xpu_smoke.py not found, skipping the XPU check"
        $script:XpuSkipReason = "xpu_smoke.py not found"
        return $false
    }
    $python = Get-VenvPython
    if (-not (Test-Path $python)) {
        Write-Warning "[setup_env] $python not found, skipping the XPU check"
        $script:XpuSkipReason = "venv python not found"
        return $false
    }
    Write-Host "[setup_env] verifying the XPU install (real GEMM + training step, timeout ${XpuCheckTimeoutSec}s)"

    # Diagnostics.Process rather than Start-Process -PassThru: the latter never populates
    # ExitCode without -Wait (so every run would look like a failure), and -Wait cannot time out.
    # UseShellExecute=$false keeps the child attached to this console, so its output stays visible.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $python
    $psi.Arguments = '-u "{0}"' -f $checkScript
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = $ProjectRoot
    $proc = [System.Diagnostics.Process]::Start($psi)

    if (-not $proc.WaitForExit($XpuCheckTimeoutSec * 1000)) {
        $proc.Kill()
        $proc.WaitForExit()
        Die "the XPU check did not finish within ${XpuCheckTimeoutSec}s: kernel compilation is hanging, which an outdated Intel GPU driver causes. Update the driver, then re-run (or pass -SkipXpuCheck to bypass)."
    }
    if ($proc.ExitCode -ne 0) {
        Die "the XPU check failed (exit $($proc.ExitCode)); see the message above. Pass -SkipXpuCheck to bypass."
    }
    return $true
}

function Get-LlamaConvertTooling {
    # Reuse any checkout whose git metadata still works, and just move it to the pinned revision.
    # `.git` is a *file*, not a directory, in a full clone left over from the old submodule layout;
    # deleting that tree because it does not match the sparse layout would throw away a real
    # checkout. Require .git to exist first, or `git -C` walks up and resolves the parent repo.
    $hasGitMetadata = $false
    if (Test-Path (Join-Path $LlamaConvertDir ".git")) {
        git -C $LlamaConvertDir rev-parse --git-dir *> $null
        $hasGitMetadata = ($LASTEXITCODE -eq 0)
    }
    if ($hasGitMetadata) {
        Write-Host "[setup_env] reusing the existing checkout: $LlamaConvertDir"
        git -C $LlamaConvertDir remote get-url origin *> $null
        if ($LASTEXITCODE -ne 0) { git -C $LlamaConvertDir remote add origin $LlamaCppUrl }
    } else {
        if (Test-Path $LlamaConvertDir) { Remove-Item -Recurse -Force $LlamaConvertDir }
        Write-Host "[setup_env] sparse checkout of the convert tooling: $LlamaCppUrl"
        git init $LlamaConvertDir
        git -C $LlamaConvertDir remote add origin $LlamaCppUrl
        git -C $LlamaConvertDir sparse-checkout set --no-cone @ConvertPaths
    }
    Write-Host "[setup_env] fetching the convert scripts (pinned to $LlamaCppRef, only $($ConvertPaths -join ', '))"
    git -C $LlamaConvertDir fetch --depth 1 --filter=blob:none origin $LlamaCppRef
    # Reusing a checkout means the checkout can be refused: git will not overwrite untracked
    # files. Say what to do instead of letting the raw git error stand — and do not "fix" it by
    # deleting the directory, which is the data loss this reuse path exists to avoid.
    git -C $LlamaConvertDir checkout --detach FETCH_HEAD
    if ($LASTEXITCODE -ne 0) {
        Die "could not check out $LlamaCppRef in $LlamaConvertDir (see the git error above). Untracked files in that directory usually cause this. Move or delete the directory, then re-run; nothing was deleted for you."
    }
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

# XPU: the torch xpu SYCL runtime ships with pip (uv sync --extra xpu), so the oneAPI Base
# Toolkit is not needed. What does matter is that the runtime version baked into the wheel
# matches the installed Intel GPU driver; when it does not, the first kernel either raises or
# hangs. Test-XpuUsable below proves it instead of assuming it.

# Resolve and validate every input up front: `uv sync` below takes minutes and downloads
# gigabytes, so a typo in any of these must be rejected before that, not after it.

# Env fallback for the timeout, so both platforms accept TRUSTA_XPU_CHECK_TIMEOUT; an explicitly
# passed -XpuCheckTimeoutSec still wins over it.
if (-not $PSBoundParameters.ContainsKey("XpuCheckTimeoutSec") -and $env:TRUSTA_XPU_CHECK_TIMEOUT) {
    $parsedTimeout = 0
    if ([int]::TryParse($env:TRUSTA_XPU_CHECK_TIMEOUT, [ref]$parsedTimeout) -and $parsedTimeout -gt 0) {
        $XpuCheckTimeoutSec = $parsedTimeout
    } else {
        Write-Warning "[setup_env] ignoring invalid TRUSTA_XPU_CHECK_TIMEOUT='$env:TRUSTA_XPU_CHECK_TIMEOUT' (want a positive integer)"
    }
}

# llama build selection (decoupled from the torch accel): env TRUSTA_LLAMA_BACKEND overrides the
# auto default. The env value goes into a plain variable, never back into $LlamaBackend: PowerShell
# enforces a parameter's ValidateSet on every assignment, so an invalid env value would raise a raw
# MetadataError here — before the check below — and the message would blame a "variable" the user
# never set.
$backend = $LlamaBackend
if ($backend -eq "auto" -and $env:TRUSTA_LLAMA_BACKEND) { $backend = $env:TRUSTA_LLAMA_BACKEND.Trim() }
if ($backend -notin @("auto", "cuda", "vulkan")) {
    Die "unsupported TRUSTA_LLAMA_BACKEND: $backend (use auto | cuda | vulkan)"
}
$script:LlamaBackendResolved = $backend
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

# Resolve the mode: env overrides the parameter, and the deprecated switch means force.
# -InstallLlama must not silently beat an explicit -Llama: doing the opposite of what was
# asked for is worse than refusing, so ask the caller to pick one.
if ($InstallLlama -and $PSBoundParameters.ContainsKey("Llama")) {
    Die "-InstallLlama is a deprecated alias for -Llama force; pass only one of them"
}
$LlamaMode = $Llama
if ($InstallLlama) {
    Write-Warning "[setup_env] -InstallLlama is deprecated, use -Llama force"
    $LlamaMode = "force"
}
if ($env:TRUSTA_INSTALL_LLAMA) {
    # Accept the same spellings as setup_env.sh, not just auto/1/0
    switch -Regex ($env:TRUSTA_INSTALL_LLAMA.Trim()) {
        '^(?i)auto$'                { $LlamaMode = "auto" }
        '^(?i)(1|true|yes|on)$'     { $LlamaMode = "force" }
        '^(?i)(0|false|no|off)$'    { $LlamaMode = "skip" }
        default { Die "unsupported TRUSTA_INSTALL_LLAMA value: $($env:TRUSTA_INSTALL_LLAMA) (use auto / 1 / 0)" }
    }
}

Set-Location $ProjectRoot
Write-Host "[setup_env] uv sync --extra $Accel"
uv sync --extra $Accel

$XpuChecked = "n/a"
if ($Accel -eq "xpu") {
    if ($SkipXpuCheck -or $env:TRUSTA_SKIP_XPU_CHECK -eq "1") {
        Write-Host "[setup_env] skipping the XPU check (-SkipXpuCheck / TRUSTA_SKIP_XPU_CHECK=1)"
        $XpuChecked = "skipped"
    } elseif (Test-XpuUsable) {
        $XpuChecked = "ok (GEMM + training step)"
    } else {
        $XpuChecked = "skipped ($script:XpuSkipReason)"
    }
}

$script:LlamaInstalled = $false
if ($LlamaMode -eq "skip") {
    Write-Host "[setup_env] skipping llama (-Llama skip)"
    $LlamaStatus = "skipped"
} else {
    # Binary
    $existingLlama = $null
    if ($LlamaMode -ne "force") { $existingLlama = Resolve-LlamaBinary }
    if ($existingLlama) {
        Write-Host "[setup_env] llama binary already present at $existingLlama — leaving it alone (-Llama force to reinstall)"
        $binStatus = "already present"
        Save-LlamaBinaryToEnv $existingLlama
    } else {
        Install-LlamaPrebuilt
        if ($LlamaInstalled) {
            $binStatus = "prebuilt $LlamaVersion ($LlamaBackendResolved)"
            Save-LlamaBinaryToEnv (Resolve-LlamaBinary)
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
Write-Host "  XPU check   : $XpuChecked"
Write-Host "  Service Dir : $ServiceDir"
Write-Host "  llama       : $LlamaStatus"
Write-Host "=========================================="
