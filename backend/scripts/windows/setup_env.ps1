# Detect the GPU type and build the Python environment with the matching uv extra
# (torch cuda / xpu variant).
# llama (release binary + GGUF convert tooling) is installed by default, but only
# what is actually missing — an existing binary or convert checkout is left alone.
# Usage:
#   .\setup_env.ps1                # auto-detect the accelerator; install llama only if missing
#   .\setup_env.ps1 -Accel xpu     # force cuda | xpu
#   .\setup_env.ps1 -Llama skip    # skip llama entirely
#   .\setup_env.ps1 -Llama force   # reinstall even if already present
#   .\setup_env.ps1 -LlamaBackend vulkan  # force the generic Vulkan build (sees every Intel/AMD/NVIDIA card)
#   .\setup_env.ps1 -LlamaBackend cpu     # CPU-only build (no GPU, or the GPU build will not install)
#   .\setup_env.ps1 -AllowLlamaFallback   # unattended: accept a degraded llama instead of stopping
#   .\setup_env.ps1 -SkipXpuCheck  # skip the post-sync XPU smoke check
param(
    [ValidateSet("cuda", "xpu")]
    [string]$Accel = "",
    # auto (default) = install only what is missing; force = reinstall; skip = do nothing.
    # env TRUSTA_INSTALL_LLAMA (auto / 1 / 0) overrides it, matching setup_env.sh.
    [ValidateSet("auto", "force", "skip")]
    [string]$Llama = "auto",
    [switch]$InstallLlama,             # Deprecated alias for -Llama force; kept so existing invocations keep working
    [string]$LlamaVersion = "",        # Pinned llama version; defaults to $LlamaVersionDefault below, or env TRUSTA_LLAMA_VERSION
    # llama inference backend, decoupled from the torch accel: auto = cuda when NVIDIA is present,
    # else vulkan (env TRUSTA_LLAMA_BACKEND also overrides it)
    [ValidateSet("auto", "cuda", "vulkan", "cpu")]
    [string]$LlamaBackend = "auto",
    # Unattended installs cannot stop to ask, so let them accept a llama whose CPU backend
    # is degraded. Interactive runs should not pass this: stopping is the point of the
    # check. Env TRUSTA_LLAMA_ALLOW_FALLBACK=1 does the same.
    [switch]$AllowLlamaFallback,
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
# only shipped with the llama.cpp sources, not in the release zip. A sparse + blobless
# fetch pulls just these paths (~1.7MB) — no C++, nothing to compile. The pinned revision is
# maintained by hand here.
$LlamaConvertDir = Join-Path $ServiceDir "utils\llama.cpp"
# The llama binaries live beside the convert tooling, under the project rather than in
# WindowsApps: they are ours to place, pin and delete, and a per-project directory keeps
# two checkouts from fighting over one global install.
$LlamaBinDir     = Join-Path $ServiceDir "utils\llama-bin"
$LlamaCppUrl     = "https://github.com/ggml-org/llama.cpp"
# The one place the pinned build is named. -LlamaVersion / TRUSTA_LLAMA_VERSION override it.
$LlamaVersionDefault = "b10107"
# Offline fallback for the convert tooling's commit. Normally resolved from the tag by
# Resolve-LlamaCppRef below, so the two cannot drift; kept because a machine with no network
# should still be able to finish a setup, and annotated with its tag so
# tests/unit/test_llama_paths_agree.py can check it belongs to the version being pinned.
$LlamaCppRefFallback = "c0bc8591e8815c63cb01dd3f051a8b0df02501c9"  # = tag b10107 HEAD
$LlamaCppRef     = $LlamaCppRefFallback
$ConvertPaths    = @("convert_hf_to_gguf.py", "convert_lora_to_gguf.py", "conversion", "gguf-py")

# Record the binary setup_env actually resolved, so the service reads a decision instead of
# repeating the search. PATH is per-process: setup runs in your shell, the service may run
# under a service manager with a different one, and then the two disagree. The file ends up with
# exactly one LLAMA_SERVER_BINARY line: an override that resolves is kept as it is, one that
# points at nothing is replaced by what was actually verified, and duplicates are collapsed.
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
    # Exactly one LLAMA_SERVER_BINARY line survives this. Appending whenever no *live* line
    # matched left .env.example's commented example sitting above a second entry, so the file
    # showed the key twice; and once two live lines exist python-dotenv silently keeps the last,
    # which is not the one someone editing the first would expect.
    $live     = '^\s*LLAMA_SERVER_BINARY\s*='
    $commented = '^\s*#\s*LLAMA_SERVER_BINARY\s*='
    $setting  = "LLAMA_SERVER_BINARY=$Resolved"

    # ReadAllText/WriteAllText rather than Get-Content/Set-Content: the former honours a BOM on
    # the way in, and UTF8Encoding($false) guarantees none on the way out. A BOM here would glue
    # itself to the first key name and hide that variable from python-dotenv.
    $text = [System.IO.File]::ReadAllText($envFile)
    $eol  = if ($text -match "`r`n") { "`r`n" } else { "`n" }

    $out = New-Object System.Collections.Generic.List[string]
    $placed = $false
    $previous = $null
    $dropped = 0
    foreach ($line in ($text -split "`r`n|`n")) {
        if ($line -match $live) {
            if ($placed) { $dropped++; continue }   # a duplicate; drop it
            $previous = ($line -replace $live, '').Trim()
            $out.Add($setting)
            $placed = $true
            continue
        }
        $out.Add($line)
    }
    if (-not $placed) {
        # Take over the commented example in place, so the key stays where the template put it
        # instead of the file growing a second block that says the same thing.
        for ($i = 0; $i -lt $out.Count; $i++) {
            if ($out[$i] -match $commented) { $out[$i] = $setting; $placed = $true; break }
        }
    }
    if (-not $placed) {
        $out.Add("")
        $out.Add("# Recorded by setup_env: the llama binary it resolved, so the service does not")
        $out.Add("# have to re-resolve it from a possibly different PATH.")
        $out.Add($setting)
    }

    [System.IO.File]::WriteAllText(
        $envFile, ($out -join $eol), (New-Object System.Text.UTF8Encoding($false)))
    if ($previous -eq $Resolved) {
        Write-Host "[setup_env] .env already records LLAMA_SERVER_BINARY=$Resolved"
    } elseif ($previous) {
        Write-Host "[setup_env] .env LLAMA_SERVER_BINARY updated to $Resolved (was $previous)"
    } else {
        Write-Host "[setup_env] recorded LLAMA_SERVER_BINARY=$Resolved in .env"
    }
    if ($dropped -gt 0) {
        Write-Host "[setup_env] removed $dropped duplicate LLAMA_SERVER_BINARY line(s) from .env"
    }
}

# llama binaries come from llama.cpp's own GitHub release, not from ggml-org/llama-install.sh.
# That installer probes CUDA -> Vulkan -> CPU and keeps the first hit, so it can hand back a
# different backend than the one asked for and still exit 0. Worse, its CUDA/ROCm presets never
# set LLAMA_INSTALL_FLAGS, so those binaries are real CUDA builds whose *CPU* backend has no
# vector ISA - 2.1x slower once any weight is computed on the CPU, 5.1x on a pure-CPU workload:
# https://github.com/samhong5668/llama-bench-lab
# Naming an asset explicitly removes the probing, and llama.cpp's own releases carry a proper
# CPU backend (GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS, dispatched at run time).
$LlamaReleaseBase = "https://github.com/ggml-org/llama.cpp/releases/download"
$LlamaApiBase     = "https://api.github.com/repos/ggml-org/llama.cpp"

# Ask the remote which commit the tag points at, so the convert scripts and the binary are
# always the same revision. Maintaining both by hand meant a version bump could update one and
# not the other, with no error - just convert scripts from a different build.
function Resolve-LlamaCppRef {
    # git ls-remote rather than the API: git is already required, and this needs no token.
    #
    # GIT_TERMINAL_PROMPT=0 and the timeout are both load-bearing: against a URL that answers
    # with an auth challenge, git blocks forever waiting for a username, which would hang the
    # whole setup instead of falling back. Measured - a wrong URL hung until killed. Run through
    # ProcessStartInfo because PowerShell has no way to time out the call operator.
    $sha = $null
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName  = "git"
        $psi.Arguments = "ls-remote --tags `"$LlamaCppUrl`" `"refs/tags/$LlamaVersion`""
        $psi.UseShellExecute        = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError  = $true
        $psi.EnvironmentVariables["GIT_TERMINAL_PROMPT"] = "0"
        $psi.EnvironmentVariables["GIT_ASKPASS"] = "echo"
        $proc = [System.Diagnostics.Process]::Start($psi)
        $stdout = $proc.StandardOutput.ReadToEndAsync()
        if (-not $proc.WaitForExit(30000)) {
            $proc.Kill()
            $proc.WaitForExit()
        } elseif ($proc.ExitCode -eq 0) {
            $first = ($stdout.Result -split "`n" | Where-Object { $_.Trim() } | Select-Object -First 1)
            if ($first) { $sha = ($first.Trim() -split "\s+")[0] }
        }
    } catch { $sha = $null }
    if ($sha -match "^[0-9a-f]{40}$") {
        $script:LlamaCppRef = $sha
        Write-Host "[setup_env] $LlamaVersion resolves to $sha"
    } else {
        Write-Warning "[setup_env] could not resolve $LlamaVersion from the remote; using the pinned fallback $LlamaCppRefFallback"
    }
}

# The CUDA asset has to match the cuBLAS that will actually load. ggml-cuda.dll imports
# cublas64_<major>.dll, and this project already ships that inside torch, so the torch wheel
# decides the major version. Falls back to nvidia-smi, then to 13.
function Get-LlamaCudaMajor {
    $venv = $env:UV_PROJECT_ENVIRONMENT
    if (-not $venv) { $venv = Join-Path $ProjectRoot ".venv" }
    if (-not [System.IO.Path]::IsPathRooted($venv)) { $venv = Join-Path $ProjectRoot $venv }
    $torchLib = Join-Path $venv "Lib\site-packages\torch\lib"
    if (Test-Path $torchLib) {
        $cublas = Get-ChildItem -Path $torchLib -Filter "cublas64_*.dll" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($cublas -and $cublas.Name -match "cublas64_(\d+)\.dll") {
            Write-Host "[setup_env] torch ships $($cublas.Name), so pairing with CUDA $($Matches[1])"
            return $Matches[1]
        }
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        # The header wording is not stable across drivers: 610.62 prints "CUDA UMD Version:
        # 13.3", older ones print "CUDA Version: 13.0", and `nvidia-smi -q` pads with spaces
        # before the colon. Accept all of those rather than one of them. Note that the plain
        # "CUDA Version" spelling is documented as going away in CUDA 14.
        # No 2>$null here: with $ErrorActionPreference = "Stop", redirecting a native command's
        # stderr turns any line it writes into a terminating NativeCommandError.
        $smi = ""
        try { $smi = (& nvidia-smi | Out-String) } catch { $smi = "" }
        $m = [regex]::Match($smi, "CUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+)\.")
        if ($m.Success) {
            $major = $m.Groups[1].Value
            Write-Host "[setup_env] no torch cuBLAS yet; nvidia-smi reports CUDA $major"
            return $major
        }
    }
    Write-Host "[setup_env] could not determine the CUDA major version, defaulting to 13"
    return "13"
}

# One asset is enough. Every backend zip is a superset of the CPU zip: checked on b10107, the
# cuda-13.3 and vulkan zips each carry all 22 executables, all 14 ggml-cpu-*.dll variants and
# libomp140, plus their own backend DLL. Downloading the CPU zip as well would only re-fetch the
# same files and then fail to overwrite the ones a virus scanner is still holding open.
# Which CUDA minor this tag actually publishes for our major. Asked of the release rather than
# hardcoded: llama.cpp has already added cuda-13.4 for arm64, and the x64 set will move too, so a
# fixed "12 -> 12.4, else 13.3" map is a future silent 404. Falls back to that map offline.
function Get-LlamaCudaAssetName {
    param([string]$Tag, [string]$Major)
    $pattern = "^llama-$([regex]::Escape($Tag))-bin-win-cuda-$([regex]::Escape($Major))\.(\d+)-x64\.zip$"
    try {
        $release = Invoke-RestMethod -UseBasicParsing -Uri "$LlamaApiBase/releases/tags/$Tag" `
            -Headers @{ "User-Agent" = "trusta-ast-backend-setup" } -TimeoutSec 30
        # Highest minor wins, compared numerically so 13.10 beats 13.9.
        $best = $release.assets.name |
            Where-Object { $_ -match $pattern } |
            Sort-Object { [int]([regex]::Match($_, $pattern).Groups[1].Value) } -Descending |
            Select-Object -First 1
        if ($best) {
            Write-Host "[setup_env] $Tag publishes $best for CUDA $Major"
            return $best
        }
        Write-Warning "[setup_env] $Tag publishes no win-cuda-$Major.*-x64 asset"
    } catch {
        Write-Warning "[setup_env] could not list $Tag's assets ($($_.Exception.Message)); guessing the CUDA minor"
    }
    $fallback = if ($Major -eq "12") { "12.4" } else { "13.3" }
    return "llama-$Tag-bin-win-cuda-$fallback-x64.zip"
}

function Get-LlamaReleaseAsset {
    param([string]$Tag, [string]$Backend)
    switch ($Backend) {
        "cuda"   { return (Get-LlamaCudaAssetName -Tag $Tag -Major (Get-LlamaCudaMajor)) }
        "vulkan" { return "llama-$Tag-bin-win-vulkan-x64.zip" }
        default  { return "llama-$Tag-bin-win-cpu-x64.zip" }
    }
}

function Install-LlamaRelease {
    $asset = Get-LlamaReleaseAsset -Tag $LlamaVersion -Backend $script:LlamaBackendResolved
    $staging = Join-Path $env:TEMP ("llama-release-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    $script:LlamaInstalled = $false
    try {
        $url = "$LlamaReleaseBase/$LlamaVersion/$asset"
        $zip = Join-Path $staging $asset
        Write-Host "[setup_env] downloading $asset"
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
        } catch {
            Write-Warning "[setup_env] could not download $asset from $url - $($_.Exception.Message)"
            return
        }
        # Extract to staging first, then move into place: expanding straight over a populated
        # directory fails on any file another process still has open.
        $unpacked = Join-Path $staging "unpacked"
        try {
            Expand-Archive -Path $zip -DestinationPath $unpacked -Force
        } catch {
            Write-Warning "[setup_env] could not extract $asset - $($_.Exception.Message)"
            return
        }
        # Some release zips nest everything under build\bin; take whichever level holds the files.
        $src = $unpacked
        $nested = Join-Path $unpacked "build\bin"
        if (Test-Path $nested) { $src = $nested }

        # Replace the directory rather than copying over it. Copy-Item only writes the files the
        # source has, so unpacking the cpu zip over a cuda install would leave ggml-cuda.dll
        # behind and the binary would keep loading CUDA - a backend switch that silently does
        # nothing. Same for a version change: files dropped between releases would survive.
        # The app-local CRT is placed after this, so clearing here does not lose it.
        if (Test-Path $LlamaBinDir) {
            Remove-Item -Recurse -Force $LlamaBinDir -ErrorAction Stop
        }
        New-Item -ItemType Directory -Path $LlamaBinDir -Force | Out-Null
        try {
            Copy-Item -Path (Join-Path $src "*") -Destination $LlamaBinDir -Recurse -Force
        } catch {
            Write-Warning "[setup_env] could not place the binaries in $LlamaBinDir - $($_.Exception.Message)"
            return
        }
    } finally {
        Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
    }
    # Do not report success on the strength of "no exception": check that the file the service
    # will actually run is there.
    $server = Join-Path $LlamaBinDir "llama-server.exe"
    if (-not (Test-Path $server -PathType Leaf)) {
        Write-Warning "[setup_env] $asset unpacked but llama-server.exe is not in $LlamaBinDir"
        return
    }
    $script:LlamaInstalled = $true
    Write-Host "[setup_env] llama $LlamaVersion ($script:LlamaBackendResolved) installed: $LlamaBinDir"
}

# The release binaries import MSVCP140 / VCRUNTIME140(_1) from the VC++ redistributable. That is
# already a prerequisite of this project - torch_cpu.dll imports the same DLLs and the wheels do
# not ship them - so the usual case needs nothing done. The gap is a uv-managed Python: it is a
# portable build that runs no prerequisite installer, so a machine that never had Visual Studio
# or the redistributable has none of them.
#
# Copying the three files next to the executables (app-local deployment) removes the dependency
# without administrator rights. Verified that the app-local copy wins the loader search, by
# corrupting it while system32's copy stayed intact and watching the process fail anyway.
$LlamaCrtDlls = @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")

function Test-LlamaStarts {
    param([string]$Binary)
    # Ask the binary for the cheapest possible thing. A missing DLL surfaces as a loader status
    # code (0xC0000135 not found / 0xC000012F bad image), not as an ordinary failure.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Binary
    $psi.Arguments = "--version"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    try {
        $proc = [System.Diagnostics.Process]::Start($psi)
        $proc.StandardOutput.ReadToEnd() | Out-Null
        $proc.StandardError.ReadToEnd() | Out-Null
        $proc.WaitForExit()
    } catch {
        return $false
    }
    # Only the loader codes mean "cannot start"; a tool that merely rejects --version did run.
    return -not (($proc.ExitCode -eq -1073741515) -or ($proc.ExitCode -eq -1073741521))
}

function Copy-LlamaCrtFromVisualStudio {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    foreach ($root in (& $vswhere -all -products * -property installationPath 2>$null)) {
        if (-not $root) { continue }
        $redist = Join-Path $root "VC\Redist\MSVC"
        if (-not (Test-Path $redist)) { continue }
        # Highest toolset first: an app-local CRT must not be older than what the binaries were
        # built against.
        $dirs = Get-ChildItem $redist -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        foreach ($dir in $dirs) {
            $crt = Get-ChildItem (Join-Path $dir.FullName "x64") -Directory -Filter "Microsoft.VC*.CRT" `
                -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $crt) { continue }
            $missing = $LlamaCrtDlls | Where-Object { -not (Test-Path (Join-Path $crt.FullName $_)) }
            if ($missing) { continue }
            foreach ($dll in $LlamaCrtDlls) {
                Copy-Item (Join-Path $crt.FullName $dll) $LlamaBinDir -Force
            }
            Write-Host "[setup_env] copied the VC++ CRT app-local from $($crt.FullName)"
            return $true
        }
    }
    return $false
}

function Copy-LlamaCrtFromUrl {
    # TRUSTA_VCREDIST_URL points at a zip holding just the three DLLs, so a packaged installer
    # can carry them without Visual Studio being present on the target. Unset by default rather
    # than pointing at a guessed location.
    if (-not $env:TRUSTA_VCREDIST_URL) { return $false }
    $staging = Join-Path $env:TEMP ("vcredist-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    try {
        $zip = Join-Path $staging "vcredist.zip"
        Write-Host "[setup_env] fetching the VC++ CRT from TRUSTA_VCREDIST_URL"
        Invoke-WebRequest -UseBasicParsing -Uri $env:TRUSTA_VCREDIST_URL -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $staging -Force
        # Resolve all three before copying any, the way the Visual Studio path already does.
        # Copying them as they are found leaves a partial CRT behind when the archive is missing
        # one, and an app-local DLL shadows the system one - so a half-applied archive can break
        # a binary whose CRT was fine, and keep it broken on every later run.
        $resolved = [ordered]@{}
        foreach ($dll in $LlamaCrtDlls) {
            $found = Get-ChildItem $staging -Recurse -Filter $dll -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if (-not $found) {
                Write-Warning "[setup_env] $dll was not in the archive at TRUSTA_VCREDIST_URL"
                return $false
            }
            $resolved[$dll] = $found.FullName
        }
        foreach ($dll in $LlamaCrtDlls) {
            Copy-Item $resolved[$dll] $LlamaBinDir -Force
        }
        Write-Host "[setup_env] copied the VC++ CRT app-local from TRUSTA_VCREDIST_URL"
        return $true
    } catch {
        Write-Warning "[setup_env] could not use TRUSTA_VCREDIST_URL - $($_.Exception.Message)"
        return $false
    } finally {
        Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
    }
}

# Cheapest step first: most machines already resolve the CRT and need nothing done at all.
function Resolve-LlamaCrt {
    param([string]$Binary)
    if (Test-LlamaStarts -Binary $Binary) { return $true }
    $name = [System.IO.Path]::GetFileName($Binary)
    Write-Host "[setup_env] $name cannot start; resolving the VC++ CRT"
    foreach ($step in @({ Copy-LlamaCrtFromVisualStudio }, { Copy-LlamaCrtFromUrl })) {
        if (& $step) {
            if (Test-LlamaStarts -Binary $Binary) { return $true }
            Write-Warning "[setup_env] the CRT was placed but the binary still will not start"
        }
    }
    return $false
}

# Verify what was installed instead of trusting it: scripts/llama_backend.py explains why
# "is it CUDA?" is not a sufficient question on its own.
function Test-LlamaBinary {
    param([string]$Binary, [string]$RequireBackend)
    $probe = Join-Path $ProjectRoot "scripts\llama_backend.py"
    if (-not (Test-Path $probe)) {
        Write-Warning "[setup_env] scripts\llama_backend.py not found, cannot verify the binary"
        $script:LlamaVerify = "unverified (probe missing)"
        return $true
    }
    $python = Get-VenvPython
    if (-not (Test-Path $python)) {
        Write-Warning "[setup_env] $python not found, cannot verify the llama binary"
        $script:LlamaVerify = "unverified (venv python missing)"
        return $true
    }
    $probeArgs = @("-u", $probe, $Binary)
    if ($RequireBackend) { $probeArgs += @("--require", $RequireBackend) }
    if ($script:LlamaAllowFallback) { $probeArgs += "--allow-degraded" }
    # Out-Host, not the output stream: the probe's report must reach the console without being
    # returned. A function that emits anything alongside its boolean returns an array, and a
    # non-empty array is truthy in PowerShell - so `if (Test-LlamaBinary ...)` would treat a
    # failed verification as a pass.
    & $python @probeArgs | Out-Host
    if ($LASTEXITCODE -eq 0) {
        $script:LlamaVerify = if ($script:LlamaAllowFallback) {
            "accepted (fallback allowed)"
        } else {
            "ok"
        }
        return $true
    }
    $script:LlamaVerify = "FAILED"
    return $false
}

# The same four options the GUI offers, so both paths speak one vocabulary.
function Show-LlamaOptions {
    Write-Host ""
    Write-Host "  How to proceed (re-run setup_env with one of these):"
    Write-Host "    1. retry after fixing the driver / toolkit / network"
    Write-Host "         -Llama force                              (TRUSTA_INSTALL_LLAMA=1)"
    Write-Host "    2. use the CPU-only build"
    Write-Host "         -Llama force -LlamaBackend cpu            (TRUSTA_INSTALL_LLAMA=1 TRUSTA_LLAMA_BACKEND=cpu)"
    Write-Host "    3. use your own binary"
    Write-Host "         set LLAMA_SERVER_BINARY in .env to its path"
    Write-Host "    4. skip llama entirely"
    Write-Host "         -Llama skip                               (TRUSTA_INSTALL_LLAMA=0)"
    # -Llama force matters on 2: without it an existing binary counts as "already present" and
    # nothing is reinstalled, so the new backend would be ignored and the same binary re-checked.
    Write-Host "  Unattended runs can accept a degraded binary with -AllowLlamaFallback"
    Write-Host "  (TRUSTA_LLAMA_ALLOW_FALLBACK=1)."
    Write-Host ""
}

# Where the llama binary is expected. An explicit LLAMA_SERVER_BINARY wins; otherwise the
# directory setup_env unpacks the release into. .env is consulted too, since that is
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
    # ggml-org/llama-install.sh's WindowsApps\llama.exe is deliberately not a fallback: its
    # CUDA builds ship a CPU backend with no vector ISA. Point LLAMA_SERVER_BINARY at one
    # explicitly if you really want it.
    return (Join-Path $LlamaBinDir "llama-server.exe")
}

# Returns the path that actually holds a usable binary — the configured/default one, or
# whatever `llama-server` resolves to on PATH — and $null when there is none. Callers report the
# path that was really found instead of the one that was merely expected.
function Resolve-LlamaBinary {
    $configured = Get-LlamaBinaryPath
    if (Test-Path $configured -PathType Leaf) { return $configured }
    # A configured path that does not exist must not end the search. setup_env may have just
    # installed a good binary at the default location, and giving up here skipped the whole
    # verify-and-record step while the run still exited 0 - leaving .env pointing at a path
    # that is not there and the service unable to start.
    $default = Join-Path $LlamaBinDir "llama-server.exe"
    if ($configured -ne $default -and (Test-Path $default -PathType Leaf)) {
        # Said once: this function is called both to test for an existing install and again
        # after one, and the same warning twice reads like something went wrong twice.
        if (-not $script:LlamaOverrideWarned) {
            Write-Host "[setup_env] LLAMA_SERVER_BINARY is set to $configured, which does not exist; using $default"
            $script:LlamaOverrideWarned = $true
        }
        return $default
    }
    $onPath = Get-Command llama-server -ErrorAction SilentlyContinue
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
if ($backend -notin @("auto", "cuda", "vulkan", "cpu")) {
    Die "unsupported TRUSTA_LLAMA_BACKEND: $backend (use auto | cuda | vulkan | cpu)"
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

# Unattended runs may accept a degraded llama; interactive ones should stop instead.
$script:LlamaAllowFallback = [bool]$AllowLlamaFallback
if (-not $script:LlamaAllowFallback -and $env:TRUSTA_LLAMA_ALLOW_FALLBACK) {
    if ($env:TRUSTA_LLAMA_ALLOW_FALLBACK.Trim() -match "^(?i)(1|true|yes|on)$") {
        $script:LlamaAllowFallback = $true
    }
}
$script:LlamaVerify = "n/a"

# Pinned version: -LlamaVersion wins, then env TRUSTA_LLAMA_VERSION, then the default
if (-not $LlamaVersion) {
    if ($env:TRUSTA_LLAMA_VERSION) { $LlamaVersion = $env:TRUSTA_LLAMA_VERSION } else { $LlamaVersion = $LlamaVersionDefault }
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
    # Convert tooling first. It is pure Python, independent of the binary, and cheap - doing it
    # before the binary means a binary failure below still leaves GGUF conversion working.
    if ($LlamaMode -ne "force" -and (Test-LlamaConvertTooling)) {
        Write-Host "[setup_env] convert tooling already present at $LlamaConvertDir — leaving it alone"
        $convertStatus = "already present"
    } else {
        # Resolve the tag to a commit first: the convert scripts have to be the same revision
        # as the binary.
        Resolve-LlamaCppRef
        Get-LlamaConvertTooling
        $convertStatus = "fetched"
    }

    # Binary
    $existingLlama = $null
    if ($LlamaMode -ne "force") { $existingLlama = Resolve-LlamaBinary }
    if ($existingLlama) {
        Write-Host "[setup_env] llama binary already present at $existingLlama — leaving it alone (-Llama force to reinstall)"
        $binStatus = "already present"
    } else {
        Install-LlamaRelease
        if ($LlamaInstalled) {
            $binStatus = "release $LlamaVersion ($LlamaBackendResolved)"
        } else {
            # Exiting non-zero, not warning: the service has no usable binary either way, and a
            # launcher that only reads the exit code would otherwise report this as a success.
            $LlamaStatus = "binary: download failed / convert tooling: $convertStatus"
            Show-LlamaOptions
            Die "could not install llama $LlamaVersion ($LlamaBackendResolved) - see the download error above. Nothing was recorded in .env."
        }
    }

    # From here on the two paths are the same: whatever binary we are about to hand the
    # service has to start, and has to be the build we asked for. An already-present binary
    # gets checked too - it may be a llama.app install from before this change.
    $llamaBin = Resolve-LlamaBinary
    if ($llamaBin) {
        if (-not (Resolve-LlamaCrt -Binary $llamaBin)) {
            Show-LlamaOptions
            # Name the download rather than the product: the VC++ runtime is not part of
            # Windows, and a message that says which component is missing without saying where
            # to get it leaves the reader exactly as stuck as before.
            Write-Host ""
            Write-Host "  The VC++ runtime is a separate Microsoft download, not part of Windows:"
            Write-Host "    https://aka.ms/vs/17/release/vc_redist.x64.exe"
            Write-Host "  Installing it needs administrator rights. Without them, set TRUSTA_VCREDIST_URL"
            Write-Host "  to a zip holding msvcp140.dll, vcruntime140.dll and vcruntime140_1.dll, which"
            Write-Host "  setup_env will place next to the binary instead."
            Write-Host ""
            Die "the llama binary at $llamaBin cannot start and the VC++ CRT could not be resolved (see above)."
        }
        # Do not require a backend for the cpu build: "cpu" is what the probe reports when no
        # accelerator is listed, which is also what a GPU build on a GPU-less host reports.
        $require = if ($LlamaBackendResolved -eq "cpu") { "" } else { $LlamaBackendResolved }
        if (Test-LlamaBinary -Binary $llamaBin -RequireBackend $require) {
            # Only record a binary that passed. Writing a rejected one into .env would make the
            # next run treat it as already present and skip the check for good.
            Save-LlamaBinaryToEnv $llamaBin
        } else {
            Show-LlamaOptions
            Die "the llama binary at $llamaBin did not pass verification (see above). It was not recorded in .env."
        }
    } else {
        # Install either succeeded or already exited above, so getting here means nothing usable
        # could be found. Falling through silently would exit 0 having verified nothing.
        Show-LlamaOptions
        Die "no llama binary could be resolved: LLAMA_SERVER_BINARY points at '$(Get-LlamaBinaryPath)', which does not exist, and llama-server is not on PATH. Clear that setting to use $LlamaBinDir."
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
Write-Host "  llama check : $script:LlamaVerify"
Write-Host "=========================================="
