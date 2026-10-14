# Detect the GPU type and build the Python environment with the matching uv extra
# (torch cuda / xpu variant).
# llama (release binary + GGUF convert tooling) is installed by default, but only
# what is actually missing — an existing binary or convert checkout is left alone.
# Usage:
#   .\setup_env.ps1                # auto-detect the accelerator; install llama only if missing
#                                  # (llama-server, llama-quantize and llama-fit-params)
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

# Downloading a large file over a link that is not reliably fast.
#
# Invoke-WebRequest is an HttpWebRequest on Windows PowerShell 5.1, and that class
# defaults to a 100-second Timeout which covers the *entire* transfer rather than the
# gaps in it. The llama runtime is 137 MB, so the default is really a demand for
# 11 Mbit/s sustained from the first byte to the last: a link slower than that fails
# with "The operation has timed out" however healthy it is. Seen on a test VM, where
# the asset answered a HEAD request in 0.7 seconds and the download still timed out.
#
# Raising the number is the wrong fix in both directions -- too small still kills a
# slow but working transfer, too large waits ten minutes on a connection that has
# actually died. What this wants is a stall detector, a retry, and a resume, and
# curl.exe has all three; it ships in Windows 10 1803 and later, so it is present on
# anything this product supports.
#
# Resume is only offered where the result is checked against a published hash, which
# is what makes a resumed file safe to use: one that came back wrong is rejected there
# rather than unpacked.
function Invoke-Curl {
    param([string]$Curl, [string]$Uri, [string]$OutFile, [bool]$Resume)
    $curlArgs = @(
        "-L", "--fail", "--silent", "--show-error",
        # Three attempts over transport errors and 5xx, two seconds apart.
        "--retry", "3", "--retry-delay", "2", "--retry-all-errors",
        # The stall detector: give up when throughput stays under 1 KB/s for 30
        # seconds. This is the condition the 100-second deadline was standing in for.
        "--speed-limit", "1024", "--speed-time", "30"
    )
    if ($Resume) { $curlArgs += @("--continue-at", "-") }
    $curlArgs += @("--output", $OutFile, $Uri)
    & $Curl @curlArgs
    return $LASTEXITCODE
}

function Get-RemoteFile {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$OutFile,
        # Only for downloads whose content is verified afterwards.
        [switch]$Resume
    )
    $curl = Join-Path $env:SystemRoot "System32\curl.exe"
    if (Test-Path -LiteralPath $curl) {
        $code = Invoke-Curl -Curl $curl -Uri $Uri -OutFile $OutFile -Resume $Resume.IsPresent
        if ($code -eq 0) { return }
        if ($Resume) {
            # A leftover part-file cannot always be resumed from: the server may ignore
            # the range, or the file may already be complete, and curl reports that as
            # an error the reader cannot act on. Start it over instead.
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            $code = Invoke-Curl -Curl $curl -Uri $Uri -OutFile $OutFile -Resume $false
            if ($code -eq 0) { return }
        }
        throw "curl.exe exited $code fetching $Uri"
    }

    # No curl: Windows older than 10 1803. TimeoutSec 0 is infinite, which is wrong in
    # the other direction, but a transfer that hangs forever is at least visible --
    # where the 100-second default silently rules out every slow connection. There is
    # no stall detector to be had here.
    $previous = $ProgressPreference
    # Without this, one progress bar is drawn per buffer, which on a slow link costs
    # more time than the transfer.
    $ProgressPreference = "SilentlyContinue"
    try {
        Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing -TimeoutSec 0
    } finally {
        $ProgressPreference = $previous
    }
}

# uv is the one thing this script cannot build for itself, and until now it simply
# assumed it: on a machine without it the first `uv sync` failed with "The term 'uv' is
# not recognized", which is a poor thing to say to someone who ran a one-click
# installer precisely so they would not have to install a toolchain.
#
# Two failures, not one. It may be genuinely absent, or it may be installed and missing
# from PATH -- the launcher is started from a shortcut, and a shortcut's environment is
# whatever Explorer had when it started, so an installer that appended to PATH after
# that will not be seen until the next sign-in.
#
# Mirrors ensure_uv in setup_env.sh. Installs per-user, which is what lets the whole
# product stay per-user with no elevation.
function Install-UvIfMissing {
    # A packaged installer carries its own uv, pinned and hashed when it was built, and
    # names it here. Taken before anything on the machine: which uv resolved the lock is
    # part of what a release is, and "whatever the user happened to have" is not an
    # answer. Unset in a checkout, where the machine's own uv is the right one.
    if ($env:TRUSTA_UV -and (Test-Path -LiteralPath $env:TRUSTA_UV)) {
        Write-Host "[setup_env] using the uv the installer carries: $env:TRUSTA_UV"
        $env:PATH = (Split-Path -Parent $env:TRUSTA_UV) + ";$env:PATH"
        return
    }

    if (Get-Command uv -ErrorAction SilentlyContinue) { return }

    $userBin = Join-Path $env:USERPROFILE ".local\bin"
    $uvExe   = Join-Path $userBin "uv.exe"
    if (Test-Path -LiteralPath $uvExe) {
        Write-Host "[setup_env] uv found at $uvExe but not on PATH; adding it"
        $env:PATH = "$userBin;$env:PATH"
        return
    }

    Write-Host "[setup_env] uv is not installed; fetching it into $userBin"
    $env:UV_INSTALL_DIR      = $userBin
    $env:UV_UNMANAGED_INSTALL = $userBin
    # To a file, then run the file. Piping the response into Invoke-Expression is the
    # documented one-liner and it does not work here: with -UseBasicParsing on Windows
    # PowerShell 5.1, .Content comes back as a byte array, and Invoke-Expression then
    # fails with "Cannot convert 'System.Byte[]' to the type 'System.String'". Measured
    # on a clean Windows 11 VM. A file also survives the script being large enough to
    # matter and gives the installer a real path to report errors against.
    $installer = Join-Path $env:TEMP "uv-install-$([guid]::NewGuid().ToString('N').Substring(0,8)).ps1"
    try {
        Get-RemoteFile -Uri "https://astral.sh/uv/install.ps1" -OutFile $installer
        if (-not (Test-Path $installer) -or (Get-Item $installer).Length -eq 0) {
            Die "downloaded an empty uv installer from https://astral.sh/uv - check the network, or install uv by hand."
        }
        & powershell -NoProfile -ExecutionPolicy Bypass -File $installer | Out-Null
    } catch {
        Die "could not install uv from https://astral.sh/uv/install.ps1 - $($_.Exception.Message). Check the network, or install it by hand."
    } finally {
        Remove-Item $installer -Force -ErrorAction SilentlyContinue
    }
    $env:PATH = "$userBin;$env:PATH"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Die "installed uv into $userBin but it is still not runnable"
    }
    Write-Host "[setup_env] uv $((uv --version) -replace '^uv\s+','') installed"
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
# The standalone tools the backend shells out to besides the server, and the .env setting it
# reads first for each: llama-quantize (service/utils/conversion_manager.py) and
# llama-fit-params (gguf_estimator.py). The release zip ships both beside llama-server.
$LlamaTools = [ordered]@{
    "llama-quantize.exe"   = "LLAMA_QUANTIZE_BIN"
    "llama-fit-params.exe" = "LLAMA_FIT_PARAMS_BIN"
}
# Where those tools go when the server is not ours (LLAMA_SERVER_BINARY or PATH): setup_env
# does not write into someone else's directory, so they get their own and are recorded in .env.
$LlamaToolsDir   = Join-Path $ServiceDir "utils\llama-tools"
$LlamaCppUrl     = "https://github.com/ggml-org/llama.cpp"
# The one place the pinned build is named. -LlamaVersion / TRUSTA_LLAMA_VERSION override it.
$LlamaVersionDefault = "b10107"
# The convert scripts come from an archive of the tag, not a checkout.
#
# git used to be a hard prerequisite of this script and of the desktop launcher's preflight
# check, purely to fetch these four paths. On a clean Windows that cost an installation of
# git, and winget's Git.Git has no per-user installer -- it elevates whatever --scope it is
# given -- so a one-click installer whose entire design is to need no administrator put a
# UAC prompt in front of the user on first run. GitHub serves every tag as a zip, which
# Expand-Archive can open with nothing installed.
#
# Nothing is lost by it: the tag is resolved by the server, so there is no sha to resolve, no
# offline fallback sha to keep in step with the tag, and no way for the convert scripts to end
# up on a different revision from the binary.
$ConvertPaths    = @("convert_hf_to_gguf.py", "convert_lora_to_gguf.py", "conversion", "gguf-py")
# Written beside the copied scripts, because an archive carries no revision of its own.
$LlamaSrcStamp   = ".trusta-llama-version"

# Record what setup_env actually resolved, so the service reads a decision instead of
# repeating the search. PATH is per-process: setup runs in your shell, the service may run
# under a service manager with a different one, and then the two disagree. The file ends up with
# exactly one line per key: an override that resolves is kept as it is, one that
# points at nothing is replaced by what was actually verified, and duplicates are collapsed.
function Save-EnvSetting {
    param([string]$Key, [string]$Resolved)
    if (-not $Resolved) { return }
    $envFile = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $envFile)) {
        # settings.py falls back to .env.example when .env is missing; creating one here would
        # change which file the service loads, so only say what could not be recorded.
        Write-Host "[setup_env] no .env, so $Key was not recorded (resolved: $Resolved)"
        return
    }
    # Exactly one line for the key survives this. Appending whenever no *live* line
    # matched left .env.example's commented example sitting above a second entry, so the file
    # showed the key twice; and once two live lines exist python-dotenv silently keeps the last,
    # which is not the one someone editing the first would expect.
    $live     = "^\s*$Key\s*="
    $commented = "^\s*#\s*$Key\s*="
    $setting  = "$Key=$Resolved"

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
        $out.Add("# Recorded by setup_env: the llama path it resolved, so the service does not")
        $out.Add("# have to re-resolve it from a possibly different PATH.")
        $out.Add($setting)
    }

    [System.IO.File]::WriteAllText(
        $envFile, ($out -join $eol), (New-Object System.Text.UTF8Encoding($false)))
    if ($previous -eq $Resolved) {
        Write-Host "[setup_env] .env already records $Key=$Resolved"
    } elseif ($previous) {
        Write-Host "[setup_env] .env $Key updated to $Resolved (was $previous)"
    } else {
        Write-Host "[setup_env] recorded $Key=$Resolved in .env"
    }
    if ($dropped -gt 0) {
        Write-Host "[setup_env] removed $dropped duplicate $Key line(s) from .env"
    }
}

function Save-LlamaBinaryToEnv {
    param([string]$Resolved)
    Save-EnvSetting -Key "LLAMA_SERVER_BINARY" -Resolved $Resolved
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
# What a tag publishes, as a name -> SHA-256 map, or $null when the release cannot be read.
#
# The hashes come from the same call that already had to be made to find the CUDA asset, and
# GitHub reports one for every asset as "sha256:<hex>". That is worth checking even though it
# arrives from the same vendor as the file: it is served by api.github.com while the download
# comes from objects.githubusercontent.com, so it catches a truncated or corrupted transfer, a
# stale mirror, and a proxy that rewrites one host but not the other. It is not a defence
# against GitHub itself, and nothing here pretends otherwise.
function Get-LlamaReleaseAssets {
    param([string]$Tag)
    try {
        $release = Invoke-RestMethod -UseBasicParsing -Uri "$LlamaApiBase/releases/tags/$Tag" `
            -Headers @{ "User-Agent" = "trusta-ast-backend-setup" } -TimeoutSec 30
    } catch {
        Write-Warning "[setup_env] could not list $Tag's assets ($($_.Exception.Message))"
        return $null
    }
    $map = @{}
    foreach ($item in $release.assets) {
        $digest = $null
        if ($item.digest -and $item.digest.StartsWith("sha256:")) {
            $digest = $item.digest.Substring(7).ToLower()
        }
        $map[$item.name] = $digest
    }
    return $map
}

# The asset to download and the hash to hold it to. Which CUDA minor a tag publishes for our
# major is asked of the release rather than hardcoded: llama.cpp has already added cuda-13.4 for
# arm64, and the x64 set will move too, so a fixed "12 -> 12.4, else 13.3" map is a future silent
# 404. Offline that map is still the guess, and then there is no hash to check against.
function Get-LlamaReleaseAsset {
    param([string]$Tag, [string]$Backend)
    $assets = Get-LlamaReleaseAssets -Tag $Tag
    $name = $null
    switch ($Backend) {
        "cuda" {
            $major = Get-LlamaCudaMajor
            $pattern = "^llama-$([regex]::Escape($Tag))-bin-win-cuda-$([regex]::Escape($major))\.(\d+)-x64\.zip$"
            if ($assets) {
                # Highest minor wins, compared numerically so 13.10 beats 13.9.
                $best = $assets.Keys |
                    Where-Object { $_ -match $pattern } |
                    Sort-Object { [int]([regex]::Match($_, $pattern).Groups[1].Value) } -Descending |
                    Select-Object -First 1
                if ($best) {
                    Write-Host "[setup_env] $Tag publishes $best for CUDA $major"
                    $name = $best
                } else {
                    Write-Warning "[setup_env] $Tag publishes no win-cuda-$major.*-x64 asset"
                }
            }
            if (-not $name) {
                Write-Warning "[setup_env] guessing the CUDA minor for $Tag"
                $fallback = if ($major -eq "12") { "12.4" } else { "13.3" }
                $name = "llama-$Tag-bin-win-cuda-$fallback-x64.zip"
            }
        }
        "vulkan" { $name = "llama-$Tag-bin-win-vulkan-x64.zip" }
        default  { $name = "llama-$Tag-bin-win-cpu-x64.zip" }
    }
    $sha = if ($assets) { $assets[$name] } else { $null }
    return [pscustomobject]@{ Name = $name; Sha256 = $sha }
}

# Defaults to the full install into $LlamaBinDir, which must yield llama-server.exe. A tools-only
# install passes its own directory and the tools it is there for.
function Install-LlamaRelease {
    param(
        [string]$Destination = $LlamaBinDir,
        [string[]]$Required = @("llama-server.exe")
    )
    $release = Get-LlamaReleaseAsset -Tag $LlamaVersion -Backend $script:LlamaBackendResolved
    $asset = $release.Name
    $staging = Join-Path $env:TEMP ("llama-release-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    $script:LlamaInstalled = $false
    try {
        $url = "$LlamaReleaseBase/$LlamaVersion/$asset"
        $zip = Join-Path $staging $asset
        Write-Host "[setup_env] downloading $asset"
        try {
            # Resumable: the hash check below is what makes that safe.
            Get-RemoteFile -Uri $url -OutFile $zip -Resume
        } catch {
            Write-Warning "[setup_env] could not download $asset from $url - $($_.Exception.Message)"
            return
        }
        # Checked before anything is unpacked, so a wrong file is never executed. This is the
        # largest binary the install fetches and was the only one arriving unverified.
        if ($release.Sha256) {
            $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLower()
            if ($actual -ne $release.Sha256) {
                Write-Warning "[setup_env] $asset does not match the hash GitHub publishes for it"
                Write-Warning "[setup_env]   expected $($release.Sha256)"
                Write-Warning "[setup_env]   got      $actual"
                Write-Warning "[setup_env] refusing to install it. Re-run to download it again."
                return
            }
            Write-Host "[setup_env] $asset matches its published SHA-256"
        } else {
            # Reached when the release could not be read, which is also when the asset name was
            # guessed. Said plainly rather than passed over: the binary is going in unchecked.
            Write-Warning "[setup_env] no published hash for $asset; installing it unverified"
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
        # Checked in staging, before anything is replaced: an asset without the files this install
        # is for must not wipe out the previous install that still works.
        foreach ($name in $Required) {
            if (-not (Test-Path (Join-Path $src $name) -PathType Leaf)) {
                Write-Warning "[setup_env] $asset unpacked but has no $name"
                return
            }
        }
        if (Test-Path $Destination) {
            Remove-Item -Recurse -Force $Destination -ErrorAction Stop
        }
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
        try {
            Copy-Item -Path (Join-Path $src "*") -Destination $Destination -Recurse -Force
        } catch {
            Write-Warning "[setup_env] could not place the binaries in $Destination - $($_.Exception.Message)"
            return
        }
    } finally {
        Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
    }
    # Do not report success on the strength of "no exception": check that the files the service
    # will actually run are there.
    foreach ($name in $Required) {
        if (-not (Test-Path (Join-Path $Destination $name) -PathType Leaf)) {
            Write-Warning "[setup_env] $asset unpacked but $name is not in $Destination"
            return
        }
    }
    $script:LlamaInstalled = $true
    Write-Host "[setup_env] llama $LlamaVersion ($script:LlamaBackendResolved) installed: $Destination"
}

# The release binaries import MSVCP140 / VCRUNTIME140(_1) from the VC++ redistributable. That is
# already a prerequisite of this project - torch_cpu.dll imports the same DLLs and the wheels do
# not ship them - so the usual case needs nothing done. The gap is a uv-managed Python: it is a
# portable build that runs no prerequisite installer, so a machine that never had Visual Studio
# or the redistributable has none of them.
#
# Copying the redistributable next to the executables (app-local deployment) removes the
# dependency without administrator rights. Verified that the app-local copy wins the loader
# search, by corrupting it while system32's copy stayed intact and watching the process fail
# anyway.
#
# These are what a source directory must contain to be accepted as a CRT; the copy itself
# takes everything in it. msvcp140_atomic_wait is here because torch_python.dll imports it
# and nothing else in this project does -- it was missing from the first version of this
# list, and the symptom was torch failing to import with "or one of its dependencies".
$LlamaCrtDlls = @("msvcp140.dll", "msvcp140_atomic_wait.dll",
                  "vcruntime140.dll", "vcruntime140_1.dll")

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

function Copy-CrtInto {
    param([string]$Source, [string]$Target)
    # Checked before anything is copied. Copying as they are found leaves a partial CRT behind
    # when one is absent, and an app-local DLL shadows the system one -- so a half-applied set
    # can break a binary whose CRT was fine, and keep it broken on every later run.
    foreach ($dll in $LlamaCrtDlls) {
        if (-not (Test-Path (Join-Path $Source $dll))) { return $false }
    }
    # Everything in the directory, not just the names checked above. The redistributable is
    # 2 MB and choosing names is how this first shipped broken: llama's binaries import
    # msvcp140 / vcruntime140 / vcruntime140_1, so those were the three copied, and
    # torch_python.dll also imports MSVCP140_ATOMIC_WAIT.dll from the same package. What the
    # user saw was "Error loading torch_python.dll or one of its dependencies".
    New-Item -ItemType Directory -Path $Target -Force | Out-Null
    Copy-Item (Join-Path $Source "*.dll") $Target -Force
    return $true
}

function Copy-LlamaCrtFromPayload {
    # TRUSTA_VCRUNTIME_DIR is the copy the installer carries, put there at build time from the
    # build machine's Visual Studio redistributable. First choice because it is the only source
    # that is certain to be present on a user's machine and is versioned with the release.
    #
    # Unset when this script is run from a source checkout, which is correct: a machine that
    # builds this has the runtime installed.
    if (-not $env:TRUSTA_VCRUNTIME_DIR) { return $false }
    if (-not (Copy-CrtInto -Source $env:TRUSTA_VCRUNTIME_DIR -Target $LlamaBinDir)) {
        Write-Warning "[setup_env] TRUSTA_VCRUNTIME_DIR=$env:TRUSTA_VCRUNTIME_DIR does not hold all of $($LlamaCrtDlls -join ', ')"
        return $false
    }
    Write-Host "[setup_env] copied the VC++ CRT app-local from the payload"
    return $true
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
            if (-not (Copy-CrtInto -Source $crt.FullName -Target $LlamaBinDir)) { continue }
            Write-Host "[setup_env] copied the VC++ CRT app-local from $($crt.FullName)"
            return $true
        }
    }
    return $false
}

# A third source used to sit here: TRUSTA_VCREDIST_URL, a zip of the DLLs fetched from wherever
# that variable pointed. It was written for a packaged installer to carry the runtime, and then
# the installer carried it in the payload instead -- so nothing ever set the variable and the
# function returned $false on every run of its life. Removed rather than left: it was the only
# path here that would download and load code from an arbitrary URL, which is a poor thing to
# leave lying unused in a script that provisions machines.

# torch needs the same three DLLs as llama: c10.dll, torch_cpu.dll and torch_cuda.dll all
# import MSVCP140 / VCRUNTIME140 / VCRUNTIME140_1, and the wheels do not ship them. That used
# to be covered by making Microsoft's redistributable a prerequisite of the installer, which
# is the administrator prompt this whole arrangement exists to remove -- so torch has to be
# given an app-local copy the same way llama is.
#
# torch/lib rather than beside python.exe: torch adds that directory with os.add_dll_directory,
# so it is searched for the dependencies of everything torch loads, and it is inside the
# virtual environment this script owns. uv's managed Python already carries vcruntime140 and
# vcruntime140_1 next to python.exe; msvcp140 is the one nothing supplies.
function Test-TorchImports {
    param([string]$Python)
    # Through ProcessStartInfo, like Test-LlamaStarts, and for the same reason: with
    # $ErrorActionPreference = "Stop", redirecting a native command's stderr in Windows
    # PowerShell wraps each line in a NativeCommandError and terminates the script. A failed
    # import is the expected answer here, not an error to stop on.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Python
    $psi.Arguments = "-c `"import torch`""
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
    return $proc.ExitCode -eq 0
}

function Resolve-TorchCrt {
    $python = Get-VenvPython
    if (-not (Test-Path $python)) { return }
    if (Test-TorchImports -Python $python) { return }

    $torchLib = Join-Path (Split-Path (Split-Path $python)) "Lib\site-packages\torch\lib"
    if (-not (Test-Path $torchLib)) { return }
    Write-Host "[setup_env] torch cannot be imported; placing the VC++ CRT beside it"
    foreach ($source in @($env:TRUSTA_VCRUNTIME_DIR)) {
        if ($source -and (Copy-CrtInto -Source $source -Target $torchLib)) {
            if (Test-TorchImports -Python $python) {
                Write-Host "[setup_env] torch imports with the app-local CRT"
                return
            }
        }
    }
    # Not fatal here. The import is retried by everything that follows, and the real diagnosis
    # belongs to whatever fails next rather than to a guess made at this point.
    Write-Warning "[setup_env] torch still cannot be imported after placing the CRT"
}

# Cheapest step first: most machines already resolve the CRT and need nothing done at all.
function Resolve-LlamaCrt {
    param([string]$Binary)
    if (Test-LlamaStarts -Binary $Binary) { return $true }
    $name = [System.IO.Path]::GetFileName($Binary)
    Write-Host "[setup_env] $name cannot start; resolving the VC++ CRT"
    foreach ($step in @({ Copy-LlamaCrtFromPayload }, { Copy-LlamaCrtFromVisualStudio })) {
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

# The value a setting has for the service: the process environment first, then .env (or
# .env.example when .env is absent, matching service/settings.py). $null when it is unset.
function Get-EnvSetting {
    param([string]$Key)
    $fromEnv = [System.Environment]::GetEnvironmentVariable($Key)
    if ($fromEnv) { return $fromEnv }
    foreach ($name in @(".env", ".env.example")) {
        $envFile = Join-Path $ProjectRoot $name
        if (-not (Test-Path $envFile)) { continue }
        $line = Select-String -Path $envFile -Pattern "^\s*$Key\s*=\s*(.+)$" |
            Select-Object -Last 1
        if ($line) {
            $value = $line.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
            if ($value) { return $value }
        }
        break
    }
    return $null
}

# Where the llama binary is expected. An explicit LLAMA_SERVER_BINARY wins; otherwise the
# directory setup_env unpacks the release into. .env is consulted too, since that is
# where the service itself reads the override from — including service/settings.py's
# fallback to .env.example when .env is absent, so setup and the service agree.
function Get-LlamaBinaryPath {
    $configured = Get-EnvSetting -Key "LLAMA_SERVER_BINARY"
    if ($configured) { return $configured }
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

# The tools the backend would fail to find next to the given server. Mirrors the backend's
# resolvers (ConversionManager._resolve_quantize_binary and GGUFEstimator._resolve_fit_binary):
# an explicit setting, the convert tree, the server's own directory, or a unified `llama` that
# runs both as subcommands.
function Get-MissingLlamaTools {
    param([string]$Server)
    $missing = @()
    if ([System.IO.Path]::GetFileName($Server) -ieq "llama.exe") { return ,$missing }
    $serverDir = Split-Path -Parent $Server
    foreach ($tool in $LlamaTools.Keys) {
        $configured = Get-EnvSetting -Key $LlamaTools[$tool]
        if ($configured -and (Test-Path $configured -PathType Leaf)) { continue }
        $dirs = @(
            (Join-Path $LlamaConvertDir "build\bin\Release"),
            (Join-Path $LlamaConvertDir "build\bin"),
            (Join-Path $LlamaConvertDir "bin"),
            $LlamaConvertDir,
            $serverDir
        )
        $found = $false
        foreach ($dir in $dirs) {
            if ($dir -and (Test-Path (Join-Path $dir $tool) -PathType Leaf)) { $found = $true; break }
        }
        if (-not $found) { $missing += $tool }
    }
    return ,$missing
}

# Whether a binary lives in the directory setup_env manages, so it may be reinstalled in place.
function Test-InLlamaBinDir {
    param([string]$Binary)
    if (-not (Test-Path $LlamaBinDir)) { return $false }
    $dir = (Resolve-Path (Split-Path -Parent $Binary) -ErrorAction SilentlyContinue).Path
    $ours = (Resolve-Path $LlamaBinDir).Path
    return ($dir -and ($dir.TrimEnd('\') -ieq $ours.TrimEnd('\')))
}

# Whether the binary is one the user chose through LLAMA_SERVER_BINARY (environment or .env)
# and setup_env must not replace. Anything else - our own install, or a prebuilt that merely
# happened to be on PATH - is setup_env's to replace.
function Test-UserLlamaBinary {
    param([string]$Binary)
    $configured = Get-EnvSetting -Key "LLAMA_SERVER_BINARY"
    if (-not $configured -or -not (Test-Path $configured -PathType Leaf)) { return $false }
    if ($configured -ne $Binary) { return $false }
    return (-not (Test-InLlamaBinDir -Binary $Binary))
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

# Unpack only $ConvertPaths from a llama.cpp source archive, keeping the archive's
# llama.cpp-<tag>\ top directory.
#
# Not Expand-Archive on the whole thing: the archive also carries a web UI whose deepest
# path is 178 characters, and under %TEMP%\llama-src-<guid>\ that passes Windows
# PowerShell's 260-character limit as soon as the user name is longer than about ten
# characters. Expand-Archive then fails on the very first file with "Cannot find path"
# -- setup stopped for any such user, and for the Administrator account the release
# build host runs as. What the conversion code needs sits at the top of the tree.
function Expand-ConvertPaths {
    param([Parameter(Mandatory)] [string] $Zip, [Parameter(Mandatory)] [string] $Destination)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        $count = 0
        foreach ($entry in $archive.Entries) {
            # <top>/<relative path>; directory entries end in / and carry no data.
            $parts = $entry.FullName -split '/', 2
            if ($parts.Count -lt 2 -or -not $parts[1] -or $entry.FullName.EndsWith('/')) { continue }
            $relative = $parts[1]
            $wanted = $ConvertPaths | Where-Object { $relative -eq $_ -or $relative.StartsWith("$_/") }
            if (-not $wanted) { continue }
            $target = Join-Path $Destination (Join-Path $parts[0] ($relative -replace '/', '\'))
            New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
            $count++
        }
    } finally {
        $archive.Dispose()
    }
    if ($count -eq 0) { throw "none of $($ConvertPaths -join ', ') is in the archive" }
}

function Get-LlamaConvertTooling {
    $stamp = Join-Path $LlamaConvertDir $LlamaSrcStamp
    if ((Test-Path $stamp) -and
        ((Get-Content $stamp -Raw -ErrorAction SilentlyContinue).Trim() -eq $LlamaVersion) -and
        (Test-Path (Join-Path $LlamaConvertDir "convert_hf_to_gguf.py"))) {
        Write-Host "[setup_env] convert tooling for $LlamaVersion is already installed"
        return
    }

    $url = "$LlamaCppUrl/archive/refs/tags/$LlamaVersion.zip"
    $work = Join-Path ([System.IO.Path]::GetTempPath()) ("llama-src-" + [guid]::NewGuid().ToString("N"))
    $zip = "$work.zip"
    Write-Host "[setup_env] downloading the convert tooling ($LlamaVersion)"
    try {
        # Not resumable: nothing publishes a hash for a tag archive, so a resumed file
        # that came back wrong would only be caught by Expand-Archive failing, and a
        # corrupt archive that still unpacks would be worse. It is a tenth the size of
        # the runtime, so starting over costs little.
        Get-RemoteFile -Uri $url -OutFile $zip
        Expand-ConvertPaths -Zip $zip -Destination $work
    } catch {
        Remove-Item -Recurse -Force $work, $zip -ErrorAction SilentlyContinue
        Die "could not download the llama.cpp $LlamaVersion sources from $url ($($_.Exception.Message)). Check the network, or pin a version that exists with -LlamaVersion."
    }

    # The archive unpacks into a single llama.cpp-<tag>\ directory.
    $root = Get-ChildItem -Directory $work | Select-Object -First 1
    if (-not $root) {
        Remove-Item -Recurse -Force $work, $zip -ErrorAction SilentlyContinue
        Die "the llama.cpp archive from $url unpacked to nothing usable."
    }

    Write-Host "[setup_env] installing the convert tooling: $($ConvertPaths -join ', ')"
    New-Item -ItemType Directory -Force $LlamaConvertDir | Out-Null
    foreach ($path in $ConvertPaths) {
        $source = Join-Path $root.FullName $path
        if (-not (Test-Path $source)) {
            Remove-Item -Recurse -Force $work, $zip -ErrorAction SilentlyContinue
            Die "the llama.cpp $LlamaVersion sources do not contain $path, which the GGUF conversion code calls. The pinned version may have moved or renamed it."
        }
        $target = Join-Path $LlamaConvertDir $path
        if (Test-Path $target) { Remove-Item -Recurse -Force $target }
        Copy-Item -Recurse -Force $source $target
    }
    Set-Content -Path $stamp -Value $LlamaVersion -Encoding utf8
    Remove-Item -Recurse -Force $work, $zip -ErrorAction SilentlyContinue
    Write-Host "[setup_env] convert tooling ready (pure Python, no build step)"
}

# Auto-detect: CUDA when nvidia-smi exists, XPU when there is an Intel display adapter,
# otherwise the CPU build.
#
# The middle branch used to be the fallback -- anything that was not NVIDIA was assumed
# to be Intel. Most Windows desktops do have an Intel iGPU, so it was usually right; on
# the ones that do not, it downloaded 2.7 GB of XPU torch that could never be used. A
# virtual machine is the clearest case: its only display adapter is the hypervisor's.
if (-not $Accel) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $Accel = "cuda"
    } else {
        $intel = $null
        try {
            $intel = Get-CimInstance Win32_VideoController -ErrorAction Stop |
                Where-Object { $_.Name -match 'Intel' } | Select-Object -First 1
        } catch { $intel = $null }
        if ($intel) {
            Write-Host "[setup_env] Intel display adapter: $($intel.Name)"
            $Accel = "xpu"
        } else {
            Die @"
no supported GPU was found on this machine.

  Looked for: an NVIDIA GPU (nvidia-smi), and an Intel display adapter.

  This build installs PyTorch for NVIDIA or Intel; there is no CPU-only build, so
  there is nothing to install here. If you do have one of those and it was simply not
  detected, say so and setup will use it:

      .\setup_env.ps1 -Accel cuda
      .\setup_env.ps1 -Accel xpu
"@
        }
    }
}
Write-Host "[setup_env] accelerator=$Accel"

# XPU: the torch xpu SYCL runtime ships with pip (uv sync --extra xpu), so the oneAPI Base
# Toolkit is not needed. What does matter is that the runtime version baked into the wheel
# matches the installed Intel GPU driver; when it does not, the first kernel either raises or
# hangs. Test-XpuUsable below proves it instead of assuming it.

# Here rather than at the first `uv` call, for the same reason as everything else in
# this block: what cannot work should be reported before the long part, not during it.
Install-UvIfMissing

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
# $ErrorActionPreference does not cover native commands in Windows PowerShell, so
# without this the script walks straight past a failed sync and reports success. It
# did: a first run under the installer's redirection-trust mitigation could not reach
# uv's managed Python, uv exited non-zero, and setup_env went on to fetch llama and
# print "Environment setup complete" over an environment with no packages in it. The
# launcher believed it, and the failure only surfaced much later as a missing
# interpreter. setup_env.sh gets this from `set -euo pipefail`; this is its equivalent.
if ($LASTEXITCODE -ne 0) { Die "uv sync --extra $Accel failed (exit $LASTEXITCODE) - see the error above" }

Resolve-TorchCrt

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
$LlamaToolsStatus = "n/a"
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
        Get-LlamaConvertTooling
        $convertStatus = "fetched"
    }

    # Binary
    $existingLlama = $null
    if ($LlamaMode -ne "force") { $existingLlama = Resolve-LlamaBinary }
    if ($existingLlama) {
        # "Present" means the whole set the backend needs, not only the server: a server brought
        # from elsewhere would otherwise be left without the tools for good, since every later
        # run stops at this check.
        $missingTools = Get-MissingLlamaTools -Server $existingLlama
        $missingText = $missingTools -join " "
        if ($missingTools.Count -eq 0) {
            Write-Host "[setup_env] llama binary already present at $existingLlama — leaving it alone (-Llama force to reinstall)"
            $binStatus = "already present"
        } elseif (-not (Test-UserLlamaBinary -Binary $existingLlama)) {
            # An older install of ours, or a prebuilt found on PATH: replace it with the full,
            # pinned release in $LlamaBinDir, so the tools share the server's version and DLLs.
            # A PATH binary is not deleted (it is not ours), only no longer used. The install
            # leaves the existing one alone until the new asset checks out.
            Write-Host "[setup_env] llama binary at $existingLlama lacks $missingText; installing the full release $LlamaVersion into $LlamaBinDir"
            Install-LlamaRelease -Required (@("llama-server.exe") + $missingTools)
            if ($LlamaInstalled) {
                $binStatus = "release $LlamaVersion ($LlamaBackendResolved), replacing $existingLlama (lacked $missingText)"
            } else {
                # Not fatal: the server still runs, only conversion / fit-params are unavailable.
                Write-Warning "[setup_env] could not install the full release; keeping $existingLlama"
                $binStatus = "already present (could not install $missingText)"
            }
        } else {
            # Set explicitly through LLAMA_SERVER_BINARY: never replaced. Unpack the release into
            # its own directory and point the backend at the tools there, rather than writing
            # beside someone else's binary.
            Write-Host "[setup_env] LLAMA_SERVER_BINARY=$existingLlama lacks $missingText; installing them into $LlamaToolsDir"
            Install-LlamaRelease -Destination $LlamaToolsDir -Required $missingTools
            if ($LlamaInstalled) {
                foreach ($tool in $missingTools) {
                    Save-EnvSetting -Key $LlamaTools[$tool] -Resolved (Join-Path $LlamaToolsDir $tool)
                }
                $binStatus = "already present; installed $missingText into $LlamaToolsDir"
            } else {
                Write-Warning "[setup_env] could not install $missingText; GGUF quantize / fit-params stay unavailable"
                $binStatus = "already present (could not install $missingText)"
            }
        }
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
            Write-Host "  The VC++ runtime is a separate Microsoft download, not part of Windows."
            Write-Host "  An installed build carries it and needs nothing; this is a source checkout,"
            Write-Host "  so it comes from Visual Studio or Build Tools with the C++ workload:"
            Write-Host "    winget install Microsoft.VisualStudio.2022.BuildTools"
            Write-Host "  Or point TRUSTA_VCRUNTIME_DIR at a directory holding the CRT DLLs, which"
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
    # Said in the summary, not only in a warning mid-install that scrolls away.
    $missingTools = Get-MissingLlamaTools -Server $llamaBin
    $LlamaToolsStatus = if ($missingTools.Count -gt 0) {
        "MISSING $($missingTools -join ' ') (GGUF quantize / fit-params unavailable)"
    } else {
        "ok ($(@($LlamaTools.Keys) -join ' '))"
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Environment setup complete"
Write-Host "  Accelerator : $Accel"
Write-Host "  XPU check   : $XpuChecked"
Write-Host "  Service Dir : $ServiceDir"
Write-Host "  llama       : $LlamaStatus"
Write-Host "  llama tools : $LlamaToolsStatus"
Write-Host "  llama check : $script:LlamaVerify"
Write-Host "=========================================="
