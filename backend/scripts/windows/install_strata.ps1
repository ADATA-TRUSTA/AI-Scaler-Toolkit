# Install the Strata engine (engine="strata") from a strata-sleep package and record it in .env.
#
# The Windows twin of scripts/linux/install_strata.sh. The backend always starts Strata with
# --sleep-mode, which only the deep-sleep build has; that build comes as a package
# (TRUSTA-SW-FAE/strata-sleep releases: the engine already compiled, plus Strata's Python
# side), so nothing is cloned or compiled here and no git credentials are needed. This unpacks
# the package, checks it against its own manifest, makes its venv, runs Strata's setup (which
# installs the model and writes its run config) and refuses an engine that has no deep sleep.
# setup_env.ps1 runs it when TRUSTA_SETUP_STRATA=1; run it alone to install or update Strata later.
#
# Usage:
#   .\install_strata.ps1
#   $env:TRUSTA_STRATA_SETUP_ARGS = "--model IQ3_XXS --gguf-dir D:\models\IQ3_XXS"; .\install_strata.ps1
#
#   TRUSTA_STRATA_PACKAGE     the package: its .zip or the folder it unpacks to (default: the
#                             newest vendor\strata-sleep-*-windows-x86_64.zip in this repository,
#                             where the installer's payload carries it)
#   TRUSTA_STRATA_DIR         where it is installed (default: strata\ beside this repository)
#   TRUSTA_STRATA_SETUP_ARGS  passed to Strata's setup as they are (default: --model IQ3_XXS,
#                             which downloads about 76 GB unless --gguf-dir names the files)
#   TRUSTA_STRATA_GGUF_DIR    the model's GGUF files already on disk: setup's --gguf-dir, as one
#                             argument (a path may hold spaces, which _SETUP_ARGS splits on)
#   TRUSTA_STRATA_DATA_DIR    where Strata keeps what it derives from a model (the expert pack,
#                             the draft layer): setup's --data-dir, also as one argument. Recorded
#                             with --set-default-data-dir, which moves nothing that already exists
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"

$script:LogTag = "[install_strata]"
function Die {
    param([string]$Message)
    Write-Host "$script:LogTag $Message"
    exit 1
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. (Join-Path $PSScriptRoot "env_file.ps1")

$package = $env:TRUSTA_STRATA_PACKAGE
$strataDir = if ($env:TRUSTA_STRATA_DIR) { $env:TRUSTA_STRATA_DIR } else { Join-Path (Split-Path -Parent $ProjectRoot) "strata" }
$strataDir = [System.IO.Path]::GetFullPath($strataDir)
$setupArgs = if ($env:TRUSTA_STRATA_SETUP_ARGS) { $env:TRUSTA_STRATA_SETUP_ARGS } else { "--model IQ3_XXS" }

if (-not $package) {
    $package = Get-ChildItem -Path (Join-Path $ProjectRoot "vendor") -Filter "strata-sleep-*-windows-x86_64.zip" -File -ErrorAction SilentlyContinue |
        Sort-Object { $_.Name -replace '^strata-sleep-(.+)-windows-x86_64\.zip$', '$1' } | Select-Object -Last 1 -ExpandProperty FullName
    if (-not $package) { Die "no strata-sleep package in $ProjectRoot\vendor; set TRUSTA_STRATA_PACKAGE to its .zip" }
}
if (-not (Test-Path -LiteralPath $package)) { Die "TRUSTA_STRATA_PACKAGE: $package does not exist" }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Die "uv is needed for Strata's venv (setup_env.ps1 installs it)" }

# Strata's tree goes about 130 characters deep below its folder (third_party\llama.cpp, the
# venv), and Windows stops at 260 unless long paths are turned on for the whole machine.
if ($strataDir.Length -gt 110) {
    Die "$strataDir is $($strataDir.Length) characters long; Strata needs a folder of at most 110 (TRUSTA_STRATA_DIR)"
}
# An install from before packages was a git checkout of the source; that folder holds
# someone's tree, not ours to overwrite.
if (Test-Path -LiteralPath (Join-Path $strataDir ".git")) {
    Die "$strataDir is a git checkout (the old source install); move it away or set TRUSTA_STRATA_DIR"
}
if ((Test-Path -LiteralPath $strataDir) -and -not (Test-Path -LiteralPath (Join-Path $strataDir "SLEEP.json")) -and
    (Get-ChildItem -LiteralPath $strataDir -Force | Select-Object -First 1)) {
    Die "$strataDir exists and is not a strata-sleep install; move it away or set TRUSTA_STRATA_DIR"
}

# Unpack beside the target and check every file against the package's own manifest before
# anything is copied in: a truncated or altered package stops here, with the install untouched.
$parent = Split-Path -Parent $strataDir
# A drive root is a parent that already exists and that New-Item still refuses to make:
# New-Item -Force -Path "D:\" throws "invalid path format", and the install stopped there with
# nothing unpacked. Make only the folders that are missing.
if (-not $parent) { Die "TRUSTA_STRATA_DIR cannot be a drive root: $strataDir" }
if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$staging = Join-Path $parent (".strata-unpack-" + [System.IO.Path]::GetRandomFileName().Replace(".", ""))
try {
    if (Test-Path -LiteralPath $package -PathType Container) {
        $source = (Resolve-Path -LiteralPath $package).Path
        Copy-Item -LiteralPath $source -Destination $staging -Recurse
        $root = $staging
    } else {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        try { [System.IO.Compression.ZipFile]::ExtractToDirectory($package, $staging) }
        catch { Die "could not unpack ${package}: $($_.Exception.Message)" }
        # The zip holds one folder named after the package.
        $top = @(Get-ChildItem -LiteralPath $staging)
        $root = if ($top.Count -eq 1 -and $top[0].PSIsContainer) { $top[0].FullName } else { $staging }
    }
    $manifest = Join-Path $root "MANIFEST.sha256"
    if (-not (Test-Path -LiteralPath (Join-Path $root "SLEEP.json")) -or -not (Test-Path -LiteralPath $manifest)) {
        Die "$package is not a strata-sleep package (no SLEEP.json / MANIFEST.sha256)"
    }
    foreach ($line in [System.IO.File]::ReadAllLines($manifest)) {
        if ($line -notmatch '^([0-9a-f]{64})  (.+)$') { continue }
        $file = Join-Path $root ($Matches[2] -replace '/', '\')
        $actual = if (Test-Path -LiteralPath $file) { (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLower() } else { "" }
        if ($actual -ne $Matches[1]) {
            Die "$package does not match its MANIFEST.sha256 ($($Matches[2])); download it again. Nothing was installed."
        }
    }
    $version = (Get-Content -LiteralPath (Join-Path $root "VERSION") -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $version) { $version = "unknown" }

    # Copied over an earlier install rather than replacing it: its venv and the run configs
    # Strata's setup wrote (strata-<model>.json) stay, and setup reuses them.
    New-Item -ItemType Directory -Force -Path $strataDir | Out-Null
    Get-ChildItem -LiteralPath $root -Force | Copy-Item -Destination $strataDir -Recurse -Force
} finally {
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
}
Write-Host "$script:LogTag strata-sleep $version unpacked into $strataDir"

$python = Join-Path $strataDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    # --seed: Strata's setup installs its packages with pip, which a bare uv venv lacks.
    uv venv -q --seed -p 3.12 (Join-Path $strataDir ".venv")
    if ($LASTEXITCODE -ne 0) { Die "could not make Strata's venv" }
}

# The package's engine\ is the ready-made engine setup takes as installed. --no-start: the
# backend starts it, not setup. --vision no: the package has no image encoder.
$extra = @($setupArgs -split '\s+' | Where-Object { $_ })
if ($env:TRUSTA_STRATA_GGUF_DIR) { $extra += @("--gguf-dir", $env:TRUSTA_STRATA_GGUF_DIR) }
if ($env:TRUSTA_STRATA_DATA_DIR) { $extra += @("--data-dir", $env:TRUSTA_STRATA_DATA_DIR, "--set-default-data-dir") }
Write-Host "$script:LogTag running Strata's setup: --yes --no-start --vision no $($extra -join ' ')"
Push-Location $strataDir
try { & $python setup.py --yes --no-start --vision no @extra; $setupExit = $LASTEXITCODE }
finally { Pop-Location }
if ($setupExit -ne 0) { Die "Strata's setup failed (see above). Nothing was recorded in .env." }

$engine = Join-Path $strataDir "engine\strata.exe"
if (-not (Test-Path -LiteralPath $engine)) { Die "Strata's setup finished but left no engine at $engine" }
# The flag's name is in the binary of every build that parses it, and in no other: the
# ready-made upstream engine has none, and the backend would fail to start it.
$flag = [System.Text.Encoding]::ASCII.GetBytes("--sleep-mode")
$bytes = [System.IO.File]::ReadAllBytes($engine)
$text = [System.Text.Encoding]::GetEncoding(28591).GetString($bytes)
if ($text.IndexOf([System.Text.Encoding]::GetEncoding(28591).GetString($flag), [System.StringComparison]::Ordinal) -lt 0) {
    Die "$engine has no deep sleep (--sleep-mode): it is not a strata-sleep engine. Nothing was recorded in .env."
}

Save-EnvSetting -Key "STRATA_HOME" -Resolved $strataDir
Save-EnvSetting -Key "STRATA_PYTHON" -Resolved $python
Save-EnvSetting -Key "STRATA_ENGINE_BINARY" -Resolved $engine

Write-Host ""
Write-Host "$script:LogTag Strata is installed: $strataDir (strata-sleep $version)"
# Load with engine="strata" and model_path: the backend finds the run config whose --native
# is that GGUF file. Listed here so the paths to register are on screen.
Write-Host "$script:LogTag installed models (load with engine=""strata"" and model_path = the GGUF):"
foreach ($cfg in Get-ChildItem -Path $strataDir -Filter "strata-*.json" -File) {
    if ($cfg.Name -like "*.shared-settings.json") { continue }
    $gguf = ""
    try {
        $run = Get-Content -LiteralPath $cfg.FullName -Raw | ConvertFrom-Json
        $i = [Array]::IndexOf(@($run.args), "--native")
        if ($i -ge 0 -and $i -lt @($run.args).Count - 1) { $gguf = @($run.args)[$i + 1] }
    } catch { }
    if (-not $gguf) { $gguf = "(no --native)" }
    Write-Host "    $gguf  <- $($cfg.Name)"
}
exit 0
