# Windows wrapper around install.py -- all the logic lives there so the two
# platforms cannot drift. Any arguments are passed straight through.
#
#   .\install.ps1
#   .\install.ps1 --comfy-home C:/Users/you/ComfyUI_windows_portable
#   .\install.ps1 --dry-run
#
# If PowerShell blocks the script:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Test-Py($exe) {
    # A bare `python` on Windows is usually the Microsoft Store stub, which
    # exits 9009 with "Python was not found" instead of running anything.
    try {
        & $exe -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

$py = $null
$candidates = @()
if ($env:PYTHON) { $candidates += $env:PYTHON }
# Hermes ships its own venv; prefer it, it is the one the skills will run under.
$local = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME 'AppData\Local' }
$hermesHome = if ($env:HERMES_HOME) { $env:HERMES_HOME } else { Join-Path $local 'hermes' }
$candidates += (Join-Path $hermesHome 'hermes-agent\venv\Scripts\python.exe')
$candidates += 'py'
$candidates += 'python3'
$candidates += 'python'

foreach ($c in $candidates) {
    if (-not $c) { continue }
    $resolved = $c
    if ($c -notmatch '[\\/]') {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $resolved = $cmd.Source
    } elseif (-not (Test-Path $c)) { continue }
    if (Test-Py $resolved) { $py = $resolved; break }
}

if (-not $py) {
    Write-Error "Need Python 3.9 or newer. Install it from python.org (not the Microsoft Store stub), or set `$env:PYTHON."
    exit 1
}

Write-Host "using $py"
& $py install.py @args
exit $LASTEXITCODE
