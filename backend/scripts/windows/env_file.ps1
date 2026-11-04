# Shared by setup_env.ps1 and install_strata.ps1; dot-sourced, never run. Needs $ProjectRoot set.
# $LogTag is the prefix of the lines it prints ([setup_env] unless the caller sets another).
if (-not $script:LogTag) { $script:LogTag = "[setup_env]" }

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
        Write-Host "$script:LogTag no .env, so $Key was not recorded (resolved: $Resolved)"
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
        $out.Add("# Recorded by setup: a path it resolved, so the service does not")
        $out.Add("# have to re-resolve it from a possibly different PATH.")
        $out.Add($setting)
    }

    [System.IO.File]::WriteAllText(
        $envFile, ($out -join $eol), (New-Object System.Text.UTF8Encoding($false)))
    if ($previous -eq $Resolved) {
        Write-Host "$script:LogTag .env already records $Key=$Resolved"
    } elseif ($previous) {
        Write-Host "$script:LogTag .env $Key updated to $Resolved (was $previous)"
    } else {
        Write-Host "$script:LogTag recorded $Key=$Resolved in .env"
    }
    if ($dropped -gt 0) {
        Write-Host "$script:LogTag removed $dropped duplicate $Key line(s) from .env"
    }
}
