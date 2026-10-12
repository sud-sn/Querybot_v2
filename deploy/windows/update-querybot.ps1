<#
.SYNOPSIS
Brings this QueryBot to the latest commit of a branch, installs what it needs and starts it again.

.DESCRIPTION
In an Administrator PowerShell, from the QueryBot folder:
    powershell -ExecutionPolicy Bypass -File deploy\windows\update-querybot.ps1                 # branch core-v2
    powershell -ExecutionPolicy Bypass -File deploy\windows\update-querybot.ps1 -Branch main

The local branch is set to the branch on GitHub every time, so the server never stays on an older copy of it by
mistake. QueryBot is stopped first, the SQLite database copied (data\querybot.db.bak-<time>), then started again
and checked. Files changed on this server are kept unless the update changes them too: git then stops, nothing
changes, and QueryBot is started again as it was. The line to go back to where the server was is printed at the end.
#>
param(
    [string]$Branch = "core-v2",
    [string]$TaskName = "QueryBot",
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Service = Join-Path $PSScriptRoot "querybot-service.ps1"
Set-Location $RepoRoot

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "git $($args -join ' ') failed (exit code $LASTEXITCODE)." }
}

# A hashtable: an array of "-Name", value strings would be bound by position, "-TaskName" as the port.
$ServiceArgs = @{ TaskName = $TaskName }
if ($Port -gt 0) { $ServiceArgs.Port = $Port }
$HasTask = [bool](Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)

$PreviousBranch = (& git rev-parse --abbrev-ref HEAD).Trim()
$Previous = (& git rev-parse --short HEAD).Trim()
Write-Host "== Now at: $PreviousBranch $(& git log --oneline -1)"
$Changed = & git status --porcelain --untracked-files=no
if ($Changed) {
    Write-Host "== Files changed on this server (kept unless the update changes them too):"
    $Changed | ForEach-Object { Write-Host "   $_" }
}

if ($HasTask) {
    Write-Host "== Stopping QueryBot"
    & $Service -Action Stop @ServiceArgs
} else {
    Write-Warning "The $TaskName task is not installed: stop QueryBot in its console now if it is running, then press Enter."
    [void](Read-Host)
}

try {
    # SQLite's own backup: recent writes may still be in querybot.db-wal, which a file copy would miss.
    $Backup = & $Python (Join-Path $RepoRoot "deploy\backup_db.py")
    if ($LASTEXITCODE -ne 0) { throw "Could not copy the database; nothing was changed." }
    if ($Backup) { Write-Host "== Database copied to $Backup" }

    Write-Host "== Fetching $Branch"
    Invoke-Git fetch origin $Branch
    Invoke-Git checkout -B $Branch "origin/$Branch"
    Write-Host "== Moved to: $(& git log --oneline -1)"

    Write-Host "== Installing what this commit needs"
    & $Python -m pip install --quiet -r requirements-windows.txt
    if ($LASTEXITCODE -ne 0) { throw "pip could not install requirements-windows.txt (exit code $LASTEXITCODE)." }
    & $Python -m playwright install chromium
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The PDF browser could not be installed: scheduled emails will go without pictures and PDF."
    }
} finally {
    if ($HasTask) {
        Write-Host "== Starting QueryBot"
        & $Service -Action Start @ServiceArgs
    } else {
        Write-Host "== Start QueryBot as before (or install the background task: querybot-service.ps1 -Action Install)."
    }
}

Write-Host ""
Write-Host "Done. If the release notes say so, run Learn again on each workspace (your decisions are kept)."
Write-Host "To go back to where this server was:"
Write-Host "  git checkout -B $PreviousBranch $Previous   (then -Action Restart)"
