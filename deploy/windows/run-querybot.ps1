<#
.SYNOPSIS
Runs QueryBot with no window, for the QueryBot background task (querybot-service.ps1 -Action Install).

.DESCRIPTION
Starts QueryBot on one worker with the settings in .env, writes what it logs to logs\querybot-<start time>.log
(and anything it prints to the matching .out.log), and starts it again ten seconds after it stops for any
reason. It is stopped by stopping the task (querybot-service.ps1 -Action Stop), never from a console, so a click
or Ctrl+C in a window can no longer pause or end it. Log files older than -KeepLogDays are removed.
#>
param(
    [int]$Port = 8000,
    [string]$BindAddress = "0.0.0.0",
    [int]$KeepLogDays = 14
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$EnvFile = Join-Path $RepoRoot ".env"
$Logs = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
Set-Location $RepoRoot

# Logs in UTF-8 whatever the system code page (an em dash in a log line otherwise fails to write), and at once.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$UvicornArgs = @("-m", "uvicorn", "main:app", "--host", $BindAddress, "--port", "$Port", "--workers", "1",
                 "--log-level", "info")
if (Test-Path $EnvFile) {
    $UvicornArgs += @("--env-file", "`"$EnvFile`"")
}

while ($true) {
    $Cutoff = (Get-Date).AddDays(-$KeepLogDays)
    Get-ChildItem -Path $Logs -Filter "querybot-*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $Cutoff } |
        Remove-Item -Force -ErrorAction SilentlyContinue

    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $ErrLog = Join-Path $Logs "querybot-$Stamp.log"
    $OutLog = Join-Path $Logs "querybot-$Stamp.out.log"
    $Process = Start-Process -FilePath $Python -ArgumentList $UvicornArgs -WorkingDirectory $RepoRoot `
        -NoNewWindow -PassThru -Wait -RedirectStandardError $ErrLog -RedirectStandardOutput $OutLog
    $Line = "{0} QueryBot stopped (exit code {1}); starting it again in 10 seconds." -f `
        (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Process.ExitCode
    Add-Content -Path $ErrLog -Value $Line
    Start-Sleep -Seconds 10
}
