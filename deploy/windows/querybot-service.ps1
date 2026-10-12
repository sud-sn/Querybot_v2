<#
.SYNOPSIS
Runs QueryBot on Windows as a background task that starts with the server and restarts itself, and manages it.

.DESCRIPTION
Install (once, in an Administrator PowerShell, signed in as the Windows user QueryBot has been running as):
    powershell -ExecutionPolicy Bypass -File deploy\windows\querybot-service.ps1 -Action Install -Port 8001

  - .env gets, only where it has no value yet, the four session and pseudonym secrets (random), the path of the
    key that encrypts stored passwords (this user's .querybot_key), and where this user's PDF browser and AI model
    files are. A value already in .env is never changed. Never change PII_PSEUDONYM_SECRET once it is set.
  - A QueryBot already running on that port from a console is stopped first.
  - A scheduled task "QueryBot" runs deploy\windows\run-querybot.ps1 with no window, as this user, at every start
    of the server whether or not anyone is signed in, and QueryBot starts again if it stops.

Then:
    ... -Action Status     is it running, on which port, what /health says
    ... -Action Logs       the newest log, followed live (Ctrl+C here stops the following, not QueryBot)
    ... -Action Restart    after changing .env
    ... -Action Stop | Start | Uninstall

Back up .env and the key file together: without the key, stored database passwords cannot be read.
#>
param(
    [ValidateSet("Install", "Uninstall", "Start", "Stop", "Restart", "Status", "Logs")]
    [string]$Action = "Status",
    [int]$Port = 0,
    [string]$TaskName = "QueryBot"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "run-querybot.ps1"
$EnvFile = Join-Path $RepoRoot ".env"
$Logs = Join-Path $RepoRoot "logs"

function Get-QbTask {
    Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}

function Resolve-QbPort {
    if ($Port -gt 0) { return $Port }
    $Task = Get-QbTask
    if ($Task -and ($Task.Actions[0].Arguments -match '-Port\s+(\d+)')) { return [int]$Matches[1] }
    return 8000
}

function Get-QbHealth([int]$OnPort) {
    try {
        return (Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$OnPort/health" -TimeoutSec 5).Content
    } catch {
        return $null
    }
}

function Get-QbProcesses([int]$OnPort) {
    # The venv's python.exe and the Python it starts both carry the uvicorn command line.
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'uvicorn' -and $_.CommandLine -match 'main:app' -and
                       $_.CommandLine -match "--port\s+$OnPort(\s|$)" }
}

function Stop-QbProcesses([int]$OnPort) {
    foreach ($P in @(Get-QbProcesses $OnPort)) {
        Stop-Process -Id $P.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Wait-QbHealth([int]$OnPort, [int]$Seconds = 150) {
    $Deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $Deadline) {
        $Health = Get-QbHealth $OnPort
        if ($Health) { return $Health }
        Start-Sleep -Seconds 3
    }
    return $null
}

function Get-QbNewestLog {
    Get-ChildItem -Path $Logs -Filter "querybot-*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "*.out.log" } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

function Assert-QbAdmin {
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $IsAdmin = ([Security.Principal.WindowsPrincipal]$Identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $IsAdmin) { throw "Run this in an Administrator PowerShell (right-click PowerShell, Run as administrator)." }
}

function Start-Qb([int]$OnPort) {
    if (-not (Get-QbTask)) { throw "The $TaskName task is not installed. Run -Action Install first." }
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Starting QueryBot on port $OnPort (the first start after an update can take a minute)..."
    $Health = Wait-QbHealth $OnPort
    if ($Health) {
        Write-Host "Running: $Health"
    } else {
        Write-Warning "QueryBot did not answer on port $OnPort yet. Its newest log:"
        $Log = Get-QbNewestLog
        if ($Log) { Get-Content -Path $Log.FullName -Tail 30 }
    }
}

function Stop-Qb([int]$OnPort) {
    if (Get-QbTask) { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue }
    # Ending the task ends its PowerShell, not the Python it started: stop that too.
    Stop-QbProcesses $OnPort
    Start-Sleep -Seconds 2
    Write-Host "QueryBot is stopped on port $OnPort."
}

$EffectivePort = Resolve-QbPort

switch ($Action) {
    "Install" {
        Assert-QbAdmin
        if (-not (Test-Path $Python)) { throw "No virtual environment at $RepoRoot\venv. Run deploy\windows\install-querybot.ps1 first." }

        Write-Host "== Settings in $EnvFile (only what has no value yet is added)"
        $Defaults = @("--default", "QUERYBOT_KEY_FILE=$env:USERPROFILE\.querybot_key")
        $Browsers = Join-Path $env:LOCALAPPDATA "ms-playwright"
        if (Test-Path $Browsers) { $Defaults += @("--default", "PLAYWRIGHT_BROWSERS_PATH=$Browsers") }
        $Models = Join-Path $env:USERPROFILE ".cache\huggingface"
        if (Test-Path $Models) { $Defaults += @("--default", "HF_HOME=$Models") }
        & $Python (Join-Path $RepoRoot "deploy\envfile.py") $EnvFile `
            --secret SESSION_SECRET --secret PORTAL_SESSION_SECRET --secret ADMIN_SESSION_SECRET `
            --secret PII_PSEUDONYM_SECRET @Defaults
        if ($LASTEXITCODE -ne 0) { throw "Could not write $EnvFile." }
        if (-not (Test-Path "$env:USERPROFILE\.querybot_key")) {
            Write-Warning "There is no $env:USERPROFILE\.querybot_key yet. QueryBot creates one on its first start; back it up then."
        }

        $Running = @(Get-QbProcesses $EffectivePort)
        if ($Running.Count -gt 0) {
            Write-Host "== Stopping the QueryBot already running on port $EffectivePort (from a console)"
            Stop-QbProcesses $EffectivePort
            Start-Sleep -Seconds 2
        }

        Write-Host "== Scheduled task $TaskName (runs as $env:USERDOMAIN\$env:USERNAME, at every start of the server)"
        if (Get-QbTask) {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        }
        $TaskAction = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $RepoRoot `
            -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Runner`" -Port $EffectivePort"
        $Trigger = New-ScheduledTaskTrigger -AtStartup
        $Trigger.Delay = "PT30S"                  # let the network come up first
        $Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
        $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
            -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
        Register-ScheduledTask -TaskName $TaskName -Action $TaskAction -Trigger $Trigger -Principal $Principal `
            -Settings $Settings -Description "QueryBot on port $EffectivePort ($RepoRoot)" | Out-Null

        Start-Qb $EffectivePort
        Write-Host ""
        Write-Host "Installed. QueryBot now starts with the server and restarts itself."
        Write-Host "  Status:  powershell -ExecutionPolicy Bypass -File deploy\windows\querybot-service.ps1 -Action Status"
        Write-Host "  Logs:    powershell -ExecutionPolicy Bypass -File deploy\windows\querybot-service.ps1 -Action Logs"
        Write-Host "  Back up together: $EnvFile and $env:USERPROFILE\.querybot_key"
    }
    "Uninstall" {
        Assert-QbAdmin
        Stop-Qb $EffectivePort
        if (Get-QbTask) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false }
        Write-Host "The $TaskName task is removed. .env, the key file, the data and the logs are kept."
    }
    "Start" { Start-Qb $EffectivePort }
    "Stop" { Stop-Qb $EffectivePort }
    "Restart" {
        Stop-Qb $EffectivePort
        Start-Qb $EffectivePort
    }
    "Status" {
        $Task = Get-QbTask
        if ($Task) {
            $Info = Get-ScheduledTaskInfo -TaskName $TaskName
            Write-Host ("Task {0}: {1} (last started {2}, result {3})" -f $TaskName, $Task.State, $Info.LastRunTime,
                        $Info.LastTaskResult)
        } else {
            Write-Host "Task ${TaskName}: not installed"
        }
        $Pids = @(Get-QbProcesses $EffectivePort | ForEach-Object { $_.ProcessId })
        Write-Host ("Port {0}: {1}" -f $EffectivePort, $(if ($Pids.Count) { "QueryBot running (process $($Pids -join ', '))" } else { "no QueryBot running" }))
        $Health = Get-QbHealth $EffectivePort
        Write-Host ("Health: {0}" -f $(if ($Health) { $Health } else { "no answer" }))
        $Log = Get-QbNewestLog
        if ($Log) { Write-Host "Newest log: $($Log.FullName)" }
    }
    "Logs" {
        $Log = Get-QbNewestLog
        if (-not $Log) { throw "No log yet in $Logs." }
        Write-Host "Following $($Log.FullName) (Ctrl+C stops following; QueryBot keeps running)"
        Get-Content -Path $Log.FullName -Tail 50 -Wait
    }
}
