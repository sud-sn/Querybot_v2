<#
.SYNOPSIS
Makes the virtual environment and installs everything Solviz needs to run. deploy.ps1 runs it for you.

.DESCRIPTION
From the Solviz folder:
    powershell -ExecutionPolicy Bypass -File deploy\windows\install-querybot.ps1
    powershell -ExecutionPolicy Bypass -File deploy\windows\install-querybot.ps1 -Python C:\Python312\python.exe

Installs the exact versions in requirements-windows.lock into venv, the name recogniser's English model
(en_core_web_md), and Chromium, which draws scheduled dashboards as pictures and PDF. Gives .env its secrets when
it has none yet. Safe to run again. The Python used is -Python, else -PythonVersion through the py launcher, else
Python 3.11 or 3.12, whichever the py launcher finds first.
#>
param(
    [string]$Python = "",
    [string]$PythonVersion = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Venv = Join-Path $RepoRoot "venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$EnvFile = Join-Path $RepoRoot ".env"
Set-Location $RepoRoot

# The Python version a program reports, or "" when it cannot be run. Windows PowerShell treats a native program's
# error output as an error under "Stop", so it is asked with errors allowed.
function Get-PythonVersion([string]$Exe, [string[]]$Prefix = @()) {
    $Saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $Out = & $Exe @Prefix -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0) { return "" }
        return ($Out | Out-String).Trim()
    } catch {
        return ""
    } finally {
        $ErrorActionPreference = $Saved
    }
}

# The virtual environment must be Python 3.11 or 3.12: the pinned versions are built for those two.
if (Test-Path $VenvPython) {
    $Have = Get-PythonVersion $VenvPython
    if ($Have -notin @("3.11", "3.12")) {
        $Moved = "$Venv.old-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Write-Host "The virtual environment is Python '$Have'; the libraries need 3.11 or 3.12. Moving it to $Moved"
        Move-Item -Path $Venv -Destination $Moved
    }
}

if (-not (Test-Path $VenvPython)) {
    if ($Python) {
        $Maker = @($Python)
    } elseif ($PythonVersion) {
        $Maker = @("py", "-$PythonVersion")
    } else {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw "No Python launcher 'py'. Install 64-bit Python 3.12 from python.org, or run deploy\windows\deploy.ps1, which installs it."
        }
        $Maker = $null
        foreach ($Version in "3.11", "3.12") {
            if ((Get-PythonVersion "py" @("-$Version")) -eq $Version) { $Maker = @("py", "-$Version"); break }
        }
        if (-not $Maker) { throw "No Python 3.11 or 3.12 found. Install 64-bit Python 3.12, or run deploy\windows\deploy.ps1." }
    }
    Write-Host "Creating the virtual environment with $($Maker -join ' ')..."
    $MakerArgs = @()
    if ($Maker.Count -gt 1) { $MakerArgs = $Maker[1..($Maker.Count - 1)] }
    & $Maker[0] @MakerArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) { throw "Could not create the virtual environment." }
}

Write-Host "Installing Solviz's libraries (the exact versions in requirements-windows.lock)..."
& $VenvPython -m pip install --quiet --upgrade pip
$Requirements = Join-Path $RepoRoot "requirements-windows.lock"
if (-not (Test-Path $Requirements)) { $Requirements = Join-Path $RepoRoot "requirements-windows.txt" }
& $VenvPython -m pip install -r $Requirements
# A half-installed environment is not a working install: stop here and say so, rather than let the failure show up
# later as missing behaviour in the app.
if ($LASTEXITCODE -ne 0) {
    Write-Error "Installing the libraries failed (pip exit code $LASTEXITCODE). Solviz is NOT installed."
    exit $LASTEXITCODE
}

# The name recogniser's English model (core\masking.py): not a PyPI package, so fetched here once. Without it names
# typed in questions are withheld by the name rules alone, and the compliance page says so.
& $VenvPython -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('en_core_web_md') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the name recogniser's English model..."
    & $VenvPython -m spacy download en_core_web_md
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The name recogniser's model could not be downloaded: names typed in questions will be withheld by rules only. To retry: $VenvPython -m spacy download en_core_web_md"
    }
}

Write-Host "Installing the browser that draws scheduled dashboards as pictures and PDF..."
& $VenvPython -m playwright install chromium
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Chromium could not be installed: scheduled emails will go without pictures and PDF, with every number. To retry: $VenvPython -m playwright install chromium"
}

$Drivers = & $VenvPython -c "import pyodbc; print('|'.join(pyodbc.drivers()))"
if ($Drivers -notmatch "ODBC Driver 18 for SQL Server") {
    Write-Warning "ODBC Driver 18 for SQL Server is not installed: Azure SQL and SQL Server sources need it (deploy\windows\deploy.ps1 installs it)."
}

# .env with its secrets: added only where it has no value yet, never changed.
& $VenvPython (Join-Path $RepoRoot "deploy\envfile.py") $EnvFile `
    --secret SESSION_SECRET --secret PORTAL_SESSION_SECRET --secret ADMIN_SESSION_SECRET --secret PII_PSEUDONYM_SECRET `
    --default "QUERYBOT_KEY_FILE=$env:USERPROFILE\.querybot_key"
if ($LASTEXITCODE -ne 0) { throw "Could not write $EnvFile." }

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "clients") | Out-Null

Write-Host ""
Write-Host "Libraries installed."
Write-Host "Next: run Solviz as a background task:"
Write-Host "  powershell -ExecutionPolicy Bypass -File deploy\windows\querybot-service.ps1 -Action Install -Port 8000"
exit 0
