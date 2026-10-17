<#
.SYNOPSIS
Checks a Windows Solviz install: Git, the virtual environment and its libraries, the ODBC driver, the name
recogniser's model, the PDF browser, the store, the running service and outbound access for updates.

.DESCRIPTION
    powershell -ExecutionPolicy Bypass -File deploy\windows\verify-querybot.ps1 [-Port 8000]
#>
param([int]$Port = 8000)

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"

function Write-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Details
    )

    $status = if ($Passed) { "PASS" } else { "FAIL" }
    $color = if ($Passed) { "Green" } else { "Red" }
    Write-Host ("[{0}] {1}: {2}" -f $status, $Name, $Details) -ForegroundColor $color
}

Write-Host "Solviz deployment check"
Write-Host "Repository: $RepoRoot"
Write-Host ""

$git = Get-Command git -ErrorAction SilentlyContinue
Write-Check "Git" ($null -ne $git) $(if ($git) { (& git --version) } else { "not installed" })

$pythonExists = Test-Path $Python
$pythonVersion = if ($pythonExists) { (& $Python --version 2>&1) } else { "venv not found" }
Write-Check "Python venv" $pythonExists "$pythonVersion"

if ($pythonExists) {
    $importResult = & $Python -c "import fastapi, uvicorn, pyodbc, duckdb, sqlglot, spacy, playwright; print('imports OK')" 2>&1
    Write-Check "Python dependencies" ($LASTEXITCODE -eq 0) "$importResult"

    $driverResult = & $Python -c "import pyodbc; print('|'.join(pyodbc.drivers()))" 2>&1
    Write-Check "ODBC Driver 18" ($driverResult -match "ODBC Driver 18 for SQL Server") "$driverResult"

    $modelResult = & $Python -c "import en_core_web_md; print('installed')" 2>&1
    Write-Check "Name recogniser model" ($LASTEXITCODE -eq 0) "$modelResult"

    $browserResult = & $Python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); p.chromium.launch().close(); p.stop(); print('starts')" 2>&1
    Write-Check "PDF browser (Chromium)" ($LASTEXITCODE -eq 0) "$browserResult"

    Push-Location $RepoRoot
    $dbResult = & $Python -c "from dotenv import load_dotenv; load_dotenv(); from store.db import init_db; init_db(); print('database OK')" 2>&1
    $dbPassed = $LASTEXITCODE -eq 0
    Pop-Location
    Write-Check "QueryBot database" $dbPassed "$dbResult"
}

try {
    $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5
    Write-Check "QueryBot HTTP" ($health.status -eq "ok") ("status={0}, version={1}" -f $health.status, $health.version)
} catch {
    Write-Check "QueryBot HTTP" $false $_.Exception.Message
}

$outboundHosts = @(
    "pypi.org",
    "files.pythonhosted.org",
    "download.pytorch.org",
    "github.com"
)

foreach ($hostName in $outboundHosts) {
    $test = Test-NetConnection $hostName -Port 443 -WarningAction SilentlyContinue
    Write-Check "Outbound HTTPS" $test.TcpTestSucceeded "$hostName`:443"
}
