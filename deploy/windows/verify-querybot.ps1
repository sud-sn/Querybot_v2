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

Write-Host "QueryBot deployment verification"
Write-Host "Repository: $RepoRoot"
Write-Host ""

$git = Get-Command git -ErrorAction SilentlyContinue
Write-Check "Git" ($null -ne $git) $(if ($git) { (& git --version) } else { "not installed" })

$pythonExists = Test-Path $Python
$pythonVersion = if ($pythonExists) { (& $Python --version 2>&1) } else { "venv not found" }
Write-Check "Python venv" $pythonExists "$pythonVersion"

if ($pythonExists) {
    $importResult = & $Python -c "import fastapi, uvicorn, psycopg2, qdrant_client, pyodbc; print('imports OK')" 2>&1
    Write-Check "Python dependencies" ($LASTEXITCODE -eq 0) "$importResult"

    $driverResult = & $Python -c "import pyodbc; print('|'.join(pyodbc.drivers()))" 2>&1
    Write-Check "ODBC Driver 18" ($driverResult -match "ODBC Driver 18 for SQL Server") "$driverResult"

    Push-Location $RepoRoot
    $dbResult = & $Python -c "from dotenv import load_dotenv; load_dotenv(); from store.db import init_db; init_db(); print('database OK')" 2>&1
    $dbPassed = $LASTEXITCODE -eq 0
    Pop-Location
    Write-Check "QueryBot database" $dbPassed "$dbResult"
}

$postgres = Test-NetConnection 127.0.0.1 -Port 5432 -WarningAction SilentlyContinue
Write-Check "PostgreSQL port" $postgres.TcpTestSucceeded "127.0.0.1:5432"

try {
    $qdrant = Invoke-RestMethod http://127.0.0.1:6333/healthz -TimeoutSec 5
    Write-Check "Qdrant" $true "$qdrant"
} catch {
    Write-Check "Qdrant" $false $_.Exception.Message
}

try {
    $health = Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 5
    Write-Check "QueryBot HTTP" ($health.status -eq "ok") ("status={0}, version={1}" -f $health.status, $health.version)
} catch {
    Write-Check "QueryBot HTTP" $false $_.Exception.Message
}

$outboundHosts = @(
    "pypi.org",
    "hub.docker.com",
    "huggingface.co"
)

foreach ($hostName in $outboundHosts) {
    $test = Test-NetConnection $hostName -Port 443 -WarningAction SilentlyContinue
    Write-Check "Outbound HTTPS" $test.TcpTestSucceeded "$hostName`:443"
}
