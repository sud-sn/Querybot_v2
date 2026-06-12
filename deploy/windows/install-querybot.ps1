param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"

Set-Location $RepoRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install 64-bit Python $PythonVersion and enable the Python launcher."
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment..."
    & py "-$PythonVersion" -m venv venv
}

Write-Host "Installing QueryBot dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements-windows.txt

Write-Host "Checking Microsoft ODBC Driver 18..."
$Drivers = & $VenvPython -c "import pyodbc; print('|'.join(pyodbc.drivers()))"
if ($Drivers -notmatch "ODBC Driver 18 for SQL Server") {
    Write-Warning "ODBC Driver 18 for SQL Server was not found. Install it before connecting QueryBot to Azure SQL."
} else {
    Write-Host "ODBC Driver 18 for SQL Server is installed."
}

if (-not (Test-Path (Join-Path $RepoRoot ".env"))) {
    Copy-Item (Join-Path $RepoRoot ".env.windows.example") (Join-Path $RepoRoot ".env")
    Write-Warning "Created .env from .env.windows.example. Replace every CHANGE_ME value before startup."
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "clients") | Out-Null
New-Item -ItemType Directory -Force -Path "C:\QueryBot\secrets" | Out-Null

Write-Host ""
Write-Host "Python setup complete."
Write-Host "Next: configure PostgreSQL and .env, start Qdrant, then run deploy\windows\start-querybot.ps1."
