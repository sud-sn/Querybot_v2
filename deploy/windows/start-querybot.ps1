param(
    [string]$BindAddress = "0.0.0.0",
    [int]$Port = 8000,
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$EnvFile = Join-Path $RepoRoot ".env"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run deploy\windows\install-querybot.ps1 first."
}

if (-not (Test-Path $EnvFile)) {
    throw ".env not found. Copy .env.windows.example to .env and configure it."
}

Set-Location $RepoRoot
& $Python -m uvicorn main:app `
    --host $BindAddress `
    --port $Port `
    --workers $Workers `
    --log-level info `
    --env-file $EnvFile

exit $LASTEXITCODE
