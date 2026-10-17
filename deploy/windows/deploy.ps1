<#
.SYNOPSIS
Installs Solviz on a Windows server in one go: everything it needs, the background task, a firewall rule and a
health check.

.DESCRIPTION
In an Administrator PowerShell, from the Solviz folder (the cloned repository):

    powershell -ExecutionPolicy Bypass -File deploy\windows\deploy.ps1
    powershell -ExecutionPolicy Bypass -File deploy\windows\deploy.ps1 -Port 8001 -AdminEmail you@company.com -AdminName "Your Name"

Every step is safe to run again, so it also repairs a server that is missing a piece:
  1. Python 3.11 or 3.12 (64-bit): kept when installed, else installed for all users (winget, else python.org).
  2. Microsoft's ODBC Driver 18 for SQL Server, with the Visual C++ runtime it needs, for Azure SQL and SQL Server
     data sources (winget, else Microsoft's download). -OdbcMsi C:\path\msodbcsql.msi installs from a file you
     have; -NoOdbc leaves it out.
  3. The virtual environment venv with the exact versions in requirements-windows.lock, the name recogniser's
     English model, and Chromium, which draws scheduled dashboards as pictures and PDF (install-querybot.ps1).
  4. .env: its secrets made once and never changed, and the path of the key that encrypts saved passwords.
     -PublicUrl sets the address people open, used in emailed links.
  5. The background task "QueryBot" that starts with the server and restarts Solviz if it stops
     (querybot-service.ps1 -Action Install), a Windows Firewall rule for the port, and a health check.
  6. Optional: the first Super admin account (-AdminEmail; asks for the password).

Updates afterwards are one command:  powershell -ExecutionPolicy Bypass -File deploy\windows\update-querybot.ps1
#>
param(
    [int]$Port = 0,
    [string]$PublicUrl = "",
    [string]$AdminEmail = "",
    [string]$AdminName = "",
    [string]$OdbcMsi = "",
    [switch]$NoOdbc,
    [switch]$NoFirewall,
    [string]$TaskName = "QueryBot"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
$EnvFile = Join-Path $RepoRoot ".env"
$Downloads = Join-Path $env:TEMP "solviz-setup"
$PythonInstaller = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
$VcRuntime = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
$OdbcDownload = "https://go.microsoft.com/fwlink/?linkid=2358430"
$OdbcDriver = "ODBC Driver 18 for SQL Server"
Set-Location $RepoRoot
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Write-Step([string]$Text) { Write-Host ""; Write-Host "== $Text" -ForegroundColor Cyan }

function Assert-Admin {
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $IsAdmin = ([Security.Principal.WindowsPrincipal]$Identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $IsAdmin) { throw "Run this in an Administrator PowerShell (right-click PowerShell, Run as administrator)." }
}

# Runs a program and returns what it printed, or $null when it failed. Its error output is dropped without stopping
# the script: Windows PowerShell treats a native program's redirected error output as an error under "Stop".
function Invoke-Quiet([string]$File, [string[]]$Arguments) {
    $Saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $Out = & $File @Arguments 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($Out | Out-String).Trim()
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $Saved
    }
}

function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

function Test-Winget { [bool](Get-Command winget -ErrorAction SilentlyContinue) }

function Install-WithWinget([string]$Id, [string[]]$Extra = @()) {
    if (-not (Test-Winget)) { return $false }
    Write-Host "Installing $Id with winget..."
    $Arguments = @("install", "--id", $Id, "--exact", "--silent", "--accept-package-agreements",
                   "--accept-source-agreements") + $Extra
    $Saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & winget @Arguments | Out-Host } finally { $ErrorActionPreference = $Saved }
    return ($LASTEXITCODE -eq 0)
}

function Save-Download([string]$Url, [string]$Name) {
    New-Item -ItemType Directory -Force -Path $Downloads | Out-Null
    $Target = Join-Path $Downloads $Name
    Write-Host "Downloading $Name..."
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Target
    return $Target
}

# Python 3.11 first (the version the test suite runs on), then 3.12: the path of its python.exe, or $null.
function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($Version in "3.11", "3.12") {
            $Exe = Invoke-Quiet "py" @("-$Version", "-c", "import sys, venv; print(sys.executable)")
            if ($Exe) { return $Exe }
        }
    }
    foreach ($Command in "python", "python3") {
        if (Get-Command $Command -ErrorAction SilentlyContinue) {
            $Found = Invoke-Quiet $Command @("-c", "import sys, venv; print('%d.%d|%s' % (sys.version_info[0], sys.version_info[1], sys.executable))")
            if ($Found -and ($Found -match '^(3\.11|3\.12)\|(.+)$')) { return $Matches[2] }
        }
    }
    return $null
}

function Test-VcRuntime {
    $Key = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
    return [bool]($Key -and $Key.Installed -eq 1)
}

function Test-Odbc18 {
    return [bool](Get-OdbcDriver -Name $OdbcDriver -Platform "64-bit" -ErrorAction SilentlyContinue)
}

Assert-Admin
if ($Port -le 0) {
    # The port the installed task runs on, else 8000.
    $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Existing -and ($Existing.Actions[0].Arguments -match '-Port\s+(\d+)')) { $Port = [int]$Matches[1] } else { $Port = 8000 }
}
Write-Host "Solviz: $RepoRoot, task `"$TaskName`", port $Port, Windows user $env:USERDOMAIN\$env:USERNAME"

# ── 1. Python ─────────────────────────────────────────────────────────────────────────────────────────────────
Write-Step "1/6 Python"
$Python = Find-Python
if (-not $Python) {
    if (-not (Install-WithWinget "Python.Python.3.12" @("--scope", "machine"))) {
        $Installer = Save-Download $PythonInstaller "python-3.12-amd64.exe"
        $Process = Start-Process -FilePath $Installer -Wait -PassThru `
            -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_test=0"
        if ($Process.ExitCode -ne 0) { throw "The Python installer stopped with exit code $($Process.ExitCode)." }
    }
    Update-SessionPath
    $Python = Find-Python
}
if (-not $Python) {
    throw "No Python 3.11 or 3.12 after installing it. Install 64-bit Python 3.12 from python.org (tick 'Add to PATH'), then run this again."
}
Write-Host "Python: $Python"

# ── 2. ODBC Driver 18 for SQL Server ──────────────────────────────────────────────────────────────────────────
Write-Step "2/6 Microsoft ODBC Driver 18 for SQL Server"
if ($NoOdbc) {
    Write-Host "Left out (-NoOdbc): Azure SQL and SQL Server data sources will not connect until it is installed."
} elseif (Test-Odbc18) {
    Write-Host "$OdbcDriver is installed."
} else {
    if (-not (Test-VcRuntime)) {
        if (-not (Install-WithWinget "Microsoft.VCRedist.2015+.x64")) {
            $Runtime = Save-Download $VcRuntime "vc_redist.x64.exe"
            Start-Process -FilePath $Runtime -ArgumentList "/install /quiet /norestart" -Wait | Out-Null
        }
    }
    $Installed = $false
    if ($OdbcMsi) {
        if (-not (Test-Path $OdbcMsi)) { throw "No file at $OdbcMsi (-OdbcMsi)." }
        Start-Process msiexec.exe -Wait -ArgumentList "/i `"$OdbcMsi`" /quiet /norestart IACCEPTMSODBCSQLLICENSETERMS=YES" | Out-Null
    } elseif (-not (Install-WithWinget "Microsoft.msodbcsql.18")) {
        try {
            $Msi = Save-Download $OdbcDownload "msodbcsql.msi"
            Start-Process msiexec.exe -Wait -ArgumentList "/i `"$Msi`" /quiet /norestart IACCEPTMSODBCSQLLICENSETERMS=YES" | Out-Null
        } catch {
            Write-Warning "The ODBC driver could not be downloaded: $($_.Exception.Message)"
        }
    }
    $Installed = Test-Odbc18
    if ($Installed) {
        Write-Host "$OdbcDriver is installed."
    } else {
        Write-Warning ("$OdbcDriver is not installed. Azure SQL and SQL Server sources need it; the rest of Solviz works. " +
                       "Download the x64 installer from https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server " +
                       "and run this again with -OdbcMsi <path to msodbcsql.msi>.")
    }
}

# ── 3. Virtual environment, libraries, name recogniser, PDF browser ──────────────────────────────────────────
Write-Step "3/6 Virtual environment and libraries (the first time takes several minutes)"
& (Join-Path $PSScriptRoot "install-querybot.ps1") -Python $Python
if ($LASTEXITCODE -ne 0) { throw "install-querybot.ps1 did not finish (exit code $LASTEXITCODE)." }

# ── 4. Settings ───────────────────────────────────────────────────────────────────────────────────────────────
Write-Step "4/6 Settings (.env)"
$Defaults = @()
if ($PublicUrl) { $Defaults += @("--default", "PORTAL_BASE_URL=$($PublicUrl.TrimEnd('/'))") }
& $VenvPython (Join-Path $RepoRoot "deploy\envfile.py") $EnvFile `
    --secret SESSION_SECRET --secret PORTAL_SESSION_SECRET --secret ADMIN_SESSION_SECRET --secret PII_PSEUDONYM_SECRET `
    --default "QUERYBOT_KEY_FILE=$env:USERPROFILE\.querybot_key" @Defaults
if ($LASTEXITCODE -ne 0) { throw "Could not write $EnvFile." }

# ── 5. Background task, firewall, health ──────────────────────────────────────────────────────────────────────
Write-Step "5/6 Background task, firewall and health check"
& (Join-Path $PSScriptRoot "querybot-service.ps1") -Action Install -Port $Port -TaskName $TaskName
if (-not $NoFirewall) {
    $RuleName = "Solviz (TCP $Port)"
    if (-not (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow | Out-Null
        Write-Host "Windows Firewall: port $Port opened."
    } else {
        Write-Host "Windows Firewall: port $Port already open."
    }
}

# ── 6. First admin ────────────────────────────────────────────────────────────────────────────────────────────
Write-Step "6/6 First admin"
if ($AdminEmail) {
    # The admin command must read the same database and key as Solviz: the settings in .env, from this folder.
    Write-Host "Super admin ${AdminEmail}: choose a password (at least 12 characters)"
    & $VenvPython -c "import runpy, sys; from dotenv import load_dotenv; load_dotenv(sys.argv.pop(1)); runpy.run_module('admin.create_admin', run_name='__main__')" `
        $EnvFile $AdminEmail $AdminName
    if ($LASTEXITCODE -ne 0) { Write-Warning "The admin account was not made. Run this script again with -AdminEmail to retry." }
} else {
    Write-Host "No -AdminEmail: the first person to open /admin sets the admin password. Do that now, or run this"
    Write-Host "script again with -AdminEmail you@company.com -AdminName `"Your Name`"."
}

$Address = if ($PublicUrl) { $PublicUrl.TrimEnd('/') } else { "http://$($env:COMPUTERNAME):$Port" }
Write-Host ""
Write-Host "Solviz is installed and running." -ForegroundColor Green
Write-Host ""
Write-Host "  Admin console   $Address/admin"
Write-Host "  Chat portal     $Address/portal/login?workspace=<workspace id>"
Write-Host "  Health          http://localhost:$Port/health"
Write-Host ""
Write-Host "Next, in the admin console:"
Write-Host "  1. Sign in (or set the admin password if this is the first visit)."
Write-Host "  2. System > AI provider: choose the provider and paste its key (Azure OpenAI, Anthropic or OpenAI)."
Write-Host "  3. Workspaces > New workspace: connect the data source, run discovery, then Learn."
Write-Host "  4. The workspace's People page: add users, their groups, their tables and any row rules."
Write-Host ""
Write-Host "Keep safe, together:  $EnvFile  and  $env:USERPROFILE\.querybot_key"
Write-Host "  (without the key file, saved data-source passwords cannot be read)"
Write-Host "Status:  powershell -ExecutionPolicy Bypass -File deploy\windows\querybot-service.ps1 -Action Status"
Write-Host "Update:  powershell -ExecutionPolicy Bypass -File deploy\windows\update-querybot.ps1"
if (-not $PublicUrl -or $PublicUrl -notmatch '^https://') {
    Write-Host ""
    Write-Host "This address is plain HTTP. Before real data goes in, put HTTPS in front (Azure Application Gateway,"
    Write-Host "Front Door or IIS with a certificate), then run this again with -PublicUrl https://<your address>."
    Write-Host "Open port $Port in the cloud firewall (network security group) if it is closed."
}
