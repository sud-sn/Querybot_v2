# QueryBot Azure Windows VM: Copy-Paste Installation

This runbook assumes:

- Windows Server 2022 or 2025, 64-bit
- Administrator access to the VM
- Repository: `https://github.com/sud-sn/Querybot_v2.git`
- QueryBot installed at `C:\QueryBot\app`
- PostgreSQL hosted on the same VM
- Qdrant hosted in WSL 2 for a POC/demo

For production, prefer Azure Database for PostgreSQL and Qdrant Cloud or a
private Linux Qdrant host. Do not expose PostgreSQL or Qdrant publicly.

## Step 1: Download the installers

Download these files onto the VM:

1. Git for Windows:
   `https://git-scm.com/download/win`
2. Python 3.12, Windows x64:
   `https://www.python.org/downloads/windows/`
3. PostgreSQL 17, Windows x64:
   `https://www.postgresql.org/download/windows/`
4. Microsoft Visual C++ Redistributable x64:
   `https://aka.ms/vs/17/release/vc_redist.x64.exe`
5. Microsoft ODBC Driver 18 for SQL Server x64:
   `https://go.microsoft.com/fwlink/?linkid=2358430`

Use PostgreSQL 17 on Windows Server 2019/2022. PostgreSQL 18 is also supported
on Windows Server 2022/2025, but version 17 is the conservative choice for this
deployment.

## Step 2: Install the prerequisites

Open PowerShell as Administrator. Change the paths below if your downloaded
filenames differ.

### 2.1 Install Git

```powershell
Start-Process `
  -FilePath "$env:USERPROFILE\Downloads\Git-64-bit.exe" `
  -ArgumentList "/VERYSILENT /NORESTART" `
  -Wait

git --version
```

If the final command is not found, close PowerShell and open it again.

### 2.2 Install Python

Example for Python 3.12:

```powershell
Start-Process `
  -FilePath "$env:USERPROFILE\Downloads\python-3.12.10-amd64.exe" `
  -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_test=0" `
  -Wait

py -3.12 --version
```

Use the actual downloaded Python filename when it differs.

### 2.3 Install the Visual C++ runtime

```powershell
Start-Process `
  -FilePath "$env:USERPROFILE\Downloads\vc_redist.x64.exe" `
  -ArgumentList "/install /quiet /norestart" `
  -Wait
```

### 2.4 Install Microsoft ODBC Driver 18

```powershell
Start-Process `
  -FilePath "$env:USERPROFILE\Downloads\msodbcsql.msi" `
  -ArgumentList "/quiet IACCEPTMSODBCSQLLICENSETERMS=YES" `
  -Wait
```

The QueryBot Azure SQL connector specifically expects:

```text
ODBC Driver 18 for SQL Server
```

### 2.5 Install PostgreSQL

Run the downloaded PostgreSQL installer:

```powershell
Start-Process `
  -FilePath "$env:USERPROFILE\Downloads\postgresql-17.x-windows-x64.exe" `
  -Wait
```

During installation select:

- PostgreSQL Server
- pgAdmin 4
- Command Line Tools
- Port: `5432`
- Locale: default
- A strong password for the `postgres` administrator

Replace the example filename with the exact downloaded installer name.

After installation:

```powershell
Get-Service postgresql*
```

The service status should be `Running`.

## Step 3: Clone QueryBot

Open a new PowerShell window as Administrator:

```powershell
New-Item -ItemType Directory -Force C:\QueryBot | Out-Null
Set-Location C:\QueryBot

git clone https://github.com/sud-sn/Querybot_v2.git app
Set-Location C:\QueryBot\app

git branch --show-current
git log -1 --oneline
```

For a private repository, sign in using Git Credential Manager when prompted.
Do not embed a GitHub token in the clone URL.

## Step 4: Install QueryBot's Python packages

```powershell
Set-Location C:\QueryBot\app

powershell -ExecutionPolicy Bypass `
  -File .\deploy\windows\install-querybot.ps1
```

This creates:

- `C:\QueryBot\app\venv`
- `C:\QueryBot\app\.env`
- required data/client directories
- `C:\QueryBot\secrets`

Confirm the packages and ODBC driver:

```powershell
.\venv\Scripts\python.exe -c "import fastapi, uvicorn, psycopg2, qdrant_client, pyodbc; print('Python packages OK')"
.\venv\Scripts\python.exe -c "import pyodbc; print(pyodbc.drivers())"
```

The second command must include `ODBC Driver 18 for SQL Server`.

## Step 5: Create the PostgreSQL application database

Locate `psql.exe`:

```powershell
$Psql = Get-ChildItem "C:\Program Files\PostgreSQL" `
  -Recurse -Filter psql.exe |
  Select-Object -First 1 -ExpandProperty FullName

$Psql
```

Open the PostgreSQL shell:

```powershell
& $Psql -U postgres -h 127.0.0.1 -p 5432
```

Enter the PostgreSQL administrator password, then execute:

```sql
CREATE ROLE querybot LOGIN PASSWORD 'REPLACE_WITH_A_STRONG_PASSWORD';
CREATE DATABASE querybot OWNER querybot;
\connect querybot
GRANT ALL ON SCHEMA public TO querybot;
\q
```

The QueryBot password must not contain unescaped URL-reserved characters in
`DATABASE_URL`. Either use an alphanumeric password with `-._~`, or
percent-encode it.

Test the new database:

```powershell
$env:PGPASSWORD = "REPLACE_WITH_A_STRONG_PASSWORD"
& $Psql -U querybot -h 127.0.0.1 -p 5432 -d querybot `
  -c "SELECT current_database(), current_user;"
Remove-Item Env:PGPASSWORD
```

## Step 6: Install WSL 2 for Qdrant

For Windows Server 2022/2025, run as Administrator:

```powershell
wsl --install -d Ubuntu
```

Restart the VM when Windows requests it:

```powershell
Restart-Computer
```

After reconnecting, launch Ubuntu once:

```powershell
wsl -d Ubuntu
```

Create the requested Linux username/password, then exit:

```bash
exit
```

If `wsl --install` reports that virtualization is unavailable, the Azure VM
size does not support nested virtualization. Use Qdrant Cloud or a private
Linux VM instead.

## Step 7: Install Docker Engine and Qdrant inside WSL

From Windows PowerShell:

```powershell
Set-Location C:\QueryBot\app

wsl -d Ubuntu -- bash `
  /mnt/c/QueryBot/app/deploy/windows/install-qdrant-wsl.sh
```

Verify Qdrant from Windows:

```powershell
Invoke-RestMethod http://127.0.0.1:6333/healthz
```

Expected response:

```text
healthz check passed
```

Useful Qdrant commands:

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/QueryBot/app && sudo docker compose -f deploy/qdrant.compose.yml ps"
wsl -d Ubuntu -- bash -lc "cd /mnt/c/QueryBot/app && sudo docker compose -f deploy/qdrant.compose.yml logs --tail=100"
wsl -d Ubuntu -- bash -lc "cd /mnt/c/QueryBot/app && sudo docker compose -f deploy/qdrant.compose.yml restart"
```

## Step 8: Configure `.env`

Generate three separate session secrets:

```powershell
Set-Location C:\QueryBot\app

.\venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
.\venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
.\venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
```

Edit the environment file:

```powershell
notepad C:\QueryBot\app\.env
```

Use:

```text
DATABASE_URL=postgresql://querybot:YOUR_ENCODED_PASSWORD@127.0.0.1:5432/querybot
QDRANT_URL=http://127.0.0.1:6333
QUERYBOT_RERANK=true

PORTAL_BASE_URL=http://YOUR_VM_IP:8000

SESSION_SECRET=FIRST_GENERATED_SECRET
PORTAL_SESSION_SECRET=SECOND_GENERATED_SECRET
ADMIN_SESSION_SECRET=THIRD_GENERATED_SECRET

QUERYBOT_KEY_FILE=C:/QueryBot/secrets/.querybot_key
LLM_AUDIT_RETENTION_DAYS=30
```

For production, change `PORTAL_BASE_URL` to the final HTTPS URL.

## Step 9: Initialize QueryBot's PostgreSQL schema

```powershell
Set-Location C:\QueryBot\app

.\venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); from store.db import init_db; init_db(); print('QueryBot PostgreSQL schema OK')"
```

Do not continue if this command reports a PostgreSQL connection or permission
error.

## Step 10: Start QueryBot

```powershell
Set-Location C:\QueryBot\app

powershell -ExecutionPolicy Bypass `
  -File .\deploy\windows\start-querybot.ps1
```

Keep this window open for the first test.

From another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected shape:

```text
status  : ok
version : 2.0.0
clients : 0
ready   : 0
```

Open:

```text
http://YOUR_VM_IP:8000/admin
```

Complete the first-time admin setup and configure Azure OpenAI and Azure SQL.

## Step 11: Open the Windows and Azure firewalls

For temporary POC access to port 8000:

```powershell
New-NetFirewallRule `
  -DisplayName "QueryBot HTTP 8000" `
  -Direction Inbound `
  -Protocol TCP `
  -LocalPort 8000 `
  -Action Allow
```

Also create an Azure Network Security Group inbound rule:

- Source: your office/public IP only
- Destination port: `8000`
- Protocol: TCP
- Action: Allow

Do not create public inbound rules for:

- PostgreSQL `5432`
- Qdrant REST `6333`
- Qdrant gRPC `6334`

For production, remove the 8000 rule and publish the application through HTTPS
443 using IIS, Azure Application Gateway, or another reverse proxy.

## Step 12: Verify all services

```powershell
Set-Location C:\QueryBot\app

powershell -ExecutionPolicy Bypass `
  -File .\deploy\windows\verify-querybot.ps1
```

Every required check should show `PASS`.

## Step 13: Start QueryBot automatically after reboot

Open Task Scheduler and create a task:

- Name: `QueryBot`
- Run whether user is logged on or not
- Run with highest privileges
- Trigger: At startup, delay 30 seconds
- Program: `powershell.exe`
- Arguments:

```text
-NoProfile -ExecutionPolicy Bypass -File C:\QueryBot\app\deploy\windows\start-querybot.ps1
```

- Start in:

```text
C:\QueryBot\app
```

Configure the task to restart every minute after failure, up to three times.

## Step 14: Future updates

Stop QueryBot, then:

```powershell
Set-Location C:\QueryBot\app

git status
git pull --ff-only

.\venv\Scripts\python.exe -m pip install `
  -r requirements-windows.txt

.\venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); from store.db import init_db; init_db(); print('Migration OK')"

powershell -ExecutionPolicy Bypass `
  -File .\deploy\windows\start-querybot.ps1
```

Back up before updating:

- PostgreSQL database
- Qdrant volume or Qdrant Cloud collection
- `C:\QueryBot\secrets\.querybot_key`
- `.env`
