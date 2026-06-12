# QueryBot on an Azure Windows VM

For a command-by-command installation, use
[`STEP-BY-STEP.md`](STEP-BY-STEP.md).

This guide deploys QueryBot on Windows with:

- QueryBot and Uvicorn running natively on Windows
- PostgreSQL running locally on the VM
- Qdrant running privately in Docker on Linux/WSL, or in Qdrant Cloud
- Azure SQL accessed through Microsoft ODBC Driver 18

## 1. Network and software prerequisites

Install:

1. Git for Windows.
2. 64-bit Python 3.12 with the `py` launcher.
3. PostgreSQL for Windows, including Command Line Tools.
4. Microsoft ODBC Driver 18 for SQL Server.

The VM needs outbound HTTPS access to PyPI, Docker Hub, the configured LLM
provider, and `huggingface.co` when local embedding and cross-encoder models
are first downloaded.

Do not expose PostgreSQL port 5432 or Qdrant ports 6333/6334 to the public
internet. QueryBot uses both through localhost.

## 2. Clone and install QueryBot

Run PowerShell:

```powershell
New-Item -ItemType Directory -Force C:\QueryBot | Out-Null
Set-Location C:\QueryBot
git clone https://github.com/sud-sn/Querybot_v2.git app
Set-Location app
powershell -ExecutionPolicy Bypass -File .\deploy\windows\install-querybot.ps1
```

For a private repository, authenticate with Git Credential Manager or use a
read-only deployment token. Do not place a token in a committed URL.

## 3. Create the PostgreSQL database

Open SQL Shell (`psql`) as the PostgreSQL administrator and run:

```sql
CREATE ROLE querybot LOGIN PASSWORD 'replace-with-a-strong-password';
CREATE DATABASE querybot OWNER querybot;
\connect querybot
GRANT ALL ON SCHEMA public TO querybot;
```

Keep PostgreSQL bound to localhost when the application is on the same VM.
The QueryBot role needs schema creation/alter privileges during first startup
because `init_db()` creates and upgrades the application tables.

Set this in `.env`, percent-encoding special URL characters in the password:

```text
DATABASE_URL=postgresql://querybot:encoded-password@127.0.0.1:5432/querybot
```

Test it:

```powershell
.\venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); from store.db import init_db; init_db(); print('PostgreSQL OK')"
```

## 4. Start Qdrant

### Recommended for Windows Server

Docker Desktop is not supported on Windows Server. Use one of these:

1. Qdrant Cloud for the simplest production deployment.
2. A small private Linux VM/container host for self-hosted production.
3. WSL 2 with Ubuntu and Docker Engine for a POC or demo, if the Azure VM size
   and Windows Server version support WSL 2 and nested virtualization.

Do not treat Docker-in-WSL as the durable production vector store. Qdrant
documents Windows/WSL mounted-storage risks; use Cloud or Linux-hosted Qdrant
when backups and recovery matter.

For WSL 2, install Ubuntu from elevated PowerShell and reboot if requested:

```powershell
wsl --install -d Ubuntu
```

Install Docker Engine and the Compose plugin inside Ubuntu using Docker's
official Ubuntu instructions. Then run from the repository:

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/QueryBot/app' && docker compose -f deploy/qdrant.compose.yml up -d"
```

The Compose file binds Qdrant only to `127.0.0.1`. Confirm Windows can reach it:

```powershell
Invoke-RestMethod http://127.0.0.1:6333/healthz
```

If Windows cannot reach WSL through localhost forwarding, obtain the WSL
address with `wsl hostname -I`, set `QDRANT_URL` to that private address, and
keep Windows Firewall restricted to the local/private interface.

For Qdrant Cloud, set:

```text
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-api-key
```

Before production, pin `QDRANT_IMAGE` to the version validated in staging:

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/QueryBot/app' && QDRANT_IMAGE='qdrant/qdrant:<tested-version>' docker compose -f deploy/qdrant.compose.yml up -d"
```

## 5. Configure QueryBot

Edit `C:\QueryBot\app\.env`:

1. Set `DATABASE_URL`.
2. Set `QDRANT_URL` and optionally `QDRANT_API_KEY`.
3. Generate unique session secrets:

```powershell
.\venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
```

4. Set `PORTAL_BASE_URL` to the final HTTPS address.
5. Keep `QUERYBOT_KEY_FILE` outside the repository and back it up securely.

API keys and client database credentials can then be configured through the
QueryBot admin UI; they are encrypted using the QueryBot key file.

## 6. Start and verify

Start QueryBot:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\start-querybot.ps1
```

Verify locally:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Test-NetConnection 127.0.0.1 -Port 5432
Test-NetConnection 127.0.0.1 -Port 6333
Test-NetConnection huggingface.co -Port 443
```

Open `http://<vm-private-or-public-ip>:8000/admin` only for initial testing.
For production, place IIS, Azure Application Gateway, or another reverse proxy
in front of QueryBot and expose HTTPS 443 rather than Uvicorn port 8000.

## 7. Run automatically after reboot

Use Windows Task Scheduler to run at system startup:

- Program: `powershell.exe`
- Arguments:
  `-NoProfile -ExecutionPolicy Bypass -File C:\QueryBot\app\deploy\windows\start-querybot.ps1`
- Start in: `C:\QueryBot\app`
- Run whether the user is logged on or not.
- Restart the task on failure.

Use a dedicated Windows service account with Log on as a batch job, read access
to the repository, and write access only to QueryBot data, client, log, and
secret locations.

## 8. Updating

Stop the scheduled task or process, then:

```powershell
Set-Location C:\QueryBot\app
git pull --ff-only
.\venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); from store.db import init_db; init_db(); print('Migration OK')"
powershell -ExecutionPolicy Bypass -File .\deploy\windows\start-querybot.ps1
```

Back up PostgreSQL, the Qdrant volume or cloud collection, and
`C:\QueryBot\secrets\.querybot_key` before upgrading.
