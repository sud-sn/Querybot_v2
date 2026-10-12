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

## 7. Run in the background, start with the server

Once QueryBot answers on its port, stop the console run (Ctrl+C) and install the
background task, in an Administrator PowerShell, signed in as the Windows user
QueryBot runs as:

```powershell
Set-Location C:\QueryBot\app
powershell -ExecutionPolicy Bypass -File .\deploy\windows\querybot-service.ps1 -Action Install -Port 8000
```

This:

- adds to `.env`, only where there is no value yet, the four session and
  pseudonym secrets (random), `QUERYBOT_KEY_FILE` (this user's
  `.querybot_key`), and where this user's PDF browser and AI model files are.
  A value already in `.env` is never changed. Never change
  `PII_PSEUDONYM_SECRET` once it is set: every alias shown before would change;
- stops a QueryBot already running on that port from a console;
- registers the scheduled task `QueryBot`. It runs
  `deploy\windows\run-querybot.ps1` with no window, as this user, at every
  start of the server whether or not anyone is signed in, and starts QueryBot
  again ten seconds after it stops for any reason. A click or Ctrl+C in a
  window can no longer pause or end it.

Logs go to `C:\QueryBot\app\logs\querybot-<start time>.log`, kept 14 days.

```powershell
.\deploy\windows\querybot-service.ps1 -Action Status    # task, process, /health, newest log
.\deploy\windows\querybot-service.ps1 -Action Logs      # newest log, followed live
.\deploy\windows\querybot-service.ps1 -Action Restart   # after changing .env
.\deploy\windows\querybot-service.ps1 -Action Stop      # also Start, Uninstall
```

(Prefix each with `powershell -ExecutionPolicy Bypass -File` if scripts are
blocked on the server.) The user needs Log on as a batch job, read access to
the repository, and write access to QueryBot's data, client, log and secret
locations.

## 8. Updating

In an Administrator PowerShell:

```powershell
Set-Location C:\QueryBot\app
powershell -ExecutionPolicy Bypass -File .\deploy\windows\update-querybot.ps1 -Branch core-v2
```

It stops QueryBot, copies the SQLite database (`data\querybot.db.bak-<time>`,
with the writes not yet folded into the file), sets the local branch to the
branch on GitHub, installs `requirements-windows.txt` and the PDF browser,
starts QueryBot again and checks it. Files changed on the server are kept
unless the update changes them too: git then stops and nothing changes. The
line to go back to where the server was is printed at the end.

Back up PostgreSQL (when `DATABASE_URL` is set), the Qdrant volume or cloud
collection, `.env` and the `.querybot_key` file before upgrading. Keep `.env`
and the key file together: without the key, stored database passwords cannot
be read.
