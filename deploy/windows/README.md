# Solviz on a Windows server

Windows Server 2019, 2022 or 2025, 64-bit. Everything below runs in an **Administrator** PowerShell, from the
Solviz folder. The scripts keep their `querybot` names so servers already running need nothing renamed.

## Install (once)

```powershell
git clone https://github.com/sud-sn/Querybot_v2.git C:\Solviz
cd C:\Solviz
powershell -ExecutionPolicy Bypass -File deploy\windows\deploy.ps1 -AdminEmail you@company.com -AdminName "Your Name"
```

`deploy.ps1` does every step, and is safe to run again to repair a server that is missing a piece:

1. **Python 3.11 or 3.12**, 64-bit: kept when installed, else installed for all users with winget, else from
   python.org.
2. **Microsoft ODBC Driver 18 for SQL Server** and the Visual C++ runtime it needs, for Azure SQL and SQL Server
   data sources: winget, else Microsoft's download. When neither can be reached it says so and carries on; download
   the x64 installer from Microsoft and run again with `-OdbcMsi C:\path\msodbcsql.msi`.
3. **The libraries** (`install-querybot.ps1`): the virtual environment `venv` with the exact versions in
   `requirements-windows.lock`, the name recogniser's English model, and Chromium for pictures and PDFs of
   scheduled dashboards.
4. **Settings** in `.env`: the session and pseudonym secrets, generated once and never changed, and the path of the
   key that encrypts saved passwords (`%USERPROFILE%\.querybot_key`).
5. **The background task** "QueryBot" (`querybot-service.ps1 -Action Install`): it runs as the Windows user who
   installed it, starts with the server whether or not anyone is signed in, and starts Solviz again if it stops.
   Then a Windows Firewall rule for the port, and a health check.
6. **The first Super admin account**, with `-AdminEmail` (it asks for the password).

| Option | Does |
|---|---|
| `-Port 8001` | The port (default 8000, or the one the task already uses) |
| `-PublicUrl https://solviz.company.com` | The address people open, used in emailed links |
| `-AdminEmail`, `-AdminName` | Make (or reset) a Super admin account |
| `-OdbcMsi <file>` | Install the ODBC driver from a file you downloaded |
| `-NoOdbc` | Leave out the ODBC driver |
| `-NoFirewall` | Add no firewall rule |

If the server cannot download installers at all, install Python 3.12 (64-bit, "Add to PATH"), the Visual C++
runtime and ODBC Driver 18 by hand first ([STEP-BY-STEP.md](STEP-BY-STEP.md)), then run `deploy.ps1`.

## Run it

```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\querybot-service.ps1 -Action Status   # running? /health?
powershell -ExecutionPolicy Bypass -File deploy\windows\querybot-service.ps1 -Action Logs     # follow the newest log
powershell -ExecutionPolicy Bypass -File deploy\windows\querybot-service.ps1 -Action Restart  # after changing .env
powershell -ExecutionPolicy Bypass -File deploy\windows\verify-querybot.ps1                   # check every piece
```

Logs are in `logs\` (kept 14 days).

## Update

```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\update-querybot.ps1                  # the release (main)
powershell -ExecutionPolicy Bypass -File deploy\windows\update-querybot.ps1 -Branch core-v2  # the next version
```

It stops Solviz, copies the SQLite database (`data\querybot.db.bak-<time>`), moves to the latest commit of the
branch, installs the exact library versions it needs, starts Solviz again and prints the line that goes back to
where the server was. Files changed on the server are kept unless the update changes them too: git then stops,
nothing changes, and Solviz starts as it was.

## HTTPS

Solviz serves plain HTTP on its port. Put HTTPS in front before real data goes in: Azure Application Gateway or
Front Door, or IIS with Application Request Routing. Forward to `http://<server>:8000`, allow WebSocket upgrades (the
chat uses one), keep the original `Host` header, and allow uploads of 100 MB. Then run `deploy.ps1` again with
`-PublicUrl https://<your address>` (or set `PORTAL_BASE_URL` in `.env`) and restart.

## Back up

Together: `.env`, `%USERPROFILE%\.querybot_key` (without it saved data-source passwords cannot be read),
`data\querybot.db`, `clients\` and `workbooks\`. To use PostgreSQL for Solviz's own store instead of SQLite, set
`DATABASE_URL` in `.env` and back that database up with `pg_dump`.
