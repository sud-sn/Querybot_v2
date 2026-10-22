# Solviz

Solviz answers business questions about a company's own data, in plain English or French, and builds dashboards,
scheduled reports and alerts from those answers. It connects read-only to the company's database, learns what the
tables mean, and answers through a governed path: the AI writes a plan, code checks the plan and writes the SQL, and
every number comes from the database.

> The product is called **Solviz** everywhere its users and admins see it. The code, the service, the
> `QUERYBOT_*` settings, the `querybot.db` store and the scripts in `deploy/` keep the name `querybot`, so a server
> already running needs nothing renamed.

## Contents

- [What you need](#what-you-need)
- [Install on Linux](#install-on-linux)
- [Install on Windows](#install-on-windows)
- [After installing](#after-installing)
- [Updating](#updating)
- [Settings](#settings)
- [HTTPS](#https)
- [Backups and restore](#backups-and-restore)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [Security and access](#security-and-access)
- [Development](#development)
- [Project structure](#project-structure)
- [Known limits](#known-limits)

## What you need

**A server**

| | Minimum | Recommended |
|---|---|---|
| Operating system | Ubuntu 22.04 or 24.04, Debian 12; or Windows Server 2019, 2022 or 2025 | Ubuntu 24.04 |
| CPU and memory | 2 vCPU, 8 GB | 4 vCPU, 16 GB |
| Free disk | 15 GB | 30 GB |
| Python | 3.11 or 3.12, 64-bit (the installers add it when missing) | 3.11 |

The installers set up everything else: Microsoft's ODBC Driver 18 for SQL Server, the Python libraries at the exact
versions the tests run on (`requirements.lock` on Linux, `requirements-windows.lock` on Windows), the name
recogniser's English model, and Chromium for pictures and PDFs of scheduled dashboards.

**Accounts and access**

- **An AI service.** An API key for Azure OpenAI (a `gpt-4o` deployment is what the product is tested with),
  Anthropic or OpenAI; or a model you host yourself behind an OpenAI-compatible endpoint. One question uses about
  14,000 tokens; size the deployment's tokens-per-minute for the number of people asking at once (about 150,000 per
  minute for 8 to 10 people asking in the same minute).
- **A read-only account on each data source:** Azure SQL or SQL Server, Snowflake (key pair or PAT), Oracle,
  PostgreSQL, or an Excel workbook (uploaded, or a SharePoint/OneDrive link).
- **Optional:** Microsoft 365 (Graph) or an SMTP server for scheduled emails and alerts; a domain name and
  certificate for HTTPS.

**Network**

- During install and updates the server reaches: `pypi.org` and `files.pythonhosted.org`, `download.pytorch.org`,
  `github.com` (the name recogniser's model), Playwright's browser download, and on Linux `packages.microsoft.com`
  (the ODBC driver).
- While running: the AI service, the data sources, and the mail service. People reach the server on its port (8000
  by default), or on 443 behind HTTPS.

## Install on Linux

On a fresh Ubuntu or Debian server, as a user with sudo:

```bash
git clone https://github.com/sud-sn/Querybot_v2.git solviz
cd solviz
sudo bash deploy/linux/deploy.sh --admin-email you@company.com --admin-name "Your Name"
```

One command does all of it, and is safe to run again:

1. System packages: Python 3.11 or 3.12 with `venv`, git, curl, unixODBC, and Microsoft's ODBC Driver 18.
2. The virtual environment `./venv` with the exact library versions. PyTorch comes as its CPU build, which saves
   about 3 GB of GPU libraries Solviz never uses.
3. The name recogniser's model and Chromium, with the system libraries Chromium needs.
4. The service: settings in `/etc/querybot/querybot.env` (root only), its secrets generated once and never changed;
   a systemd unit that starts at boot and restarts after a crash; a health check.
5. The first Super admin account (it asks for the password), and HTTPS when you give a domain.

Options:

| Option | Does |
|---|---|
| `--port 8001` | The port Solviz listens on (default 8000) |
| `--user solviz` | The Linux user the service runs as (default: the user who ran sudo) |
| `--domain solviz.company.com --tls-email you@company.com` | HTTPS through nginx with a Let's Encrypt certificate; Solviz then listens only on this server |
| `--public-url https://...` | The address people open, used in emailed links |
| `--admin-email` and `--admin-name` | Make (or reset) a Super admin account |
| `--no-odbc` | Leave out the ODBC driver (no Azure SQL or SQL Server sources) |
| `--skip-system` | Install no system packages, for a distribution the script does not know (install them first) |

Run as a service, Solviz is managed with `sudo systemctl status|restart|stop querybot`, and its logs are read with
`sudo journalctl -u querybot -f`.

## Install on Windows

In an **Administrator** PowerShell:

```powershell
git clone https://github.com/sud-sn/Querybot_v2.git C:\Solviz
cd C:\Solviz
powershell -ExecutionPolicy Bypass -File deploy\windows\deploy.ps1 -AdminEmail you@company.com -AdminName "Your Name"
```

(Without git, install it first: `winget install Git.Git`, or download it from git-scm.com.)

One command does all of it, and is safe to run again:

1. Python 3.11 or 3.12, if missing, for all users (winget, else python.org).
2. Microsoft's ODBC Driver 18 for SQL Server and the Visual C++ runtime it needs (winget, else Microsoft's download).
   If neither works, it says so and carries on; download the driver yourself and run again with
   `-OdbcMsi C:\path\msodbcsql.msi`.
3. The virtual environment `venv` with the exact library versions, the name recogniser's model, and Chromium.
4. `.env` in the Solviz folder, its secrets generated once and never changed.
5. A background task, "QueryBot", that starts with the server and restarts Solviz if it stops; a Windows Firewall
   rule for the port; a health check.
6. The first Super admin account (it asks for the password).

Options: `-Port 8001`, `-PublicUrl https://...`, `-AdminEmail` and `-AdminName`, `-OdbcMsi <file>`, `-NoOdbc`,
`-NoFirewall`.

Manage it with `deploy\windows\querybot-service.ps1 -Action Status|Restart|Stop|Start|Logs`. Logs are in `logs\`.

## After installing

1. Open `http://<server>:8000/admin` and sign in with the admin account you made. Without `--admin-email` /
   `-AdminEmail`, the first person to open `/admin` sets the admin password: do that straight away, or run the
   installer again with the option.
2. **System → AI provider**: choose the provider and paste its key. Keys are stored encrypted.
3. **Workspaces → New workspace**: connect the data source with its read-only account, run discovery, then **Learn**. Learn
   studies the tables, links, dates and metrics, and takes a few minutes on a large database.
4. Check what Learn found on the workspace's data pages (Relationships, Metrics, Dates, Knowledge base) and correct
   anything that is wrong. Your decisions are kept through every later Learn.
5. In the workspace, **People**: add users with a role (Viewer, Asker or Creator), their group and tables, and any row rules.
   **Compliance**: choose Standard or Regulated, and the data posture (Cloud, Private or Air-gapped).
6. Optional: **System → Email** (Microsoft 365 or SMTP) for scheduled dashboards and alerts.

People then sign in at `http://<server>:8000/portal/login?workspace=<workspace id>`.

## Updating

Linux, as the user the service runs as:

```bash
cd ~/solviz && bash deploy/linux/update.sh                    # the release (main)
bash deploy/linux/update.sh --branch core-v2                  # the next version, to try it
```

Windows, in an Administrator PowerShell:

```powershell
cd C:\Solviz; powershell -ExecutionPolicy Bypass -File deploy\windows\update-querybot.ps1
```

Each update copies the Solviz database first (SQLite: `data/querybot.db.bak-<time>`), moves to the latest commit,
installs the exact library versions that commit needs, restarts, checks health, and prints the line that goes back
to where the server was. When the release notes say so, run **Learn** again on each workspace; your decisions are
kept.

## Settings

Linux keeps them in `/etc/querybot/querybot.env` (root only); Windows in `.env` in the Solviz folder. The
installers only ever add to the file. Restart after changing it. Most configuration (AI keys, data sources, email,
people) is done in the admin console and stored encrypted, not here.

| Setting | Meaning |
|---|---|
| `PORTAL_BASE_URL` | The address people open, used in emailed links. An `https://` address also makes sign-in cookies HTTPS-only. |
| `DATABASE_URL` | Solviz's own store in PostgreSQL (`postgresql://user:pass@host:5432/db`). Unset, it is SQLite in `data/querybot.db`, right for one server. |
| `QUERYBOT_KEY_FILE` | The key that encrypts saved passwords and keys. Back it up with the settings file. |
| `SESSION_SECRET`, `PORTAL_SESSION_SECRET`, `ADMIN_SESSION_SECRET` | Sign the session cookies. Generated by the installer; changing one signs everyone out. |
| `PII_PSEUDONYM_SECRET` | Keys the aliases shown for masked people. Never change it once set. |
| `QUERYBOT_QUERY_TIMEOUT_SECONDS` | The time limit for one query (default 300). |
| `LLM_AUDIT_RETENTION_DAYS`, `KB_EGRESS_RETENTION_DAYS` | How long the AI call log (default 30) and data-egress log (default 365) are kept. |
| `QUERYBOT_WORKBOOKS_DIR` | Where uploaded Excel workbooks are kept (default `workbooks/`). |

## HTTPS

Do not put real data on a plain-HTTP address.

- **Linux:** `sudo bash deploy/linux/deploy.sh --domain solviz.company.com --tls-email you@company.com`. The domain's
  DNS must already point at the server, with ports 80 and 443 open. nginx passes the chat's WebSocket, takes 100 MB
  uploads, and waits up to ten minutes for slow databases; the certificate renews itself.
- **Windows, or any cloud gateway** (Azure Application Gateway, Front Door, IIS with ARR): end TLS there, forward to
  `http://<server>:8000` with WebSocket upgrades allowed and the original `Host` header kept, then set
  `PORTAL_BASE_URL=https://...` (`-PublicUrl` on Windows) and restart.

## Backups and restore

Back up these together:

| What | Linux | Windows |
|---|---|---|
| Solviz's store | `data/querybot.db` (or your PostgreSQL database) | `data\querybot.db` |
| Workspace files and uploads | `clients/`, `workbooks/` | `clients\`, `workbooks\` |
| Settings | `/etc/querybot/querybot.env` | `.env` |
| Encryption key | the `QUERYBOT_KEY_FILE` path (`~/.querybot_key` of the service user) | `%USERPROFILE%\.querybot_key` |

Without the key file, saved data-source passwords cannot be read. For a consistent SQLite copy while running, use
`venv/bin/python deploy/backup_db.py` (it uses SQLite's own backup); for PostgreSQL, `pg_dump`. To restore: stop the
service, put the files back, start it.

## Troubleshooting

| Seen | Do |
|---|---|
| The page does not open from another machine | Open the port in the cloud firewall (network security group) and the server's firewall. `curl -s http://localhost:8000/health` on the server shows whether Solviz itself is up. |
| `/health` says `kb_rebuild_needed` | Open that workspace in the admin console and run Learn (and the knowledge-base rebuild, if offered). |
| Answers fail with a rate-limit message | The AI deployment's tokens-per-minute is too low for the number of people asking. Raise it. |
| Azure SQL or SQL Server will not connect | ODBC Driver 18 is missing: run the installer again (Windows: with `-OdbcMsi`), and allow the server's IP in the database firewall. |
| Scheduled emails arrive without pictures or PDF | Chromium is missing or cannot start: run the installer again. On Linux: `sudo venv/bin/python -m playwright install-deps chromium`. |
| The compliance page says the name recogniser is off | `venv/bin/python -m spacy download en_core_web_md` (Windows: `venv\Scripts\python.exe`), then restart. |
| Locked out of the admin console | On the server, as the service user: `python -m admin.create_admin you@company.com "Your Name"` with the service's settings, or `python -m admin.reset_password`. |

## How it works

```text
People ── portal, admin console, Teams/Slack/Zoom, email
   │
Access gate ── sign-in, role and workspace, checked for every page, API call and chat message
   │
Learn ─── studies the database: tables, links, dates, metrics, names
Answer ── question → plan (AI) → checked → SQL (code) → run under the person's access → answer, chart, notes
Dashboards, scheduled emails, alerts ── saved plans run again on live data, with no AI
   │
Governed executor ── read-only check, time limit, the person's tables, row rules and masked columns
   │
Your databases (read-only)        AI service (gets names and the question, returns a plan)
```

- The AI returns a structured plan, not SQL; code checks every name in it against the model Learn built, compiles
  SQL for the database's dialect, and runs it read-only.
- One exception, off unless an admin turns it on in Settings ("Answer with a direct query when no analysis fits"):
  when no plan can express a question, the AI writes one query from the approved tables, links and metrics only.
  Code checks it before it runs (one read-only SELECT; only the tables and fields the person may see; only approved
  links; nothing added up twice; default filters and a row limit put on by code), and it runs through the same
  governed executor. The answer is marked "Direct query", only Creators see the query, and it cannot be added to a
  dashboard or a schedule. A query that breaks a rule never runs.
- Every answer shows the period, the dates and every condition it applied, and "How it was counted" shows the query.
- More: [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/core-v2/DESIGN.md](docs/core-v2/DESIGN.md).

## Security and access

- **Roles.** Portal users are Viewers (open shared dashboards), Askers (also ask, export, save their own dashboards)
  or Creators (also build, share and schedule dashboards). Admins are Super admins, Workspace admins or Data
  stewards. A role never widens data access.
- **Data access.** Groups decide which tables a person sees; row rules limit rows by a fixed value or the person's own
  attribute (their region, for example), added inside the query; personal columns are masked unless released by a
  recorded confidentiality attestation.
- **The AI boundary.** The AI is given names and the question, never rows to plan from. In a Regulated workspace no
  data value reaches it at all; elsewhere the optional summary under an answer is written from the answer as shown.
  People's names typed in a question are found on the server and replaced before anything is sent.
- **Records.** Admin changes, governed queries (hash-chained in Regulated workspaces), every AI call with its cost,
  data sent to the AI, exports and emails are logged.
- **Sign-in.** Salted PBKDF2 password hashes, signed HttpOnly cookies, cross-site requests refused, slower tries after
  repeated wrong passwords. Single sign-on and two-factor sign-in are not built yet.

## Development

```bash
python3.11 -m venv venv
venv/bin/pip install "torch==$(sed -nE 's/^torch==([^ ;]+).*/\1/p' requirements.lock)" --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -r requirements.lock -r requirements-dev.txt
venv/bin/python -m spacy download en_core_web_md
venv/bin/uvicorn main:app --reload --port 8000
```

| Check | Command |
|---|---|
| Tests (about 22,000; every test uses a temporary database) | `venv/bin/python -m pytest -q` |
| Lint | `venv/bin/ruff check .` |
| Types | `venv/bin/mypy` |
| Demo rehearsal, English and French | `venv/bin/python -m evals.sample_rehearsal --strict` |

The dashboard charts are Vue components in `frontend/`, built into `static/dist/` (committed). After changing them:
`cd frontend && npm ci && npm run build`.

Changing a library version: edit `requirements.txt`, regenerate `requirements.lock` and
`requirements-windows.lock` (each file's header says how), and let the full suite pass on the result.

## Project structure

```text
main.py            The web application: routes, startup, the background scheduler
admin/             Admin console: workspaces, Learn, data pages, people, compliance, email, AI settings
portal/            The users' portal: chat, dashboards, data guide, notifications, settings
gateway/           Teams, Slack and Zoom webhooks, the portal chat WebSocket, the question API
core2/             The engine: plan/ (the AI's plan), resolve/ and compile/ (checks and SQL), answer/,
                   bootstrap/ (Learn), warehouse/ (governed execution), dashboards, recipes, agent/
core/              Shared services: data-source connections, compliance (policies, masking, row rules),
                   mail, scheduled dashboards and alerts, AI providers and their audit
store/             Solviz's own store (SQLite or PostgreSQL): people, workspaces, decisions, logs
static/, frontend/ Styles, scripts and the Vue chart components
deploy/            Installers and update scripts for Linux and Windows, backup and settings helpers
tests/             The test suite
evals/             Question sets and the demo rehearsal
docs/              Design notes and runbooks
```

## Known limits

- Answers depend on what Learn found and the admins approved: links between tables, how each metric adds up,
  and which date each table is counted by. Review the data pages after the first Learn.
- Requests that combine several analyses in one sentence, shares within a group, and "did A but not B" between two
  periods are still being improved; Solviz says what it could not do rather than guess.
- One server process serves everyone; the AI service's tokens-per-minute is the usual limit on how many can ask at
  once.
- SQLite suits one server. Use PostgreSQL for Solviz's own store when you need managed backups or high availability.
- No single sign-on or two-factor sign-in yet; HTTPS has to be put in front (see [HTTPS](#https)).
- Compliance features reduce risk; they do not on their own establish regulatory compliance.

No open-source license is declared in this repository. Treat the code as proprietary unless the repository owner
provides separate licensing terms.
