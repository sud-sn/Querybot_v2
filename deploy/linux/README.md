# Solviz on a Linux server

Ubuntu 22.04 or 24.04, or Debian 12. The scripts keep their `querybot` names so servers already running need nothing
renamed. Every script here is safe to run again.

## Install (once)

```bash
git clone https://github.com/sud-sn/Querybot_v2.git solviz
cd solviz
sudo bash deploy/linux/deploy.sh --admin-email you@company.com --admin-name "Your Name"
```

`deploy.sh` does every step, and repairs a server that is missing a piece when run again:

1. **System packages**: Python 3.11 or 3.12 with `venv`, git, curl, unixODBC, and Microsoft's ODBC Driver 18 for SQL
   Server (`install_linux_odbc18.sh`; leave it out with `--no-odbc`).
2. **The libraries**: `./venv` with the exact versions in `requirements.lock`. PyTorch is installed as its CPU build
   first (the default build brings about 3 GB of GPU libraries Solviz does not use); if PyTorch's download site
   cannot be reached, the default build comes from PyPI instead.
3. **The name recogniser's English model** (`en_core_web_md`) and **Chromium**, with the system libraries it needs,
   for pictures and PDFs of scheduled dashboards.
4. **The service** (`install-service.sh`, below), then a health check.
5. **The first Super admin account** with `--admin-email` (it asks for the password), and **HTTPS** with
   `--domain solviz.company.com --tls-email you@company.com`: nginx in front with a Let's Encrypt certificate that
   renews itself, Solviz listening only on this server. The domain must already point at the server, with ports 80
   and 443 open.

Other options: `--port`, `--user` (the user the service runs as; default the one who ran sudo), `--name` (the
service's name), `--public-url`, `--skip-system` (install no system packages). `sudo bash deploy/linux/deploy.sh
--help` lists them.

## The service

`install-service.sh` (run by `deploy.sh`; run it alone to change the port, address or user):

```bash
sudo bash deploy/linux/install-service.sh --port 8000 --user <the user Solviz runs as>
sudo bash deploy/linux/install-service.sh --host 127.0.0.1 --set PORTAL_BASE_URL=https://solviz.company.com
```

- **Settings** live outside the repository, in `/etc/querybot/querybot.env`
  (root only, mode 600). They are only ever added to, never changed:
  - the four session and pseudonym secrets (`SESSION_SECRET`,
    `PORTAL_SESSION_SECRET`, `ADMIN_SESSION_SECRET`, `PII_PSEUDONYM_SECRET`),
    random, created once. Sign-ins then survive a restart and pseudonyms
    stay the same. Never change `PII_PSEUDONYM_SECRET` once it is set;
  - `QUERYBOT_KEY_FILE`: the key that encrypts stored passwords (the run
    user's `~/.querybot_key` unless already set);
  - anything given with `--set NAME=VALUE`, and any `Environment=` line of the
    unit it replaces, such as `PORTAL_BASE_URL`.
  Add your own settings (`DATABASE_URL` and so on) to this file with
  `sudo nano`, then `sudo systemctl restart querybot`.
- **The unit** `/etc/systemd/system/querybot.service` starts Solviz at boot
  on one worker, restarts it after a crash, and logs to the journal. The unit
  it replaces is kept beside it as `querybot.service.bak-<time>`.
- **The port and address** are the ones given; else the current unit's; else
  port 8000 on every address (`--host 127.0.0.1` when a proxy on this server is
  the only way in).

```bash
sudo systemctl status querybot
sudo journalctl -u querybot -f
sudo systemctl restart querybot
```

## Update

As the user the service runs as (it asks for sudo only to restart):

```bash
bash deploy/linux/update.sh
```

It copies the SQLite database (`data/querybot.db.bak-<time>`, taken with
SQLite's backup so writes not yet folded into the file are kept), sets the
local branch to the branch on GitHub, installs `requirements.lock` (PyTorch as
its CPU build), installs the browser that draws scheduled dashboards as pictures and PDF and the name
recogniser's English model (`en_core_web_md`, once), restarts the
service, checks `/health` and shows any error logged since. Files changed on
the server are kept unless the update changes them too: git then stops and
nothing changes. The line to go back to where the server was is printed at
the end.

It takes `main`, the release. The next version is built on `core-v2`:
`bash deploy/linux/update.sh --branch core-v2` moves a test server to it, and
the plain command moves it back.

If it says Chromium will not start, install its system libraries once:

```bash
sudo venv/bin/python -m playwright install-deps chromium
```

## Learn on a large or busy database

Learn reads four tables at a time and lets each query run for two minutes. A
table of more than a million rows that the database cannot read whole in that
time is read on a sample of about a million rows, and the progress log says so. If QueryBot restarts while Learn
runs (an update, a reboot), press Learn again within a day: it starts where it
was and reads again only the tables whose columns, size or masking changed.

On a small database that others use during the day, read fewer tables at once
and give each query longer, in `/etc/querybot/querybot.env`:

```
QUERYBOT_LEARN_WORKERS=2
QUERYBOT_LEARN_QUERY_TIMEOUT_SECONDS=300
```

then restart QueryBot. On Azure SQL or SQL Server, `deploy/sql/columnstore_suggestions.sql`
lists the large tables that would answer faster with a columnstore index, with the
statement to create each. It changes nothing itself: run it in the database
QueryBot reads and agree the indexes with whoever runs that database.

## Back up

Keep `/etc/querybot/querybot.env` and the key file together: without the key,
stored database passwords cannot be read. Back up `data/querybot.db` (or
PostgreSQL with `pg_dump` when `DATABASE_URL` is set), `clients/` and
`workbooks/`. `venv/bin/python deploy/backup_db.py` takes a consistent copy of
the SQLite database while Solviz runs.
