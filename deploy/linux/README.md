# QueryBot on a Linux server

Two scripts run QueryBot as a systemd service and keep it up to date. Both are
safe to run again.

## Run as a service (once)

From the repository on the server, with a virtual environment in `venv/`:

```bash
sudo bash deploy/linux/install-service.sh --port 8000 --user <the user QueryBot runs as>
```

- **Settings** live outside the repository, in `/etc/querybot/querybot.env`
  (root only, mode 600). They are only ever added to, never changed:
  - the four session and pseudonym secrets (`SESSION_SECRET`,
    `PORTAL_SESSION_SECRET`, `ADMIN_SESSION_SECRET`, `PII_PSEUDONYM_SECRET`),
    random, created once. Sign-ins then survive a restart and pseudonyms
    stay the same; left unset, every start made new ones. Never change
    `PII_PSEUDONYM_SECRET` once it is set;
  - `QUERYBOT_KEY_FILE`: the key that encrypts stored passwords (the run
    user's `~/.querybot_key` unless already set);
  - any `Environment=` line of the unit it replaces, such as
    `PORTAL_BASE_URL`.
  Add your own settings (`AZURE_OPENAI_*`, `DATABASE_URL` and so on) to this
  file with `sudo nano`, then `sudo systemctl restart querybot`.
- **The unit** `/etc/systemd/system/querybot.service` starts QueryBot at boot
  on one worker, restarts it after a crash, and logs to the journal. The unit
  it replaces is kept beside it as `querybot.service.bak-<time>`.
- **The port** is the one given; else the one the current unit runs on; else
  8000.

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
local branch to the branch on GitHub, installs `requirements.lock`, installs
the browser that draws scheduled dashboards as pictures and PDF, restarts the
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
PostgreSQL with `pg_dump` when `DATABASE_URL` is set) and `clients/`.
