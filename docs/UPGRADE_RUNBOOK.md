# Upgrade runbook: deploy a new build and bring a workspace to ready

For the person with SSH access to the customer's server and admin access to
QueryBot. It takes a running deployment to the current `main`, rotates a
portal password that was exposed, and walks the workspace to the readiness
gate's **Ready for questions**.

Plan on about an hour for a model of around twenty tables. Rediscovery probes
every join against the warehouse, so it takes longer on a large mart.

> **Credentials.** Never paste a production password, key or connection string
> into chat, a ticket or a shell command line. Read them from the customer's
> secret store or environment variables, and use `read -s` when a script needs
> one. `deploy/live_smoke.py` takes its credentials from the environment for
> this reason.

---

## 0. Before you start

- [ ] SSH to the server (the service runs as `azureuser` from
      `/home/azureuser/querybot`; see `querybot.service`).
- [ ] Admin sign-in to `http://<host>:8000/admin`.
- [ ] The workspace's account id (Admin → Clients).
- [ ] The commit to deploy: `main` at `<sha>`. Write it down.
- [ ] A maintenance window agreed with the customer. Readers can keep asking
      questions during most of it, but answers change as the model is settled.

## 1. Take the exposed password out of use (before the deploy)

A portal password was shared in plain text during testing. Treat it as
compromised.

1. Admin → the workspace → **Users**. For each account that used it, press
   **Deactivate** (the eye icon). This takes effect at once, puts no
   password anywhere, and ends every session the account had open.
2. If the admin password was the same one, change it now:
   Admin → **System** → Password. That signs out every other admin session.

Do **not** reset the password on the old build. It puts the new password in
the redirect URL (`temp_pw=…`), and the service's access log writes that URL
to the journal. The build you are deploying shows the password once on the
page instead (step 4).

Find out whether earlier resets already leaked passwords:

```bash
sudo journalctl -u querybot | grep -c 'temp_pw='
```

If this is non-zero, every user created or reset through the admin page has
their password in the journal. Reset those users too in step 4. Clearing the
journal (`journalctl --vacuum-*`) is the customer's IT decision: raise it with
them rather than doing it yourself.

## 2. Record what is running, and back up

```bash
cd /home/azureuser/querybot
git rev-parse HEAD          # the commit running now: your rollback target
git status --short          # local edits on the server? stop and ask before overwriting them
sudo systemctl status querybot --no-pager | head -5

# A consistent copy of the SQLite store, taken while the service runs: data/querybot.db
# unless the service sets QUERYBOT_DB_PATH (use pg_dump if DATABASE_URL points at PostgreSQL).
stamp=$(date +%Y%m%d-%H%M)
python3 -c "import sqlite3; s = sqlite3.connect('data/querybot.db'); d = sqlite3.connect('data/querybot.db.bak-$stamp'); s.backup(d); d.close()"
tar czf ~/clients-bak-$stamp.tgz clients/

ls -la ~/.querybot_key      # losing this key loses every stored credential: make sure a copy exists off the server
```

## 3. Deploy

```bash
git fetch origin main
git checkout <sha>                 # the commit you wrote down
bash deploy.sh                     # installs requirements.lock and restarts the service
curl -s http://localhost:8000/health
sudo journalctl -u querybot -n 80 --no-pager
```

The store upgrades itself at startup (for example, the business-meaning table
and the metric certification columns). There is nothing to run by hand. Read
the startup lines for errors before going on.

## 4. Rotate the password on the new build

1. Admin → the workspace → **Users** → **Reset password** (the circular
   arrow) for each account from step 1.
2. The page shows the temporary password **once**, with no password in the
   URL. Give it to the user over a separate channel (not the one the old
   password leaked on). They must choose their own at first sign-in.
3. **Activate** the account again (the eye icon). Sessions opened before the
   deactivation stay closed: reactivating lets the user sign in again, with
   the new password, and brings back nothing issued under the old one.
4. Check that the old password no longer signs in.

## 5. Settings that must be right before rediscovery

- **Compliance profile** (Admin → the workspace → Compliance → Industry
  profile). Choose **Standard** unless the customer is regulated. A workspace
  with no profile is treated as regulated and fails closed: its value index
  keeps no values, so discovery can read meanings from names only. A country
  code column cannot be recognised by its values, for example. A regulated
  workspace must also confirm the Field Masking review in Setup before
  discovery will run.
- **Source system and industry vocabulary** (Setup). Choose the ERP pack (for
  example Infor M3) and **Wholesale distribution**. The French terms for stock
  questions ("stock disponible par groupe d'articles") come with the
  distribution pack.
- **Tables** (Setup): only the tables questions are about.

## 6. Rediscover

Setup → **Discover schema**. Discovery now:

- probes every suggested join against the data (match, orphan and empty-key
  rates). A key that is empty or orphaned on some rows turns its join LEFT;
- finds each dimension's placeholder members (0, -1, 777, "NULL value
  provided");
- builds the value index, then proposes business meanings with their evidence;
- lists columns of one table that readers see under the same name.

Follow it with `sudo journalctl -u querybot -f`. Look for
`Business meanings proposed for <account>: {...}` and the join profiling
counts.

## 7. Review what discovery found

Do this before the knowledge base is rebuilt: the KB is written with the
readings you confirm here.

1. **Business Meanings** (Data & Model → Business Meanings).
   - **Confirm all strong readings** (confidence 80 or more), then read the
     rest one by one: confirm, edit or reject.
   - **Codes nothing reads**: write what each one means, or leave it as
     spelled.
   - **Columns shown under one name**: write what each one is. On an item
     table with `PDC_GRP_DMS_KEY` and `PRU_GRP_DMS_KEY`, ask the customer what
     PRU is. The PDC one joins the product-group table, so its name stands.
2. **Graph** (Data & Model → Graph): every join the data refuted (broken,
   matching nothing, or a warning), and the ones the data cannot settle.
   Fix, confirm or reject each.

## 8. Rebuild the knowledge base

Setup → **Rebuild Knowledge Base**, with the business description. This also
writes the semantic model, and with it the business dates found on each fact.
Those dates have nothing to review until the rebuild has run.

## 9. Settle the dates and the metrics

1. **Dates** (Data & Model → Dates): approve the date each fact is counted
   on. Where a fact has several, choose its default, for example the balance
   date for a daily balance fact.
2. **Metrics** (Data & Model → Metrics): accept the starter proposals that
   fit, or define the key metrics. Check each one's number against a figure
   the business already trusts (last month's stock value from their own
   report, say), then **Certify** it. Changing a formula, table, grain or date
   later removes the certification.

## 10. The readiness gate

Open Quality → **What To Model Next**. The card at the top must say **Ready
for questions**. Until it does, it lists what is blocking, each item with a
**Fix** link:

| Check | Typical blocker | Where it is fixed |
|---|---|---|
| Checked time axes | A fact has no approved date, or its join to the date table was never checked | Dates, Graph |
| Checked joins | A join is broken, matches nothing, has never been probed, or its tables changed | Graph |
| No ambiguous dates | A fact has several approved dates and no default | Dates |
| Certified metrics | A fact has no metric, or a live metric is not certified | Metrics |

The same verdict is in the `gate` field of
`GET /admin/api/clients/<account_id>/readiness`.

## 11. Smoke test

```bash
python -m deploy.preflight_live <account_id>

read -r QB_EMAIL && read -rs QB_PASSWORD && export QB_EMAIL QB_PASSWORD   # a test user
python deploy/live_smoke.py http://<host>:8000 <account_id> --ask "stock on hand by warehouse"
```

Ask these as an English reader and as a French reader:

| Question | What the answer should show |
|---|---|
| stock on hand by warehouse | Totals per unit of measure, and the card says so |
| top 10 suppliers by stock value | The supplier's name, once the supplier meaning is confirmed; no placeholder member ranked |
| stock disponible par groupe d'articles | French labels where the warehouse has them, English otherwise, with a caveat saying so |
| stock value by month this year | The default date, with no question about which date |
| quantité en stock par fournisseur | The supplier column (once the supplier meaning is confirmed) |

## 12. Rollback

```bash
git checkout <the commit from step 2>
bash deploy.sh
```

The upgrade only adds tables and columns, so the older build runs on the store
as it is, and the review decisions made since are kept. Restore the backup only
if the store itself is the problem, and only with the service stopped
(`sudo systemctl stop querybot`, then
`cp data/querybot.db.bak-<stamp> data/querybot.db`); that also undoes those
decisions.

## 13. If the admin password is lost

The setup page runs once: after the first admin password is saved, it is
closed, even when the stored password can no longer be read because
`~/.querybot_key` changed (the sign-in page then says so). Set a new password
on the server, as the service's user, from the application directory:

```bash
cd /home/azureuser/querybot
venv/bin/python -m admin.reset_password     # asks twice, without echo
```

It also signs out every admin session and ends any wait that failed sign-ins
built up (section 14).

If the service sets `QUERYBOT_DB_PATH`, `DATABASE_URL` or `QUERYBOT_KEY_FILE`,
export the same values first, or the command writes to a different store.

## 14. HTTPS, the reverse proxy and the question API

Session cookies are marked Secure, so they travel over https only, when the
request arrived over https or `PORTAL_BASE_URL` starts with `https://`. Behind
a proxy that ends TLS (nginx, an Azure Application Gateway or Front Door), set
`PORTAL_BASE_URL` to the public https address, and let uvicorn trust the
proxy's forwarded headers so it also sees the real scheme and client address:

```ini
ExecStart=/home/azureuser/querybot/venv/bin/uvicorn main:app \
    --host 127.0.0.1 --port 8000 --workers 1 \
    --proxy-headers --forwarded-allow-ips=127.0.0.1
```

Use the proxy's own address in `--forwarded-allow-ips` if it runs on another
machine. The admin console counts failed sign-ins per client address: without
the forwarded headers every attempt seems to come from the proxy, and one
person guessing makes everyone wait.

The proxy must pass the `Host` header through (nginx: `proxy_set_header Host
$host;`). Writes to /admin and /portal, and the chat socket's handshake, are
refused when the browser says they came from another site; a browser too old
to say so is judged by comparing its `Origin` with `Host`.

"Too many attempts" at sign-in: five failures in fifteen minutes are free,
then each doubles the wait, up to fifteen minutes. The wait ends by itself; a
password reset by an admin ends a portal user's, and
`python -m admin.reset_password` ends the admin console's.

`POST /api/ask` (for Copilot Studio or Power Automate) is off until
`QUERYBOT_API_KEY` is set in the service's environment; callers send the same
value as `api_key`. It answers without per-user restrictions, so keep the key
as carefully as the admin password.
