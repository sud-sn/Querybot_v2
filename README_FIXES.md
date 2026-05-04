# QueryBot v18.7 — Clarification Loop Hardened

This is your v18.7 codebase with the clarification loop hardened. Unzip,
drop onto your VM (or `cp -r` over your existing working copy), and run
the normal deploy path.

## What's different from vanilla v18.7

Six fixes, all scoped to the clarification loop. See `CHANGES.md` for the
full rationale per fix.

| # | Fix                                    | Files touched                               |
|---|----------------------------------------|---------------------------------------------|
| 1 | LLM constrained-menu ambiguity check   | `core/clarification.py`                     |
| 2 | `selected_option_id` passthrough       | `core/clarification.py`, `main.py`          |
| 3 | Zero-rows clarification gate           | `main.py`                                   |
| 5 | Tolerant JSON parsing of LLM responses | `core/clarification.py`                     |
| 7 | "Your clarification expired" hint      | `core/clarification.py`, `main.py`          |
| 8 | Webhook idempotency                    | `core/webhook_dedup.py` (new), `main.py`    |
| 9 | WebSocket free-text clarifications     | `main.py`                                   |

## How to apply

You have two paths.

### Path A — replace the whole tree
If you haven't made local changes since v18.7:
```bash
# on the VM
sudo systemctl stop querybot
cd /home/azureuser
mv querybot querybot.v18.7.bak
unzip querybot_v18_7_clarification_hardened.zip
mv querybot_v18_7_clarification_hardened querybot
sudo systemctl start querybot
```

### Path B — cherry-pick the changed files
If you have local changes you want to keep, only four files are new or modified:
```
core/clarification.py          (rewritten — replace)
core/webhook_dedup.py          (new file — add)
main.py                        (surgical edits — see CHANGES.md)
tests/test_clarification_fixes.py  (new tests — add)
```

## Verify it works

From the project root after unzip:
```bash
python3 -c "import ast; [ast.parse(open(f).read()) for f in ('main.py','core/clarification.py','core/webhook_dedup.py')]"
python3 -m unittest tests.test_clarification_fixes
```

Both should succeed. The unit tests cover tolerant JSON parsing, option-id
passthrough, expiry grace trail, webhook dedup, and the LLM constrained
menu path with mocked LLM responses — no live API calls.

## Important caveats — read before shipping

1. **Tests were written but not executed by me.** I drafted them and ran
   syntax checks on all three modified files, but I ran out of tool
   budget before running `python -m unittest`. Run them yourself first.

2. **No database schema changes.** Every fix is code-only. Existing
   `pending_clarification` rows keep working. No migration needed.

3. **Fix #7 uses in-process state.** The "recently expired" trail is a
   module-level dict. Fine for the current single-worker systemd setup.
   If you ever scale to multiple uvicorn workers, move this to SQLite or
   Redis or the grace hint will be inconsistent across workers.

4. **Fix #8 uses in-process state.** Same caveat — dedup cache is per
   process. A duplicate webhook routed to a different worker won't be
   caught. Single worker today, so not an issue.

5. **Fix #3 changes user-facing behaviour.** Previously every empty
   result set triggered an LLM ambiguity check. Now it only does so when
   the question matched ambiguous glossary terms or multiple metrics.
   Other empty results get a plain "no rows — try broadening the filter"
   message. This is the intended behaviour but worth knowing if support
   gets a ticket asking why the bot stopped asking follow-up questions
   on empty results.

6. **The admin/portal/security items from your earlier production review
   are NOT in this drop.** This zip only hardens the clarification loop.
   Admin password hashing, CSRF, pin-refresh ACL, Teams JWT etc. are
   still as they were in v18.7.

## Rolling back

The new clarification logic reads the same `clarification_meta` JSON shape
as v18.7, so rollback is just:
```bash
sudo systemctl stop querybot
rm -rf /home/azureuser/querybot
mv /home/azureuser/querybot.v18.7.bak /home/azureuser/querybot
sudo systemctl start querybot
```
No DB cleanup required.
