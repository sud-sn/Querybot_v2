# QueryBot v2 — Staged Rollout Guide

> **Living document** — update this file alongside ARCHITECTURE.md, README.md,
> and ARCHITECTURE_VISUAL.html whenever a new flag or rollout step is added.
>
> **Cross-links**:
> - Full feature reference → [README.md](README.md)
> - Technical deep-dive    → [ARCHITECTURE.md](ARCHITECTURE.md)
> - Visual diagram         → [ARCHITECTURE_VISUAL.html](ARCHITECTURE_VISUAL.html)

---

## Overview

The learning loop and Genie engine are **additive, backward-compatible features**
controlled by two per-tenant database flags stored in the `client` table.
Every existing tenant continues to operate as before with both flags off.

```
Feature flags in client table:
  enable_feedback_collection   0|1   (default: 0)
  enable_genie_suggestions     0|1   (default: 0)
```

The flags are **independent** — you can enable either one without the other.
The recommended order below maximises signal quality before exposing ranking to users.

---

## Rollout Steps

### Step 0 — Pre-flight checks

Before enabling anything, verify the deployment is healthy:

```bash
# 1. All tables exist
python -c "from store.db import init_db; init_db(); print('DB OK')"

# 2. Qdrant is reachable (only needed for Step 2)
python -c "from core.governed_store import get_governed_count; print(get_governed_count('your_account_id'))"

# 3. Run the full learning-loop test suite
python -m pytest tests/test_learning_store.py tests/test_governed_store.py \
                 tests/test_genie_ranker.py tests/test_learning_loop_integration.py -q
```

Expected: all tests green, no import errors, Qdrant responds (or returns 0 gracefully).

---

### Step 1 — Enable feedback collection

**What it does**

- Thumbs-up / thumbs-down UI appears in the portal chat window.
- `POST /portal/api/feedback/<question_id>` endpoint becomes active.
- Answered queries are scored by `core.quality_scorer` and stored as `learning_candidate` rows.
- The admin Learning Queue (`/admin/learning`) is populated.
- Net feedback adjusts `final_score` and reclassifies candidate type in real time.

**How to enable** (one tenant at a time recommended)

```sql
-- In your admin panel or directly in SQLite:
UPDATE client
SET enable_feedback_collection = 1
WHERE account_id = 'your_account_id';
```

Or via the admin setup UI:

1. Go to `/admin/setup/<account_id>`
2. Toggle **"Enable Feedback Collection"** → Save

**What to verify**

| Check | How |
|-------|-----|
| Thumbs UI visible | Open portal chat, ask a question, confirm thumbs icons appear below the answer |
| Feedback saves | Click thumbs-up; check `answer_feedback` table row created |
| Candidate created | Check `learning_candidate` table — status should be `pending_review` |
| Score adjusts | Click thumbs-down on the same question as a second user; confirm `final_score` drops and `candidate_type` may change |
| Admin queue live | Open `/admin/learning?account_id=<id>` — pending candidates listed |

**Edge cases at this step**

| Scenario | Behaviour |
|----------|-----------|
| Same user votes twice | Upsert — previous vote is overwritten, not double-counted |
| Qdrant unavailable | No impact — governed upsert only fires on admin approve (Step 2) |
| feedback flag off (control group) | `/portal/api/feedback` returns 403; no candidates created |
| Empty SQL answer | Candidate is skipped — `sql_text=""` guard in `create_candidate` path |

---

### Step 2 — Enable Genie suggestions (requires Step 1 data)

> **Recommended**: Run Step 1 for at least 1–2 weeks to accumulate meaningful
> impression and click data before enabling this step.  The cold-start guard
> (< 10 impressions) falls back to source-quality boost automatically, so
> enabling early is safe but provides no behavioral benefit until data exists.

**What it does**

- Suggestion chips are re-ranked by `core.genie_ranker` using behavioral signals
  (CTR 30%, exec_rate 40%, success_rate 25%, dismissal penalty 5%).
- `POST /portal/api/suggestions/event` endpoint becomes active for browser events
  (clicked, executed, successful, dismissed).
- `_record_suggestions_displayed` fires on every portal_chat page load,
  recording one `displayed` event per chip (server-side only — cannot be inflated
  by clients).

**How to enable**

```sql
UPDATE client
SET enable_genie_suggestions = 1
WHERE account_id = 'your_account_id';
```

**What to verify**

| Check | How |
|-------|-----|
| Suggestions appear ranked | Open portal chat; confirm suggestion chips load |
| Click event recorded | Click a chip; check `recommendation_event` table for `event_type='clicked'` |
| Displayed events recorded | Reload chat page; check `recommendation_event` table for `event_type='displayed'` |
| Browser cannot send displayed | `POST /portal/api/suggestions/event` with `event_type='displayed'` → must return 400 |
| Cold-start: governed > static | Check `_score` field on suggestions; any governed/admin_correction chip should outscore static chips at 0 impressions |
| Genie disabled falls back gracefully | Temporarily raise an exception inside `rank_suggestions`; suggestion list should still be returned unranked |

**Edge cases at this step**

| Scenario | Behaviour |
|----------|-----------|
| < 10 impressions | Cold-start guard: score = source boost only (governed=0.10, auto=0.05, static=0.00) |
| Ranker DB error | `rank_suggestions` exception caught; original order preserved, page renders normally |
| `displayed` event from browser | 400 Bad Request — only server-side recording allowed to prevent inflation |
| Genie flag off | `rank_suggestions` never called; `recommendation_event` rows not created |

---

### Step 2b — Promote approved examples to Qdrant (admin action)

**This step is independent of Step 1/2 flags** — it uses the admin Learning Queue.

When an admin approves a candidate in `/admin/learning`:

1. `update_candidate_status(candidate_id, "approved")` is called.
2. `_fire_governed_upsert` embeds the question and SQL into `querybot_governed`.
3. The Qdrant point ID is written back to `learning_candidate.qdrant_id`.
4. On future queries, `retrieve_governed_examples` includes this example in few-shot context.

**Recovery: Qdrant was down during approval**

If `qdrant_id` is empty for an approved candidate, the Qdrant upsert failed silently.
Recover without re-approving:

```python
from core.governed_store import backfill_approved_candidates
n = backfill_approved_candidates("your_account_id")
print(f"Backfilled {n} candidates")
```

This is idempotent — safe to run multiple times.

**Revoke a mistaken approval**

```python
from store.learning_store import update_candidate_status
update_candidate_status(candidate_id, "revoked", reviewer_id="admin1")
# → fires _fire_governed_delete; point removed from Qdrant
```

Revoking after re-approve is safe: the same deterministic point ID is used, so delete
removes exactly the right point.

---

## Rollback Procedure

### Rollback Step 2 (Genie) — zero-downtime

```sql
UPDATE client SET enable_genie_suggestions = 0 WHERE account_id = 'your_account_id';
```

Effect: immediate. `rank_suggestions` is never called on the next request.
No data is deleted — events in `recommendation_event` are preserved for future analysis.

### Rollback Step 1 (Feedback) — zero-downtime

```sql
UPDATE client SET enable_feedback_collection = 0 WHERE account_id = 'your_account_id';
```

Effect: immediate. The thumbs UI disappears. Existing candidates in `learning_candidate`
are not deleted and remain in the admin queue.

### Full rollback — both flags off

```sql
UPDATE client
SET enable_feedback_collection = 0,
    enable_genie_suggestions   = 0
WHERE account_id = 'your_account_id';
```

QueryBot operates exactly as it did before the sprint with no degradation.

---

## Enabling for All Tenants at Once

Only do this after successful single-tenant validation:

```sql
-- Step 1 for all
UPDATE client SET enable_feedback_collection = 1;

-- Step 2 for all (after data accumulates)
UPDATE client SET enable_genie_suggestions = 1;
```

Or use the admin bulk-update endpoint if one exists for your deployment.

---

## Monitoring Checklist

After enabling each step, watch these for 24–48 hours:

| Metric | Table / endpoint | Alert if |
|--------|-----------------|----------|
| Feedback save errors | Application logs `ERROR portal_answer_feedback` | > 0 in 1h |
| Candidate creation rate | `SELECT COUNT(*) FROM learning_candidate WHERE created_at > datetime('now','-1 day')` | 0 (no answers being scored) |
| Event recording failures | Application logs `DEBUG _record_suggestions_displayed failed` | Sustained warnings |
| Qdrant upsert failures | Application logs `WARNING _fire_governed_upsert failed` | > 0 per day |
| Suggestion stats growth | `SELECT COUNT(*) FROM recommendation_event WHERE event_type='displayed'` | Flat after page loads |
| Cold-start proportion | % of suggestions with `impressions < 10` | > 90% after 2 weeks means users aren't clicking |

---

## Feature Flag Reference

| Flag | Default | Enables |
|------|---------|---------|
| `enable_feedback_collection` | `0` | Thumbs UI, feedback API, candidate creation, admin queue |
| `enable_genie_suggestions`   | `0` | Behavioral ranking, browser event API, impression recording |

Both flags are per-tenant columns in the `client` table. Neither requires a restart.
