# QueryBot v2

**Natural-language analytics bot for enterprise ERP and data warehouse.** Users ask questions in plain English — QueryBot generates SQL, executes it, and returns a narrative answer with charts.

**Last updated:** 2026-06-07  
**Status:** Production-ready. Learning loop sprint complete (Days 1-9). Day 10 (integration tests + rollout docs) pending.

---

## Quick navigation

| I want to... | Go to |
|---|---|
| **See the interactive architecture diagrams** | **[ARCHITECTURE_VISUAL.html](ARCHITECTURE_VISUAL.html)** — open in any browser |
| Understand the full technical flow (text) | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Onboard a new tenant | [Tenant Setup](#tenant-setup) |
| Enable the learning loop | [Self-Learning Loop](#self-learning-loop) |
| Configure a metric formula | [Metric Registry](#metric-registry) |
| Set up user access control | [User Access Control](#user-access-control) |
| Understand what a feature flag does | [Feature Flags](#feature-flags) |
| Debug a wrong SQL / wrong answer | [Debugging Guide](#debugging-guide) |
| See all implemented features | [Full Feature List](#full-feature-list) |

---

## What QueryBot v2 does

1. A user asks a business question in Zoom, Teams, Slack, or the web portal chat.
2. QueryBot retrieves relevant schema knowledge, resolves JOIN paths, looks up approved metric formulas, and fetches proven few-shot examples.
3. It generates SQL using Claude (or OpenAI), validates it, and executes it against the tenant's database.
4. The answer is returned as a narrative + table + chart with follow-up action chips.
5. Every answer is scored and queued for admin review. Approved answers become trusted few-shot examples for future queries — the bot improves from its own history.

---

## Supported platforms and databases

### Chat platforms
| Platform | Notes |
|---|---|
| Zoom | Native webhook; unregistered users get a one-time portal registration link |
| Microsoft Teams | HMAC-signed webhook; service account messages skipped |
| Slack | HMAC-verified; automatic dedup (Slack retries on HTTP error) |
| Web portal chat | WebSocket; full portal authentication; schema tab selector |

### Databases
| Database | Notes |
|---|---|
| Snowflake | Full support including warehouse/database/schema/role |
| Oracle | Full support including service name / SID |
| Azure SQL | Full support including managed identity option |

---

## Tenant Setup

### Step 1 — System settings
Admin → System → set LLM API key (Anthropic or OpenAI), admin password.

### Step 2 — Add a platform
Admin → Platforms → Add platform (Zoom / Teams / Slack). Paste webhook credentials.

### Step 3 — Add a database
Admin → Databases → Add database. Select type, enter connection details. Use "Test connection" to verify.

### Step 4 — Create a client (tenant)
Admin → Clients → New. Set `account_id` (must match Zoom accountId / Teams tenantId / Slack team_id), link platform + database.

### Step 5 — Discover schema
Admin → Client → Setup → Discover Schema. Reads the DB information schema, creates `.md` files per table.

### Step 6 — (Optional) Configure masking
Admin → Client → Setup → Configure column masking before building KB. Mark any PII/sensitive columns.

### Step 7 — Build Knowledge Base
Admin → Client → Setup → Build KB. Generates table descriptions, question/SQL pairs, embeds everything into Qdrant. Client state moves to `READY`.

### Step 8 — Create users
Admin → Client → Users → Create user. Set role (admin = all tables; analyst = group-scoped).

### Step 9 — (Optional) Enable advanced features
Enable feature flags: Chat UI, Feedback collection, Genie suggestions. See [Feature Flags](#feature-flags).

---

## Full Feature List

### Natural Language to SQL
- Question → KB retrieval → entity graph JOIN resolution → semantic plan → LLM → SQL → execution → answer
- Supported aggregations: GROUP BY, time series, ranking, period comparison, drill-down, anomaly detection
- Follow-up context: consecutive questions in the same session share SQL context (DuckDB cache re-route)

### Multi-tenant isolation
- Each tenant (`account_id`) has completely separate: KB files, Qdrant vectors (filtered by account_id), SQLite rows, users, and DB connections
- No cross-tenant data leakage possible by design

### Knowledge Base (KB)
- **Stage 1** — Schema discovery: table/column descriptions, ERP abbreviation expansion, join map, data egress log
- **Stage 2** — Query pattern generation: LLM generates question→SQL pairs, validated against real DB, stored as few-shot examples
- **Incremental rebuild** — KB can be rebuilt per-table; existing vectors are overwritten (upsert), not duplicated
- **Stop KB build** — `POST /clients/{account_id}/setup/stop-kb` sends a cancellation signal to an in-progress build
- **Delete KB** — wipes all KB files and Qdrant vectors for a tenant; client state returns to `SCHEMA_READY`

### Schema Enrichment
- ERP column name expansion (e.g. `CUS_IVC_DT_DMS_KEY` → "Customer Invoice Date Dimension Key")
- Column role classification: metric, dimension, filter, date, identifier
- Naming convention taxonomy: detects ERP-style table/column naming patterns and generates richer descriptions

### Entity Graph
- Admin-defined business object model: entities (Customer, Order) → DB tables
- Relationship edges define JOIN paths (FK → PK, with join type and label)
- BFS resolver picks the shortest correct path for any question
- **Auto-suggest**: reads schema FK metadata and proposes graph rows; admin confirms
- **Bulk relationship manager**: import multiple relationships at once
- **Relationship validation**: verify all graph edges against live DB schema
- **Graph health score**: % of entities with relationships, % of relationships validated
- **Audit column purge**: removes spurious `created_by`/`updated_by` → USER table joins from auto-suggest

### Date Role Disambiguation
- 12 built-in date roles: Invoice, Order, Delivery, Ship, Request, Due, Creation, Posting, Effective, Expiry, Approval, Payment
- Each role has synonyms and regex patterns matching ERP FK column names
- Virtual date-role entities in the graph ensure "invoice month" always picks the invoice date FK
- Admin UI to review and override auto-detected date roles

### Semantic Layer & Metric Registry
- Admin-defined approved metric formulas (e.g. Revenue = `SUM(AMOUNT) WHERE STATUS='POSTED'`)
- Formula enforcement at validation time — LLM cannot free-form an approved metric
- Collision detection: warns before saving a metric name that conflicts with glossary terms
- Metric harvest: auto-extracts metric candidates from KB content
- Synonyms, grain, allowed dimensions, example questions per metric
- **Formula test**: live test a metric formula against the real DB before saving
- Deprecated metrics: soft-delete with deprecation note; no crash when queried

### Business Glossary
- Admin-managed glossary of dimensions, metrics, filters, and entities
- Injected into system prompt for term disambiguation
- `auto-populate` endpoint: extracts glossary candidates from KB content
- Clarification UI: terms with `requires_clarification=1` always prompt the user to choose an interpretation

### Clarification Engine
- Detects ambiguous terms using the glossary before generating SQL
- If ambiguous, returns a numbered option list to the user and waits
- Clarification state persists per-user with TTL (clears on timeout or new question)
- `_looks_like_new_query()` word-overlap detection prevents clarification context from leaking into unrelated follow-up questions

### DDL Guard
- Rejects any message containing `CREATE`, `DROP`, `ALTER`, `TRUNCATE`, `INSERT`, `UPDATE`, `DELETE`
- Hard block — never reaches the LLM

### SQL Validation
- Structural checks: FQN format, forbidden keywords, ACL enforcement
- Metric formula mismatch detection: one automatic repair retry with exact formula injected
- CTE support: validator scans all SELECT nodes including CTE bodies
- DuckDB validator: additional layer for portal-originated follow-up queries

### Answer Confidence
- Per-answer confidence score: schema compliance + semantic compliance + entity graph alignment
- Anomaly callouts: flags outlier values and zero-value metrics
- Decision signal: plain-English insight derived from the result stats

### Answer Narrative
- LLM-generated 1-2 sentence summary of the result (configurable off)
- Zero-row RCA: distinguishes "table empty" from "filter too narrow" by probing table row counts
- Business-language error messages (no SQL errors exposed to users)

### Follow-Up Action Chips
| Chip | When shown |
|---|---|
| Explain | Always (non-empty result) |
| Analyze trends | Time series ≥ 3 points |
| Compare prior period | Time series with significant change |
| Predict | Declining trend ≥ 3 points |
| Drill by dimension | Ranking/aggregation result |
| Contribution analysis | Ranking with known leader share |
| Outlier detection | Ranking with high concentration |
| Decide | Result with numeric decision dimension |
| Export CSV | Always |
| Alert me | Any result |

### Period Comparison
- `core/period_comparison.py` — compares current vs prior period for time-series results
- Detects MoM, QoQ, YoY based on detected date grain

### Drill by Dimension
- `core/drill_dimension.py` — re-runs the last query grouped by a new dimension
- Uses DuckDB cache to avoid re-querying the source DB

### Alert Engine
- `core/alert_engine.py` — "Alert me when this changes" chip
- Stores threshold + SQL; periodically checks and notifies user via chat platform

### Export
- `GET /portal/api/export-csv` — download last result as CSV
- `core/export.py` — handles formatting, NULL handling, header quoting

### Charts
- `core/chart.py` — auto-detects chart type from result schema: bar, line, scatter, pie, heatmap
- `core/chart_spec.py` — chart config per result shape
- Charts included in portal chat answers and pinned dashboard

### Pinned Dashboard
- Users can pin any chart to their personal dashboard
- Drag-and-drop grid layout (configurable width/height per card)
- Charts re-execute on dashboard load (live data)
- Pin flow: bot sends a one-time pin token link in chat; user confirms via portal

### LLM Audit Log
- When `enable_llm_audit = 1`, every LLM call is logged: prompt hash, token count, model, component, status
- 30-day rolling retention (configurable via `LLM_AUDIT_RETENTION_DAYS` env var)
- Viewable in Admin → Client → Traces

### Answer Traces
- Full per-request trace with step timings (authorization, context, generation, execution, answer)
- Viewable in Admin → Client → Traces
- Stored in `answer_trace` + `answer_trace_step` tables

### Evals (Golden SQL)
- Admin → Client → Evals — run a suite of golden question/SQL test cases
- Reports: total cases, pass rate, per-case SQL diff
- `POST /clients/{account_id}/evals/run` — trigger a run via API

### Model Health Dashboard
- Admin → Client → Model Health
- Shows: answer success rate, repair rate, avg confidence, top failing questions
- Data from `answer_trace` table, computed on demand

### Billing & Usage
- Per-tenant monthly query count and token count
- Hard limit blocking + 80% warning threshold
- Editable per-model LLM pricing rates (USD/1M tokens) in Admin → System
- `GET /clients/{account_id}/billing/export.csv` — billing data export

### External Log Export
- Scheduled export of query logs + LLM audit logs to the tenant's own DB
- Used for compliance, billing reconciliation, and custom dashboards
- State tracked in `external_log_export_state`; viewable per DB config

### Data Egress Log
- Every KB build logs: tables processed, columns sent vs masked, sample mode (synthetic/real/none)
- Full audit trail for data governance / compliance
- Viewable in Admin → Client → Egress Log

### PII Masking
- Admin marks sensitive columns before KB build
- Masked columns receive synthetic replacement values: `redact`, `fake_name`, `fake_email`, `fake_date`, `constant`
- Column sensitivity preview available before committing to a build

### Semantic Field Feedback
- Users submit corrections to column meaning/use-case from the portal KB view
- Pending count badge shown in admin sidebar
- Admin approves/rejects; approved submissions update the KB chunk

---

## Self-Learning Loop

The loop turns every answered query into a potential improvement without requiring manual SQL writing.

### How it works (end-to-end)

```
1. User asks question → answer sent
2. Answer scored automatically (0-100) based on SQL quality signals
3. Candidate row created in learning_candidate table (status=pending_review)
4. User can give thumbs up/down → adjusts score
5. Admin reviews queue at Admin → Client → Learning Queue
6. Admin approves → SQL embedded into querybot_governed Qdrant collection
7. Future queries retrieve this approved example as a few-shot prompt → better SQL
```

### Enabling the loop
Set `enable_feedback_collection = 1` on the client. This enables:
- Automatic candidate creation after each answer
- Thumbs up/down buttons in portal chat answers
- Admin learning queue tab

### Admin Learning Queue
Located at: Admin → Client → Learning Queue

**Filter tabs:** pending_review | approved | rejected | known_failure | all  
**Score color coding:** green ≥85, amber 60-84, red <60

**Actions per candidate:**
- **Approve** — embeds the SQL into the governed Qdrant collection immediately
- **Reject** — marks as rejected; excluded from future retrieval
- **Known Failure** — marks as a known bad pattern; excluded from suggestions
- **Correct SQL** — admin writes the correct SQL; system sets score=85 and source=admin_correction; approving embeds the corrected SQL (not the original)

### What "approved" means at query time
The next user who asks a similar question gets this approved example as a few-shot example in the SQL generation prompt. The LLM sees: "Here is a proven question and SQL pair — use this as a pattern."

Governed examples are retrieved first (before legacy examples) and deduplicated by question text.

### Score factors
| Factor | Max pts |
|---|---|
| SQL validation passed | 10 |
| Execution success | 20 |
| Non-empty result | 15 |
| No SQL repair needed | 15 (repair succeeded = 10) |
| Metric formula compliance | 20 |
| Entity graph compliance | 10 |
| Schema ACL compliance | 10 |
| **Total** | **100** |

---

## Genie Suggestion Engine

Ranks the suggestion chips shown in portal chat using real engagement data.

### Enabling
Set `enable_genie_suggestions = 1` on the client.

### What changes when enabled
- Suggestion chips are sorted by a behavioral score (most useful first) instead of static order
- Each page load records a `displayed` event per suggestion
- Browser sends `clicked`, `executed`, `successful`, `dismissed` events via `POST /portal/api/suggestions/event`

### Scoring
- **Cold-start** (< 10 impressions): score = source quality boost only  
  (admin-corrected = 0.10, auto-harvested = 0.05, static = 0.00)
- **Warm-start** (≥ 10 impressions):  
  `CTR×0.30 + execution_rate×0.40 + success_rate×0.25 − dismiss_rate×0.05`  
  Blended with source boost based on confidence (`min(impressions/100, 1.0)`)

---

## User Access Control

### Roles
| Role | Table access |
|---|---|
| `admin` | All tables in the connected DB |
| `analyst` | Tables assigned to their group + individual overrides |

### Setup
1. Admin → Client → Groups → Create group → add tables
2. Admin → Client → Users → Create user → assign group

### User registration (Zoom)
When an unregistered Zoom user messages the bot:
1. Bot sends a one-time registration link in chat
2. User visits the link, sets a password
3. User is linked to their Zoom identity and can query immediately

---

## Feature Flags

Set on the `client` row in Admin → Client → Edit.

| Flag | When to enable |
|---|---|
| `chat_ui_enabled` | Activate the portal chat page for this tenant |
| `enable_llm_audit` | Log every LLM call (for compliance / debugging) |
| `enable_feedback_collection` | Enable the learning loop (thumbs, scoring, admin queue) |
| `enable_genie_suggestions` | Enable behavioral ranking for suggestion chips |

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DB_PATH` | `data/querybot.db` | SQLite database path |
| `SESSION_SECRET` | insecure | Admin session signing key |
| `PORTAL_SESSION_SECRET` | insecure | Portal session signing key |
| `ADMIN_SESSION_SECRET` | insecure | Admin session signing key (overrides SESSION_SECRET) |
| `ENCRYPTION_KEY` | — | AES key for credential encryption (required in production) |
| `LLM_AUDIT_RETENTION_DAYS` | `30` | How long to keep LLM call logs |
| `PORTAL_BASE_URL` | `http://localhost:8000` | Used for generating registration links in chat |

---

## Debugging Guide

**Wrong SQL generated:**
1. Check Admin → Client → Traces for the request — see which KB chunks were retrieved and what graph context was injected
2. If a metric formula is wrong → check metric_registry for that term
3. If JOIN is wrong → check entity graph relationships; run validation
4. If date join is wrong → check Admin → Client → Date Roles
5. If example poisoned it → check Admin → Client → Learning Queue; reject the bad candidate

**Zero rows returned:**
- See the RCA hints in the answer — it tells you whether the table is empty or the filter is too narrow
- Check `_count_tables_for_zero_row` log output for table sizes

**Suggestion chips not ranking:**
- Confirm `enable_genie_suggestions = 1` on the client
- Check that `recommendation_event` table has rows (impressions must reach ≥ 10 for behavioral scoring to kick in)
- Check the `_score` key on suggestions via the `rank_suggestions()` debug output

**Learning candidate not appearing in queue:**
- Confirm `enable_feedback_collection = 1`
- Check that the query completed successfully (zero-row answers also create candidates)
- Check `learning_candidate` table for the row; check `_create_learning_candidate` log output

**Governed example not retrieved:**
- Confirm the candidate status is `approved` in the learning queue
- Check `qdrant_id` column on the candidate row — it should be non-empty after approve
- Run `backfill_approved_candidates(account_id)` from Python if qdrant_id is empty (Qdrant was down during approve)

---

## File Map

| File | Responsibility |
|---|---|
| `main.py` | App entry point, query pipeline, webhook handlers |
| `admin/routes.py` | All admin UI routes |
| `portal/routes.py` | All portal UI and API routes |
| `core/llm.py` | LLM provider resolution, prompt construction, SQL generation |
| `core/knowledge.py` | KB chunk retrieval (RAG) |
| `core/vector_store.py` | Qdrant client singleton, upsert/retrieve for querybot_kb |
| `core/governed_store.py` | Qdrant CRUD for querybot_governed collection |
| `core/genie_ranker.py` | Behavioral suggestion scoring and ranking |
| `core/schema.py` | Schema discovery, KB generation, query execution |
| `core/graph_resolver.py` | Entity graph BFS + JOIN skeleton builder |
| `core/date_roles.py` | Date role detection patterns and helpers |
| `core/semantic_planner.py` | Metric formula lookup and injection |
| `core/validator.py` | SQL structural validation + ACL enforcement |
| `core/metric_validator.py` | Metric formula compliance check |
| `core/quality_scorer.py` | Deterministic 0-100 scoring of answer traces |
| `core/response_builder.py` | Confidence, anomaly callouts, chip eligibility |
| `core/examples.py` | Few-shot example retrieval (dual collection) |
| `core/suggestions.py` | Question suggestion generation for portal chat |
| `core/chart.py` | Chart type detection and payload builder |
| `core/masking.py` | PII column masking for KB builds |
| `core/alert_engine.py` | Alert me chip backend |
| `core/export.py` | CSV export formatting |
| `store/db.py` | SQLite schema and connection management |
| `store/learning_store.py` | Feedback, learning candidate CRUD, event recording |
| `store/config_store.py` | DB config and credential management |
| `store/semantic_store.py` | Metric registry CRUD |
| `store/user_store.py` | User, group, table access CRUD |
| `gateway/` | Platform adapter normalisation (Zoom, Teams, Slack, Web) |

---

*Keep this file updated when new features ship. Add a row to the feature list, update the file map if new modules are added, and update the debugging guide if a new class of issue is identified.*
