# QueryBot v2 — System Architecture

**Last updated:** 2026-06-07  
**Coverage:** All layers from gateway to persistence, including every edge case and the full learning loop sprint.

> **How to use this doc:**  
> Each section names the exact file(s) responsible. When a bug is reported, find the matching section, read the edge-cases table, then open only the files listed. No codebase scan needed.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Entry Points](#2-entry-points)
3. [Query Pipeline (happy path)](#3-query-pipeline-happy-path)
4. [Step-by-Step Pipeline Detail](#4-step-by-step-pipeline-detail)
5. [Knowledge Base Build Pipeline](#5-knowledge-base-build-pipeline)
6. [Semantic Layer & Metric Registry](#6-semantic-layer--metric-registry)
7. [Entity Graph & JOIN Resolution](#7-entity-graph--join-resolution)
8. [Date Role Disambiguation](#8-date-role-disambiguation)
9. [Self-Learning Loop](#9-self-learning-loop)
10. [Genie Suggestion Engine](#10-genie-suggestion-engine)
11. [Portal Chat UI](#11-portal-chat-ui)
12. [User Access Control](#12-user-access-control)
13. [Persistence Layer (SQLite Tables)](#13-persistence-layer-sqlite-tables)
14. [Feature Flags (per-client toggles)](#14-feature-flags-per-client-toggles)
15. [External Integrations](#15-external-integrations)
16. [Data Security & Masking](#16-data-security--masking)

---

## 1. System Overview

QueryBot v2 is a multi-tenant natural-language-to-SQL analytics bot. Users ask questions in plain English via Zoom, Teams, Slack, or a web portal. The bot generates SQL, executes it against the connected database (Snowflake / Oracle / Azure SQL), and returns a narrative answer with charts and follow-up chips.

```
User (chat or portal)
  ↓
Gateway adapters  — normalise platform events
  ↓
handle_query()    — main.py:553, the full pipeline
  ↓
Answer + Chart    — sent back via adapter or WebSocket
```

**Key design principle:** every layer degrades gracefully. A Qdrant failure, a DB timeout, or a missing config never surfaces an unhandled exception to the user — it falls back to a simpler path and logs a warning.

---

## 2. Entry Points

| Route | File | Purpose |
|---|---|---|
| `POST /webhook/zoom` | `main.py:2698` | Zoom chat webhook |
| `POST /webhook/teams` | `main.py:2726` | Teams webhook |
| `POST /webhook/slack` | `main.py:2744` | Slack webhook |
| `GET/WS /ws/chat/{account_id}` | `main.py:2769` | Portal WebSocket chat |
| `GET /portal/*` | `portal/routes.py` | End-user portal (login, dashboard, chat, KB view) |
| `GET /admin/*` | `admin/routes.py` | Admin UI (multi-tenant setup, KB, metrics, graph) |
| `GET /health` | `main.py:3934` | Health check (used by load balancer) |

### Gateway adapters (`gateway/`)

Each adapter normalises a platform-specific webhook payload into a common `PlatformEvent`:

```python
@dataclass
class PlatformEvent:
    account_id: str
    user_id: str
    channel_id: str
    text: str
    platform: str
    schema_hint: str = ""   # portal schema-tab selection
    table_hint: str = ""    # portal table-pin hint
```

| Adapter file | Platform | Key edge cases handled |
|---|---|---|
| `slack_adapter.py` | Slack | Dedup via `webhook_dedup.py` (Slack retries duplicates); bot message filtering (`bot_id` check) |
| `teams_adapter.py` | Teams | HMAC signature verification; service account message skip |
| `zoom_adapter.py` | Zoom | `accountId` → `account_id` mapping; unregistered user → one-time link flow |
| `web_adapter.py` | WebSocket portal | Direct session authentication; `schema_hint` and `table_hint` forwarded |

---

## 3. Query Pipeline (happy path)

```
User question
  │
  ├─ 1. Rate / token limit check
  ├─ 2. Webhook dedup check (Slack/Teams only)
  ├─ 3. Clarification check (ambiguity detection)
  │        ↓ if ambiguous → ask user, save pending
  │        ↓ if clarifying → combine with original Q
  ├─ 4. DDL guard (reject CREATE/DROP/ALTER)
  ├─ 5. Schema scoping (schema tab selection narrows tables)
  ├─ 6. RAG — KB context retrieval (knowledge.py)
  ├─ 7. Entity graph resolution — deterministic JOIN skeleton (graph_resolver.py)
  ├─ 8. Semantic plan — metric formula injection (semantic_planner.py)
  ├─ 9. Few-shot examples retrieval — governed first, legacy fill (examples.py)
  ├─ 10. LLM SQL generation (llm.py → Claude API)
  ├─ 11. SQL validation (validator.py + duckdb_sql_validator.py)
  │        ↓ if metric_formula_mismatch → repair retry (1×)
  ├─ 12. Cache check / DuckDB re-route (result_cache.py + query_router.py)
  ├─ 13. Execute SQL against real DB (schema.py:run_query)
  │        ↓ if zero rows → RCA hints + table count probe
  ├─ 14. Response builder (response_builder.py + answer_formatter.py)
  ├─ 15. Chart detection + build (chart.py)
  ├─ 16. Follow-up chip eligibility (chip_eligibility, response_builder.py)
  ├─ 17. Send answer (adapter or WebSocket)
  ├─ 18. Log query (query_log + answer_trace)
  └─ 19. Create learning candidate (main.py:_create_learning_candidate, background)
```

---

## 4. Step-by-Step Pipeline Detail

### 4.1 Rate & Token Limits
**File:** `main.py:180-194`

| Check | Field | Behaviour |
|---|---|---|
| Monthly query count | `client.query_limit_monthly` | Hard block at limit; warning at 80% |
| Monthly token count | `client.token_limit_monthly` | Hard block if limit > 0; uncapped if 0 |

### 4.2 Webhook Dedup
**File:** `core/webhook_dedup.py`

- Bloom-filter in memory keyed on `(account_id, event_id)`.
- 5-minute TTL. Zoom re-delivers on HTTP error; dedup prevents double-responses.

### 4.3 Clarification Engine
**File:** `core/clarification.py`

Checks glossary ambiguity before generating SQL. If a key term maps to multiple interpretations (e.g. "revenue" could mean gross or net), returns a numbered option list to the user.

| Edge case | Handling |
|---|---|
| User replies to a clarification with a new unrelated question | `_looks_like_new_query()` detects it via word-overlap (main.py:202); treats as fresh query |
| Clarification expires without response | `was_recently_expired()` guard prevents stale context from being reused |
| Multi-term ambiguity | Only the highest-priority ambiguous term triggers; others resolved in context |

### 4.4 DDL Guard
**File:** `core/llm.py:is_ddl_attempt`

Rejects any message containing DDL keywords (`CREATE`, `DROP`, `ALTER`, `TRUNCATE`, `INSERT`, `UPDATE`, `DELETE`). Returns a fixed user message. **Never reaches LLM.**

### 4.5 Schema Scoping
**File:** `main.py:624`

When user selects a schema tab in the portal (e.g. "HR"), `schema_hint` is set. All tables not in that schema are removed from `effective` (the allowed table set used for RAG, SQL generation, and validation). This prevents cross-schema confusion for users who work in one schema only.

### 4.6 RAG — KB Context Retrieval
**File:** `core/knowledge.py`, `core/vector_store.py`

- Embeds the question and does cosine search over `querybot_kb` Qdrant collection.
- Returns top-k KB chunks (table descriptions, column definitions, join hints).
- Filtered by `effective` (user's allowed tables) so no out-of-scope table docs appear.
- Synonym injection: `_extract_kb_synonym_injection()` reads "Business Synonyms" sections from KB chunks and prepends them as hints (main.py:252).

### 4.7 Entity Graph Resolution
**File:** `core/graph_resolver.py`

BFS over the entity graph (SQLite `entity_graph` + `entity_relationships` tables) to find the minimal JOIN path between entities detected in the question.

| Edge case | Handling |
|---|---|
| Ambiguous table references | `detect_entities()` scores each relationship label against the question; picks highest scorer |
| Date dimension FK ambiguity | Date role scoring (+35 boost) overrides generic label match — see §8 |
| Circular joins | BFS with visited set; cycle detection prevents infinite loops |
| Missing entity | Falls back to empty graph context; LLM still runs with KB context only |
| Multiple fact tables | BFS finds shortest path; admin can define bridge entities for M:N |

### 4.8 Semantic Plan
**File:** `core/semantic_planner.py`, `core/semantic_model.py`

Looks up metric registry for any metric terms in the question. If found, injects the approved formula and required tables into the system prompt so the LLM uses the canonical formula instead of guessing.

`_merge_semantic_plans()` (main.py:142) merges up to 3 independent semantic plans (metric, field-level, runtime) removing duplicates.

### 4.9 Few-Shot Examples
**File:** `core/examples.py` (active implementation at line 356)

Dual retrieval — governed examples ranked first:
1. `querybot_governed` (Qdrant) — admin-approved candidates → highest trust
2. `querybot_kb` (Qdrant) — auto-harvested validated examples → fill remaining slots

Deduplication by question text (case-insensitive) ensures the same Q/SQL pair never appears twice in the prompt.

| Edge case | Handling |
|---|---|
| Legacy `chroma_dir` path string | Path parts extraction: `parts[1]` = account_id (e.g. `"clients/tenant1"` → `"tenant1"`) |
| Qdrant down | `try/except` around each retrieval; partial results returned gracefully |
| Both collections empty | Returns `[]`; prompt still works with KB context only |

### 4.10 SQL Generation
**File:** `core/llm.py`

`build_sql_system_prompt()` assembles:
- DB schema (table/column descriptions from KB)
- Entity graph JOIN skeleton
- Semantic field plan (metric formulas)
- Few-shot examples
- Clarification context (if applicable)
- Business description and glossary terms

Provider resolution (`resolve_provider()`) supports per-client LLM override:
- `client.llm_provider` = `anthropic` or `openai`
- `client.llm_model` = specific model string
- Falls back to system-level API key if no per-client key configured

### 4.11 SQL Validation
**File:** `core/validator.py`, `core/duckdb_sql_validator.py`

Two-pass validation:
1. **Structural** — checks FQN format, forbidden DDL patterns, table access (ACL enforcement)
2. **Metric formula** — checks that the generated SQL uses the approved formula, not a free-form equivalent

| Edge case | Handling |
|---|---|
| Metric formula mismatch | One repair retry: injects the exact formula into a repair prompt; if still fails → returns partial answer with warning |
| CTE with metric formula | Validator scans all SELECT nodes (including CTE body), not just the outermost SELECT |
| Schema not in effective tables | Validator rejects → error message sent; no query executed |

### 4.12 Cache + DuckDB Router
**Files:** `core/result_cache.py`, `core/query_router.py`, `core/duckdb_sql_validator.py`

If the previous answer for this session has the same tables and aggregation pattern, a DuckDB re-query against the cached result set avoids a round-trip to Snowflake/Oracle/Azure SQL. Particularly useful for follow-up "drill by X" questions.

### 4.13 SQL Execution
**File:** `core/schema.py:run_query`

Supports Snowflake, Oracle, Azure SQL. Connection is per-query (not pooled). Query timeout enforced at DB connector level.

### 4.14 Zero-Row Handling
**File:** `main.py:_build_zero_row_message`, `_zero_row_rca_hints`

When zero rows are returned:
1. Counts rows in each referenced table (`_count_tables_for_zero_row`) to distinguish "table empty" from "filter too narrow".
2. Generates business-language RCA hints (e.g. "ORDERS table has 0 rows today — check if data has been loaded").
3. Sends structured `format_zero_row_business_response` message instead of a blank table.

### 4.15 Response Builder
**File:** `core/response_builder.py`, `core/answer_formatter.py`, `core/answer_confidence.py`

Builds:
- **Confidence signal** — `build_answer_confidence()` scores the answer on schema compliance, semantic compliance, entity graph alignment
- **Anomaly callouts** — `_build_anomaly_callouts()` flags unusual values (outliers, zero-value metrics)
- **Decision signal** — `_build_decision_signal()` turns stats into plain-English business insight
- **Narrative** — `_generate_result_narration()` uses LLM to write 1-2 sentence summary of results

### 4.16 Follow-Up Chips
**File:** `core/response_builder.py`, `core/chip_eligibility.py`

Chips are action buttons attached to the answer. Eligibility is computed from the result shape:

| Chip | Eligibility condition |
|---|---|
| Explain | Always (if result non-empty) |
| Analyze trends | Time series with ≥ 3 data points |
| Compare prior period | Time series with ≥ 2 points and significant change |
| Predict | Declining trend with ≥ 3 points |
| Drill by dimension | Ranking/aggregation result |
| Contribution analysis | Ranking with known leader share |
| Outlier detection | Ranking with high concentration (top-1 > 50% total) |
| Decide | Any result with a numeric decision dimension |
| Export CSV | Always |
| Alert me | Any result (triggers alert engine) |

### 4.17 Learning Candidate Creation
**File:** `main.py:_create_learning_candidate` (line 1907)

Called **after the response is sent** (fire-and-forget). Only runs when `client.enable_feedback_collection = 1`.

Calls `score_trace()` → `create_candidate()`. Score factors:
- SQL validation passed (10 pts)
- Execution success (20 pts)
- Row count > 0 (15 pts)
- No repair needed (15 pts) vs repair succeeded (10 pts) vs repair failed (0 pts)
- Metric compliance (20 pts)
- Entity graph compliance (10 pts)
- Schema ACL compliance (10 pts)

---

## 5. Knowledge Base Build Pipeline

**Triggered by:** Admin → Setup → Build KB

### Stage 1 — Schema Discovery
**File:** `core/schema.py`

Connects to the DB and reads the information schema. For each table:
1. Writes `{table}_schema.json` (column names, types, nullable)
2. Writes `{table}_kb.md` (markdown description with sample rows)
3. Detects and expands ERP abbreviations (`core/erp_column_dict.py`, `core/schema_enrichment.py`)
4. Builds `_join_map.md` with:
   - Standard FK → PK relationships
   - **Pass 3:** Role-playing date dimension joins (separate entry per FK → DIM_DATE pair)
5. Logs every table processed to `kb_data_egress_log` (fields sent, masked fields, sample mode)

### Stage 2 — Query Pattern Generation
**File:** `core/schema.py` → LLM generates `{table}_queries.md`

Natural-language question → SQL pairs generated per table. Validated against the real DB (single connection for the whole batch to prevent Snowflake connection floods). Valid pairs stored in `validated_examples` and embedded into `querybot_kb` Qdrant collection.

### Masking & Data Security
**File:** `admin/routes.py:4128` (`/kb-tables/masking`), `core/masking.py`

Admin can mark columns as PII/sensitive before KB build. Masked columns get synthetic replacement values instead of real sample rows. Mask replacement strategies: `redact`, `fake_name`, `fake_email`, `fake_date`, `constant`.

### Egress Log
**File:** `core/log_export.py`, `store/db.py:kb_data_egress_log`

Every KB build operation records: which tables were processed, which columns were sent vs masked, sample mode (`synthetic` / `real` / `none`). Viewable in Admin → Client → Egress Log.

---

## 6. Semantic Layer & Metric Registry

**Files:** `core/semantic_layer.py`, `core/semantic_registry.py`, `core/metric_builder.py`, `store/semantic_store.py`

### What it does
Prevents the LLM from free-forming a metric formula. If "revenue" is defined as `SUM(AMOUNT) WHERE STATUS='POSTED'`, that exact formula is injected into the system prompt and enforced at validation time.

### Admin Workflow
1. Admin defines a metric (name, synonyms, SQL template, formula type, grain, dimensions)
2. Metric stored in `metric_registry` table
3. At query time: `build_semantic_field_plan()` matches question terms to metric names/synonyms
4. Matched metrics: formula injected into prompt + required tables added to effective set

### Collision detection
`GET /clients/{account_id}/metrics/check-collision` — checks if a new metric name conflicts with existing glossary terms. Prevents ambiguous metric resolution.

### Metric harvest
`POST /clients/{account_id}/metrics/harvest` — extracts metric candidates from KB content and creates draft entries for admin review.

### Formula validation at query time
**File:** `core/metric_validator.py`

Parses the generated SQL AST. Checks that the formula matches the approved template. Works on CTEs by scanning all SELECT nodes, not just the outermost one.

| Edge case | Handling |
|---|---|
| LLM rewrites formula to equivalent SQL | Validator catches mismatch → repair retry with exact formula injected |
| Metric in CTE | Scanner walks full AST tree, not just outermost SELECT |
| Multiple metrics in one query | Each metric checked independently; first mismatch triggers repair |
| Deprecated metric queried | Graceful fallback — treat as unknown term; no crash |

### Semantic Field Feedback
**Files:** `store/semantic_feedback.py`, `admin/routes.py:1434`

Users can submit corrections to column meaning/use-case from the portal KB view. Admin reviews pending submissions and approves/rejects. Approved submissions update the KB chunk for that column.

---

## 7. Entity Graph & JOIN Resolution

**Files:** `core/graph_resolver.py`, `admin/routes.py` (graph API routes)

### What it does
The entity graph is a business object model: entities (Customer, Order) mapped to DB tables, connected by relationships (FK → PK). The graph resolver runs BFS to find the shortest JOIN path between entities mentioned in a question, then injects a deterministic JOIN skeleton into the SQL prompt.

### Data model (SQLite)
- `entity_graph` — entity name, table name, schema, PK column, entity type (fact/dimension/bridge)
- `entity_relationships` — from_entity, to_entity, from_column, to_column, join_type, label, join_conditions
- `entity_properties` — column roles per entity (metric / dimension / filter / date / identifier / ignore)

### Auto-suggest
**File:** `admin/routes.py:2671` (`/graph/api/suggest`)

Reads the schema FK/PK metadata and proposes entity graph rows. Admin clicks "Confirm" to accept each suggestion. FCT_ and FACT_ prefix tables are detected as fact entities.

### Relationship validation
`POST /clients/{account_id}/graph/api/relationships/validate-all` — verifies every relationship references real columns in the connected DB.

### Bulk relationship manager
`POST /clients/{account_id}/graph/api/relationships/bulk` — import/create multiple relationships at once from a CSV or JSON payload.

### Graph health
`GET /clients/{account_id}/graph/api/health` — returns a health score: % of entities with at least one relationship, % of relationships validated.

| Edge case | Handling |
|---|---|
| Audit-column joins (created_by, updated_by → USER table) | `purge-audit` endpoint removes spurious joins created during auto-suggest |
| Entity not found in question | BFS returns empty; LLM still gets KB context |
| Multiple possible join paths | BFS returns shortest; admin can add bridge entities to force a specific path |
| Schema not matching current DB state | Relationship validation endpoint flags stale FK/PK references |

---

## 8. Date Role Disambiguation

**Files:** `core/date_roles.py`, `core/schema.py`, `core/graph_resolver.py`, `core/schema_enrichment.py`

### Problem
Enterprise fact tables have many date FKs all pointing to the same physical date dimension (DIM_DATE): invoice date, order date, delivery date. Without role awareness, the LLM picks the wrong FK and returns wrong date-based aggregations.

### Solution
1. **Schema discovery** (`core/schema.py:_build_join_map`) — Pass 3 creates a role-playing date dimension section in `_join_map.md`. Each fact FK → DIM_DATE pair gets its own entry with the business role name.
2. **Virtual entities** — `build_entity_graph_from_schema()` creates virtual entities ("Invoice Date", "Order Date") that point to the same physical DIM_DATE table but with different FK columns.
3. **Resolver scoring** — `detect_entities()` scores each relationship label against the question. "invoice month" → +35 for "Invoice Date" entity. Only the winning entity's FK appears in the JOIN skeleton.

### 12 built-in date roles
Invoice, Order, Delivery, Ship, Request, Due, Creation, Posting, Effective, Expiry, Approval, Payment

Each role has synonyms (e.g. "ship" → "dispatch", "sent", "shipped") and a regex pattern matching ERP column naming conventions (e.g. `CUS_IVC_DT_DMS_KEY` → invoice role).

### Admin date-roles UI
**File:** `admin/routes.py:3484`

Detects role-playing date dimensions from the schema and shows which columns map to which roles. Admin can approve/override the auto-detected roles.

| Edge case | Handling |
|---|---|
| Multiple FKs match the same role | Highest column-pattern specificity wins |
| New ERP naming convention not in patterns | Admin can add synonyms via date-roles UI |
| No date dimension table detected | Fallback to generic join; no crash |

---

## 9. Self-Learning Loop

**Sprint:** Days 1-7. **Status:** Complete.

### Overview

```
Answer sent
  ↓
_create_learning_candidate()        ← main.py:1907 (background, non-blocking)
  ↓
score_trace()                       ← core/quality_scorer.py
  ↓
create_candidate()                  ← store/learning_store.py
  ↓
[learning_candidate row, status=pending_review]
  │
  ├── User thumbs up/down           ← portal/routes.py:1209
  │     ↓ recompute_candidate_score()
  │
  └── Admin reviews queue           ← admin/routes.py:5091
        ├── Approve  → _fire_governed_upsert()  → querybot_governed (Qdrant)
        ├── Reject   → status=rejected           (no Qdrant write)
        ├── Correct SQL → set_candidate_corrected_sql() → new score=85, source=admin_correction
        └── Revoke   → _fire_governed_delete()  → remove from querybot_governed
```

### Quality Scorer
**File:** `core/quality_scorer.py`

Deterministic 0–100 score from execution trace signals. No LLM involved.

| Signal | Points | Description |
|---|---|---|
| SQL validation passed | 10 | Structural + ACL check passed |
| Execution success | 20 | DB returned result without error |
| Row count > 0 | 15 | Non-empty result |
| No repair needed | 15 | First-pass SQL was used (best) |
| Repair succeeded | 10 | One retry fixed it (acceptable) |
| Repair failed | 0 | Could not fix (worst) |
| Metric compliance | 0–20 | % of required metrics using approved formulas |
| Entity graph compliance | 0–10 | JOIN path matched entity graph |
| Schema ACL compliance | 10 | No out-of-scope table accessed |

### Feedback delta
User thumbs-up (+N) / thumbs-down (−N) adjusts `final_score` from the base `technical_score`. Net negative feedback (thumbs-down > thumbs-up) forces `candidate_type = "negative"` regardless of technical score.

### Admin Learning Queue
**File:** `admin/routes.py:5091`, `admin/templates/client_learning_queue.html`

- Filter tabs: pending_review / approved / rejected / known_failure / all
- Score pills: green ≥85, amber 60-84, red <60
- Per-row: SQL preview, evidence chips, voter counts, corrected SQL form
- Actions: Approve / Reject / Known Failure / Correct SQL

### Governed Qdrant Collection
**File:** `core/governed_store.py`

Collection: `querybot_governed` (separate from `querybot_kb`)

| Function | Behaviour |
|---|---|
| `upsert_governed_example()` | Idempotent — deterministic UUID from MD5("governed::{candidate_id}"). Re-approve = safe overwrite |
| `delete_governed_example()` | Called on revoke. No-op if point missing. Never raises |
| `retrieve_governed_examples()` | Tenant-filtered (account_id), best-effort — returns [] if collection missing |
| `backfill_approved_candidates()` | Idempotent recovery utility. Re-runs all approved candidates through upsert |

### Write hooks in learning_store
**File:** `store/learning_store.py`

`_fire_governed_upsert` and `_fire_governed_delete` are called inside `update_candidate_status()` **after** the SQLite commit. They are wrapped in `try/except` — a Qdrant failure never rolls back the DB write.

SQL preference: `corrected_sql` > `sql_text`. Admin-corrected SQL is what gets embedded.

| Edge case | Handling |
|---|---|
| Qdrant down during approve | `_fire_governed_upsert` logs warning, returns. DB status = approved. `backfill_approved_candidates()` can recover later |
| Re-approve after SQL correction | Deterministic point ID → safe upsert overwrites old embedding |
| Revoke then re-approve | Delete point on revoke; upsert recreates it on re-approve |
| Empty SQL at approve time | `upsert_governed_example` returns `""` without writing; logs warning |

---

## 10. Genie Suggestion Engine

**Sprint:** Days 8-9. **Status:** Complete.

### Overview

When `enable_genie_suggestions = 1`, suggestion chips in the portal chat are ranked by behavioral signals rather than served in static order.

```
portal_chat GET
  ↓
_build_chat_suggestions(user)
  ├── get_suggestions()              ← core/suggestions.py (Tier 1: validated examples, Tier 2: metric registry)
  ├── _guess_safe_metric_suggestions() ← fallback if < 4 results
  └── rank_suggestions()             ← core/genie_ranker.py (when flag on)
        ↓ for each suggestion:
        get_suggestion_stats()       ← store/learning_store.py
        score_suggestion()           ← genie_ranker.py
  ↓
_record_suggestions_displayed()     ← fire-and-forget impression events
  ↓
render portal_chat.html with ranked suggestions
```

### Scoring algorithm
**File:** `core/genie_ranker.py`

```
impressions = stats["displayed"]

Cold-start (impressions < 10):
  score = source_boost
    governed / admin_correction / pre_governed → +0.10
    learned / auto                             → +0.05
    static                                     → +0.00

Warm-start (impressions ≥ 10):
  behavioral = 0.30×CTR + 0.40×exec_rate + 0.25×success_rate − 0.05×dismiss_rate
  confidence = min(impressions / 100, 1.0)
  score = confidence × behavioral + (1 − confidence) × source_boost
```

### Browser event API
`POST /portal/api/suggestions/event`

Accepted: `clicked`, `executed`, `successful`, `dismissed`  
Blocked: `displayed` (recorded server-side on page load to prevent client-side inflation)

**All DB failures return 200** — a recording error must never break the chat.

### `_record_suggestions_displayed`
Called in `portal_chat` handler after `_build_chat_suggestions()` returns the ranked list. Uses a synthetic `uuid4()` page-load ID (not the session cookie value — security). One `displayed` row per suggestion per page load.

| Edge case | Handling |
|---|---|
| Ranker exception | `try/except` in `_build_chat_suggestions`; falls back to unranked list |
| DB failure in stats lookup | Per-suggestion `try/except`; that suggestion scored as 0.0 (cold-start static) |
| Missing impression data | Cold-start path returns source_boost only; no division-by-zero |
| `source_map` not provided | Falls back to `sug.get("source", "static")` on each dict |
| Suggestions > 6 | Trimmed to 6 after ranking (same cap as before genie) |

---

## 11. Portal Chat UI

**File:** `portal/routes.py`, `portal/templates/portal_chat.html`

### WebSocket protocol
The portal chat uses a persistent WebSocket (`/ws/chat/{account_id}`). The browser sends:
```json
{"text": "what is revenue?", "table_hint": "DB.DBO.SALES", "schema_hint": "DBO"}
```
The server streams back structured `stage_update` events (authorization → context → generation → execution → answer) and a final `answer` event.

### Schema selector
Multiple schema tabs are built from `_get_available_schemas()`. Selecting a tab sets `schema_hint`, which scopes the entire pipeline to that schema.

### Suggestion chips
`.suggestion-chip` buttons prefill the composer. `sendSuggestion(btn)` reads `btn.dataset.fqn` and passes it as `table_hint`. The `shuffleSuggestions()` button Fisher-Yates shuffles the chips for variety.

### Pinned dashboard
`GET /portal/dashboard` — drag-and-drop grid of pinned charts. Charts are live-queries re-executed on page load via `GET /portal/api/pin-chart` + DuckDB cached results.

### Export
`GET /portal/api/export-csv` — downloads last result as CSV. Filtered by user's allowed tables.

### History
`GET /portal/api/history` — last 20 questions and answers for this user.

---

## 12. User Access Control

**Files:** `store/user_store.py`, `admin/routes.py` (groups + users), `main.py:610`

### ACL model

```
client (tenant)
  └── user_group (e.g. "Sales Team")
        ├── group_table_access — tables this group can see
        └── portal_user
              ├── user_table_access — individual overrides (additive)
              └── role: admin | analyst
                    admin  → allowed_tables = None → effective = all_known
                    analyst → allowed_tables = group + individual overrides
```

### Enforcement points

| Layer | File | What is enforced |
|---|---|---|
| RAG retrieval | `core/knowledge.py` | Only KB chunks for allowed tables returned |
| SQL validation | `core/validator.py` | Tables in generated SQL must be in `effective` |
| Result export | `portal/routes.py:1148` | Export refused if table not in allowed set |
| Suggestions | `core/suggestions.py` | Questions filtered by `allowed_tables` |
| Governed examples | `core/governed_store.py` | Retrieved by `account_id` only (admin approved each) |

### One-time registration
When an unknown Zoom user messages the bot, a registration token is created and a link sent in chat. User visits `/portal/register?token=xxx`, sets password, and is bound to their Zoom user ID. Token is single-use and expires in 24 hours.

---

## 13. Persistence Layer (SQLite Tables)

**File:** `store/db.py`

| Table | Purpose |
|---|---|
| `system_config` | Global API keys, model settings, admin password (AES encrypted) |
| `platform_config` | Chat platform credentials (Zoom/Teams/Slack) per workspace |
| `db_config` | DB connection credentials (Snowflake/Oracle/Azure SQL) |
| `client` | One row per tenant. Holds feature flags, state, LLM overrides |
| `query_log` | Every query: question, SQL, tokens, cost, duration |
| `answer_trace` + `answer_trace_step` | Full per-request trace with step timings |
| `eval_run` + `eval_case_result` | Structured eval runs (golden SQL regression tests) |
| `llm_call_log` | LLM call audit: payload hash, token counts, model, component (30-day rolling) |
| `portal_user` | User accounts: email, password_hash (bcrypt), role, zoom_user_id |
| `user_group` + `group_table_access` | Row-level security groups |
| `user_table_access` | Individual user table overrides |
| `registration_token` | One-time portal registration links |
| `pinned_chart` | User dashboard charts (question, SQL, chart_type, grid position) |
| `pin_token` | Short-lived tokens for pinning charts from chat |
| `metric_registry` | Admin-defined metric formulas |
| `validated_examples` | Proven question→SQL pairs from Stage 2 validation and query harvesting |
| `pending_clarification` | Active clarification state per user (expires automatically) |
| `business_term` | Glossary: dimensions, metrics, filters, entities |
| `semantic_field_feedback` | User column-meaning corrections pending admin review |
| `llm_pricing` | Editable cost rates per model (USD per 1M tokens) |
| `kb_data_egress_log` | Full audit of what data was sent to LLM during KB builds |
| `entity_graph` | Business entity definitions (name → table mapping) |
| `entity_relationships` | JOIN edges between entities |
| `entity_properties` | Column roles per entity |
| `external_log_export_state` | State for scheduled external log export |
| `answer_feedback` | User thumbs up/down per answer |
| `learning_candidate` | Scored answer candidates pending admin review |
| `recommendation_event` | Suggestion display/click/execute/dismiss events |

---

## 14. Feature Flags (per-client toggles)

All flags live on the `client` table. Set in Admin → Client → Edit.

| Flag | Column | Effect when = 1 |
|---|---|---|
| Chat UI | `chat_ui_enabled` | Enables portal chat page |
| LLM Audit | `enable_llm_audit` | Logs every LLM call to `llm_call_log` |
| Feedback collection | `enable_feedback_collection` | Creates learning candidates after each answer; shows thumbs UI in portal |
| Learned retrieval | `enable_learned_retrieval` | *(reserved)* Future: controls whether governed examples are used in retrieval |
| Genie suggestions | `enable_genie_suggestions` | Activates behavioral ranking for suggestion chips; records impression events |

---

## 15. External Integrations

### LLM Providers
**File:** `core/llm.py:resolve_provider`

- **Anthropic (Claude)** — default. Uses `anthropic` SDK. Models: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-3-5, etc.
- **OpenAI** — per-client override. Uses `openai` SDK. Models: gpt-4o, gpt-4-turbo, etc.
- **Azure OpenAI** — configured via `AZURE_OPENAI_*` env vars.

Resolution order: per-client API key → system-level API key. Missing key → error sent to user.

### Qdrant (Vector DB)
**Files:** `core/vector_store.py`, `core/governed_store.py`

Two collections:
- `querybot_kb` — KB docs + legacy auto-harvested examples (read-only for learning loop)
- `querybot_governed` — admin-approved learning candidates (written by learning loop only)

Connection: single shared singleton from `core/vector_store._qdrant()`. Both collections reuse it.

### External Log Export
**File:** `core/log_export.py`

Scheduled task (startup) exports query logs + LLM audit logs to the tenant's own DB table at a configured interval. Used for billing reconciliation and compliance reporting.

---

## 16. Data Security & Masking

**Files:** `core/masking.py`, `admin/routes.py:4128`, `core/synthetic.py`

### PII masking during KB build
1. Admin marks sensitive columns before KB build.
2. `core/masking.py` replaces real sample values with synthetic equivalents.
3. Replacement strategies: `redact` (→ `[REDACTED]`), `fake_name`, `fake_email`, `fake_date`, `constant`.
4. `kb_data_egress_log` records `masked_fields` and `mask_replacement_map` for audit.

### Column sensitivity preview
`GET /clients/{account_id}/setup/column-sensitivity` — shows a preview of which columns would be flagged as sensitive by the auto-detector before committing to a build.

### Synthetic sample guard
`core/synthetic.py` generates plausible synthetic rows when a column is marked as PII. The LLM sees realistic-looking data but no real values.

### Credential encryption
All database credentials and API keys are AES-encrypted at rest using `store/crypto.py`. The encryption key is set via `ENCRYPTION_KEY` env var.

---

*End of architecture document. Update this file whenever a new module, feature flag, or edge-case guard is added.*
