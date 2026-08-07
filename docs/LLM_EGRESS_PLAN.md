# LLM Data Egress — Boundary Hardening & Per-Question Egress Log

**Researched:** 2026-08-04 against `origin/main` @ `207fa1c`
**Scope:** what actually reaches an LLM when answering a question, and what we can prove about it.

## Context

The product claims: *for regulated tenants, actual data records are never sent to the LLM.*
This document records what is actually true today, where the claim does not hold, and what to
build so it holds and is demonstrable to an auditor.

Two deliverables:
- **Part A — close the boundary holes**, so the claim is true on every path.
- **Part B — a per-question egress log**, so for any question we can show: *this question,
  these tables, these columns went out; zero data values.*

---

## 1. What is already true (do not rebuild)

- **`llm_complete` is the single funnel** (`core/llm.py:1288`) and unconditionally calls
  `record_llm_call` (`core/llm_audit.py:210`).
- **Prompt text is never stored raw.** `llm_call_log` (`store/db.py:197-215`) stores a SHA-256
  of the prompt plus a *sanitized, truncated* preview — emails/GUIDs/phones/long numbers/quoted
  literals masked, table and column names deliberately preserved
  (`core/llm_audit.py:170-203`). Same for responses.
- **Four result-narration features are genuinely gated** by
  `result_llm_features_allowed()` (`core/compliance/policy_engine.py:15-31`), each writing a
  `record_llm_blocked` proof row: auto why-insight (`core/query_pipeline.py:355`), all analysis
  action buttons (`core/response_builder.py:1511`), follow-up suggestions
  (`core/insight.py:1341`), period comparison (`core/period_comparison.py:401`).
- **The metadata follow-up planner is strict and correct.** `core/result_planner.py:339-399`
  binds every quoted literal, number, email, GUID and phone to `VALUE_REF_n`; bindings stay
  in-process; manifest is columns + row_count only. `report_planner.py` uses the same discipline
  with `METRIC_REF_n`.
- **Result-chat narration is governed** — routes to `core/governed_result_followup.py:81-93`
  which asserts `rows_sent_to_llm: 0`.
- **KB sample rows are masked before reaching KB markdown** (`core/schema.py:130-243`), with
  `mode="none"` force-downgraded to `"auto"` and a final NER/regex pass.
- **Admin UI already groups LLM calls by question** (`admin/templates/client_detail.html:955-1140`),
  with retention (`purge_old_llm_calls`, 30d) and daily export to the tenant's own warehouse
  (`core/log_export.py`).

That is a real foundation. The gaps below are additions to it, not a rewrite.

---

## 2. Boundary holes (Part A)

Ranked by risk to a regulated tenant.

### A1 — Value index injects verbatim cell values into the SQL prompt · HIGH · by design
`core/value_resolver.py:281-289` emits a block headed
`VERIFIED FILTER VALUES (matched against actual database contents)` containing lines like
`user text 'emco' -> DB.SCH.CUSTOMER.NAME = 'EMCO Corporation'`, injected at
`core/query_pipeline.py:1394-1409` / `:1486`. `_sanitize` (`value_resolver.py:272`) only strips
newlines and caps length — no redaction.

The gate is `value_index_enabled(state)` — a **client feature flag**
(`core/value_index.py:85-89`), **not a compliance check**. Protection today rests entirely on
index-build-time heuristics (masked fields skipped, name-pattern PII skipped, value-level PII
scan) — the same heuristic reliance that `policy_engine.py:21-23` itself calls insufficient.

**Fix:** gate the injection on compliance mode, not just the feature flag. For regulated
tenants either (a) suppress the block entirely, or (b) inject *column-scoped* verified values
only for columns whose classification is non-sensitive **and** admin-reviewed. Recommend (b) —
it keeps most of the accuracy benefit. Decision needed; see §5.

### A2 — Raw DB error text goes into the repair prompt · HIGH · unintentional
`core/query_pipeline.py:3173-3177` interpolates `f"Error: {exec_error}"` verbatim, plus the
failed SQL (which carries WHERE literals). Same at `gateway/webhooks.py:2896`. Drivers echo
offending values — *"Conversion failed when converting the varchar value 'Lipitor 40mg'"*.

**Fix — cheapest high-value item in this document:** `core/failure_messages.py:109` already has
`sanitize_db_error()`. It is used for user-facing text but not on this path. Route the repair
prompt through it, and additionally strip literals from the echoed SQL for regulated tenants.

### A3 — Few-shot KB examples embed literal values · MEDIUM-HIGH · by design
`core/llm.py:1229-1236` instructs KB generation to write Q→SQL pairs *"using actual distinct
values from the KB"*. Those examples are retrieved and concatenated into every SQL-generation
call (`core/query_pipeline.py:1342`, `gateway/webhooks.py:2657`). Mitigated only upstream:
masked columns have distinct values stripped at build (`core/schema.py:1703`), high-cardinality
capped at 30.

**Fix:** for regulated tenants, filter retrieved examples through a literal-scrubber before
prompt assembly (reuse the `VALUE_REF` approach or replace literals with `<VALUE>` placeholders —
few-shot examples teach *shape*, and placeholders preserve that).

### A4 — `compute_data_brief`'s docstring is false · MEDIUM
`core/insight.py:300` documents "NEVER contains raw row values" but carries them:
`brief["value"]` is the literal single-cell result (`:346`), `category_breakdown.top_5`/`bottom_3`
are actual label strings + values (`:387-388`), `time_series` period labels (`:431-432`). The
only filter is `_is_sensitive_field()` (`:242`) — **column-name keyword matching**, so
`PRODUCT_NAME`/`DRUG` survive as literal values.

Regulated tenants are protected because the four consumers are gated — but the docstring is
actively misleading to anyone adding a fifth consumer. **Fix:** correct the docstring, and
rename to reflect reality (e.g. `compute_data_brief(..., include_values: bool)`).

### A5 — `drill_dim` escapes the boundary · MEDIUM
`core/drill_dimension.py:192` (via `gateway/webhooks.py:3264`) is a result-follow-up action
**not** behind `result_llm_features_allowed`. It sends no rows, but it does send the original
SQL including WHERE literals. **Fix:** put it behind the same gate as its sibling actions.

### A6 — Four different "is regulated" predicates · MEDIUM
1. `profile["mode"] == "regulated"` (canonical) — `policy_engine.py:31`, `query_pipeline.py:744`
2. `mode == "regulated" AND enforcement_mode == "enforce"` — `core/result_renderer.py:771`,
   `portal/routes.py:1929`. **A regulated tenant in `shadow` mode fails OPEN on both.**
3. `mode != "standard"` — `core/agent_runtime.py:47`
4. `industry in ("banking","healthcare_pharmacy")` — `core/masking.py:300`, drives NER scrubbing

**Fix:** one predicate, one helper (`is_regulated(account_id)`), used everywhere; keep the
industry axis only where it genuinely means industry (NER model selection).

### A7 — `result_llm_features_allowed` fails open on a missing profile · MEDIUM
`store/compliance_store.py:21` synthesizes a default with `"mode": "standard"` when no row
exists, so an unprovisioned tenant gets LLM features enabled. **Fix:** fail closed for unknown
tenants, or require an explicit provisioning step before any LLM feature is permitted.

### A8 — Smaller items
- `core/analysis_code_planner.py:61-63` puts the raw user request in the prompt with no
  `VALUE_REF` sanitisation — protected only by the regulated-only, regex-shaped question scrub.
- Clarification options are harvested from KB `distinct_values`
  (`core/clarification.py:452-472`) with no compliance re-check.
- Both the **raw** and scrubbed question are written to the trace store
  (`core/query_pipeline.py:742-754`) — the audit store therefore holds PHI-bearing text. Justified
  as tenant-local, but it should be a documented, deliberate decision, not incidental.
- Question scrubbing (`core/masking.py:272-302`) silently degrades to regex-only if
  spaCy/Presidio are absent (`:236-239`). An installation-dependent control should log loudly
  and surface in readiness.
- `core/result_renderer.py:309` (`_generate_result_narration`) would send 5 raw rows verbatim.
  It is dead code with no callers — **delete it** rather than leave a loaded gun.

---

## 3. Logging gaps (Part B)

- **Three answer-path LLM calls are entirely unaudited**: `gateway/webhooks.py:2792` (result-chat
  fallback SQL generation), `:2841` (validation retry), `:2903` (execution retry). The nearest
  scope closes at `:2426`. Also unaudited: `evals/run.py:232`.
- **No structured statement of what was sent.** The sanitized preview is good forensics but
  cannot answer *"were any data values sent?"* without a human reading it.
- **No join from a question to its LLM calls.** `answer_trace` and `llm_call_log` both have
  `question_id` but nothing enforces or queries the link; the admin UI is browse-only with no
  deep-link, date range, or search.
- **KB build shares one `request_id`** across every table (`admin/routes.py:8940-8955`). Only
  `kb_table_doc` is disambiguated, via a `question=table_name` override
  (`core/knowledge.py:904-908`); `kb_query_examples` and `kb_business_vocab` remain untraceable.
- **`kb_data_egress_log` has no `request_id`/`question_id`** (`store/db.py:664-690`) — cannot be
  correlated to `llm_call_log`.
- **Export omits the response side** — `LLM_COLUMNS` (`core/log_export.py:38-42`) drops
  `response_hash`, `response_preview_sanitized`, `response_chars`.
- **`kb_data_egress_log` has no purge** (only `llm_call_log` does, startup-only).

### The log to build: a per-call egress manifest

Add a structured `egress_manifest` JSON column to `llm_call_log`, populated by `record_llm_call`
from data the prompt builder already has:

```json
{
  "tables":  ["PHARMA_LAB.F_RX_FILL", "PHARMA_LAB.D_PRODUCT"],
  "columns": ["FILL_DATE", "PRODUCT_ID", "NET_REVENUE"],
  "content": ["question", "schema", "kb_docs", "few_shot", "semantic_model"],
  "values_sent": false,
  "value_sources": [],
  "value_count": 0
}
```

`values_sent` is computed, not asserted — set true when the value-index block, a literal-bearing
few-shot example, or an unsanitised DB error is included. That single boolean is what makes the
compliance claim checkable rather than a matter of trust, and it turns the admin view into a
plain-language statement:

> **Q: "net revenue by product last quarter"** → 2 LLM calls
> • `sql_generation` — sent: question, 2 tables, 8 columns, 3 KB docs, 3 examples. **0 data values.**
> • `sql_repair` — sent: question, previous SQL, sanitized error. **0 data values.**

---

## 4. Phasing

| Phase | Content | Size |
|---|---|---|
| **0** | A2 (sanitize DB error) + A5 (gate drill_dim) + delete dead `_generate_result_narration` + wrap the 3 unaudited call sites | Small — highest value/effort |
| **1** | A6 (single `is_regulated` predicate) + A7 (fail closed) + A4 (docstring/signature) | Small |
| **2** | Egress manifest column + population + admin per-question view + `answer_trace`↔`llm_call_log` link | Medium |
| **3** | A1 (value-index gating — needs the §5 decision) + A3 (few-shot literal scrubbing) | Medium |
| **4** | KB-build per-component `request_id`, `kb_data_egress_log` correlation + purge, export response columns, auditor CSV/PDF export | Medium |

Phase 0 alone closes both unintentional leaks and the audit blind spots.

## 5. Decision needed before Phase 3

**A1 — how much accuracy to trade for the boundary.** Verified filter values measurably improve
SQL correctness (they exist because the LLM otherwise invents literals). Options:
- **(a) Suppress entirely for regulated tenants** — safest, measurable accuracy loss.
- **(b) Allow only for columns classified non-sensitive *and* admin-reviewed** — keeps most of the
  benefit, relies on the classification workflow being complete (which readiness already gates on).
- **(c) Send masked/pseudonymised values** — breaks the feature's purpose; not recommended.

Recommend **(b)**, with **(a)** as the setting for tenants that have not completed classification.

## 6. Verification

- **The missing test class is negative-space on the assembled SQL prompt.** Today no test asserts
  a sensitive value is absent from a built SQL-generation prompt. Add, for a regulated tenant:
  value-index block suppressed/filtered; retrieved few-shot examples literal-free; repair prompt
  carrying a sanitized error. Model on `tests/test_masking_privacy.py:498-501` and
  `tests/test_governed_result_exclusion.py:33-34`, which already do this well elsewhere.
- **Convert the two source-scan boundary tests** in `tests/test_regulated_llm_boundary.py:159-190`
  (follow-ups, result-chat narration) to runtime assertions like their four siblings.
- **Manual on Demo_2:** ask a question with a filter on a real product name; confirm the egress
  manifest reports `values_sent: false`, and that the sanitized preview contains no product name.
- Full suite green (baseline 3,711 on `207fa1c`).
