# QueryBot core v2: technical design

Status: draft 1, branch `core-v2`. This is the contract every core-v2 change is
built and reviewed against. When the code and this document disagree, one of
them is wrong; fix whichever it is in the same change.

The product-level plan (what changes for users, phases, targets) is the
"QueryBot core plan" document shared with the product owner. This file is the
engineering design behind it.

---

## 1. Principles

1. **Learn once, plan once, compile by rules.** The warehouse is studied when it
   is connected (bootstrap). Each question becomes one typed plan (the AI's only
   job at question time). A compiler turns the plan into SQL by rules.
2. **Evidence over names.** Every inference (a key, a join, a date role, a
   measure's behaviour) records the evidence it rests on. Names are one signal
   among several; vocabulary packs are weak evidence only. No English word list
   decides meaning anywhere on the question path.
3. **One representation per concept.** One semantic model (versioned), one plan
   type, one join-path rule, one time model, one compiler, one answer builder.
4. **The AI never writes SQL for a question the plan can express**, and never
   sees rows: it sees names, profiles (aggregates computed in the warehouse) and,
   where allowed, a few common values of short code and name columns.
5. **Admin decisions always win** and survive rebuilds. Imported definitions
   (data dictionary, dbt, LookML, Power BI) rank next. Nothing waits for an admin.
6. **Ask rather than guess** when the evidence is genuinely tied. Never answer
   with a different measure than the one asked for.
7. **Deterministic.** Same question, same model version, same conversation
   state: same plan, same SQL, same answer. Temperature 0, canonical prompts,
   stable ordering everywhere (dict iteration order is never a tie-break).
8. **Generic.** Nothing in the code assumes one ERP's naming, one industry or
   English column names.

Targets (measured by the eval harness, §11): at least 90% of golden questions
answered correctly on every dataset; under 2% answered wrongly (the rest ask or
decline); a typical answer in about 5 seconds.

---

## 2. Shape of the system

```
          connect / schedule                       question (web portal)
                 │                                          │
        ┌────────▼─────────┐                       ┌────────▼────────┐
        │    bootstrap     │  profiles, joins,     │  value lookup   │ deterministic
        │ (core2.bootstrap)│  dates, measures,     │ (core2.plan)    │
        └────────┬─────────┘  labels, quality      └────────┬────────┘
                 │                                          │
        ┌────────▼─────────┐   cached prefix (CAG)  ┌───────▼────────┐
        │ semantic model v2├───────────────────────►│  AI planner    │ one LLM call
        │ (versioned, +    │                        │  → typed plan  │
        │  admin overrides)│◄──────── checks ───────┤  (core2.plan)  │
        └────────┬─────────┘                        └───────┬────────┘
                 │                                          │
                 │                                 ┌────────▼────────┐
                 └────────────────────────────────►│    resolver     │ joins, dates,
                                                   │ (core2.resolve) │ windows, grains
                                                   └────────┬────────┘
                                                   ┌────────▼────────┐
                                                   │    compiler     │ sqlglot AST,
                                                   │ (core2.compile) │ 3 dialects
                                                   └────────┬────────┘
                                           existing validator, governance, executor
                                                   ┌────────▼────────┐
                                                   │ answer builder  │ from the plan
                                                   │ (core2.answer)  │
                                                   └─────────────────┘
```

Package `core2/` (top level, beside `core/`), so nothing in it is imported by
today's pipeline and it can be removed or renamed in one step:

```
core2/
  ids.py                 canonical identity keys (§3)
  model/schema.py        pydantic: SemanticModel and its parts (§4)
  model/catalog.py       the planner-facing rendering of a model (CAG prefix, §7.2)
  model/overrides.py     admin/import overrides applied on top of a built model
  model/values.py        member value index: lookup, distinct values (§7.3)
  warehouse/dialect.py   per-dialect SQL fragments (§9.2)
  warehouse/runner.py    Warehouse protocol; DuckDB and QueryBot-connection runners
  bootstrap/             inventory, profiler, keys, joins, calendar, dates,
                         kinds, measures, quality, labels, build (§5, §6)
  plan/ir.py             the typed plan (§7.1)
  plan/prompt.py         planner prompt builder
  plan/planner.py        LLM call, parse, validate, one repair, optional tool round
  plan/followup.py       conversation state, clarification replies, cached-result path
  resolve/               resolver, join paths (§6.4), time (§8)
  compile/               compiler and query shapes (§9)
  answer/                answer payload, findings, formats (§10)
  service.py             answer_question(): the channel-independent entry point
store/core2_store.py     persistence (§4.6)
evals/core2/            synthetic domains, golden questions, runners (§11)
tests/core2/            unit and gate tests
```

---

## 3. Identity

One identity key per warehouse object, fixed here and used by every writer and
reader in core2 (a mismatch between writer and reader keys is the quietest bug
this product has had):

* **Table key**: `"{database}.{schema}.{table}"` in the warehouse's stored
  spelling. Comparison is always through `core2.ids.norm()` (casefold, strip
  quotes/brackets). The stored spelling is kept for SQL generation.
* **Column key**: `"{table key}.{column}"`.
* **Join key**: `"{from table}({c1,c2})->{to table}({c1,c2})"` plus `"#{role}"`
  when the same columns are used in more than one role.
* **Date role key**: `"{table key}.{column}"` (a date role is a column's use as
  a date; the calendar join is part of the role, not of its key).
* **Measure key**: `"m:{table key}.{name}"` for column-derived measures,
  `"metric:{metric id}"` for approved metrics, `"import:{source}:{name}"` for
  imported ones.

Planner-facing names (**slugs**) are separate: short, readable, unique within a
model version (`revenue`, `invoice_date`, `customer.name`). Slugs are generated
from business names at build time and kept stable across versions for the same
key (§4.5). The plan the AI writes uses slugs; the bound plan stored with a
conversation uses keys.

---

## 4. Semantic model v2

### 4.1 Objects

All objects carry `evidence: list[Evidence]`, `confidence: float (0..1)`,
`provenance` (`profile | ai | heuristic | admin | import | metric`) and `status`
(`verified | proposed | approved | rejected | needs_review`).

`Evidence = {kind, detail, weight, data}`; `detail` is one human-readable line
("99.7% of 20,000 rows match; target unique").

* **Table**: key, database, schema, name, business_name, description,
  `kind` (`fact | snapshot | dimension | bridge | calendar | other`), grain
  (column keys whose combination is unique, with `grain_text` "one row per
  invoice line"), row_count, primary_key (declared or inferred and verified),
  default_date (date role key), default_filters (approved only are applied),
  slug, profiled_at.
* **Column**: key, table, name, data_type (normalised: `integer | decimal |
  float | text | date | timestamp | boolean | other`), raw_type, `role`
  (`measure | attribute | label | code | key | foreign_key | date | date_key |
  period_key | status | flag | audit | identifier | text | unknown`),
  business_name, description, synonyms (per language), unit, `format`
  (`currency | percent | count | number | integer | text | date | datetime`),
  label_of (the key column this column names), sensitivity
  (`none | pii | confidential`), profile (§5.2).
* **Join**: from table and columns, to table and columns, `cardinality`
  (`many_to_one | one_to_one | one_to_many | many_to_many`), match_rate (share of
  non-null, non-placeholder from-rows that find a match), null_rate,
  orphan_rows, to_unique, max_fanout, role (business name of the role when the
  same pair of tables is joined more than one way), `trust`
  (`admin | declared | verified | proposed | rejected`).
* **Calendar**: table, key column, date column, attribute map (canonical
  attribute → column key, each verified against the date), first/last date,
  contiguous, week_start (of its own week columns, if any), fiscal year start
  month (inferred from fiscal columns), placeholder keys.
* **DateRole**: key, table, column, business name, via_calendar (the column is a
  key into a calendar), granularity (`day | month | year | timestamp`),
  `kind` (`event | snapshot | planned | due | validity | audit`), is_default,
  score with its evidence, coverage, placeholder share, first/last date, slug.
* **Measure**: key, slug, business_name, description, synonyms, base table,
  `expr` (structured, §4.2), `additivity` (`additive | semi_additive |
  non_additive`), `time_aggregation` for semi-additive (`last | average | first`),
  format, unit, unit_column, default_date, filters, provenance
  (`metric | import | model | proposed | column`), tested (dry-run result).
* **Entity** (a dimension): slug, business name, table, key column, label
  column, code column, members count, parent entity (hierarchy), role (for
  role-playing variants), value-indexed flag.
* **Attribute**: slug (`entity.attr`), column key, business name, members
  count, value-indexed flag. Degenerate attributes live on fact tables
  (`invoice.status`).
* **QualityFlag**: object key, `kind` (`outlier_period | negative_values |
  constant | low_match_rate | status_column | placeholder_dates |
  listed_vs_active | load_timestamp | unit_mix`), message, data, severity.

`SemanticModel = {client_id, db_id, db_type, version, built_at, source_hash,
settings, tables, columns, joins, calendars, date_roles, measures, entities,
attributes, quality, review_queue, slugs}`.

### 4.2 Measure expressions

Structured, never SQL text from the model:

```
expr := {"agg": "sum|count|count_distinct|avg|min|max", "column": <column key>|null,
         "filters": [Filter]}                      # count with column null = count rows
      | {"op": "ratio|subtract|add|multiply", "args": [expr, expr], "scale": 1|100}
      | {"measure": <measure key>}                 # reuse another measure
      | {"sql": <expression>, "columns": [...]}    # imported approved metric that has
                                                   # no structural form: parsed with sqlglot,
                                                   # every column resolved, rendered via AST
```

Ratios compile to `numerator / NULLIF(denominator, 0)` with each side aggregated
over the same rows (never a ratio of row-level values summed).

### 4.3 Precedence

For any measure the question names: approved metric → imported/semantic-layer
measure → model measure (column-derived, verified) → proposed derived measure
(shown as proposed in the answer) → ask. A different measure is never
substituted.

For any fact about the data: admin override → import → verified by data →
AI label → heuristic.

### 4.4 Versions and diffs

Every build writes a new version. A build diff (tables/columns added or removed,
joins promoted/demoted, defaults changed, measures added) is stored with it and
shown on the "What QueryBot learned" page. Conversations keep the model version
they were answered with; a follow-up after a rebuild re-binds the previous
plan's keys to the new version and says so if anything it used disappeared.

### 4.5 Overrides

`core2_overrides(client, db, object_key, field, value, author, at, note)`: one row
per decision (rename, set default date, approve/reject a join, mark audit, set a
label column, set a default filter, hide a column, add a synonym). Applied after
every build, so they survive rebuilds. Today's approvals (confirmed joins,
approved date roles and defaults, certified metrics, field overrides, synonyms)
are imported once into overrides (§12.4).

### 4.6 Storage

`store/core2_store.py`, direct SQL like the rest of `store/`, idempotent DDL at
startup:

* `core2_models(id, client_id, db_id, version, status, built_at, source_hash,
  model_json, diff_json, stats_json)`; the last 10 versions are kept.
* `core2_overrides(...)` as above, unique on (client, db, object_key, field).
* `core2_profiles(client_id, db_id, table_key, schema_hash, profiled_at,
  profile_json)`: lets a rebuild re-profile only what changed.
* `core2_values(client_id, db_id, attribute_key, refreshed_at, values_json)`:
  member values for lookup (bounded; §7.3).
* `core2_turns(session_key, turn, at, question, plan_json, model_version,
  result_json)`: conversation state for follow-ups.

---

## 5. Bootstrap: learning the data

Runs in the background when a database is connected, on schema change, and on a
schedule. Each step reads the previous steps' output only; each writes evidence.

### 5.1 Inventory
Tables, columns, types, declared primary and foreign keys, comments, row counts,
within the admin's chosen scope (schemas/tables). Reuses today's discovery
output when present; otherwise queries `INFORMATION_SCHEMA` (per dialect).

### 5.2 Profile (SQL in the warehouse; only aggregates come back)
Per table one aggregate query (all columns): `COUNT(*)`, per column
`COUNT(col)`, approximate distinct, `MIN`, `MAX`; numeric: negatives, zeros,
integer-valued share; text: min/avg/max length; temporal: distinct timestamps
and time-of-day spread. Per low-cardinality column (≤ 50 distinct) one top-k
query (value, count). Pattern detection from min/max/lengths and top values:
`yyyymmdd`, `yyyymm`, `yyyy`, 0/1 flag, Y/N flag, short code, free text.

Cost controls: tables over 5 M rows are profiled on a sample (dialect sampling,
§9.2); every query has a timeout; at most 4 run concurrently; a per-build budget
stops profiling and marks the model partial rather than overloading the
warehouse. All SQL is generated by code (no AI), read-only, tagged.

### 5.3 Keys
Primary key: declared, else the smallest column set with distinct = rows and no
nulls, searched over key-like columns (single columns first; composites of up to
3 foreign-key/date-key columns for facts and snapshots). Verified by a
`COUNT(*) vs COUNT(DISTINCT ...)` query.

### 5.4 Calendar detection
A table is a calendar when: one column is a date, unique per row, contiguous
(rows = days between min and max); a key column maps 1:1 to it (integer
`yyyymmdd` or surrogate); and at least two other columns verify as calendar
attributes. Attribute verification is by SQL against the date (month number =
`EXTRACT(MONTH)`, year, quarter, day of week (ISO or Sunday-first), month start,
week start, month name in any language via consistency within each month).
Fiscal columns: the offset between fiscal month and calendar month gives the
fiscal year's start month; the year label convention (named by the year it ends
or starts) is read off the data. Placeholder keys: key values with no date or a
sentinel date (1900-01-01, 9999-12-31) or an "Unknown" label.

### 5.5 Joins (§6 has the rules)

### 5.6 Table kinds
* calendar: §5.4.
* dimension: has a unique key referenced by at least one verified join, and
  label-like text columns.
* bridge: two or more foreign keys, no measures, its key pairs are many-to-many.
* snapshot: a fact whose grain includes a date/period key, with few distinct
  dates each holding many rows at regular spacing (daily, month-end, period),
  and balance-like measures (the same entity repeats each period).
* fact: measures plus foreign keys, many rows, an event date.
* other: everything else (shown, never used for joins until reviewed).

### 5.7 Date roles (§8.1 has the scoring)

### 5.8 Measures
Numeric columns on facts and snapshots that are not keys, codes, flags or
date keys (decided by profile: distinct ratio, integer share, range, join
membership; a name is only extra evidence). Additivity: additive on facts;
semi-additive (last value over time) on snapshots for balances and counts on
hand; non-additive for unit prices, rates and percentages (averaged with a
note, or recomputed from parts when the parts are known). Units: a quantity
measure records its unit column when the fact or a joined dimension has a
low-cardinality unit-of-measure column. Each fact gets a row-count measure
("number of invoice lines") and a distinct count per business identifier
("number of invoices"). Derived measures are proposed (margin % from margin and
revenue, average price from amount and quantity) and dry-run before they are
offered.

### 5.9 Labels (the only AI step in bootstrap)
One call per table (batched, bounded concurrency): names, types, profile
summary, evidence-derived facts; common values only for short code/name
columns and only where the tenant allows it (never for regulated tenants).
The AI returns business names, descriptions, grain sentence, synonyms (English
and French), units/currency, and chooses among the roles the evidence allows; it
cannot overrule evidence (a non-unique column cannot become a key). Without a
configured AI provider the deterministic fallback expands names into words
(abbreviation dictionary + splitting) so a model is always produced; the page
shows which labels came from where.

### 5.10 Quality
Outlier periods (robust z-score of period totals for each main measure),
negative values in count-like measures, constant measures, low join match
rates, status columns that look like cancellations/credits (proposed default
filter, asked once), placeholder dates, listed vs active members (members with
no fact rows), audit timestamps, mixed units.

---

## 6. Joins

### 6.1 Candidates
1. Declared foreign keys (enforced or not).
2. Name evidence, in any style: `customer_id → customers.id`,
   `CUST_KEY → DIM_CUSTOMER.CUST_KEY`, `CustomerKey → DimCustomer.CustomerKey`,
   same-named columns where one side is unique.
3. Value containment: column A's distinct values mostly found in column B,
   where B is unique, types compatible and A's range within B's. Pre-filtered
   from profiles so only plausible pairs are tested.
4. AI-proposed (labels step), tested like the rest.

### 6.2 Test-run
For each candidate one query: rows, non-null rows, matched rows, distinct
unmatched values, and target duplicates (`COUNT(*) - COUNT(DISTINCT)` on the
target key). Placeholder keys are excluded from the match rate and counted
separately.

### 6.3 Trust
`admin` (approved) > `declared` (declared and passes) > `verified` (match rate
≥ 99%, target unique, many-to-one) > `proposed` (passes partially: shown in the
review queue, used only when no better path exists, and named in the answer) >
`rejected` (admin rejected or fails: never used). The calendar is joined only as
a date role (§8.2), never as a link between other tables.

### 6.4 One path rule
From a base table F to the table holding the attribute A:

1. Only paths that never multiply F's rows: every step many-to-one or one-to-one
   (a one-to-many or many-to-many step under an aggregate is never taken; a
   bridge is used only with explicit allocation, which v2 does not do yet: it
   asks).
2. The measure table's own key first: a direct edge from F wins over any longer
   path.
3. Then higher trust, then higher match rate, then fewer steps, then the
   lexicographically smallest join key (deterministic, and reported).
4. When two valid paths mean different things (the invoice's profit centre
   vs the customer's profit centre), the chosen one is named in the answer and
   the catalog lists the other as a qualified attribute the user can ask for
   ("by the customer's profit centre").

Every grouping and filter on the same attribute uses the same path, in every
query shape: rankings, trends and filters agree.

---

## 7. Planning a question

### 7.1 The plan (typed IR)

The planner returns exactly one JSON object validated by pydantic (strict, no
extra fields). Names are slugs from the catalog.

```
Plan {
  kind: "query" | "clarify" | "describe_data" | "unsupported" | "smalltalk"
  intent: "value" | "breakdown" | "trend" | "compare" | "rank" | "share" | "drivers"
        | "list" | "count" | "forecast" | null
  measures: [slug]
  proposed_measures: [{name, agg, column: slug, filters: [Filter]}]
  group_by: [slug | "time:day_of_week" | "time:month_of_year" | ...]
  via: {attribute slug: qualifier slug}        # path qualifier, rarely used
  filters: [Filter]
  time: {
    date: slug | null                          # null = the measure's / table's default
    grain: "day"|"week"|"month"|"quarter"|"year"|"fiscal_month"|"fiscal_quarter"|"fiscal_year"|null
    window: {kind: "all"|"between"|"since"|"until"|"last"|"this"|"to_date"|"previous",
             start: date|null, end: date|null,  # inclusive dates as the user means them
             unit: grain|null, n: int|null, fiscal: bool}
    compare: null | {kind: "previous_period"|"same_period_last_year"|"window",
                     window: <window>|null}
  }
  sort: [{by: slug|"change"|"pct_change"|"share", desc: bool}]
  limit: int | null
  forecast: {periods: int} | null
  drivers: {dimensions: [slug] | null} | null
  clarify: {about: "measure"|"date"|"member"|"other", question: str, options: [str]} | null
  follow_up: "new" | "refine"
  notes: [str]                                 # assumptions, in plain words
}
Filter { field: slug, op: "in"|"not_in"|"eq"|"ne"|"gt"|"gte"|"lt"|"lte"|"between"
                          |"contains"|"starts_with"|"is_null"|"not_null",
         values: [str|number|bool] }
```

Relative windows are written relatively (`last 6 months`) and resolved by code
(§8.3), never by the AI's date arithmetic. Explicit dates are written as dates.

### 7.2 The prompt
* **Stable prefix (cached, CAG)**: role and rules, the plan's JSON schema, and
  the model catalog: measures (slug, name, definition in words, format, default
  date, synonyms), entities and attributes (slug, name, members count, path
  qualifiers), date roles (slug, table, default flag, data range), time
  attributes, the fiscal calendar, and verified example questions with their
  plans. Byte-identical for a model version: no timestamps, no question, sorted.
  Joins are not in the prompt: paths are the compiler's job.
* **Dynamic tail**: today's date, the conversation (previous plan rendered with
  current slugs, previous question, pending clarification), value matches from
  §7.3, and the question.

Very large models (catalog over ~60k tokens) are trimmed to the subject areas a
cheap first pass selects; the eval harness measures the loss.

### 7.3 Values
Before the AI is called, the question is matched against the member index:
normalised (case, accents, quotes/dashes) n-grams of up to 8 tokens, keeping
`#`, digits and punctuation inside names; exact matches first, then whole-token
containment, longest first; no stop-word lists. Matches are given to the
planner as candidates (`"distribution 58" → customer.name = "Acme Distribution
58"`). Attributes with too many members for the index are looked up at question
time with a bounded `LIKE` query (a tool the planner may call once). For
regulated tenants matched values are replaced by placeholders before the
prompt and substituted back after planning: the AI never sees a member value.

### 7.4 Checks (before anything runs)
Every slug exists; measures are measures and attributes are attributes; filter
values exist (exact or a single normalised match, else ask with the closest
members); the date role belongs to a table reachable from the measure; the
grain is supported by the date's granularity; the window is valid. One repair
round: the errors are sent back with the allowed options. A plan that still
fails becomes a clarification, never a guess.

### 7.5 Follow-ups
The planner sees the previous plan and returns a full new plan with
`follow_up: "refine"` or `"new"`. Code computes the change ("measure → gross
profit") for the answer. When the new plan differs from the previous one only
in sort, limit or filters on columns of the result on screen, and that result
was complete, it is answered from the session's cached result (DuckDB), with no
warehouse query. A reply to a clarification is matched to its options by code
(exact, ordinal, unique prefix) and completes the original plan without
re-planning.

---

## 8. Time

### 8.1 Date-role scoring
Candidates: date and timestamp columns, integer/text columns that join a
calendar or match `yyyymmdd`/`yyyymm`/`yyyy`. Evidence (weights tuned on the
synthetic domains, recorded in the role's evidence):
* part of the table's grain or key (snapshot periods);
* joins a calendar;
* coverage (non-null, non-placeholder share);
* load clustering: few distinct timestamps relative to rows, or values lagging
  the row's other dates by months (audit stamps);
* the role's name in business terms (AI label): matches the table's event
  ("invoice date" on invoice lines), or says planned/due/modified/loaded;
* order relations between dates (delivered on or after shipped).

The default is the highest-scoring non-audit role if it beats the runner-up by a
margin; snapshots default to their snapshot/period key. An audit stamp is never
a default. Close calls are listed in the review queue; the question path still
answers with the default and names it, and asks only when the question's
measure has no default and two business dates tie.

### 8.2 Calendars and role-playing dates
A date key column joins the calendar once per role under its own alias
(`invoice_date`, `order_date`); two roles in one question are two joins.
Placeholder keys are excluded from windows and counted ("12% of orders not
delivered yet"). Period buckets come from the calendar's verified columns
(week-start, month-start, fiscal columns) or are computed from the date
(§9.2). The calendar never links two facts.

### 8.3 Windows
Resolved by code to half-open `[start, end)` dates against `today` (injected,
for tests) and the date role's data range:
* `between a and b` → `[a, b+1 day)` at the unit of the dates given;
* `since a` → `[a, latest data]`; `until b` → `[earliest data, b]`;
* `last n units` → the n complete units before the current one (the current,
  partial unit is included only when asked: "including this month");
* `this unit`, `unit to date` (calendar or fiscal);
* `previous` → the unit before the current one.
If the data ends before the window starts, the window is re-anchored on the
latest data date and the answer says so ("your data ends 30 June 2026"). The
first and last periods of a series are flagged partial when the data does not
cover them fully; findings never compare a partial period with full ones.

### 8.4 Grains
`day, week (Monday start everywhere), month, quarter, year` and fiscal
`month/quarter/year`. Every period carries its start date, is ordered by it and
is labelled with its year ("Mar 2026", "Week of 29 Jun 2026", "Q1 2026",
"FY2026 Q3").

---

## 9. Resolve and compile

### 9.1 Resolver (plan + model → logical query)
1. Measures → expressions and base tables; group by base table.
2. Date role per base table: the plan's, if it belongs to that table; otherwise
   the measure's default, then the table's default.
3. Attributes and filters → join paths (§6.4); the same attribute always the
   same path.
4. Time window → concrete dates (§8.3); grain → bucket expression (§8.4).
5. Default filters (approved) and measure filters → predicates; listed in notes.
6. Semi-additive measures: per period (or for the whole window) only the rows of
   the last snapshot date in it.
7. Several base tables (drill-across): each aggregated separately at the shared
   grain (conformed attributes), then joined on the grouping columns; an
   attribute not reachable from every base table is a clarification.
8. Shape-specific expansion (§9.3).

The logical query records, for the answer: the date role used, the paths used
(and alternatives), filters, default filters, placeholders excluded, partial
periods, model version.

### 9.2 Dialects
The compiler builds sqlglot expressions and prints them per dialect
(`snowflake`, `tsql`, `oracle`, and `duckdb` for tests). Anything sqlglot does
not translate faithfully is a dialect fragment in `core2/warehouse/dialect.py`:

| Fragment | Snowflake | Azure SQL (T-SQL) | Oracle | DuckDB |
|---|---|---|---|---|
| date literal | `'2026-01-01'::DATE` | `CAST('2026-01-01' AS DATE)` | `DATE '2026-01-01'` | `DATE '2026-01-01'` |
| month start | `DATE_TRUNC('MONTH', d)` | `DATEFROMPARTS(YEAR(d), MONTH(d), 1)` | `TRUNC(d, 'MM')` | `DATE_TRUNC('month', d)` |
| quarter start | `DATE_TRUNC('QUARTER', d)` | `DATEFROMPARTS(YEAR(d), (DATEPART(QUARTER, d)-1)*3+1, 1)` | `TRUNC(d, 'Q')` | `DATE_TRUNC('quarter', d)` |
| year start | `DATE_TRUNC('YEAR', d)` | `DATEFROMPARTS(YEAR(d), 1, 1)` | `TRUNC(d, 'YYYY')` | `DATE_TRUNC('year', d)` |
| Monday week start | `DATEADD(DAY, 1 - DAYOFWEEKISO(d), d::DATE)` | `DATEADD(DAY, -((DATEPART(WEEKDAY, d) + @@DATEFIRST + 5) % 7), CAST(d AS DATE))` | `TRUNC(d, 'IW')` | `DATE_TRUNC('week', d)` |
| top n | `LIMIT n` | `TOP n` | `FETCH FIRST n ROWS ONLY` | `LIMIT n` |
| nulls last | `NULLS LAST` | `CASE WHEN x IS NULL THEN 1 ELSE 0 END, x` | `NULLS LAST` | `NULLS LAST` |
| approx distinct | `APPROX_COUNT_DISTINCT` | `APPROX_COUNT_DISTINCT` | `APPROX_COUNT_DISTINCT` | `approx_count_distinct` |
| sample | `SAMPLE (n ROWS)` | `TABLESAMPLE (p PERCENT)` | `SAMPLE (p)` | `USING SAMPLE n ROWS` |
| case-insensitive like | `ILIKE` | `LIKE` (CI collation) / `UPPER()` | `UPPER(x) LIKE UPPER(p)` | `ILIKE` |

(sqlglot 30 prints Oracle month truncation as `TIMESTAMP_TRUNC`, which Oracle
rejects, and T-SQL weeks as `DATETRUNC(WEEK, ...)`, which depends on the
server's `DATEFIRST`; hence the explicit fragments.) Identifiers are quoted only
when the stored spelling requires it. Division casts to a decimal on T-SQL.
Every compiled statement is re-parsed in its dialect before it leaves the
compiler.

### 9.3 Query shapes
* **value**: one row of measures over the window.
* **breakdown**: measures by attributes.
* **trend**: measures by period (+ attributes), ordered by period start.
* **compare**: conditional aggregation over two windows (`SUM(CASE WHEN d in P1
  ...)`), with change and percentage change (`NULLIF`-guarded); ranked by change
  for "grew/fell the most".
* **share**: `m / NULLIF(SUM(m) OVER (partition), 0)`.
* **rank**: order by the measure (then label, for determinism), top n.
* **drivers**: the compare shape run per candidate attribute (the plan's, or
  the top attributes by members and coverage, at most 4), contributions sorted by
  absolute change; one query per attribute, run concurrently.
* **list / count**: members of an entity or attribute (optionally only those
  with rows in a fact: "active"); counts are `COUNT(DISTINCT key)`.
* **forecast**: the trend shape, then the existing forecast models in Python on
  complete periods only; forecast periods are labelled in the series' grain.

### 9.4 Fallback
A question the plan cannot express (`kind: "unsupported"`) falls back to today's
pipeline when the switch allows it (§12.2), labelled as such. core2 itself never
asks the AI for SQL in v2.0.

---

## 10. The answer

Built from the plan and the logical query, never from column names or question
words:
* headline from the intent and measure ("Revenue by month, Jan–Jun 2026");
* formats from measure formats and units; periods labelled by grain;
* chart from the intent: trend → line (date order); rank/breakdown → bars
  sorted; share → bars (pie only for ≤ 6 parts); compare → bars with change;
  value → number tile;
* findings from the whole series: peak, trough, robust trend, spikes, partial
  periods excluded from comparisons, constant measures;
* "How this was answered": measure definitions, date used, join path(s),
  filters and default filters, placeholders excluded, model version, SQL;
* data-quality notes for the objects touched (one line each);
* next steps from the plan (drill into the top contributor, compare with the
  previous period, by week).

The payload matches what the web portal renders today (§12.3).

---

## 11. Evaluation

* **Synthetic domains** (`evals/core2/domains/`): retail, inventory, finance,
  purchasing, HR, subscriptions; each generated in DuckDB from a seeded logical
  spec in three naming styles: descriptive (`sales_order_lines.order_date_key`),
  warehouse codes (`SLS_ORD_LN_FCT.ORD_DT_KEY`), generic (`T07.C04`, no declared
  keys). Each injects the quirks real warehouses have: role-playing dates with a
  calendar, placeholder date keys, audit timestamps loaded months late, snapshot
  tables, a cancelled-status column, an outlier period, a constant measure,
  orphan rows, mixed units, names containing analysis words and symbols.
* **Ground truth** per domain: true kinds, keys, joins (with roles), calendar,
  date roles and defaults, measures with additivity/format/unit.
* **Golden questions** per domain (`golden.yaml`): question, expected plan in
  logical names, reference SQL in logical names (rewritten per naming style),
  expected properties (asks, partial periods flagged, notes), issue tags.
* **Levels**:
  1. *learn*: the learned model against ground truth (join precision/recall,
     default date accuracy, kind and measure accuracy). Deterministic. CI gate.
  2. *compile*: golden plans → resolver → compiler → DuckDB → compared with the
     reference SQL's result; plus parse-checks of the Snowflake, T-SQL and Oracle
     SQL. Deterministic. CI gate.
  3. *plan*: question → the configured AI → plan → result. Needs a provider; runs
     on demand and before each release; recorded outputs can be replayed.
  4. *issues*: each of the 43 issues from the plan maps to cases at levels 1–3
     or to unit tests; the gate reports them by id.
  5. *private acceptance*: the product owner's datasets, run locally only;
     nothing from them is committed.
* Result comparison reuses `evals/result_compare.py`.

---

## 12. Integration

### 12.1 Entry point
`core2.service.answer_question(question, ctx) -> AnswerPayload`, channel
independent. `ctx` carries client, user, session key, db config, governance
scope, regulated flag, language, `today`, and injectable LLM and warehouse
callables (tests inject DuckDB and recorded plans).

### 12.2 The switch
Per workspace: `legacy` (today's pipeline only), `compare` (both run; the portal
shows today's answer and the new core's answer side by side, and both are kept
for comparison), `core2` (the new core answers; today's pipeline answers what the
new core reports as unsupported, labelled as such). Web portal only; other
channels unchanged.

### 12.3 Where it plugs in

Today a workspace (`client`, keyed by `account_id`) has one database
(`client.db_config_id`), one discovery folder (`clients/<account>/schema/`) and
one KB folder. core2 keys everything by `(account_id, db_config_id)` so a second
source per workspace needs no migration later.

**The switch** is a column on `client`: `query_engine` = `legacy | compare | core2`
(migration tuple in `store/db.py::_run_migrations`, validated in
`update_client_meta`, normalised by a reader in `core2.service`, set from the
workspace page), modelled on `analysis_mode`.

**Web portal turns.** In `gateway/webhooks.py::ws_chat`, where a user text frame
is parsed and before the local pre-routes, a `compare` or `core2` workspace
starts `core2.service.answer_web_turn(...)` as its own task. In `compare` the
existing flow continues untouched and the new core's answer arrives as a second
`assistant_response` frame marked `engine: "core2"`, which the portal shows
beside today's card with a "New core (preview)" badge. Hooking the frame rather
than `handle_query` is deliberate: today's pre-routes (cached-result commands,
why-insights) answer some turns before `handle_query` runs, and the preview must
answer every typed question. A `clarification_response` frame whose `pending_id`
starts with `core2:` goes to core2. Teams, Slack, Zoom, `/api/ask` and evals are
untouched.

**Governance core2 calls (never re-implements):**
* workspace and database: `store.get_client`, `core.pipeline_context.get_client_db`;
* table access: `store.get_allowed_tables(portal_user)` intersected with the
  tables the model knows (`None` means unrestricted);
* regulated tenants: `scrub_question_pii` on the question; the `llm_context`
  policy evaluation and `result_llm_features_allowed` before any prompt that
  could carry values; matched member values replaced by placeholders (§7.3);
* the model call: `resolve_provider(client, purpose="query")`, then
  `llm_complete(..., temperature=0.0, **extra)` inside
  `llm_audit_scope(account_id=..., question=..., enabled=client.enable_llm_audit,
  request_id=..., question_id=..., component="core2_plan", egress={...})`; the
  catalog is the `CachedPrompt.stable` half; there is no JSON mode, so the plan is
  parsed and validated by core2 with one repair round; any new value-bearing
  prompt block gets its header added to `llm_audit._VALUE_BEARING_MARKERS`;
* execution: `core.compliance.governed_query.execute_governed_query(credentials,
  db_type, sql, context=resolve_context(...), known_tables=load_known_tables(...),
  allowed_tables=effective, semantic_context={"production_sql": True},
  max_rows=N)`, which validates (read-only, single statement, known and allowed
  tables, dialect rules, production shape), applies policies and row filters,
  executes and masks. `table_columns` is not passed: its heuristics (for example
  the null-aggregate diagnostic) reject correct compiled SQL, and the compiler
  only references columns the model holds. Compiled SQL uses `COUNT(1)`, never
  `COUNT(*)` (a star on a classified table is refused), and never names an alias
  after a word the validator's DDL screen matches;
* usage: in `core2` mode `store.log_query` per question (quota and billing);
  in `compare` mode the preview's tokens are recorded in `core2_turns` only, so
  testing does not consume the workspace's question quota.

**Bootstrap reads** what discovery already wrote: `_schema.json` (tables,
columns, types, primary keys, `__db_fk_constraints__`, masking fields) with full
names rebuilt from each entry's database/schema/owner fields (the file's own keys
differ by warehouse). Profiling queries are generated by code, SELECT-only
(checked by parsing before they run) and run on the admin connection like
today's join profiling (`core.relationship_validator.run_probe`); only
aggregates come back. Common values are read only for columns today's value
index is allowed to read (`core.value_index` privacy and compliance gates).

**Member lookup** reuses the per-workspace `value_index.sqlite`
(`lookup_exact`, `lookup_fuzzy`), adding attributes it lacks to `core2_values`.

**Answer payload** is today's `assistant_response` (keys `answer`, `chart`,
`kpi`, `data`, `trust`, `confidence`, `insight_summary`, `anomaly_callouts`,
`coverage_caveats`, `follow_up_suggestions`), built by core2. In `compare` the
card has no `result_id`, so today's chip handlers never act on it; its
suggestions are plain follow-up questions. Preview cards are not restored on a
page reload (they write no `answer_trace` row) until `core2` mode.

**Admin page** "What QueryBot learned" at `/admin/clients/{account_id}/learned`,
using `_is_auth`, `store.get_client`, `_resp` and the workspace nav
(`_client_workspace_nav.html`), with POST forms that write overrides and redirect
back with `?saved=`.

### 12.4 Import of today's approvals

Once per workspace, then on demand, into `core2_overrides` with provenance
`admin`: approved date roles and defaults and approved relationships from
`_semantic_model.json`; confirmed and rejected relationships from
`entity_relationships`; confirmed properties from `entity_properties`;
validated/published metrics from `metric_registry` (structural when the builder
config has one, else `SqlExpr` parsed and checked) with their default dates from
`metric_date_context`; meanings and synonyms from `field_overrides.json`,
`table_description` and decided `business_meaning` rows. Identity is mapped to
core2 keys through `_schema.json` (bare and two-part names are resolved against
it, and an unresolvable name is reported, never guessed).

### 12.5 The first cut, as built

Where it differs from §12.3, the first cut is simpler on purpose:

* **The hook** is `gateway/core2_bridge.py`, called from `_run_main_question` in
  `ws_chat`: in `core2` mode before today's pipeline (which answers when the new
  core reports `unsupported` or fails), in `compare` mode after it, under a
  120-second limit. Side by side, a new core that times out or fails says so on
  its preview card instead of leaving the reader waiting for one.
* **A "why" about the answer on screen** is its own turn (`_run_why_question`):
  in `core2` mode the new core answers it first, then today's analysis of the
  result on screen, then today's pipeline; in `compare` mode today's analysis
  answers and the new core's answer follows it as a preview. After a new-core
  answer with data, today's last result is set aside (`forget_current_result`):
  it is the answer before, and today's follow-up routes would otherwise explain
  or re-cut it under the older question's name. Other commands on the result on
  screen are still today's.
* **Questions about the data itself** ("what can you tell me about my data?",
  "what does gross profit mean?", "how far back does the data go?") are planned as
  `describe_data` with what they ask about in `about` (measure, group or date
  slugs, or a subject's catalog name), and answered from the model, never written
  by the AI (`core2/answer/describe.py`): a one-sentence lead, a section per
  subject or thing asked about (measures, breakdowns, dates and the period
  covered; a measure's definition and how it adds up), and example questions,
  each a measure and a breakdown that reach each other in a year with data. Only
  the reader's tables are described.
* **Refusals** say what was looked at: the measure and its table, the breakdown,
  the filters and the dates (`trust.considered`), and why it stopped
  (`trust.stopped`), under "How this answer was produced".
* **A name that is a member and a measure** ("cost of goods sold": a ledger
  account, and the cost on the invoice lines) is read as the member only when
  the member's table can be broken down as asked (planner rule 8). A plan that
  fails because a breakdown is not linked to its measure's table goes to the
  repair round with the measures that do reach it, those sharing the question's
  words first -- the question as the AI saw it, so for a regulated workspace a
  placeholder's words are not used, and there the reading cannot be made.
* **Clarifications** from the new core are written in its answer card with their
  options (no chips, no `pending_id`); the reply is planned with the question it
  answers, which the conversation history carries.
* **Member names** come from core2's own index, read once per model version under
  the workspace's service connection (entities' names and codes first, within a
  30-second budget, only columns whose values may be read), not from
  `value_index.sqlite`.
* **Recording**: every new-core answer is a row in `core2_answer` (question,
  status, headline, SQL, rows, plan, duration, model version), listed on the
  learned page beside the switch.
* **`core2` mode** keeps its answers as today's are: an `answer_trace` row (the
  thread's history, the full CSV export, the audit link), a `store.log_query` row
  (usage and limits) and the reader's thumbs; at a monthly limit today's
  pipeline answers and says so.
* **Regulated tenants**: the question is scrubbed with `scrub_question_pii`, matched
  members are replaced by placeholders, and the catalog carries no values.
* **Production SQL rules** the compiler meets, checked in tests against the real
  governed executor: output columns are listed (never `alias.*`), `COUNT(1)`, no
  alias a statement screen matches.
* **Evaluation level 3** (`evals/core2/plan_eval.py`) runs a domain's golden
  questions through a configured provider and records its answers for replay.

## 13. Delivery order

1. Model schema, identity, store, warehouse runner and dialects.
2. Eval framework and the first synthetic domains; ground truth.
3. Bootstrap: inventory, profile, keys, calendar, joins, kinds, dates, measures,
   quality; then labels. Level-1 gate.
4. "What QueryBot learned" page (read, review queue, overrides).
5. Plan IR, catalog, value index, planner prompt; resolver and compiler for
   value/breakdown/trend/rank/share; answer builder; service and switch
   (first cut).
6. Compare, drivers, follow-ups and cached results, clarifications, partial
   periods, quality notes; remaining domains; all 43 issues (full preview).
7. Forecasts, three-dialect hardening, switch-over.
