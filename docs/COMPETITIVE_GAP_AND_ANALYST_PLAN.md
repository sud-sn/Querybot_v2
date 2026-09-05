# Answers, Analysis, and Provable Governance — Competitive Gap & Implementation Plan

**Researched:** 2026-09-05 against `fix/value-grounding-governance-and-sweep` @ `03d1347`
**Reference build analysed:** AskSenseOps "Valor" 0.1.0 (`version.json`: build 2026-09-04,
platform windows, torch cpu, **mode online**), supplied as a compiled Windows distribution —
289 Cython `.pyd` modules, 56 readable `.py` files, one 3 MB bundled `public/main.js`.
**Ground rule:** no code from any analysed product is copied, adapted, or transliterated.
What follows is an analysis of *behaviour* and a plan built on QueryBot's own foundations.

---

## 0. The short version

Three things decide whether we win a deal against this class of product:

1. **Does it answer the question** — including when the answer lives somewhere the user
   did not name, and when the question is a variation nobody wrote down.
2. **Does the answer come with analysis** — not a table, a *reading* of the table.
3. **Can the buyer's risk function sign it off** — with evidence, not assurances.

On (3) we are already substantially ahead and do not know it. On (1) we are behind in two
specific, fixable places. On (2) we are ahead on machinery and behind on delivery — and, in the
one segment we most want to sell to, we currently ship the *worst* version of it.

The strategic move in this plan: **make the analytical narrative a computed artefact rather than
a generated one.** Everyone else asks a model to read the rows and write a paragraph. If we
compute the findings deterministically and let the model only *phrase* what was computed, then
the regulated tenant gets the same quality as everyone else — with zero rows leaving — and every
sentence in every answer becomes traceable to a number. That single decision resolves the
answer-quality axis, the analysis axis and the governance axis at once.

---

## 1. What the reference build actually does

Read from the 56 readable modules, the exported symbol tables of ~40 compiled modules, the MCP
toolkit headers, and the strings compiled into the bundled UI.

### 1.1 Shape

FastAPI + LangChain/LangGraph (with checkpointing) + MCP 1.21. SQLAlchemy/asyncpg/aiomysql for
its own store. `sentence-transformers` + `torch` + `transformers` for **local** embeddings — and
**no vector database at all**; similarity is in-process (scikit-learn/scipy are the only numeric
deps). Providers: Ollama, OpenAI, Anthropic, Groq — so a fully local model path exists. The
`"mode": "online"` key in `version.json` says plainly that an offline build of the same product
is shipped.

### 1.2 Two flows, and why the second one exists

- **Lite** (`src/agents/flows/lite/__init__.py`, 562 lines, readable): a ReAct loop over a small
  curated read-only tool set, answering from **synced metadata** with compacted payloads.
- **Deep** (`src/agents/flows/deep/__init__.py`, 556 lines, readable): the same loop over the
  **full MCP tool surface**, full payloads, reading the platform live.

The comment that matters is in the Deep module: it replaced a four-node LangGraph planning graph
because that graph *"froze the tool choice before any data had been seen."* They tried the
deterministic planner first and abandoned it. We should read that as a warning about *rigid*
planning, not about planning — our answer is a bounded re-plan (§5, Phase F), not a free loop.

Supporting machinery worth naming: per-agent retry budgets
(`{"semantic": 2, "association": 2, "expression": 2, "execution": 1}`) with typed `retry_hints`;
`TurnBudget.for_model()`; a `SessionScratchpad` keyed by app + session; result-handling tools
(`get_result`, `list_results`, `paginate_result`, `grep_result`, `filter_rows`, `extract_field`,
`calculate`); an `ask_user` tool that lets the agent ask a clarifying question **mid-loop**, with
the budget ticking so a thinking human is not mistaken for a hung model; a regex freshness
fast-path that answers "how current is this?" with no LLM call at all; and
`mark_superseded_attempts`, which relabels a failed tool call once a later attempt succeeds so a
reloaded transcript reads as a correction rather than a defect. That last one is pure product
polish and worth stealing as an *idea* — our trace has the same problem.

### 1.3 The tool surface

Platform families (`qlik`, `powerbi`, `looker`, `database`, `rag`) resolve through a registry;
their own docstring says adding a platform is *"a folder, not an edit to shared code"*. The Qlik
toolkit alone exposes on the order of sixty tools across read, write, selection and navigation —
search, describe app, fields, field values, table data, chart data and chart *image* analysis,
expression check/evaluate, selections and locks, bookmarks, sheet and chart authoring, dataset
profile/quality/trust-score/lineage/freshness, data products, and a full glossary CRUD. Deep mode
scopes the catalogue before the loop starts: authoring tools are ~69% of it and are dropped
unless the question contains an authoring verb.

### 1.4 The multi-app behaviour — what it actually is

This is the capability the evaluation flagged, so it is worth being precise about.

Each turn binds exactly **one** app: `bound_app_id = metadata_id`
(`deep/__init__.py:309`), and the scratchpad is keyed by that app. Apps are selected and synced
per user, with a per-app metadata overlay and a `has_section_access` flag for apps that use Qlik
section access — the UI's own words: *"For selected apps, the existing per-user metadata flow is
used — each user reads their own Qlik-filtered file."*

What makes it look multi-app is that in **Deep** mode the bound app is a starting point, not a
boundary: the loop holds `qlik_search` (*"find apps, datasets, spaces by name; query='*' to browse
all"*) and `qlik_describe_app(app_id)`, so within one turn the agent can discover other apps in
the tenant, describe them, and read chart or field data from them. Cross-app answering is
therefore **emergent agent behaviour over a live tool surface**, not a planned federation: there
is no cross-app join, no reconciliation step, and no statement of which app an answer came from
beyond what the model chooses to say.

That is genuinely more than we do — and it is also a lower bar than it appears. An agent that
reads two apps and narrates the result has no mechanism to notice that the two disagree.

### 1.5 What "governance" means there — the central finding

`src/governance/` is **metadata governance, not data protection**:

- `field_roles` — classify fields, compute an effective class (including `opaque`), withhold
  names, warn on breakdowns, carry disclosures. Its purpose is stopping a 22,000-row breakdown
  on a surrogate key. Admin pins beat the classifier; sample values weigh as much as the name.
- `scoring` — `score_name_clarity`, `score_description_quality`, `score_synonym_coverage`,
  `score_category_assigned`, `score_sample_expression_coverage`, `score_recent_feedback_hits`.
- `retrieval.apply_status_weights` — **curation status weights retrieval**.
- `overlay` — admin edits layered over synced metadata, plus `compute_metadata_version`.
- `mcp_acl` / `mcp_access` / `mcp_registry` / `mcp_gateway_keys` — tool ACLs, client approval,
  audit. `cost_policy` — budgets and `resolve_effective_model` downgrade. `scope_resolver` —
  most-specific merge with source attribution.

Now the negative space. Across all of `src/` and the bundled UI: `redact`, `pseudonym`,
`hash chain`, `decision log`, `retention`, `data residency` return **zero** hits. Every `mask`
match is CSS `mask-image`. `pii` appears once, as a label in a curation taxonomy
(`["Customer","Financial","Geography","Time","Product","Operational","PII"]`). The single
`row-level` hit is a UI caption stating their position outright:

> *"ACLs gate capability, not data: Qlik section access / row-level security still governs what
> the signed-in user sees."*

They delegate data protection to the BI platform. There is no boundary of their own between the
customer's data and the model: in Deep mode the loop reads live chart data, table data and field
values, and those payloads go to whichever provider is configured. Their strongest privacy story
is the Ollama path — *run it all locally* — which is a real answer, and one we cannot currently
match (§3, item 7).

---

## 2. The wider field

**Vanna AI** — MIT-licensed RAG over question→SQL pairs in a vector store. No semantic layer, no
orchestration; the thesis is that well-retrieved examples are enough. Accuracy tracks the quality
and quantity of curated pairs, and where no close match is retrieved it degrades to
schema-and-docs reasoning. The recurring criticisms in the 2026 write-ups are exactly our
strengths: RAG over DDL breaks under schema drift, training examples do not enforce row-level
access, and single-shot SQL cannot carry multi-step analysis.

**TextQL (Ana)** — an agent framed as a hired analyst: writes SQL, runs Python, searches the web,
produces charts, reports, CSVs, PDFs. Its **Ontology** stores metric definitions, table
preferences and business logic so they need not be re-explained per thread, and for stricter
governance it constrains which tables and fields the agent may touch. This is the closest
competitor to where we are heading. Note what its governance layer is: *scope restriction*. Same
category as the reference build.

**Wren AI** — open source, semantic layer first, with a "Modeling Definition Language" for
metadata, relationships, calculations and aggregations. The strongest open-source statement of
the position we already hold (semantic layer over raw RAG).

**Snowflake Cortex Analyst** — semantic model in YAML plus a **Verified Query Repository**:
named, dated, human-verified question→SQL pairs injected as few-shot exemplars. The lesson is
governance of the *examples*, not just the schema — a verified exemplar carries an author and a
date and can be revoked. Our `core/examples.py` retrieval has no such status.

**Databricks Genie / ThoughtSpot Spotter** — Genie's unit is a curated *Space*, and the
enterprise pattern is an agent routing a question to the right Space; Spotter leans on an agentic
semantic layer across warehouses. The whole industry has converged on *scoped subject areas plus
a router*. That is the shape Phase A adopts.

**What the research says actually moves accuracy** (this matters more than any feature list):

- Spider 1.0 is ~86% and solved; **Spider 2.0 — enterprise schemas averaging ~800 columns —
  sits around 21% under multi-step agentic evaluation**, and BIRD around 73%. Nobody's marketing
  numbers survive contact with a real warehouse. Accuracy claims should be made against a
  tenant's own golden set, which is what `evals/` already is.
- **Execution-guided selection and self-correction is where the gains are.** LitE-SQL reports
  72.10% on BIRD with vector schema linking plus execution-guided self-correction, and the first
  correction pass carries most of the gain. Execution guidance is reported to cut schema-linking,
  join and logic errors by 20–40%.
- **Schema linking recall is the dominant lever** — a context-aware bidirectional retrieval
  approach gains 5.4–5.8% over full-schema baselines *with no query fixing at all*.
- **Self-consistency is a weak selector**: the most-agreed answer is often not the correct one,
  with an upper bound ~14% above what majority voting achieves. So: generate candidates, but
  select with a *verifier*, not a vote. We already own the verifier (`core/result_verifier.py`).

---

## 3. Where QueryBot actually stands

### Ahead — and we should be selling this

| | Us | Them |
|---|---|---|
| Data-protection boundary | `core/compliance/` — policy engine, `is_regulated()` single predicate that **fails closed** on an unprovisioned tenant, `result_guard` HMAC pseudonymisation, `sql_guard`, `classifier`, `readiness` | delegated to the BI platform |
| Proof of non-egress | `llm_call_log.egress_manifest` with a **computed** `values_sent`; 7 production `record_llm_blocked` refusal-proof sites; per-question egress log; daily export to the tenant's own warehouse (`core/log_export.py`) | none |
| Value grounding under policy | `core.value_resolver.filter_resolved_for_compliance` — regulated tenants get verified values only for admin-**reviewed**, non-sensitive columns, and an incomplete classification degrades to *full suppression*, with evidence returned | n/a |
| Join safety | graph-validated joins, raw fact-to-fact rejected, `execute_governed_query` with argument-independent guarantees | expression check only |
| Retrieval egress | embeddings are local (`SentenceTransformer` in `core/vector_store.py`) — retrieval text never leaves | also local |
| Deterministic analysis stack | `stat_signals`, `result_verifier`, `contribution_analysis`, `anomaly_detection`, `correlation_analysis`, `distribution_analysis`, `period_comparison`, `window_analytics`, `forecast`, `analysis_sandbox` — all no-LLM | LLM narration end to end |
| NL metric authoring without model-written SQL | `core/metric_authoring.py` — slot filling against `TABLE_REF_n`/`COL_REF_n` bindings; no allow-listed key can carry SQL text | authoring via platform APIs, model-written expressions |
| Localisation | 1,278 message ids, both languages proven by executing tests | English only |

All five phases of `docs/LLM_EGRESS_PLAN.md` have landed (`tests/test_llm_egress_phase0..4.py`).
That work is the moat. Nothing in the analysed field has an equivalent.

### Level

Semantic layer (our metric registry + entity graph vs their curated app metadata vs Wren's MDL
vs TextQL's Ontology); result-follow-up actions; scheduling and delivery; dashboards. KB
retrieval is ahead of the field on mechanism and level on curation.

### Behind — the honest list

1. **One workspace = one connection = one schema scope.** `client.db_config_id` is a single
   integer (`store/config_store.py:373`). A tenant with several subject areas needs several
   workspaces, each with its own KB, graph, metrics and value index. No question can span two,
   and nothing ever checks an answer against a second source. **This is the multi-app gap.**
2. **Single-shot SQL.** We generate once and repair on error. No candidate generation, no
   execution-guided selection — the two techniques the literature is clearest about.
3. **Regulated tenants get a static string for analysis.** `_regulated_analysis_fallback`
   (`core/response_builder.py:2271`) returns a fixed title and body. The segment we most want to
   sell to receives the weakest product. This is the most damaging single line in the codebase.
4. **No in-loop clarification.** `core/clarification.py` is a pre-flight gate. We cannot ask a
   question halfway through and continue; they can (`ask_user`, with the budget ticking).
5. **Retrieval ignores curation status, and the example store is a second, weaker one.**
   KB retrieval is already strong — dense HNSW + BM25 + reciprocal-rank fusion + a cross-encoder
   reranker with a relevance floor and a `weak_retrieval` flag (`core/vector_store.py`). What it
   has no notion of is *curation status*: a reviewed, described, synonym-rich table ranks level
   with a raw one at equal relevance, where they weight by status. Separately,
   `core.examples.retrieve_similar_examples` is plain dense similarity over a **second, older
   ChromaDB collection** — no BM25, no fusion, no reranker, no floor — and validated examples
   carry no author, verified-at date or revocation path.
6. **No freshness contract.** No `metadata_version` on an answer, no staleness disclosure, no
   auto-resync. KB rebuilds are manual and invisible to the answer.
7. **No local-model posture.** `core/llm.py` supports `anthropic`, `openai`, `azure_openai` —
   all hosted. We cannot today offer "nothing leaves your network", which is the one governance
   claim they can make and we cannot.
8. **No MCP surface, in either direction.** We cannot be called by another agent, and we cannot
   consume a BI platform's tools. They are MCP-native both ways.
9. **No modelling-debt backlog.** We have `kb_quality`, `quality_scorer`, `graph_health`,
   `compliance/readiness` — four scores, no single prioritised list of "fix these twelve things
   and answers get better".

---

## 4. The three claims this plan buys us

Everything below serves one of these. If a task serves none, it is not in the plan.

- **C1 — "It finds the answer wherever it lives, and tells you when two sources disagree."**
  (Phases A, B, F)
- **C2 — "Every sentence of the analysis is a computed number, not a model's impression."**
  (Phase C)
- **C3 — "Your data never reaches a model, and here is the evidence, per question, signed."**
  (Phases C, G)

Plus the throughline the whole product is judged on: **C4 — "You describe a metric in chat; it
answers every real variation of the question people will ask about it."** (Phases D, E)

---

## 5. The plan

Phases are ordered by value per unit of risk. A, C and G are independent and can proceed in
parallel; B depends on nothing; D depends on nothing; E feeds B and D; F is last because it is
the easiest to get wrong.

Every phase lists its tests. The house rule holds throughout: **a test executes the real function
and asserts on its return value; no test asserts on source text.**

---

### Phase A — Domains and sources: answering across a tenant, and corroborating

**A1 · Domains within a workspace** *(medium)*

A *domain* is a named, addressable slice of a workspace: a set of tables, the graph subgraph they
span, the metrics that own measures in them, a default date role, and a description. New table
`domain` + `domain_table`, and a `domain_id` carried in the pipeline context.

Nearly all the plumbing exists: `allowed_tables` is already threaded through
`core.semantic_layer.table_allowed`, `core.value_index._table_allowed`,
`core.workspace_guide._visible_table`, `core.semantic_planner._table_allowed_for_display` and
`core.clarification._scoped_active_terms`. A domain is, mechanically, a named `allowed_tables`
set with metadata attached.

Routing reuses what `core/source_resolution.py` already does for facts: score the question
against each domain's vocabulary, description and metric synonyms. One clear winner → route.
Two close → §A3. None → the existing "cannot generate" hint, now naming the domains it looked in.

**A2 · Multiple connections per workspace** *(large)*

Replace the single `client.db_config_id` with a `client_source` table (`account_id`, `db_config_id`,
`name`, `is_default`, `domain_id`). The pipeline carries a `source_id` end to end; every existing
call site that reads `db_config_id` resolves through one helper so the change is auditable.

**Cross-source questions are never federated in SQL.** No warehouse-side join across connections.
The compiled plan becomes N single-source queries plus a **local combine** in
`core/analysis_sandbox.py`, which already runs governed Python over released rows under hard
row/column/time/output bounds. This is slower than a federated engine and dramatically more
governable — and, unlike the reference build's approach, it produces a stated, auditable
combination step rather than a model's summary of two tool outputs.

**A3 · Corroboration — the capability the evaluation actually noticed** *(medium)*

When more than one domain or source can answer the same question, answer from the primary and
**check** against the secondary:

- compile the same analytical request against both
- execute both under `execute_governed_query`
- compare with a tolerance (reuse and promote `evals/result_compare.py` into `core/`)
- agreement → one line: *"Confirmed against <secondary>."*
- disagreement → both numbers, both sources, and the difference, as a first-class answer element

This is strictly better than reading two apps and narrating. An agent loop cannot notice a
disagreement it was never asked to look for; a comparison step cannot miss one.

**Tests.** Domain routing picks the right domain for questions whose vocabulary belongs to each,
and refuses when neither scores; `allowed_tables` derived from a domain actually excludes the
other domain's tables from planner output, value-index resolution and the workspace guide; a
two-source question produces two governed executions and one combine; corroboration returns
`agreement=True` on matching numbers and a populated `difference` on divergent ones; a
cross-source plan never emits SQL naming tables from two connections.

---

### Phase B — Candidate generation with execution-guided selection

**B1 · Candidates** *(medium)*. Where the semantic plan leaves a genuine choice open — an
ambiguous grain, two reachable date roles, two candidate join paths — emit up to *k* SQL
candidates (default 3) rather than picking silently. Where the plan is fully determined (the
majority of questions), *k* = 1 and nothing changes: **cost is spent only where there is real
ambiguity.**

**B2 · Selection by verifier, not by vote** *(medium)*. Execute each candidate under
`execute_governed_query` with a small `max_rows` probe, then select:

1. drop candidates failing validation or execution;
2. score survivors with `core/result_verifier.py` — it already compares metadata-only intent
   against output columns, value types and row counts, and it never calls an LLM or logs values;
3. break ties on agreement of the headline figure;
4. **all candidates disagree and none verifies → clarify.** Never pick one at random.

The literature is explicit that majority voting is a weak selector (~14% below the achievable
upper bound) while execution guidance cuts join and logic errors by 20–40%. We already own a
verifier; using it as the *selector* rather than a post-hoc check is the whole change.

**B3 · Execution-guided self-correction** *(small)*. On a first-pass failure, feed the sanitised
error (`core.failure_messages.sanitize_db_error`, already on the repair path) plus the verifier's
structured complaint back into a single correction attempt. One pass; the research says the
first pass carries most of the gain.

**Tests.** A question with a genuinely ambiguous grain produces >1 candidate and a determined one
produces exactly 1; the selector prefers the verifier-passing candidate over a
verifier-failing one that would otherwise win on agreement; all-fail routes to clarification and
returns no answer; every candidate execution passes through `execute_governed_query`; the
selection decision appears in `core/pipeline_trace.py` with the reason.

---

### Phase C — The analytical narrative, computed rather than generated

**This is the centrepiece.** New module `core/analysis_narrative.py`, three separated steps.

**C1 · Evidence (no LLM)** *(medium)*. Run the deterministic analysers over the governed rows —
`stat_signals`, `contribution_analysis`, `period_comparison`, `anomaly_detection`,
`distribution_analysis`, `correlation_analysis`, `window_analytics`, `forecast_gate` — and emit a
typed `AnalysisEvidence`: a list of findings, each carrying a kind, the numbers behind it, the
columns involved, a materiality score, and the coverage caveats that already exist
(`core/date_coverage.py`, `core/join_coverage.py`).

**C2 · Selection (no LLM)** *(small)*. Rank findings by materiality — share of total, effect
size, significance — and take the top three to five, deduplicated by dimension so the summary
does not say the same thing about the same column twice.

**C3 · Phrasing, behind one interface** *(medium)*.

- **`TemplatePhraser`** — sentences from the i18n catalogue with named placeholders. Offline,
  deterministic, reproducible, and already bilingual because the catalogue is. **This becomes the
  default for every tenant, not the regulated fallback.**
- **`LLMPhraser`** — receives the **evidence object only**, never rows, and returns prose. Then
  the check nobody else performs: **every number in the model's output must appear in the
  evidence**, or the template output is used instead and the substitution is recorded.

Consequences, in order of commercial value:

- `_regulated_analysis_fallback` stops being a static string. A regulated tenant gets a genuine
  analytical summary with `rows_sent_to_llm = 0` and `values_sent = false` — provably.
- Every sentence in every answer, for every tenant, is traceable to a computed number. That is
  the C2 claim, and it is not a claim any analysed competitor can make.
- The summary is reproducible: same rows in, same sentences out. Auditors care about this more
  than they care about eloquence.
- It works with a small local model, because the model is only choosing words.

**Tests.** Evidence built from a fixed row set returns the expected findings with expected
materiality ordering; the selector deduplicates by dimension; `TemplatePhraser` output for the
same evidence is byte-identical across runs and differs between `en` and `fr` on every line that
is not tenant data; `LLMPhraser` output containing a number absent from the evidence is rejected
and the template output returned; a regulated tenant receives a populated narrative **and** a
`record_llm_blocked` row is absent (nothing was blocked, because nothing was attempted) while the
egress manifest for the turn reports `values_sent: false`.

---

### Phase D — Metrics authored in chat that survive real questions

`core/metric_authoring.py` is already the right design and is already wired to admin
(`admin/routes.py:5874`) and to chat (`gateway/webhooks.py:1488`). Four additions turn it from a
feature into the C4 claim.

**D1 · Draft from the question that failed** *(small)*. When no metric owns the requested measure
and the "cannot generate" hint fires, offer *"define it now"* with a draft pre-filled from the
question. The user's first encounter with metric authoring should be at the moment they need it.

**D2 · Mandatory dry-run before save** *(small)*. `core/metric_dryrun.py` exists. In the chat path
make it compulsory: compile → run over a bounded window → show the number and the row count →
then offer save. A metric the user has seen a number for is a metric they will trust.

**D3 · Variation coverage — "works for every real-world variation"** *(medium)*. On save, derive
the question variations the metric must be able to answer and check each one resolves *offline*
through the existing planner:

- **grain** — day, week, month, quarter, year (`core/multi_period.py`)
- **dimension** — each dimension reachable from the owning fact (`core/drill_dimension.py`)
- **comparison** — versus prior period and versus same period last year
  (`core/period_comparison.py`, `core/relative_date_range.py`)
- **ranking** — top/bottom N by each reachable dimension
- **filter** — each low-cardinality dimension value

Store the outcome as a **coverage report** on the metric: *"resolves for 41 of 47 variations;
the six failures all need a date role on `D_CUSTOMER`."* That turns a vague aspiration into a
number the user can watch go up, and it names the exact missing modelling — which is Phase E's
input.

**D4 · Synonyms from failure** *(small)*. Every question that should have bound to a metric and
did not — evidenced by feedback or by a subsequent successful rephrase — becomes a proposed
synonym on that metric, queued for one-click acceptance.

**Tests.** A failed measure question yields a draft whose bound refs exist in the schema; save is
refused until a dry-run has produced a number; the variation generator emits the expected
variation set for a metric with two reachable dimensions and a date role, and the coverage report
counts resolvable and unresolvable variations correctly; an unresolvable variation names the
specific missing asset; a rejected metric plan referencing an invented column is refused (the
existing guarantee — re-asserted at the chat entry point).

---

### Phase E — Modelling debt as a shrinking, prioritised backlog

The goal is not to remove modelling. It is to make the product name exactly what is missing, in
priority order, and draft most of it — so modelling becomes **review, not authoring.**

**E1 · One readiness score** *(medium)*. Unify `core/kb_quality.py`, `core/quality_scorer.py`,
`core/graph_health.py` and `core/compliance/readiness.py` into `core/model_readiness.py`, scoring
each table, column and metric on the six axes the whole field has converged on: name clarity,
description quality, synonym coverage, category/role assignment, sample-expression coverage,
recent feedback hits. One list, ordered by *how many failing questions this asset would fix* —
sourced from D3 coverage reports and from real failed questions in the trace store.

**E2 · Retrieval weighted by curation status** *(small — highest leverage per line changed)*.
KB retrieval already does the hard part: dense HNSW, BM25, reciprocal-rank fusion, a
cross-encoder reranker and a relevance floor that sets `weak_retrieval` rather than confidently
returning irrelevant tables (`core/vector_store.py`). The one thing it cannot express is
curation status. Add a status weight applied to the fused ranking so a reviewed, described,
synonym-rich asset outranks a raw one *at equal relevance* — never enough to promote an
irrelevant document past a relevant one, which is why it belongs after the reranker rather than
inside it. This is the reference build's `retrieval.apply_status_weights`, and it is the
cheapest accuracy win in this document.

**E2b · Bring example retrieval up to KB retrieval** *(small)*.
`core.examples.retrieve_similar_examples` is dense-only over a separate ChromaDB collection
while the KB has long since moved to Qdrant with hybrid retrieval. Examples are already
execution-validated at build (`validate_and_store_examples`), already literal-scrubbed
(`scrub_example_sql_literals`) and already stale-checked — but they are retrieved with the
weakest mechanism in the codebase and injected into every SQL prompt. Move them onto the same
store and the same hybrid path, and give each example an author, a verified-at date and a
revocation flag, taking the lesson from Cortex Analyst's Verified Query Repository: an exemplar
whose table has since changed shape should be demoted, not trusted equally.

**E3 · Draft everything; the human confirms** *(medium)*. `core/graph_autopopulate.py` and
`core/table_description_author.py` already draft. Extend the same
propose-with-provenance-and-`status='suggested'` discipline to date roles, synonyms harvested
from the value index, and metric proposals mined from repeated question shapes in the trace
store. Surface a single review queue with a diff and one-click accept.

**E4 · Freshness contract** *(small)*. A `metadata_version` per workspace — a hash over schema,
graph, metrics and KB build id — stamped on every answer. When the live schema has drifted since
the last build, disclose it on the answer rather than silently answering from stale metadata. The
reference build does exactly this and it is plainly right.

**Tests.** Readiness scoring returns the expected ordering for fixtures differing on one axis at a
time; status weighting changes retrieval order for two documents with equal similarity and
different curation status, and leaves order unchanged when status is equal; a drafted date role
is written with `status='suggested'` and provenance and never overwrites a confirmed row;
`metadata_version` changes when the graph changes and is stable when nothing changes; a drifted
schema produces the staleness disclosure on the answer.

---

### Phase F — Bounded recovery

Deliberately **not** a free-form ReAct loop. Their own Deep-mode comment says the rigid
four-node graph failed because it chose tools before seeing data; the fix is not to abandon
planning, it is to allow a *bounded re-plan once data has been seen*. Entered only on a **named**
failure, at most two re-plans, counted and visible in `core/pipeline_trace.py`:

- **empty result where the plan expected rows** → re-check filter values against the value index,
  widen to the observed business date window, retry once — and **say what changed**
- **validator rejection** → re-plan the join path from the graph, retry once
- **verifier shape mismatch** → re-plan the grain, retry once
- **still failing** → clarify with a specific question, never a generic apology

**F1 · In-loop clarification** *(medium)*. Let the recovery step ask one focused question and
resume with the answer, rather than abandoning the turn. The reference build's `ask_user` earns
its place; the detail worth copying is that its budget keeps ticking, so a thinking human is not
mistaken for a hung model.

**F2 · Superseded-attempt labelling** *(small)*. When a retry succeeds where an earlier attempt
failed, mark the earlier attempt superseded so a reloaded trace reads as a correction rather than
a defect. Pure polish, cheap, and it changes how the product feels under demo conditions.

**Tests.** An empty result on a filter that the value index can correct produces exactly one
retry with the corrected filter and an explanation of the change; the re-plan budget stops at two
and then clarifies; a validator rejection re-plans the join path from the graph and never
constructs a fact-to-fact join; superseded attempts are labelled and the successful attempt is
the one rendered.

---

### Phase G — Deployment postures and the proof pack

**G1 · A local-model provider** *(small — outsized commercial value)*. Add one OpenAI-compatible
local provider (Ollama / vLLM / llama.cpp server) alongside `anthropic`, `openai`,
`azure_openai` in `core/llm.py`. Embeddings are **already** local. With Phase C's phrasing step
demanding only word choice rather than analysis, a mid-size local model is sufficient for the
narrative — which is precisely why C and G belong in the same plan.

Once this lands, a fully air-gapped posture exists: **no bytes leave the tenant's network.** That
is the strongest governance claim available in this market, and today it is the one claim the
reference build can make and we cannot.

**G2 · Egress posture as a declared, enforced setting** *(small)*. Three postures, stored, shown
in the UI, and enforced by a single predicate in the style of `is_regulated()`:

| Posture | Model | What leaves the tenant |
|---|---|---|
| `cloud` | hosted API | metadata only — table and column names, question text |
| `private` | hosted API under zero-retention terms, region-pinned | metadata only |
| `airgapped` | local model on tenant hardware | **nothing** |

`egress_posture(account_id)` fails closed exactly as `is_regulated()` does, and every refusal
writes a `record_llm_blocked` proof row.

**G3 · The proof pack** *(medium — this is what closes regulated deals)*. A generated, signed,
per-period artefact the customer hands to their auditor:

- every LLM call in the period with its **computed** `values_sent` (already in
  `llm_call_log.egress_manifest`), summarised as a headline: *"0 data values were sent in 1,284
  of 1,284 calls."*
- every blocked call with its reason (from the 7 production `record_llm_blocked` sites)
- classification coverage, and exactly what value grounding suppressed — `filter_resolved_for_compliance`
  already returns that evidence and it is currently thrown away
- the model endpoints used, their retention terms, and their region
- for `airgapped`, the attestation that no external endpoint was configured at all
- a hash chain over the log rows so tampering is detectable, and a stable document id
- the narrative evidence ids from Phase C, so a reviewer can pull any sentence in any answer and
  see the computation behind it

Note what this is: not a policy document, an **artefact generated from logs the product already
writes**. The distance between here and there is a report generator and a hash chain.

**Tests.** Each posture permits and refuses the expected call classes, and an unknown posture
fails closed; the local provider completes through the same `llm_complete` funnel and writes the
same audit row as a hosted one; a proof pack over seeded log rows reports the correct counts,
`values_sent` totals and blocked reasons; altering one log row breaks the hash chain; an
`airgapped` workspace's pack asserts no external endpoint.

---

## 6. Engines and models worth incorporating

**Build, not buy:**

- **Candidate selection (B2).** Nobody sells this; it depends on our verifier and our plan.
- **The evidence engine (C1).** We already own eight analysers. The engine is the typed
  aggregation over them.
- **Variation coverage (D3).** Directly serves the C4 claim and no product does it.
- **The proof pack (G3).** Our differentiator by definition.

**Adopt:**

- **A local inference server** (Ollama or vLLM behind an OpenAI-compatible endpoint) — G1.
- **Nothing further for schema linking.** Retrieval recall is the dominant accuracy lever
  (5.4–5.8% from retrieval alone) and this is already built: dense + BM25 + RRF + cross-encoder
  rerank + relevance floor, all local. The remaining retrieval work is E2/E2b — status weighting
  and lifting example retrieval onto the same path — not a new engine.
- **MCP, both directions** — expose QueryBot as an MCP server (governed `ask` / `list_metrics` /
  `describe_domain` tools, subject to the same policy engine) so other agents can call us, and
  add an MCP client so we can read a BI platform's metadata where a tenant has one. Sequence this
  *after* G2: an MCP surface without a declared egress posture is a governance hole.

**Explicitly not adopting:**

- A federated query engine. Local combine over governed rows (A2) is slower and auditable;
  federation is fast and unprovable.
- A free-form agent loop over a wide tool surface. It is their design, it is why they cannot make
  a data-protection claim, and Phase F gets most of the recovery benefit inside a budget.
- Fine-tuning on tenant data. It would contradict C3 outright, which is worth more than the
  accuracy.

---

## 7. What the customer is actually shown

The end state, in the buyer's language:

> **Where the answer came from.** Domain, source, tables, joins, date basis, and the row count —
> and where a second source could answer the same question, whether it agreed.
>
> **Why the analysis says what it says.** Each sentence carries the computation behind it:
> the number, the columns, the method. The narrative is computed; the model, when one is used at
> all, only chooses the words — and any number it introduces that we did not compute is rejected
> before you see it.
>
> **What left your network.** Per question, per call: the tables and columns named, and a computed
> `values_sent` flag. Not a policy statement — a log line, hash-chained, exportable to your own
> warehouse, and summarised in a signed pack for your auditor. In the air-gapped posture the pack
> states that no external endpoint was configured at all.

---

## 8. Decisions needed before implementation

1. **A2 scope.** Do we need multiple *connections* per workspace now, or do domains within one
   connection (A1) cover the near-term pipeline? A1 is a fraction of the cost and covers most of
   what "multiple apps in a tenant" means for a warehouse-backed customer. *Recommend A1 first,
   A2 when a deal requires it.*
2. **Candidate budget (B1).** *k* = 3 on ambiguous plans only, or *k* = 3 always? Always is
   simpler to reason about and roughly triples SQL-generation cost. *Recommend ambiguity-gated,
   with the gate visible in the trace so we can measure what it skips.*
3. **Default phraser (C3).** Template-by-default for everyone, or template only for regulated?
   Template-by-default is reproducible, cheaper, offline-capable and bilingual today; the LLM
   phraser reads better. *Recommend template-by-default with the LLM phraser as an opt-in per
   workspace* — and note that the checked-numbers rule makes the LLM path safe either way.
4. **Local model target (G1).** Which local model do we certify? This decides how much reasoning
   the phrasing step may assume. *Needs a bake-off against `evals/` once C1 exists.*
5. **MCP server exposure (§6).** Real demand exists, and it widens the attack surface. *Recommend
   deferring until G2 is enforced.*

---

## 9. Verification

- **Every claim in this plan becomes a golden question.** `evals/` already runs golden-question
  SQL accuracy regressions per client and per schema (`python -m evals.run --client … --cases …`),
  with fixtures for banking, finance, HR, manufacturing and compounding pharmacy. Accuracy claims
  are made against a tenant's own golden set, never against a public benchmark — Spider 2.0 at
  ~21% is the standing reminder of why.
- **A per-phase evaluation gate.** Phase B ships only if execution-guided selection beats
  single-shot on the existing golden sets; Phase E2 ships only if status weighting does not
  regress them.
- **The narrative gets its own eval.** A fixed row set produces fixed evidence; the assertion is
  on the evidence and on template output, not on prose.
- **Governance is verified negatively.** Extend the existing pattern in
  `tests/test_regulated_llm_boundary.py` and `tests/test_llm_egress_phase0..4.py`: assert what is
  *absent* from an assembled prompt, executing the real assembly.
- **Full suite green before each commit** (`python -m pytest tests -q`, 292 test modules), with
  the four known environment failures as the baseline.

---

## 10. Suggested first commit

Phase C1 — the evidence engine — with no phrasing and no wiring: a typed `AnalysisEvidence`
built from the analysers we already own, and tests that execute it over fixed row sets. It is
self-contained, it unblocks C2/C3 and G3, it has no schema change, and it is the piece the other
two claims are built on.
