# Multi-period wiring — implementation plan and handoff

Branch: `fix/value-grounding-governance-and-sweep`

## The defect

`core/insight.py::detect_analytical_intents` returns fifteen intent keys.
Fourteen have an `if _intents.get(...)` branch in `core/query_pipeline.py` that
appends a SQL hint. `multi_period` has none — the string does not appear in
that file at all. A question naming two years therefore produces a one-year
query, and the answer explains a change the query never fetched.

Chasing it found something upstream and worse:

```
question_has_temporal_intent("Compare 2025 against 2024 by revenue category ...")
  -> False
```

so `resolve_contextual_date_binding` took its early return, no date role bound,
and the field plan carried no date field at all. A hint about pivoting by year
would have been prose about a column the prompt never names.

## Target question

> Compare 2025 against 2024 by revenue category which grew, which shrank, and
> what did each contribute to the overall change?

Per category it needs four things: a 2025 value, a 2024 value, the delta, and
that delta's share of the total change.

## Decision

One governed query pivots the named periods into side-by-side **columns**.
Not N executions.

The N-execution design was already half-built in `core/multi_period.py` and was
**deleted, not wired**, for two independent reasons. Its rewrite prompt would
have sent the row-policy-*injected* SQL to the model, carrying user id, group
membership and policy literals past the compliance boundary. And N separately
validated statements are each legal without being commensurable, so arithmetic
across them can be confidently wrong.

Ownership: `core/multi_period.py` owns question-time named-period detection and
predicate compilation. `core/period_comparison.py` stays narrow — it derives
its second period by shifting a window backwards from an existing result, has
no parameter for a period the user named, and runs from the `compare_prior`
chip after an answer. Do not drag it to question time.

---

## Done — steps 1 to 9

| Commit | What |
|---|---|
| `004a33a` | Detector hardened; nine dead symbols deleted |
| `c5647a1` | Date-binding gate widened; `PeriodPlan` compiler added |
| `b6c7962` | Step 6 — `build_multi_period_sql_hint` and its pipeline branch |
| `eff1b64` | Step 7 — `annotate_period_change`, the miss caveat, the renderer line |
| `a3ea540` | Step 8 — the answer surface leads with the change |
| `ebe11ac` | Step 9 — chart series and the carried period formats |

**Step 1 — detector hardened.** It called part numbers years. Executed against
the unguarded version: `list SKUs 2001 2002 2003` gave 3 "periods";
`show items priced 2020, 2030 and 2040` gave 3; `compare warehouse 2024 stock to
warehouse 2025 stock` gave 2. A bare year now needs a plausible calendar range
(1990-2039, a literal, not derived from `date.today()`) **and** a cue word in
front of it. A separating comma only counts once some period has already been
named. `MultiPeriodIntent.source` records provenance, because the obvious guard
— "is this label in the question?" — fails open on
`compare revenue for the last 2 years for warehouse 2025 and warehouse 2024`,
where both clock-derived labels appear as warehouse IDs.

**Step 2 — nine dead symbols deleted.** `PeriodResult`, `MultiPeriodResult`,
`merge_multi_period_results`, `build_multi_period_chart_payload`,
`build_multi_period_rewrite_prompt`, `ContribSummary`,
`build_contribution_summary`, `detect_comparison_intent`, plus `_infer_columns`
and `_to_float` whose callers went. Every reference was a definition, a
docstring or a test.

`build_contribution_summary` was worse than dead: its section header says "no
raw data sent to LLM", its docstring repeats it, a comment claims the top label
is redacted — and `rows` carries the whole enriched row set while the label is
only truncated. Wiring it would have breached the compliance boundary.

**Step 3 — `format_calendar_attribute_ref`** lifted out of an inner closure in
`format_period_bucket_expression` so both quote attributes from one source of
truth. Bucket output unchanged, pinned by test.

**Step 4 — `PeriodPlan` + `build_period_plan`.** Compiles named periods into
governed predicates:

```
invoice_date.[CALENDAR_YEAR] = 2024
invoice_date.[CALENDAR_YEAR] = 2024 AND invoice_date.[CAL_QUARTER] = 1
invoice_date.[YEAR_MONTH] = 202401
invoice_date.[FULL_DATE] >= '2024-01-01' AND invoice_date.[FULL_DATE] < '2025-01-01'
```

Refuses (returns `None`, meaning "behave exactly as before") on clock-derived
periods, mixed grains, no comparison word, part numbers, duplicate labels, and
more than six periods.

**Step 5 — date-binding gate widened**, narrowly: only for questions naming two
or more comparable periods that also ask for a comparison. Single-period
absolute questions (`revenue in 2024`) are untouched.

Suite: **5810 passed, 4 skipped**.

---

## Execution evidence gathered while planning

These were established by running the code, not by reading it. Do not re-derive.

**The validator already permits every two-period shape.** `validate_sql_detailed`
with generic table names, both `azure_sql` and `snowflake`: long/tidy
`GROUP BY year, category`, the conditional-aggregation pivot, the pivot plus
delta plus share-of-change (window function `SUM(SUM(...)) OVER ()`), and a
derived `GROUP BY YEAR(...)` — all return `ok`. No validator change is needed.
Caveat: run with `semantic_context=None`, so the graph-plan and fan-out gates
were not exercised.

**The long/tidy shape cannot be charted correctly.** `infer_chart_spec` on six
rows / three categories / two years returns `x=CATEGORY`, `series=None`,
`y=['total']` — two bars per category, no year split. Identical whether the
period column is int or text. Cause: `core/chart_spec.py:493` assigns
`series_col` in **exactly one** branch, `elif temporals and measures and
trend_q`. A comparison question is not a trend question. This is why the
**pivoted** shape is the right one — one row per category, each period its own
measure — and it needs no chart change.

**Two citations in the original plan were wrong; corrected here.**

- `wants_comparison` is a **dict key** returned by
  `core.query_semantics.analyze_query_intent`, not a function. Call it as
  `analyze_query_intent(q)["wants_comparison"]`. It already knows *against*,
  *relative to*, *contrast*, *delta*, *variance*, *benchmark* — do not add a
  sixth parallel vocabulary.
- The native-date fallback must be **alias-qualified**
  (`invoice_date.[FULL_DATE]`), matching how `core/pipeline_helpers.py` builds a
  date reference. Naming the physical table produces SQL that will not resolve,
  because the dimension is joined under the role alias.

---

## Steps 6 to 9 — shipped, and step 10 — remaining

### Step 6 — core/multi_period.py, core/query_pipeline.py

**Where.** core/multi_period.py: new build_multi_period_sql_hint(plan, db_type). core/query_pipeline.py: new branch in the _analytic_hints block, inserted after the anomaly branch (ends 4439) and BEFORE the contribution branch (4441); modify 4441's condition; extend the event stash at 4540-4541.

**Why.** This is where the intent finally gets a consumer (`grep -c multi_period core/query_pipeline.py` goes 0 -> non-zero). Placement matters three ways: after the analysis-contract append at 4422 so the contract stays hint index 0 and the governing frame; before the contribution branch so the plan flag is set when that condition is evaluated; and the hint must NOT contradict the contract — 'Return one row per requested business category' (core/analysis_contract.py:203, validator-enforced as composition_shape at core/validator.py:309-329) is satisfied verbatim by periods-as-columns, and fought by periods-as-rows. The contribution hint MUST be suppressed when the plan fires: build_contribution_sql_hint emits `SUM(metric_col) OVER ()` with NO PARTITION BY, which on a widened result becomes each category's share of the COMBINED 2024+2025 total — neither a per-period mix nor a share of the change.

**Change.** build_multi_period_sql_hint(plan, db_type) -> str emits a 'MULTI-PERIOD COMPARISON HINT:' block: the period_totals CTE with one `SUM(CASE WHEN <plan.predicates[i]> THEN <measure> ELSE 0 END) AS <MEASURE>_<alias[i]>` per period, an OR-ed WHERE of the same predicates, GROUP BY the category, then an outer SELECT with an EXPLICIT column list (never SELECT *, which trips select_star in EVERY Select scope including CTEs — I executed _production_shape_errors and confirmed; production_sql is always set at core/query_pipeline.py:4709) deriving CHANGE_ABS, CHANGE_PCT (denominator ABS(oldest) so a negative baseline does not flip the sign) and SHARE_OF_CHANGE_PCT. Period columns are named <MEASURE>_<LABEL> — NOT a generic P_ prefix: I executed infer_chart_spec and confirmed a name matching _CURRENCY_RE/_PERCENT_RE/_COUNT_RE is classified 'measure' while a bare P_2024 with whole-number values is demoted to 'identifier' and the chart disappears entirely. Bullet lines start with '*', never '- ', and the block must contain none of the 23 strings in core/llm.py:43-66 — hints sit AFTER _KB_SECTION_MARKER, the last boundary the rule filter knows, so a marker string there makes _filter_sql_rules_for_compiled_plan (llm.py:227-245) delete from the match to len(prompt). Text states: compare exactly these named periods; do NOT derive them from MAX(year), from the newest row, or from today (this explicitly overrides step 2 of the YEAR-OVER-YEAR rule at core/llm.py:908-925, which may or may not be in the prompt); use the period expressions exactly as written, they come from this tenant's approved date role; keep the measure, joins, filters and row grain exactly as the field plan and entity graph below specify — add no table and no join. Keep under ~450 tokens (hints are the first bytes discarded by the 120000-char tail clamp at 4623). In query_pipeline: `_mp_plan = None` before the try; the new branch has its OWN try/except logging `log.warning(..., exc_info=True)` — NOT the shared handler at 4543, whose except wipes _intents and silently kills the analysis contract plus the entire post-execution dispatch at log.debug; on success append the hint, set `_mp_plan`, `log.info('analytic_intent: multi_period periods=%d grain=%s', ...)`; on a withheld plan `log.info(... 'hint withheld')`. Change 4441 to `if _intents.get('contribution') and not _mp_plan:`. Stash BOTH: `event.__dict__['_analytic_intents'] = _intents` and `event.__dict__['_multi_period_plan'] = _mp_plan`.

**Test.** NEW tests/test_multi_period_named_periods.py::test_hint_text_is_grounded_and_prompt_safe — call build_multi_period_sql_hint on the plan from step 4 and assert the returned string contains both grounded predicates verbatim, contains no 'SELECT *', no line starts with '- ', and contains none of core.llm._OPTIONAL_SQL_RULE_MARKERS. ::test_hint_survives_the_rule_filter — build the real system prompt by calling core.llm.build_sql_system_prompt with a table_context ending in the hint, then call _filter_sql_rules_for_compiled_plan on the result and assert the hint text and the FIELD PLAN block that follows it are both still present (i.e. the hint did not eat the prompt tail). ::test_prescribed_shape_validates — call validate_sql_detailed on the exact SQL the hint prescribes with known_tables/table_columns/allowed_tables and semantic_context={'production_sql':True,'analysis_contract':{'enabled':True,'mode':'composition'}} and assert ok is True, code=='ok' (I ran this: it passes); and call core.compliance.sql_guard.analyze_sql on it and assert has_star is False and aggregate_outputs contains every measure column (this pins the aggregate-only lineage proof against future changes).

### Step 7 — core/multi_period.py, core/query_pipeline.py, core/result_renderer.py

**Where.** core/multi_period.py: new annotate_period_change(rows, plan, truncated). core/query_pipeline.py: new branch inserted as the FIRST branch inside the existing post-execution try at 6473, plus the contribution condition at 6474 and the _confidence_context dict at ~6424. core/result_renderer.py: next to the forecast_caveats extend at ~892.

**Why.** This is the half that does not depend on the model, and it resolves two flaws the judges found fatal in Design 1. (a) BOTH consumers must key off the SAME gated object. Design 1 gated the hint on whether the hint was emitted but gated the post-processor on the raw intent, which I confirmed silently deletes contribution_pct from ordinary questions: I executed 'what did each region contribute to revenue in 2023, 2024 and 2025' -> multi_period intent truthy, wants_comparison FALSE, contribution TRUE. Reading `_multi_period_plan` (None there) leaves that question untouched. (b) A DETECTED MISS MUST REACH THE USER. Design 1 logged a warning and shipped a confident single-period answer — the original defect surviving its own fix. Here a miss falls through to the existing contribution post-processor (so contribution_pct is not lost) AND appends a coverage caveat. Separately this disarms a proven wrong-answer path: infer_numeric_col's metric-word tie-break at core/contribution_analysis.py:145 picks the EARLIER column, so on a NET_AMOUNT_2024/NET_AMOUNT_2025 shape compute_contribution would ship a share-of-2024 number under the label contribution_pct.

**Change.** annotate_period_change(rows, plan, truncated) -> list[dict] | None: locate the plan's period columns by exact alias match (not a regex over arbitrary names — this is what protects ERP columns like P_CODE/P_QTY); return None if fewer than 2 are present. Recompute CHANGE_ABS = newest - oldest and SHARE_OF_CHANGE_PCT = change / sum(changes) * 100 in Python; never mutate in place. GUARDS: a row with a masked string cell (protect_rows output — _to_float returns None) is carried through with derived values None and does not enter the total; when abs(net) < 0.05 * sum(abs(changes)) emit SHARE_OF_CHANGE_PCT as None for every row and add a warning, because gains and losses cancelling makes shares of the net meaningless (this is the requires_positive_total guard core/analysis_contract.py:192-193 declares and nothing reads); when truncated is True emit NO derived columns and add a warning, because a share over a 200-row prefix is exactly what core/compliance/governed_query.py:26-31 tells consumers to refuse — this makes the pipeline the FIRST reader of GovernedQueryResult.truncated for this purpose. Pipeline branch: `_mp_plan_post = getattr(event, '_multi_period_plan', None)`; if set, call annotate_period_change(rows, _mp_plan_post, _rows_truncated); on a list, replace rows and log.info; on None, log.warning('multi_period hint did not land — the result carries no side-by-side period columns') and append 'This answer covers only part of the periods you asked about (<labels>); the query returned a single period.' to a new `_mp_caveats` list. Change 6474 to `if _post_intents.get('contribution') and not _mp_rows_applied and not any('contribution_pct' in r for r in rows[:1]):` — note it is gated on whether the multi-period annotation ACTUALLY applied, so the fall-through preserves contribution_pct on a miss. Add `'multi_period_caveats': _mp_caveats` to _confidence_context. In result_renderer add one line mirroring forecast_caveats exactly: `coverage_caveats.extend(str(n) for n in (confidence_context.get('multi_period_caveats') or []) if n)`.

**Test.** NEW tests/test_multi_period_named_periods.py::test_annotate_period_change — call annotate_period_change directly and assert: (i) compliant wide rows -> SHARE_OF_CHANGE_PCT values sum to 100.0 and CHANGE_ABS matches newest-oldest per row; (ii) single-period rows -> returns None; (iii) mixed-sign near-cancelling rows -> every SHARE_OF_CHANGE_PCT is None and a warning is present; (iv) a row whose period cell is the string 'REDACTED' -> no exception, that row's derived values None, other rows' shares still sum to 100; (v) truncated=True -> no CHANGE_ABS or SHARE_OF_CHANGE_PCT key in any row; (vi) rows keyed P_CODE/P_QTY/QTY with a plan whose aliases are NET_AMOUNT_2024/NET_AMOUNT_2025 -> returns None (no hijack). ::test_contribution_survives_a_multi_period_miss — drive the post-execution helper with a plan set and a single-period result and assert the returned rows still gain contribution_pct AND a caveat string is produced.

### Step 8 — core/response_builder.py

**Where.** New _period_comparison_summary() after _period_comparison_from_rows (~807); a period-pair branch inside build_answer's `if numeric_cols and text_cols:` block (1046-1088), ahead of the existing ranking path; a call in _build_insight_summary (~1466, where _period_comparison_from_rows already sits); a guard in _build_decision_signal; the mode-downgrade guard at 2041-2045.

**Why.** Fixing the SQL alone does not fix the answer — proven twice. compute_data_brief hardcodes value_column = numeric_cols[0] (core/insight.py:407), and I confirmed by reading that build_answer does the same at line 1048 and emits headline='Pumps leads at 3,800,000.', short_value='3,800,000', comparison='1,400,000 above the next result'. That is the LEAD of the card (portal_chat.html renders short_value + headline in the .answer-value/.answer-headline slot), while insight_summary is a tier-'note' footnote. Design 1 fixed only the footnote, so the card would still open with the 2024 leaderboard and a cross-category gap chip that reads exactly like a year-over-year delta. The target question is not causal (is_causal_question is False), so no LLM narration runs at all — these deterministic sentences ARE the answer the user reads.

**Change.** _period_comparison_summary(rows, plan_labels, column_formats, display_formats) -> str: returns '' unless >=2 of the plan's exact period aliases are present and every one parses as a real period. Computes grew/shrank/flat counts, top grower and top shrinker with their absolute and percent change, and the top grower's share of the change; renders every number through the existing format_value closure so column_formats/display_formats are honoured; redacts every category label through `from core.insight import _display_label as _redact_value_label` — ALIASED, because core/response_builder.py:572 already defines a one-argument _display_label column prettifier used at :663, :721, :1018, :1445 and :1465, and an unaliased import would shadow it and raise TypeError in an unprotected path. Falls back to a label-free sentence when redaction fired. build_answer gets a matching branch returning headline/short_value/comparison describing the CHANGE (e.g. short_value the newest-period total, headline naming the top mover, comparison the period-over-period percent) before the existing ranking path. _build_insight_summary returns the summary sentence when non-empty. _build_decision_signal returns '' for this shape rather than a 2024 concentration claim. Amend 2041-2045 so a period-comparison result is not clobbered by the keep_top/sort/contribution mode downgrade. Every branch returns '' / falls through on any non-matching shape.

**Test.** NEW tests/test_multi_period_answer_surface.py::test_period_comparison_answer_and_summary — call build_assistant_response on the wide two-period rows with display_context carrying the plan labels, and assert on the RETURNED dict: answer['headline'] does NOT contain 'leads at', answer['headline'] and answer['comparison'] mention the change or the period labels, and insight_summary contains a grew count, a shrank count and a share-of-change figure. ::test_ordinary_results_unchanged — call build_assistant_response on (a) a plain single-measure ranking result and (b) a result with columns P_CODE/P_QTY and assert headline/short_value/comparison/insight_summary are byte-identical to the pre-change values captured in the same test from the untouched code path. ::test_sensitive_label_column_redacted — rows keyed EMPLOYEE_NAME with two period columns; assert no employee name appears in the summary and 'redacted segment' or the label-free form does.

### Step 9 — core/chart_spec.py, core/query_pipeline.py, core/result_renderer.py

**Where.** core/chart_spec.py: the y_cols selection in infer_chart_spec's ranking/breakdown branch (~514-525). query_pipeline/result_renderer: carry an explicit format map for the plan's period columns through display_context into build_column_formats' explicit_formats argument.

**Why.** I executed infer_chart_spec on the exact target shape and found two real defects, both of which make the chart worse than the table. (a) y_keys comes back as ['NET_AMOUNT_2024','NET_AMOUNT_2025','CHANGE_PCT','SHARE_OF_CHANGE_PCT'] — money and percent on one axis. (b) With a generically-named measure the period columns are DEMOTED: rows keyed TONNAGE_2024/TONNAGE_2025 with whole-number values give role='identifier' (via _looks_identifier's all-integer + high-uniqueness rule, which only spares names matching _CURRENCY_RE/_PERCENT_RE/_COUNT_RE), so the chart loses both money series. I verified that passing column_formats={'TONNAGE_2024':'number','TONNAGE_2025':'number'} restores role='measure'. Design 1 claimed the chart was free and predicted the opposite failure mode; it is neither free nor as predicted.

**Change.** (a) In the ranking/breakdown branch, when >=2 non-percent measures exist, exclude percent-formatted measures from the default y_cols (they stay in allowed columns and in the table; they simply stop being default series). This is a general improvement for any result mixing a money and a rate column, not a multi-period special case. (b) Have the multi-period step publish {alias: 'number'} (or the base measure's own resolved format when known) for its period columns via display_context, and merge that into the explicit_formats passed to build_column_formats at core/result_renderer.py:719 and core/response_builder.py:2047 — the existing explicit_formats channel, no new mechanism.

**Test.** tests/test_chart_spec.py::test_percent_measures_are_not_default_series — call build_chart_payload on the wide rows and assert y_keys == ['NET_AMOUNT_2024','NET_AMOUNT_2025'] exactly (today it returns four keys — assert the pre-state in the same test). ::test_generic_measure_name_stays_a_measure — call build_chart_payload on the TONNAGE_2024/TONNAGE_2025 variant WITH the carried formats and assert y_keys == ['TONNAGE_2024','TONNAGE_2025'] (today, without formats, infer_chart_spec returns recommended_type 'table' and build_chart_payload returns None for the two-column case — assert that pre-state too). ::test_unrelated_single_measure_chart_unchanged — a result with one currency measure and one pct measure and one dimension: assert y_keys is unchanged from today.

### Steps 6 to 9 as built

Each was implemented as specified above, with these notes.

**Step 6.** `build_multi_period_sql_hint(plan, db_type)` in `core/multi_period.py`;
the branch sits between the anomaly and contribution branches with its own
`try/except` at `log.warning(exc_info=True)`, and `_intents.get('contribution')`
became `_intents.get('contribution') and not _mp_plan`. The SQL the hint
prescribes was executed through `validate_sql_detailed` (ok / code `ok`, with
`production_sql` and a composition contract) and through
`compliance.sql_guard.analyze_sql` (`has_star` False; all five measure columns
in `aggregate_outputs`). Hint length is roughly 480 tokens.

**Step 7.** `annotate_period_change(rows, plan, truncated)`. Its guard messages
go on `PeriodPlan.warnings` — the mutable channel the frozen dataclass already
declares — and the pipeline copies them into the `multi_period_caveats` entry of
`_confidence_context`, the same list object it keeps appending to, which
`core/result_renderer.py` renders beside the forecast caveats. Two additions
beyond the specification: `CHANGE_PCT` is recomputed alongside `CHANGE_ABS` so
the derived columns cannot be half the model's and half ours, and a plan whose
periods only PARTLY came back computes over the ones present and says which are
missing, rather than returning `None` as if nothing had landed.

**Step 8.** `_period_pair_facts` holds the arithmetic; `_period_comparison_summary`,
`build_answer`, `_build_insight_summary` and `_build_decision_signal` all read
it. Which columns are the plan's periods is decided only by
`core/multi_period.period_columns_by_alias`, shared with the post-processor and
the pipeline. The labels reach `build_assistant_response` on
`display_context['period_comparison']`, published only when the annotation
actually applied.

**Step 9.** `_default_series` in `core/chart_spec.py` drops percent-formatted
measures from the default y-columns once at least two others remain; the period
columns' formats are merged into `build_column_formats`' explicit-format map
from the `display_context` both of its call sites already pass. Both defects
were reproduced before the change (`y_keys` of four keys; `build_chart_payload`
returning `None` for the TONNAGE variant) and both pre-states are pinned in the
tests.

Suite: **5847 passed** locally, four pre-existing failures unrelated to this
work (that container runs Python 3.11, where `dis.Instruction.line_number` does
not exist, and duckdb 1.5, which returns `datetime` where two tests expect
`date`).

Every test executes the real function and asserts on its return value, and each
was watched going red with its defect reintroduced.

### Step 10 — (no file — live verification)

**Where.** A real tenant, the target question, through the running portal.

**Why.** THE STEP THAT DECIDES WHETHER THIS IS REAL. Everything above is a prompt plus deterministic Python around it; nothing forces the model to emit the pivot. The repo's own standing notes ('Tests Must Execute the Path', 'Verify After Editing Control Flow') were written after exactly this class of miss, and the fan-out judges were right that the wiring — not the pure functions — is what historically rots. A green suite here proves the pieces work, not that the feature fires.

**Change.** Run 'Compare 2025 against 2024 by revenue category which grew, which shrank, and what did each contribute to the overall change?' against a real tenant. Capture and inspect: (1) the resolved date binding (did a role bind, or did step 5 produce a date-role clarification instead?); (2) the generated SQL — does it carry both grounded predicates and two period columns, or did the model ignore the hint?; (3) the post-execution log line — 'change contribution added' or the miss warning; (4) the rendered card — headline, insight_summary, caveats, chart series. Do not declare the defect fixed on any weaker evidence. If the SQL comes back single-period, the honest outcome is that the caveat fired correctly and the next commit is a deterministic compiler (see open questions), not more hint prose.

**Test.** Manual live run, evidence captured in the commit message. Not substitutable by a unit test.

---

## Do not do

Each of these was verified by execution during planning.

- Do not use SELECT * anywhere, including inside a CTE. _production_shape_errors walks every Select scope and production_sql is always set at core/query_pipeline.py:4709; I executed it and confirmed the refusal. It also trips classified_select_star in the governed executor on any classified tenant.
- Do not add the new hint branch inside the shared try at core/query_pipeline.py:4403-4545. Its except at 4543 logs at log.debug, wipes _intents and resets the analysis contract — silently disabling the composition validator and the entire post-execution analytics dispatch at 6471+. Give the branch its own try/except with log.warning(exc_info=True).
- Do not gate the post-execution branches on _intents['multi_period']. Gate on the stashed PeriodPlan. I executed 'what did each region contribute to revenue in 2023, 2024 and 2025': the intent is truthy, wants_comparison is False, contribution is True — gating on the raw intent silently deletes contribution_pct from a question that literally asks 'what did each contribute'.
- Do not guard the clock with a substring test. 'compare revenue for the last 2 years for warehouse 2025 and warehouse 2024' produces clock-derived labels that DO appear in the question as warehouse IDs. Use the provenance flag set by the branch that produced the specs.
- Do not start any hint line with '- ' or include any string from core/llm.py:43-66. Hints sit after _KB_SECTION_MARKER, the last boundary the rule filter knows, so a marker there makes _filter_sql_rules_for_compiled_plan delete from the match to len(prompt).
- Do not import core.insight._display_label unaliased into core/response_builder.py. That file already defines a one-argument _display_label column prettifier used at :663, :721, :1018, :1445 and :1465; shadowing it raises TypeError in an unprotected path.
- Do not name the period columns with a generic prefix like P_2024. I executed infer_chart_spec: a name not matching _CURRENCY_RE/_PERCENT_RE/_COUNT_RE with whole-number values is classified 'identifier' and the chart disappears. Derive the alias from the measure column name.
- Do not reuse the column name contribution_pct for a share of the NET CHANGE. Every other answer in the product uses that name for share-of-current-total. Use SHARE_OF_CHANGE_PCT and an explicit suppression flag; overloading the name to exploit the idempotency guard at 6474 creates the next divergence.
- Do not wrap the date column in YEAR()/DATEPART()/FORMAT()/CONVERT(). Reference the calendar dimension's own attribute through the approved date-role join (validator.py:3237-3244 builds acceptable_cols = {plan_col} | calendar_attributes.values()), or use a half-open range on the approved date value. That keeps the surrogate_date_conversion rules at validator.py:1609 and :1692 armed and never triggered.
- Do not emit a hint with a placeholder predicate when no governed date field resolves. Forbidding YEAR()/DATEPART() while supplying no sanctioned alternative is an unfollowable instruction. Return None and emit nothing.
- Do not widen the date-binding gate for single-period absolute questions ('revenue in 2024') in this change. That is an unbounded behaviour change on every tenant — it can newly produce date-role clarification prompts and new fail-closed refusals — and it is not needed to fix the reported defect.
- Do not add a sixth comparison vocabulary. Reuse core/query_semantics.py:168 wants_comparison, the one live detector that already lists 'against', 'relative to', 'contrast', 'delta', 'variance' and 'benchmark'.
- Do not add a bare \bor\b to any comparison vocabulary — 'revenue for Q1 or Q2', 'customers or suppliers' would fire constantly.
- Do not write any test that asserts on source text. Every test above runs the real function and asserts on its return value; the wiring is proved by step 10's live run.
- Do not declare the defect fixed on a green suite. The hint persuades; it does not compel. Step 10 is not optional.

---

## Open questions for the product owner

- Should the date-binding gate also widen for SINGLE-period absolute questions ('revenue in 2024', 'revenue for March 2026')? I verified these bind no date role today, which means the LLM guesses a date column with no governed join — arguably a larger correctness problem than the reported defect. But widening it changes behaviour on every tenant and can convert answers into date-role clarification prompts. My plan deliberately leaves it out. If the owner wants it, it should ship behind a flag with a soak against a corpus of real tenant questions, and nobody can bound the clarification-rate impact from reading alone.
- Should the feature fire on legitimate multi-period questions that carry no comparison word? I executed 'what did each region contribute to revenue in 2023, 2024 and 2025' and 'revenue for Jan 2024, Feb 2024 and Mar 2024 by product': both name real periods and both fail wants_comparison, so no hint is emitted and they behave exactly as today. I chose precision over recall because a wrong pivot instruction is actively harmful while a missed pivot degrades to current behaviour. That is a judgement call the owner may want the other way.
- Is the 6-period cap right? Above 6, portal_chat.html:1746-1749 silently drops series past the palette length, so a 12-period request would render fewer series than asked — the same class of silent-incompleteness as the original defect. Capping in SQL is honest but narrows the user's request. The alternatives are raising the palette or rendering a table instead of a chart past N.
- SUM(CASE WHEN … THEN measure ELSE 0 END) makes 'no rows in this period' indistinguishable from 'genuinely zero'. A category that did not exist in 2024 reports as +100% growth from zero. A COUNT(CASE WHEN …) per period would separate them; I left it out to keep the SQL short (hints are the first bytes truncated). Worth deciding whether that distinction matters enough to spend the tokens.
- If step 10's live run shows the model ignoring the hint, the next commit should be a deterministic compiler generalising the shipped period_over_period_entity_change branch at core/pipeline_helpers.py:1013-1082 from count-distinct to any single-root-aggregate measure — with an explicit window-kind allowlist and no implicit else, and with columns enumerated rather than SELECT *. That is a larger, riskier change and should not be bundled here.
- Four independently-verified defects surfaced that are outside this fix's scope and deserve their own commits: core/period_comparison.py:496 passes semantic_context=None to its governed executor, silently disabling most of the validator for its second query; core/period_comparison.py:499-504, core/drill_dimension.py:359-363 and core/insight.py:1551-1553 each execute completely ungoverned when query_executor is None; store/compliance_store.py:554-593 writes the policy hash chain unlocked, so any concurrent governed execution can fork it; and the assistant_analysis 'headline' field — the only place compare_prior states its percentage — is computed and never rendered (portal_chat.html:3882-3919 never reads it).

---

## What the reader should see when this is finished

One governed query returns one row per revenue category with `NET_AMOUNT_2024`, `NET_AMOUNT_2025`, `CHANGE_ABS`, `CHANGE_PCT` and `SHARE_OF_CHANGE_PCT`, ordered biggest-mover first — all four things the question asks. The card's LEAD line describes the change ("Revenue rose 5.7% from 2024 to 2025; Pumps added the most, +400,000") instead of today's "Pumps leads at 3,800,000." The note below it reads roughly: "Across 12 revenue categories, 7 grew and 5 shrank between 2024 and 2025. Pumps added the most (+400,000, 46% of the total increase); Valves fell the most (-180,000)." The chart is a two-series grouped bar of 2024 vs 2025 by category, with the percent columns in the table but off the axis. The SQL shown in the trust panel is the single statement that produced those rows. If the model ignores the hint and returns one period, the user sees a coverage caveat saying the answer covers only part of the periods asked about, `contribution_pct` still appears (the existing post-processor is not suppressed on a miss), and a `log.warning` names the miss — instead of today's silent, confident explanation of a change the query never fetched.
---

## Standing constraints (product owner)

- Do not relax the SQL validators.
- Do not hardcode tenant table names — EMCO is the client, the fix must be generic.
- Do not bypass graph validation.
- Do not create direct fact-to-fact joins.
- Every SQL execution goes through `execute_governed_query` with its
  argument-independent guarantees intact.
- Tests must **execute the path**, never assert on source text. A fix once
  landed in dead code here and the suite stayed green.
- Do not add `Co-Authored-By` trailers to commits.

## Step 10 cannot be done in a cloud session

It needs a live tenant: `.env` is gitignored, so a clone carries no database
credentials and no route to the EMCO server. Steps 6-9 are ordinary repo work
and can be done anywhere. Step 10 must be run by the owner, and the defect is
not fixed until it is — the hint persuades the model, it does not compel it.
