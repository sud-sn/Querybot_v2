# French localisation and the reporting-space UI — plan and handoff

Branch: `fix/value-grounding-governance-and-sweep`

Two features were asked for together:

1. **Language selection.** Stored **per user**. The client's customers are French:
   they will **type their questions in French** and everything they read must be
   French.
2. **The reporting-space UI** — the "Add to dashboard" picker and the dashboard
   page — "they don't look good".

They are written up together because they collide in the same lines, and the
order they ship in decides whether the second one has to be redone.

---

## The finding that reshapes feature 1

"Everything in French" reads like a translation job. It is not. The product
understands a question by running hand-written **English regexes** over the raw
text, and every one of them runs **before any model sees the question** — the
first `llm_complete` on a fresh question is SQL generation at
`core/query_pipeline.py:5111`.

Executed, five English questions and their French equivalents:

| Question | EN intents | FR intents |
|---|---|---|
| Compare 2025 against 2024 by revenue category | `multi_period` | — |
| Show me the top 10 customers by margin | — | — |
| What is the trend of sales over the last 6 months | `multi_period`, `relative_date` | — |
| What did each region contribute to revenue | `contribution` | `contribution` |
| Forecast revenue for the next 3 months | `forecast` | — |
| **total** | **5 intents / 5 semantic flags** | **1 / 1** |

The single French survivor is the cognate *contribution*, and even it fires a
**different** semantic flag (`wants_share`, not `wants_grouping`).

**None of this errors.** A French question with no intents detected produces a
plain grouped `SELECT` and a confident narrative answer.
*"Montre-moi les 10 meilleurs clients par marge"* loses `wants_top_n`, so no row
limit is ever requested and the user gets every customer — presented as the
answer to "top 10". A French portal that answers the wrong question fluently is
worse than an English one that answers the right question.

Two further measured facts constrain the design:

- `core/date_roles.py:287` normalises with `re.sub(r"[^a-z0-9]+", " ", text.lower())`,
  which **shreds** accented French (`année` → `ann e`) rather than merely failing
  to match it. Adding French words to the vocabularies would not help there.
- Retrieval is a second, independent failure. BM25 (`core/vector_store.py:438`)
  strips every non-`[A-Za-z0-9_]` character, and the embedder is
  `all-MiniLM-L6-v2`, English-only. Measured token overlap between a French
  question and the English KB: **zero**.

---

## Decision — canonicalise at the front door, never translate in place

A **canonicalising normaliser** turns the French question into the product's own
canonical English analytics phrasing before the detectors run. Every detector
stays English and unedited.

Measured: literal machine translations of the same French corpus, through the
**untouched** detectors, recover **12/15 analytical intents and 19/19 semantic
flags**.

Rejected: **French twins for every detector vocabulary.** Measured at ~1,173
distinct English lexical items across 484 regex patterns and 101 vocabulary sets
in 28+ modules. It is also the failure mode this codebase documents against
itself — `core/multi_period.py:438-445` ("a documented habit of growing parallel
detectors for the same concept and letting them drift") and the post-mortem at
`core/llm.py:164-178`, where two *English* vocabularies for one concept drifted
by 13 phrasings and silently inverted an anti-join. And it still would not work,
because of the accent-shredding and the retrieval failure above.

**The normalised text is an added field, never a replacement.** This is the
whole design, and it is the difference between working and a French user's
dashboard filling with English tile names.

- **Canonical English** goes to: the 15 analytical detectors, `analyze_query_intent`,
  `question_has_temporal_intent`, `detect_temporal_window`, `resolve_source_scope`,
  `extract_candidate_phrases`, BM25/embedding retrieval, and the routing gates.
  `_semantic_plan_question` (`core/query_pipeline.py:2614`) is already the repo's
  established name for "text the deterministic planners should see, as distinct
  from what the LLM and the user see" — it is the correct carrier.
- **The user's own French** goes to: `extract_display_question`, and therefore the
  chart title and the dashboard tile name (`core/response_builder.py:2286-2289`,
  persisted by `store/dashboard_store.py:233`), the clarification wrapper, the CSV
  filename, the answer trace, and `llm_audit_scope(question=…)`.

It is gated on the per-user language, so English tenants pay no latency.
A deterministic lexicon handles the ~40 highest-value terms
(`chiffre d'affaires`→revenue, `marge`→margin, `T1`→Q1, `mois dernier`→last month,
`écart budgétaire`→budget variance, `valeurs aberrantes`→outliers) *before* the
model, so the common cases never depend on model behaviour.

---

## Decision — one catalogue, two deliveries

`core/i18n.py`: `MESSAGES: dict[msg_id, dict[lang, str]]` with **named**
`str.format` placeholders, a `t(msg_id, **kw)` helper, and a `ContextVar` locale
defaulting to `"en"` — modelled line-for-line on `core/vocab_packs.py`, which
already solves per-request state in this async app. Default-English means every
un-converted call site behaves byte-identically today.

The same catalogue is delivered to templates as a Jinja global and to the browser
as `const I18N = {{ catalogue|tojson }}` — the pattern already at
`portal_chat.html:793`.

Rejected: **gettext/Babel.** `xgettext` cannot extract f-strings, and 426 of the
1,306 user-facing strings are f-strings (263 interpolating mid-sentence), so the
toolchain that justifies gettext buys nothing here; `babel` is not installed; and
`gettext.install()`/`setlocale` are process-global in an app that has two
languages in flight on one event loop.

Rejected: **post-hoc translation of the finished payload.**
`sanitize_response_text_fields` looks like the perfect single hook. Proven by
execution that it cannot work: by the time the payload reaches it the field holds
`'Acme SA leads at 1,250,000.'` — copy, tenant entity name and formatted number
fused into one opaque string.

Message ids are **whole sentences, never fragments**. `_period_comparison_summary`
currently glues clauses with `'; '.join()` and sentence-cases with
`detail[0].upper()`; French word order and agreement make that impossible.

---

## Decision — two formatting seams, not one and not eight

`core/locale_format.py` (Python) and `static/js/qb-format.js` (browser), mirroring
each other, both reading one locale resolved once per request.

Rejected: **Python's `locale` module.** Measured: `locale.setlocale` corrupted
**12 of 20** concurrent requests on this repo's own FastAPI threadpool, and
`fr_FR.UTF-8` is not generated in the container at all. A hand-rolled formatter
(`f"{abs(n):,.{d}f}"`, then `,`→U+202F and `.`→`,`) reproduced ICU `fr-FR` exactly
on **1,535 of 1,555** sampled renderings; the 20 exceptions are Python's
banker's rounding vs JS half-expand, a divergence that **already exists** between
this app's server and browser in English.

The de-formatters are the trap. `_parseDisplayNumber` (`portal_chat.html:2800`)
and `sortDashboardTable` (`portal_dashboard.html:281`) parse formatted text back
out of the DOM. Verified: `'1 234 567,89'` → `123456789`. The Sigma total and the
column sort go silently wrong the moment cells render French.

---

## Sequence, and why this order

The two features **collide**: the picker modal holds 47 English strings and the
dashboard page ~60 more. A redesign that ships first hardcodes a fresh set of
English strings in exactly the region the catalogue then has to re-extract. So
the mechanism lands first and the redesign is built French-ready.

| Stage | What | Why here |
|---|---|---|
| 1 | `portal_user.lang` + setter + switcher; `core/i18n.py`; `core/locale_format.py` + `qb-format.js` | Foundation. Small. Nothing visible changes. |
| 2 | Picker modal: the proven defects, then the redesign | The user's complaint. Uses the catalogue from line one. |
| 3 | Dashboard page redesign | Sits on a tile that has never had number formatting; needs stage 1. |
| 4 | The question normaliser + a French eval corpus | The part that decides whether the product works at all. |
| 5 | Translate the chrome, the deterministic answers, the prompt directives | The bulk. Mechanical once 1–4 exist. |

---

## Status

| Stage | State | Where |
|---|---|---|
| 1 — mechanism | **done** | `core/i18n.py` (~350 ids), `portal_user.lang` (v39), `POST /portal/api/language`, the Jinja context processor, `window.QB_I18N` |
| 2 — picker modal | **done** | `portal_chat.html` picker: 8 defects fixed, then redesigned French-ready |
| 3 — dashboard page | **done** | `portal_dashboard.html` restructured; 49 strings; `static/css/dashboard.css` |
| — portal shell | **done** | `portal_base.html`: sidebar, mobile bar, confirm dialog, live toast, `<html lang>` |
| — language switcher | **done** | Sidebar footer, third column of `.portal-side-actions`. A plain form: no JS, so nothing has to re-translate the shell in the browser |
| — retroactive flip | **done** | `GET /portal/api/history/{thread}` rebuilds every turn from `answer_trace.result_rows` under the reader's language, and re-checks the table grant while it is at it |
| — answer card | **done** | `build_answer`, `infer_result_scope`, insight summary, anomaly callouts, decision signal, action chips, analysis title |
| — narration language | **done** | `core/insight._language_rule`, applied in `build_insight_prompt_from_contract` and the period-comparison prompt |
| 4 — French normaliser + eval corpus | **done** | `core/question_normalizer.py`, `evals/french_questions.yaml`, `evals/french_parity.py`. Intent parity 100%, against 1.6% without it |
| — the chat page | **done** | `portal_chat.html`: 315 `t()` calls, up from 31 |
| — browser tab titles | **done** | All ten portal templates |
| — the remaining portal pages | **done** | sign-in, registration, change-password, pin-confirm, new-report, notifications, Semantic Layer |
| 5 — the rest of the chrome | **portal done; server-side copy remains** | See below |

### The normaliser, and what it had to get right

`core/question_normalizer.py` canonicalises a French question into the
product's own English analytics phrasing before the detectors run, so all 484
regex patterns stay unedited. It is an ADDED name (`_analysis_question`), never
a replacement: `question` stays the reader's own words and keeps flowing to the
chart title, the dashboard tile, the trace and the audit log.

Five things it had to get right, each found by executing it:

* **"contribute", not "contributed".** `core/contribution_analysis.py` matches
  `contribut(?:ion|e|es|ing)?` — the past participle is the one form that
  pattern does not read, so the obvious translation of "a contribué" is the one
  that fails.
* **A single quote is a delimiter only at a word boundary.** A naive
  `'[^']*'` pairs the apostrophe in "d'affaires" with the opening quote, so the
  real customer value goes UNPROTECTED and half the question goes untranslated.
* **The elided article outlives its word.** "l'évolution" matches "évolution"
  and comes out "l'trend". Stripped after the lexicon, so "aujourd'hui" and
  "chiffre d'affaires" are already consumed.
* **"ça" and "CA" fold to the same two letters,** so the abbreviation for
  chiffre d'affaires is deliberately absent from the lexicon.
* **The determiner is translated, not dropped.** "over last 30 days" misses
  `over the last \d+` by one word, and the reader gets a flat aggregate where
  they asked for a series.

One deliberate departure from the list above: **value resolution keeps the
reader's text.** Candidate phrases are customer VALUES, and a customer called
"Marge" or a product called "Premier" is a French word that must not be
rewritten. Quoted spans are protected in the normaliser, but an unquoted one is
not, and this is the one consumer where a rewrite changes which rows come back
rather than which words are searched.

### What is still English

Ordered by how often a reader sees it.

1. **Number formatting.** `_format_number` in `core/response_builder.py`
   groups thousands with `,` and puts the decimal point at `.`; French does
   the opposite, so "1,234" reads as one and a bit. `i18n.format_count`
   exists for whole counts and the truncation caveat uses it, but the
   display-format pipeline (currency symbols, compact `K`/`M`, fraction
   digits) is a separate pass and is untouched.
2. **`core/answer_formatter.py`'s section markers** — deliberately English
   and deliberately out of the catalogue; see the note beside the
   `ui.chat.diag.*` ids.
3. **`core/conversational.py`'s `build_reply`** — the login greeting, and
   `core/clarification.py`'s `clarification_rejection_message`. Both are
   called from `gateway/webhooks.py` but written elsewhere; the socket now
   activates the language around them, so each is a catalogue pass in its own
   module away from being done.
4. **`core/result_renderer.py`'s `_build_cannot_generate_hint`** — the
   "I could not build SQL for that" hint, sent as `result_chat_error`.
5. **The platform webhooks** — Zoom/Teams/Slack signature and identity errors,
   and the `/api/ask` JSON error contract. These do not reach a portal reader
   and have no `portal_user` to take a language from.
6. **`admin/`** — out of scope by design. It has its own `Jinja2Templates`
   with no context processor, and ~1,546 strings.

### The chat socket, and what it needed

`core/drill_dimension.py` and the whole user-visible surface of
`gateway/webhooks.py` are done: 113 payload strings plus the locals that feed
them, under `reply.*`.

The prerequisite was that no language was active on that path at all.
`core/query_pipeline.py` activates one for the duration of ONE answer, which
covers the answer and nothing around it — every refusal, nudge and dashboard
confirmation is sent from the socket loop, outside that scope. So `ws_chat`
activates the reader's language once, right after the session cookie resolves:

* **Once per connection, not per message.** Each websocket connection is its
  own asyncio task and a task gets a COPY of the context, so one reader's
  language cannot reach another's socket. `tests/test_chat_socket_language.py`
  holds two sockets open at different languages and asserts it.
* **Never reset.** The context dies with the connection. Switching language
  re-renders the portal page, which drops the socket and opens a new one, so a
  stale value is not reachable.
* **Proven over a real websocket, not asserted from the source.** Wiring like
  this is exactly what can be written, reviewed, merged and never execute:
  every `_t()` in `webhooks.py` is English until that ContextVar is actually
  set on that task.

Four defects that were already live in English:

* `f"{operation.title()} analysis completed."` title-cased a wire token into a
  word, and the unnamed operation read "Analysis analysis completed."
* `"" if n == 1 else "s"` on the analysis row label — English again, and
  French takes the singular at zero.
* `"Monday Tuesday ...".split()[day_of_week]` built the report schedule line
  out of a hardcoded English week.
* The hand-built `result_scope` on the analysis card said "Returned result
  only", a badge no other surface used; it now takes `answer.scope.returned`
  like everything else.

Two things were deliberately left in English and are pinned as such by tests:
`drill_question` (`"{question} — broken down by {dim}"`) and the follow-up
suggestions, because both are read back by the English intent detectors and
cached as the question the NEXT action reads. `_dx_follow_up` is a prompt to
the model and stays English for the same reason.

### The analysis card and the coverage caveats, and what they needed

Both are done. `build_analysis_response` is 100% catalogue, and the caveats
are translated where they are PRODUCED — `core/forecast_gate.py`,
`core/date_coverage.py`, `core/join_coverage.py`, `core/multi_period.py`, plus
the truncation sentence in `core/result_renderer.py` and the model-fallback
note in `core/query_pipeline.py`. `core/result_renderer.py` only concatenates,
and the pipeline's `activate_language` covers all six producers.

Four defects that were already live in English, found only because the strings
had to be pulled apart:

* `f"{pct:.1f}% {direction} than the starting period"` with a three-way
  higher/lower/**flat** direction produced "0.0% flat than the starting
  period" on a series that did not move.
* `f"{missing} of {expected} {grain}s are missing"` agreed the verb with
  `expected`: "1 of 24 days **are** missing".
* `metric_subject.lower()`, applied to fit mid-sentence, lowercased the
  tenant's own metric name — an `EBITDA` metric was reported as "ebitda
  records". So did `"The selected metric"`, capitalised mid-sentence.
* The forecast gate was handed the reader's raw question rather than the
  canonical English one, so its English grain-mismatch regex matched nothing
  and a French reader was never told the projection had changed grain — the
  caveat was translated and unreachable at once.

Three patterns worth recognising elsewhere:

* **A wire token pluralised by suffix.** `f"{grain}s"`, `date_label += "s"`,
  `f"{n} day{'s' if n != 1 else ''} ago"`. "mois" is already its own plural
  and French takes the singular at zero. `i18n.grain_label()` reads the
  catalogue and lets an unrecognised tenant grain through untranslated,
  because that one is data.
* **A module-level constant.** `_TRUNCATED_WARNING` was built at import time,
  which is before any request and therefore before any answer language. The
  lookup has to happen where the guard fires.
* **A placeholder that only one language needs.** English compounded
  "6-month period"; French counts "période de 6 mois". Passing both `{unit}`
  and `{units}` and letting each language pick is exactly what
  `test_placeholders_match_across_languages` cannot check, so both languages
  count the grain instead.

A tenant `business_role` is English data ("invoice", "posting"), and
`"date de invoice"` needs an elision no rule can decide from an arbitrary word
— a mute h and an aspirated h take opposite forms. The French label quotes it:
`dates « invoice »`.

Every portal template's body is done. `tests/test_portal_pages_language.py`
holds the list and asserts it matches what is on disk, so a page added later
and not translated fails there rather than shipping.
`tests/test_analysis_card_language.py`,
`tests/test_coverage_caveat_language.py`,
`tests/test_chat_socket_language.py` and
`tests/test_drill_dimension_language.py` do the same for the surfaces above, by
executing the real producers in both languages — including the post-processing
block compiled out of `core/query_pipeline.py` and a real `_send_results`
render, so a sentence translated at its source but concatenated again
downstream still fails.

### The chat page, and what it needed

`portal_chat.html` is done: 315 `t()` calls, up from 31. Everything a reader
sees — the shell, the composer, the stage trail, every toast, the answer card,
the trust disclosure, the diagnostic card, the clarification cards, the history
panel, the chart legends.

Four things there could not be translated in place, and each is a pattern worth
recognising elsewhere:

* **A sentence assembled from fragments.** `subject + ' was ' + verb + ' by
  admin.'` has no seam in French, which puts the participle after the auxiliary
  and agrees it with the subject. Whole sentence per outcome.
* **A count with an inline `s`.** French takes the singular at zero, so every
  `n !== 1 ? 's' : ''` is wrong on an empty result. `plural()` on both sides —
  and `window.qbPlural` exists precisely because the browser builds sentences
  the server never sees. A test executes both and compares.
* **A string that is also a wire value.** The diagnostic card's markers
  ("Most likely reason:", "SQL tried:") are written by
  `core/answer_formatter.py` and scanned by the page; the forecast chart's
  "95% interval" is both a legend entry and the seriesName the tooltip filters
  on. Split the two jobs or keep one constant, never translate in place.
* **A callback that shadows `t`.** `.map(t => escHtml(t))` binds the table name
  over the translator. Two of these were live in the file; any lookup added
  inside either would have silently returned a table name.

### Two things deliberately NOT translated

Both are tested for, because translating either is a silent behaviour change
rather than a visible one.

- **Anything compared by name.** A chip `id`, an analysis `action`, a
  callout's `severity`, a decision signal's `tone` and `basis`, and the
  structural labels `HEADLINE:`/`BODY:`/`DETAIL:`/`SECTION:`/`NEXT:` that
  `parse_insight_response` matches literally. `infer_result_scope` publishes
  `badge_key` next to the translated `badge` for exactly this reason: the
  narration prompt reads the key.
- **Customer data and schema.** Column names, category values, dimension
  names, group names. A translated header stops matching the table under it.

---

## Do not do

- **Do not put the language on `display_context`.** At
  `core/query_pipeline.py:1534` it is `dict(_cache_snapshot.get("metadata"))` — a
  cached snapshot — so a user who switched to French would keep getting English on
  every governed-cache hit. Ride `portal_user`, which is a required positional of
  `_send_results` at all 4 call sites, and re-renders from cached **rows**.
- **Do not default the column to `'fr'`.** It would silently flip every existing
  tenant on upgrade. Default `'en'`; give the French client a per-client default
  separately.
- **Do not add `lang` to both `_SCHEMA` and the migration list.** `init_db()` runs
  migrations on fresh databases too (that is how `last_active_at` reaches a new
  DB). Two definitions drift.
- **Do not write `user['lang']`.** On a pre-migration database the dict simply
  lacks the key, with no exception. Always `(user or {}).get('lang') or 'en'`.
- **Do not add a `CHECK` constraint.** SQLite enforces one on `ADD COLUMN` but
  cannot drop it; a third language would need a table rebuild. Validate in the
  setter.
- **Do not translate `'redacted segment'`.** It is a PII sentinel produced at
  `core/insight.py:278` and compared **by equality** at
  `core/response_builder.py:847` and `core/insight.py:1615`. Translating it makes
  the equality fail and the real label reaches the user. Same for `_MASKED_MARKERS`
  and `sanitize_db_error`'s `cleaned` field.
- **Do not tell the model "respond in French" without pinning its control tokens.**
  `parse_insight_response` matches literal `HEADLINE:`/`SECTION:`/`BODY:`, and
  `clarification.py:1490` compares status to literal `AMBIGUOUS`. A model writing
  French will translate those labels and the parser silently degrades.
- **Do not put a blanket French directive on `llm_complete`.** Of its 31 call
  sites only 9 produce user-visible prose; 7 generate SQL and 7 generate JSON the
  pipeline parses.
- **Do not translate `_plural`.** French agreement depends on the gender of each
  tenant column, which is a property of the semantic layer
  (`entity_properties.display_name`), not of the message.
- **Do not rely on a `ContextVar` across `loop.run_in_executor`.**
  `core/vocab_packs.py`'s own docstring warns of it and
  `core/query_pipeline.py:6956` records a real incident. The pipeline's five
  executor hops all run SQL and produce no prose, but anything else takes the
  locale explicitly.
- **Do not move `_consume_pin_token` earlier** in the picker flow, and do not add
  an inline retry to the 409 path — the token is already consumed there, so the
  retry is a guaranteed 400.
- **Do not restyle the dashboard page without restructuring it.** A third of its
  declared styles are already overridden by `production.css` and would stay
  overridden.
- **Do not trust the existing template tests during a partial translation.**
  Proven: `'Add to dashboard'` exists at six sites in `portal_chat.html` (markup
  771, 782; JS 926, 1012, 3405, 3606). Translating only the markup leaves
  `tests/test_dashboard_agent.py:175` green with four English occurrences still
  rendering.

---

## Known defects found while mapping, to fix in passing

- `.dashboard-picker-new[hidden]` has no author rule, so the "Create new" panel is
  not reliably hidden — most of "it doesn't look good".
- Two "Add to dashboard" buttons render at once above 520px (`portal_chat.html`
  771 and the injected 3405); only one is hidden, and only below 520px.
- The 409 path consumes the pin token before failing.
- Nothing marks a result as pinned, so a second click walks into
  "This pin request is invalid or has expired."
- `data-chart='...|safe'` on the dashboard breaks on the first apostrophe in a
  value — which a French dataset will produce immediately. `|tojson` fixes it.
- The dashboard table tile renders raw Python floats through Jinja
  (`portal_dashboard.html:229`) — it has never had any number formatting.
- Dragging a card unpublishes a team dashboard.
- `_fmtNum` has already drifted between the two templates: `1.2e12` gives `'1.2T'`
  in chat and `'1200B'` on the dashboard.

## Standing constraints (carried from MULTI_PERIOD_WIRING_PLAN.md)

- Do not relax the SQL validators; do not hardcode tenant table names.
- Every SQL execution goes through `execute_governed_query`.
- Tests must **execute the path**, never assert on source text.
- No `Co-Authored-By` trailers.
