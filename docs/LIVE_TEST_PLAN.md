# Live test plan — `fix/value-grounding-governance-and-sweep`

Everything on this branch, as cases a tester can run against a **live warehouse,
a live model and a real browser**. It is deliberately not a restatement of the
unit suite: 7,500 automated tests already run on every commit, and what they
cannot reach is exactly what this plan covers — a real Snowflake/Oracle/Azure
SQL connection, a real LLM with its own latency and refusals, real Qdrant
retrieval, a real browser rendering real fonts, and real multi-tenant data.

## How to use it

Each case is `ID · what it proves · setup · steps · pass · **false pass**`.

The **false pass** line is the point. Most of the defects fixed on this branch
were things that *looked* fine: a chat page whose script never parsed still
served its shell, a corroboration that never ran still showed a confident
answer, an analysis card in French still had an English body. A tester who only
checks "did something appear" will sign off on all of them again. Read that line
before recording a pass.

Where a case says **regression**, it reproduces a defect that shipped. Those are
the highest-value cases in the document: run them first, and run them again
after any change to the area.

### Environment

| | |
|---|---|
| Workspace | A tenant with a real warehouse connection, KB built, state `READY` |
| Data | At least two fact tables, one shared dimension, ≥ 3 months of dates |
| Users | One admin; one portal user with a **restricted** table ACL; one unrestricted |
| Languages | Both users exercised in English and French |
| Model | The tenant's configured provider, plus one run against a local provider |

Record the workspace's **model version** (Data & Model → What To Model Next) at
the start. Several cases compare answers across it.

---

## 0 · Regressions — run these first

These four shipped broken. Each takes under two minutes.

**L0-1 · The chat page's JavaScript parses** — *regression*
Setup: any portal user.
Steps: open `/portal/chat`. Open the browser console before the page loads.
Pass: the composer accepts a question, a live stage appears, an answer renders,
and the console shows **no** `SyntaxError`.
**False pass:** the page renders its header, sidebar and empty thread perfectly
with a dead script — that is exactly what the defect looked like. You must send
a question and see an answer come back over the websocket.

**L0-2 · A governed Python analysis with a comprehension runs** — *regression*
Setup: a tenant with `enable_python_analysis` on.
Steps: ask a question that returns ≥ 3 rows; run an analysis whose code ends in
a list comprehension (e.g. "round each region's revenue to 2 decimals").
Pass: derived rows come back.
**False pass:** a plain non-comprehension analysis succeeding. The defect only
fired inside a comprehension, generator or lambda. Insist on the comprehension.

**L0-3 · A second opinion actually returns a comparison** — *regression*
Setup: two subject areas that can both answer one question (see L1-4).
Steps: ask that question; open the answer trace.
Pass: the `corroboration` step reports `checked: true` with two figures.
**False pass:** the step exists and reports `not_checked` / `execution_failed`.
Before the fix it reported exactly that, every time, because the corroborating
query executed under the *primary* area's tables and was refused `access_denied`.

**L0-4 · A CSV export of a question naming non-ASCII data** — *regression*
Steps: ask a question whose text contains `Škoda` or `Łukasz` (or any customer
name with a letter outside Latin-1); download the CSV.
Pass: a file downloads, named readably (accents folded in the fallback name,
preserved in the browser's chosen name).
**False pass:** testing with `région` only. Accented French is Latin-1 and never
crashed; the 500 needed a letter *outside* it.

---

## 1 · Subject areas and corroboration (Phase A)

**L1-1 · An area can be created at all** — *regression*
Steps: Data & Model → **Subject Areas** → create "Sales" with 2–3 tables and
synonyms `bookings, orders`.
Pass: it saves and is listed.
**False pass:** none — before this branch there was no writer at all, so any
success here is real. Confirm the row survives a page reload.

**L1-2 · Routing narrows what the answer can see**
Steps: ask a question using a declared synonym. Open the trace.
Pass: `domain_routing` shows `applied: true`, the area's name, and a reduced
table count; `allowed_tables_snapshot` contains only that area's tables.
**False pass:** the step present with `applied: false`. That is the un-routed
path and proves nothing.

**L1-3 · An area cannot widen a user's access**
Setup: the restricted portal user, whose ACL excludes one of the area's tables.
Steps: same question.
Pass: the trace's table list is the **intersection**, never the area's full list.
**False pass:** testing as the unrestricted admin.

**L1-4 · Two plausible areas produce a compared answer**
Setup: two areas whose synonyms both match one word (e.g. both declare
`revenue`), over **different** facts.
Steps: ask that one word.
Pass: the answer carries a confidence note naming the second area; the trace's
`corroboration` step shows both figures.
**False pass:** two areas that share their fact tables. They will always agree,
so agreement proves nothing. Use disjoint facts.

**L1-5 · A disagreement is visible without clicking** — *regression*
Setup: as L1-4, with facts that genuinely differ (e.g. gross vs net).
Pass: the confidence verdict is **Low**, and the sentence naming both figures is
visible beside the pill, not only inside the "How this answer was produced"
disclosure.
**False pass:** finding the sentence after opening the disclosure. The cap was
one point too high and hid it there.

**L1-6 · A failed second opinion degrades nothing**
Steps: break the second area (remove its tables from the user's ACL), re-ask.
Pass: the primary answer is unchanged and the trace says `not_checked`.
**False pass:** the answer showing a disagreement. "Not checked" must never be
reported as a conflict.

**L1-7 · The route preview matches reality**
Steps: Subject Areas → "Try a question" with the question from L1-2.
Pass: the preview names the same area the trace later shows.

**L1-8 · Governance is not bypassed by the second query**
Steps: with two areas over two facts, ask something that could join both.
Pass: the trace shows the corroborating attempt refused with
`raw_fact_to_fact_join`, not executed.
**False pass:** no corroborating attempt at all — check it was generated first.

---

## 2 · Candidate selection (Phase B)

**L2-1 · An ambiguous question is asked twice**
Setup: a question whose plan reaches two date roles or two facts.
Steps: ask it; open the trace.
Pass: two `generate_candidate` entries and one `candidate_selection` step.
**False pass:** a question the plan fully determines — `k` is 1 by design and
nothing is spent. Confirm the trace records `ambiguity`.

**L2-2 · Two verified candidates that disagree refuse to pick**
Pass: confidence **Low**, with the "asking a second way produced a different
figure" warning.
**False pass:** one candidate failing validation. That is `no_candidate_verified`
and a different branch.

**L2-3 · A disagreement survives a verifier crash** — *regression*
Steps: hard to force in production; verify instead in the trace that
`candidate_selection` is present on a run where `result_shape_verification`
reports `unavailable`.
Pass: both present.
**False pass:** never seeing an `unavailable` verification. Ask an operator to
confirm from logs rather than recording a pass you did not observe.

---

## 3 · The computed analysis (Phase C)

**L3-1 · A regulated tenant gets a real analysis**
Setup: a workspace in `regulated` mode.
Steps: ask a question with a clear leader; press **Why**.
Pass: a card with computed sentences — a share, a spread, a trend — and a note
that no values were sent to the model.
**False pass:** the static apology. That is the old behaviour.

**L3-2 · Every figure in the narrative was computed** — *governance*
Steps: cross-check each number in the card against the result table.
Pass: every figure is in the table or derivable from it.
**False pass:** a plausible number you did not check. This is the whole point of
the checked-numbers rule.

**L3-3 · No data value leaks for a regulated tenant** — *governance*
Steps: Compliance → **AI Egress Log** for that question.
Pass: `values_sent` is 0 and the manifest lists no cell values.

---

## 4 · Metrics (Phase D)

**L4-1 · Coverage is reachable and honest** — *new surface*
Steps: Data & Model → Metrics → **Coverage** on a metric with no dimensions.
Pass: a score, and gaps grouped by the asset that would close them.
**False pass:** an empty panel. Empty reads as "no gaps"; a failed lookup says
"could not be checked".

**L4-2 · Closing a gap moves the number**
Steps: add the dimension the panel names; re-open Coverage.
Pass: the score rises and that group disappears.

**L4-3 · Mined synonyms appear after real failures** — *new*
Setup: as a portal user, ask for a measure using a word the metric does **not**
carry (e.g. "turnover"), let it fail, then ask again using a word it does
("revenue"). Repeat with a different question, same pair of words.
Steps: Metrics page → the "Words readers used" card.
Pass: `turnover → <metric>` listed with `occurrences: 2` and both questions as
evidence.
**False pass:** doing it once. One occurrence is deliberately below the bar.
Also confirm the two attempts were **within ten minutes** and by the same user.

**L4-4 · Accepting a synonym changes what the product understands**
Steps: click **Add synonym**; then ask the failing question again.
Pass: it now binds to the metric and returns the number.
**False pass:** checking only that the synonyms field changed. The matcher is
what matters.

---

## 5 · Modelling debt (Phase E)

**L5-1 · The backlog is reachable and actionable** — *new surface*
Steps: Quality → **What To Model Next**.
Pass: rows ordered by questions unblocked, each with a working **Fix** link that
lands on the right editor.
**False pass:** an empty list on a workspace that is genuinely incomplete —
check the counts in the hero against reality.

**L5-2 · Fixing the top item shortens the list**
Steps: apply the top remedy; reload.
Pass: that row is gone and the totals move.

**L5-3 · Curation weighting changes which table is chosen**
Setup: two plausible tables for one question; describe one richly, leave the
other bare.
Pass: the trace's retrieval telemetry ranks the described table above the bare
one.
**False pass:** a workspace whose KB header is bracketed `[SCHEMA].[TABLE]` —
that spelling used to make the boost a silent no-op, so this case is also a
regression check. Confirm the header form with an operator.

---

## 6 · Recovery (Phase F)

**L6-1 · A corrected run reads as one correction**
Setup: a question that fails validation once and succeeds on repair.
Pass: the trace labels the first failure `superseded` and the run reads as
recovered.

**L6-2 · A run that never recovered keeps both failures** — *regression*
Setup: a question that fails, is repaired, and fails again.
Pass: **both** failures remain `error`; neither is relabelled.
**False pass:** only checking L6-1. The defect was in this direction — a failure
was quietly hidden on a run that answered nothing.

---

## 7 · Deployment posture and the proof pack (Phase G)

**L7-1 · A local model answers a question end to end**
Setup: point the local provider at a running Ollama/vLLM; set the workspace to
that provider.
Pass: a question is answered, and the egress log shows the local endpoint.

**L7-2 · An air-gapped posture refuses a hosted provider**
Steps: set posture **Air-gapped**, then set the provider to a hosted one.
Pass: the question is refused with a posture message; no outbound call.
**False pass:** the refusal appearing only in the UI. Check the network.

**L7-3 · The refusal is recorded with audit logging OFF** — *regression*
Setup: `enable_llm_audit` **disabled** (the default).
Steps: cause a refusal as in L7-2; download the proof pack.
Pass: the refusals section counts it.
**False pass:** testing with audit logging on. That was the whole defect —
the default workspace recorded nothing and the pack said "0 refused".

**L7-4 · The proof pack's hash chain verifies**
Pass: the integrity section reports an unbroken chain.
**False pass:** a `chain_forked` result read as a pass — forked is distinct from
broken and is still a finding.

---

## 8 · French — the reader's whole journey

Run **every** case in this section as a French user, comparing against the same
case in English. Where a case says *side by side*, take a screenshot of both.

**L8-1 · A visitor can choose French before signing in** — *new*
Steps: clear cookies; open `/portal/login` on an English-first browser.
Pass: a language control is present; clicking it renders the login page in
French.
**False pass:** a browser whose `Accept-Language` already prefers French. Use an
English-first browser or the page will be French without the control working.

**L8-2 · Signing in keeps that choice** — *regression*
Steps: from L8-1, sign in with an account that has never set a language.
Pass: the dashboard is French, and the preference persists after logout/login
on a **different** browser.
**False pass:** an account whose row already says `fr`.

**L8-3 · A stored preference still wins**
Setup: an account whose language is French; a browser whose cookie says English.
Pass: signing in gives French.

**L8-4 · The chat toggle is in the corner and keeps the thread** — *new*
Steps: open a thread with several turns; click **FR** top right.
Pass: the whole page — chrome, answers, caveats — comes back French, **in the
same thread**.
**False pass:** landing on a new empty thread. The return path is the case.

**L8-5 · The answer card is French to the bottom** — *side by side*
Steps: ask a question that returns a trend with a clear leader; open "How this
answer was produced".
Pass: headline, insight, caveats, chips, **and the confidence verdict with all
its reasons and watch-outs**, are French.
**False pass:** the top half only. The confidence panel was the largest English
hole and it is behind a click.

**L8-6 · Numbers are French inside French sentences** — *side by side*
Pass: `12,3 %` with a no-break space, `1 234` grouped with a narrow space, in
the narrative, the table cells, the KPI and the chart tooltip alike.
**False pass:** checking the table only. The narrative built its own numbers.

**L8-7 · A failure explains itself in French** — *side by side*
Steps: force three failures — a bad column, a database error, an empty result.
Pass: headline, reason and next step are French, and the card still shows its
correct kicker (Query failed / Validation problem / No rows).
**False pass:** the kicker reading "No rows returned" on a database error. That
is the symptom of the headline being used as a machine key.

**L8-8 · The database's own error text is NOT translated** — *governance*
Pass: under "Technical details", the driver's message is verbatim English.
**False pass:** reporting this as a bug. Support searches on that string.

**L8-9 · The front door speaks French**
Steps: as a French user type `status`, then `whoami`, then a question in a
not-yet-ready workspace.
Pass: all three replies are French, with the data (state, table list, counts)
intact.

**L8-10 · The regulated analysis card is French** — *regression*
Setup: a regulated workspace, French user.
Pass: title **and body and bullets** are French.
**False pass:** a French title over an English body — the exact defect.

**L8-11 · The dashboard formats for the reader**
Pass: KPI tiles read `1 000k` / `2,5Md`, not `1,000K` / `2.5B`.

**L8-12 · Search finds things without accents** — *new*
Steps: in the thread history, the result-table filter, and the Knowledge Base
search, type the unaccented form of an accented word (`region`, `cout`,
`soeur`).
Pass: each finds the accented entry.

**L8-13 · Nothing renders a raw message id**
Steps: walk every French screen.
Pass: no text of the form `ui.something.something` or `fail.v.something`.

**L8-14 · A digest arrives in the recipient's language** — *new*
Setup: two users on the same account subscribed to the same scheduled report,
one with `lang='fr'` and one with `lang='en'`. Let the scheduler deliver it —
do **not** trigger it from a signed-in browser session.
Pass: each recipient's digest is in their own language, and the French one
writes figures as `1 234,50`, not `1,234.50`.
**False pass:** triggering the digest from your own logged-in session. The
scheduler runs on its own thread with no request behind it, so a browser-
triggered send can pick up a language that the real 8am delivery never sees —
which is the exact bug this case exists to catch. Check the delivery timestamp
matches the schedule, not your click.

**L8-15 · An alert arrives in the recipient's language** — *new*
Setup: an alert owned by a French user on a metric you can move; trip it.
Pass: the notification reads `⚠️ ALERTE : … est maintenant à 1 234,50 (en
hausse de 23,4 %…)`. The direction word is French too, not "increased" inside a
French sentence.
**False pass:** reading only the figures. The sentence template, the direction
word and the numbers are three separate translations and the first version of
this shipped with only the last two done — a French-looking number inside an
English sentence passes a careless look.

**L8-16 · An alert for a deleted user still fires** — *new*
Setup: an alert whose `user_id` no longer resolves to a row.
Pass: the alert still delivers, in English, and the log carries a warning
naming the alert. It must not be dropped.

**L8-17 · Chart labels are drawn in French** — *new*
Steps: as a French user, produce one of each: a time series with a biggest
drop/gain annotation, a pie, a funnel, a cohort heatmap, a treemap, a
histogram. Repeat on a **pinned dashboard tile**, not only in chat.
Pass: every word drawn on the canvas is French — `↓ -23,4 % Baisse`, `Part du
total`, `Abandon`, `Rétention`, `Nombre`, `Non précisé` — and every percentage
uses a comma and a no-break space (`23,4 %`).
**False pass:** checking the tooltip only. The tooltip name and the label drawn
on the canvas are two different strings and shipped in two different languages
once already. Hover *and* look at the chart itself. Equally, checking chat only:
the dashboard builds its charts from its own copy of the code.

**L8-18 · A slice with no category says so in French** — *new*
Steps: a pie over a dimension with NULLs.
Pass: the slice reads `Non précisé`, in the legend, the label and the tooltip.

**L8-19 · The chart and the sentence round the same way** — *new*
Steps: find a figure that lands exactly on a half at the displayed precision
(a 2.5% share, an 80.5% retention).
Pass: the number on the chart and the same number in the narrative, the table
and any digest agree exactly. `2.5%` must not read `3%` on the chart and `2%`
in the sentence under it.
**False pass:** a value that is not an exact tie. `23.45` is stored just below
its own decimal literal, so it rounds down everywhere regardless and proves
nothing. Use a value that is exactly representable: 0.5, 2.5, 80.5.

---

## 9 · Cross-cutting

**L9-1 · Tenant isolation under every new surface**
Steps: as an admin for tenant A, attempt to open tenant B's Subject Areas,
What To Model Next, coverage API and synonym-proposal API by editing the URL.
Pass: refused or empty; never tenant B's data.

**L9-2 · Every new admin surface refuses an unauthenticated request**
Steps: log out; request each new URL directly.
Pass: redirect to login or 401. No JSON body containing tenant data.

**L9-3 · A restricted user's ACL holds through every new path**
Steps: as the restricted user, exercise domains, corroboration, exports.
Pass: no table outside their ACL appears in any answer, trace or export.

**L9-4 · Answers are stamped with the model version**
Steps: note the version; change a metric; ask again.
Pass: the two answers carry different versions.

**L9-5 · Load and cost sanity**
Steps: ask 20 mixed questions.
Pass: candidate generation and corroboration fire only on the questions that
warrant them (check the trace `ambiguity` and `domain_routing` fields), not on
every question. Compare token spend against a pre-branch baseline if available.

---

## Sign-off

A case is **not** passed until the false-pass line has been considered. Record
for each: tester, workspace, date, language, and the trace id of the run that
demonstrated it — the trace is the evidence, the screenshot is the illustration.

Anything that fails should be reported with its **trace id** and the workspace's
**model version**, because both determine whether the same run is reproducible.
