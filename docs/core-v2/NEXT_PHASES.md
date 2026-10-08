# QueryBot: the next phases (agreed 8 October 2026)

This is the plan the product owner approved for after the new core's current
work. The full version, with mock-ups, was shared as a PDF ("QueryBot plan:
projects, models, access, scale, dates, dashboards and UI"). This file keeps it
in the repository so the next session starts from it.

The principle throughout: **readers just ask, Learn proposes everything, and the
admin only corrects what is wrong** (a wrong date, a wrong reading of a question,
or an edit to the semantic layer).

Sizes: S small, M medium, L large (two or three commits). Each step is its own
commit with a test that fails on the code before it, a run on the private copies
of the data, a real-browser check for anything visual, and the full suite against
its baseline before pushing.

## Phase 0 (first): UI/UX clean-up and a smaller admin panel

Requested 8 October 2026, before everything below: take out what the admin panel
does not need, fix the product's UI and UX problems, chart rendering included.
The sweep's findings and the removal list are agreed with the product owner
before anything is removed or changed.

The sweep (8 October 2026, shared as "QueryBot UI/UX sweep and admin clean-up
plan") ran the real app locally on invented data: 160 admin pages, 15 chat
answers, every chart kind, a dashboard, desktop and phone. Main findings:

- Charts never appear in the answer: a side panel squeezes the chat to about
  450 px, keeps an older answer's chart, and covers the answer on a phone.
- "Add to dashboard" on a new-core answer fails ("can no longer be pinned").
- Answer cards: the number repeated in the sentence; three large follow-up
  cards; a question back styled as an answer; technical stats for readers;
  a trend by region answered "30 rows".
- Charts: line/area offered for categories; a palette picker everywhere; only
  the first of several measures drawn; prior period as strong as current;
  dashboard notes colliding with the axis.
- Dashboard page: uneven, touching tiles; "DASHBOARD ARTIFACT" wording; edit
  controls always on; readers land on a page of usage counters.
- Admin: 7 tabs and about 30 pages per workspace (11 under Data & Model);
  counters repeated on 5 pages; pages titled "Overview" that are not; developer
  internals shown; today's semantic-layer pages beside the new core's learned
  page, contradicting it.

| Step | What | Size |
|---|---|---|
| U0 | Keep the sweep as a repeatable screenshot check (desktop and phone; fails on console errors, sideways overflow, overlapping tiles) | M |
| U1 | Answer card: chart inside the answer with Expand; panel follows its answer; number once; compact chips; real question card; no technical stats; fix the "N rows" sentence | L |
| U2 | Pin new-core answers (A1 pulled forward) | M |
| U3 | Chart rules (C1, C3 pulled forward): fitting shapes only, every measure drawn, prior muted, notes clear of axes, no palette picker for readers | M |
| U4 | Dashboard page: grid, view/edit mode, plain header and captions | M |
| U5 | Portal pages: land in chat; dashboards list; reader's semantic layer from the new core; standard forms; one start screen | M |
| U6 | Admin stage 1 after approval: Overview, Data, Questions, People, Compliance, Settings (about 10 pages); merges, moves, Diagnostics for internals; pages hidden by engine | L |
| U7 | Phone pass: no sideways scrolling anywhere | S |
| U8 | Sign-in link per workspace | S |
| Later | Admin stage 2: delete pages hidden for the new core once every workspace has switched | M |

Admin proposal (each row needs the product owner's approval): keep Dashboard,
Platforms, Databases, Clients, System (minus the database-backend switch; model
prices move here), Overview, Setup, What QueryBot learned (the home of the
semantic layer), Relationships (until merged), Metrics, Flagged Answers (the
review list), Users, Groups, Compliance (scoped by profile), Usage; merge Queries
with Timing, Configuration with Advanced, Access Requests into Users; move Source
Mapping, AI Egress Log, Evaluations; hide then remove Knowledge Base, Dates,
Glossary, Business Meanings, What To Model Next, Drafted For Review, Subject Areas
(returns as Models); remove Model Health, Conflict Inbox and Versions from the
admin view (Diagnostics only); Reports is the product owner's call.

Open decisions for this phase: the admin list; hide-then-delete versus delete
now; charts in the answer versus panel only; the palette picker; Reports; where
readers land; whether these shared templates also go to main.

## What already exists (do not build again)

- Dates declared once per table (kind, calendar, grain, span), a default date per
  measure, every date condition compiled (windows, comparisons, fiscal years,
  last snapshot per period, partly covered periods, days between two dates).
- The learned page: decide and undo per object; today's setup decisions flow in.
- Answers across two fact tables: each totalled on its own, lined up on the shared
  groupings (compiler `_line_up`).
- Today's pipeline: "domains" (tables + synonyms, phrase routing with a
  second-opinion check), several "sources" per workspace, schema chips in chat.
- Access: roles admin and analyst, groups with table access, extra tables per
  user, compliance rules (deny, mask, totals only).
- Dashboards: governed saved queries, scheduled refresh as the viewer with an
  encrypted cache, headline-first layout, publish, versions, rollback.
- Look: navy palette, Inter, Lucide icons, colour tokens, one ECharts renderer
  that puts measures with different units in separate panels.

## 1. Projects, models and user access

Organization (tenant: sign-in, Teams/Slack/Zoom, org admins, limits) holds
projects (today's clients: connection, learned model, use-case models, members,
dashboards, settings). A model is a use case (facts, the shared dimensions they
reach, measures, dates, description, example questions, who may use it): a view
on the one learned project model, never a copy.

Connecting every table: Learn takes it all, but one flat model answers worse as
it grows (same words in different areas, a longer catalog for the AI, more join
paths, different teams). So: learn once, and Learn proposes the use-case models
by clustering facts on shared dimensions; the admin only renames, merges, splits.
A small one-subject database gets one model.

| Step | What | Size |
|---|---|---|
| P1 | Projects instead of clients; organization layer; existing clients become projects with the same id | M |
| P2 | Models as views on the learned model; Learn proposes them; page to rename, merge, split | L |
| P3 | Model picker in chat; router for "All models" (clear winner answers and says so; two close: answer and offer the other's figure) | M |
| P4 | Cross-model questions on shared dimensions (same mechanism as two facts today) | M |
| P5 | Project roles (Admin, Author, Viewer) and access per model | M |
| P6 | Row-level access rules on every query, refresh, export and cached answer | L |
| P7 | Entra ID sign-in and group sync | M |
| P8 | Two databases in one project (later) | L |

## 2. Production scale and many users

Large data already scales (every number computed in the warehouse, date
conditions on the stored column, Learn samples about 1M rows per table in
parallel, one AI call per question with a cached prefix). Limits found in the code:
the AI reads the whole catalog (about 250 tokens per table: 3,200 for 14 tables);
one process (`--workers 1`) with 4 new-core threads; conversations, member names
and discovery kept in process memory; the model reloaded per question; a new
warehouse connection per query; no result cache; SQLite by default.

| Step | What | Size |
|---|---|---|
| S1 | The AI sees only the relevant slice (model + matched measures, groupings, dates), under about 10k tokens | M |
| S2 | 16-32 new-core threads per process; fair queue per workspace with a "busy" message | S |
| S3 | Conversations and member names in shared storage; several workers/servers | M |
| S4 | Model cache per learned version and decisions stamp | S |
| S5 | Warehouse connection pool per workspace with a concurrency cap | M |
| S6 | Short-lived result cache keyed by SQL, access scope and data freshness | M |
| S7 | Postgres for production app data (already supported) | S |
| S8 | Incremental Learn (changed tables only), scheduled | M |
| S9 | Timings, queue, cache and error metrics; load test with 50 and 100 users before go-live | M |

## 3. Dates: a correction loop

| Step | What | Size |
|---|---|---|
| B1 | Two plausible dates: answer with the default and show the other's figure with a switch button; with no default, ask with the figure on each option | M |
| B2 | The pick is remembered for the thread, and the answer says so | S |
| B3 | Readers' picks become proposals on the learned page ("7 of 9 chose Order date") | M |
| B4 | "Fix how this was counted" on the answer for admins; thumbs-down "wrong date/number" goes to a review list | M |
| B5 | Verified questions: fixes and confirmed answers become question-and-plan pairs, used as examples and replayed after each Learn | M |
| B6 | Time-phrase check against the AI's period (safety net) | S |

## 4. Self-built dashboards

"Build a dashboard for gross profit", a "Make a dashboard from this" button, or a
model's ready overview. Default: up to 4 headline numbers with their change; a
12-month trend; 2-3 breakdowns (top 10 + "All others"); a donut only for 6 parts
or fewer; a top-10 table. Every tile is compiled and run once; empty or failing
tiles are left out and named. Changed by chat ("add margin %", "only Region A").

| Step | What | Size |
|---|---|---|
| A1 | Pin new-core answers: the bridge issues the pin token (none today); tiles drawn from the answer's own chart spec | M |
| A2 | The composer: "dashboard" intent, outline rules, run and check every tile, layout, chat card | L |
| A3 | Changes by chat; dashboard filters re-run each tile's plan (today they cut returned rows: wrong top 10) | M |
| A4 | "Add to dashboard" sheet: preview, title, tab, headline-row toggle, link | S |

## 5. Charts in answers

| Step | What | Size |
|---|---|---|
| C1 | A card per number for one-row answers with 2-4 measures (today a one-row table) | S |
| C2 | "Total and average": two cards and monthly bars with an average line | M |
| C3 | Rules: top 10 + "All others" past about 12 groups; horizontal bars for long names; donut only for shares of 6 or fewer; bars for 3 periods or fewer; prior period muted | S |

## 6. The product's UI

| Step | What | Size |
|---|---|---|
| D1 | Answer card: model and date badge, numbers first, Key insights, "How it was counted", the admin's fix button, SQL and plan behind "Details" | L |
| D2 | Dashboard page: headline row on top, Auto-arrange, save only moved tiles, refresh time, friendly errors | M |
| D3 | App shell: one sidebar; top bar with organization, project, model picker, search | M |
| D4 | Admin pages: the learned page as the one place for corrections, plus models and access | L |

## Order

0. UI/UX clean-up and a smaller admin panel (first, as requested).
1. A1, B1, B2, B4, P1, S2, S4.
2. P2, P3, S1, P5.
3. A2, A3, C1, C3, D1.
4. S3, S5, S6, S7, P6, P7, S9 (before many users).
5. P4, B3, B5, S8, C2, A4, D2-D4, B6.
Later: P8.

## Decisions still open

1. Two plausible dates or models: answer with the default and offer the switch (recommended), or always ask with numbers?
2. Names: Organization, Project, Model?
3. Roles: Admin, Author, Viewer per project, plus organization admins?
4. Who can fix from the answer: project admins only (recommended), or authors too?
5. Row-level rules: from per-user attributes, Entra ID groups, or both?
6. Self-built dashboard default contents.
7. Production target: readers and questions per hour, for the load test.
8. main: do the shared-code pieces (projects, access, scale, dashboards, UI) also go to main, or stay on core-v2 until the switch?

## Housekeeping found while planning

Three committed files contain bare totals from a private copy of a customer's
data (no names): `core2/service.py` (a docstring), `core2/answer/builder.py` (a
docstring) and `tests/test_core2_names_every_measure_asked_for.py` (its
docstring). Replace them with invented figures; rewriting the branch history is
the product owner's call.
