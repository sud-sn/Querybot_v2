# QueryBot Analytical Agent — Remaining Production Plan

## Objective

Make QueryBot compile each natural-language question into an authoritative,
tenant-specific analytical request before SQL generation. The resulting SQL
must follow the selected measure, fact, dimensions, date semantics, approved
relationships, and output shape. Ambiguity must result in a focused
clarification rather than a guessed query.

This plan is dataset-neutral. All table names, field meanings, terminology,
metrics, date roles, calendar settings, and joins must come from the active
client's governed semantic assets.

## Current foundation

The application already has:

- semantic field and metric resolution;
- approved Date Roles and runtime calendar-column discovery;
- observed business-date anchoring;
- a preliminary analytical request plan;
- confirmed Entity Graph relationship enforcement;
- raw fact-to-fact join rejection;
- persisted result handles and date preferences; and
- local governed result formatting.

The remaining problem is that these components can make independent decisions.
The request plan must become the single contract used by generation,
validation, execution, verification, and response rendering.

## Phase 1 — Authoritative analytical request compiler

Extend the analytical request plan to represent:

- requested business measures and their ownership;
- dimensions and grouping grain;
- filters and comparison periods;
- explicit or inferred source facts;
- date role and calendar basis;
- independent subrequests for compound questions;
- required result shape;
- required joins and prohibited joins; and
- unresolved decisions that require clarification.

Required compilation order:

1. Resolve business concept or approved metric.
2. Select the fact that owns the measure.
3. Resolve dimensions reachable from that fact.
4. Resolve date role and calendar basis.
5. Compile approved join paths.
6. Generate SQL from the compiled plan.
7. Validate and verify SQL against the same plan.

## Phase 2 — Measure-first source and fact selection

- An approved metric's source fact has highest priority.
- A semantic measure field has the next-highest priority.
- An explicitly named business source is authoritative.
- Output grain such as “by month” must not select a monthly source by itself.
- A dimension such as warehouse must not introduce an unrelated inventory fact
  into a revenue request.
- When the same concept exists at several grains, ask which business source or
  grain the user intends.
- Decompose genuine multi-source requests into separately governed subplans.

Raw facts must never be joined. For valid multi-source comparisons, aggregate
each fact independently to a governed shared grain and join only those results.

## Phase 3 — Calendar and fiscal-quarter semantics

Add a governed workspace Calendar Profile with:

- calendar mode: calendar, fiscal, or both;
- fiscal-year start month;
- fiscal-year label convention: starting year or ending year;
- approved calendar/date dimension;
- calendar and fiscal year, quarter, month, and date columns when available;
- business timezone and week-start day; and
- approval status.

### Q1/Q2 clarification contract

For an ambiguous request such as “Show revenue for Q1 2026,” QueryBot must ask:

> Should I interpret Q1 as calendar Q1 or your fiscal Q1?

Choices must be business-facing and explicit, for example:

- Calendar Q1 — January through March
- Fiscal Q1 — April through June; FY named by ending year
- Specify another reporting calendar

Rules:

- Explicit “calendar Q1” does not require clarification.
- Explicit “fiscal Q1” uses the approved Fiscal Calendar Profile.
- Bare Q1/Q2 requires clarification unless one approved workspace default exists.
- If fiscal start is unknown, ask when the fiscal year starts.
- If FY naming is unknown, ask whether FY2026 starts or ends in 2026.
- Prefer approved fiscal columns on the date dimension.
- Otherwise calculate fiscal boundaries from the native date and approved start
  month.
- Never silently default fiscal-year start to January.
- Persist the answer in thread metadata for the relevant metric/fact and resume
  the original question without requiring it to be repeated.
- State the chosen interpretation in the final answer.
- Distinguish reporting quarters from statistical quartiles using the full
  question context.

## Phase 4 — Production join planning

For every requested dimension:

1. Start from the measure-owning fact.
2. Find an approved path to the dimension.
3. Rank paths by confirmation, validated cardinality, fanout risk, bridge hops,
   and business-role match.
4. Compile exact join columns and join type.
5. Require generated SQL to prove every required join.
6. Clarify or refuse when no safe approved path exists.

Suggested or unreviewed relationships remain onboarding evidence and must not
become executable joins. When no safe shared grain exists between facts,
QueryBot must explain why the comparison cannot be produced.

## Phase 5 — Conversational result lineage

Route short follow-ups to the active governed result when unambiguous, including:

- Which one is highest or lowest?
- What about the second one?
- Remove that row.
- Keep only these categories.
- Sort by revenue.
- Change the date to MMM-YY.
- Display values as currency with two decimals.
- Make this a bar chart.
- Compare this with last month.
- Why did this one fall?

If multiple result artifacts could be referenced, ask which result the user
means. Local formatting, filtering, sorting, and chart changes must retain
result lineage and avoid a new database query.

## Phase 6 — Portal run-state repair

Replace independent UI processing flags with one run-state reducer:

- `running`: lock the composer and show progress;
- `waiting_for_user`: enable the clarification controls and stop progress;
- `completed`: enable the composer and clear progress;
- `blocked`, `failed`, `cancelled`: enable the composer and show the terminal
  state; and
- ignore late events belonging to an older run.

Clarification submission must never fail silently. This phase covers the
currency-option no-op, dropped messages, and stale “Working” or “Repairing”
states.

## Phase 7 — Result narration and rendering

- Response helpers must always return the declared type; never send an object
  where the portal expects summary text.
- Sort a separate analytical copy of temporal results chronologically while
  preserving the displayed table order.
- Describe two temporal observations as a comparison, not a sustained trend.
- Select the headline measure from the request plan and metric metadata.
- Exclude diagnostic columns such as matched/non-null row counts from KPI
  selection.
- Do not headline the final period when the user requested a total.
- Require a governed year context for month-only requests when multiple years
  are possible.

## Phase 8 — Observability

Record a structured trace containing:

- parsed measures, dimensions, filters, dates, and operations;
- candidate facts with scores and rejection reasons;
- selected source and measure ownership;
- calendar basis and Date Role provenance;
- join path and cardinality;
- generated SQL and validation contract;
- repair changes; and
- post-execution semantic verification.

## Regression acceptance suite

1. Revenue by invoice month cannot select monthly inventory.
2. Explicit ERP/M3 source wording selects that source and only relevant dates.
3. Latest two observed data days returns exactly two fact-scoped periods.
4. Monthly-versus-daily inventory uses two isolated aggregations.
5. A direct fact-to-fact join is rejected with a specific safety error.
6. Bridge allocation follows fact → bridge → dimension.
7. “Which one is highest?” uses the active result without a new query.
8. Date and currency formatting continues after clarification selection.
9. Two temporal points are narrated as a comparison.
10. Diagnostic columns never become KPIs.
11. No object can enter a text-only response property.
12. Bare Q1 asks calendar versus fiscal.
13. Explicit calendar Q1 does not ask.
14. Explicit fiscal Q1 uses configured fiscal boundaries.
15. Missing fiscal configuration never assumes January.
16. Calendar clarification resumes the pending request and remains thread-scoped.
17. A multi-year “January” request asks for or resolves a governed year.
18. Unsafe or unapproved relationship paths are refused before execution.

Every phase requires unit tests plus live Azure SQL and browser/WebSocket
verification. A green unit suite is necessary but not sufficient for release.
