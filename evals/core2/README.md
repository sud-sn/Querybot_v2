# core2 evaluation

Synthetic warehouses with known answers. Design: `docs/core-v2/DESIGN.md` §11.

```
evals/core2/
  naming.py          the four naming styles
  framework.py       Domain, ground truth, materialize() into DuckDB
  domains/<name>.py  one business each: build() -> Domain
  golden/<name>.yaml golden questions for that domain
  learn_eval.py      level 1: the learned model against the ground truth
  compile_eval.py    level 2: golden plans -> SQL -> rows, against the reference SQL
  plan_eval.py       level 3: golden questions -> a real AI -> rows (recorded for replay)
  benchmark.py       accuracy of joins, metrics and dates: right with no input, sent to review,
                     right among what is not sent to review (target 85% / 15% / 95%)
  public.py          public sample warehouses (Chinook, Northwind, Sakila): fetched, hash-checked,
                     cached outside the repository, with hand-written truth
  conversations/<name>.yaml   hand-labelled conversations: each turn a follow-up, a new question
                     or honestly unclear; conversations/heldout/ is never used to tune the rules
  followup_eval.py   follow-up or new: the conversations through a real AI (recorded for replay)
```

```
python -m evals.core2.learn_eval inventory            # every naming style
python -m evals.core2.compile_eval retail warehouse
QUERYBOT_EVAL_API_KEY=... python -m evals.core2.plan_eval retail --provider azure_openai \
    --model <deployment> --endpoint https://<resource>.openai.azure.com
python -m evals.core2.plan_eval retail --replay evals/core2/recorded/retail.<deployment>.json
```

Levels 1 and 2 run in CI (`tests/test_core2_*`); level 3 needs a provider and is
run on demand. The synthetic warehouses hold no customer data.

## The accuracy benchmark

```
python -m evals.core2.public fetch                      # once: the public warehouses, into ~/.cache/querybot/benchmark
python -m evals.core2.benchmark                         # every domain and style, ~30 seconds
python -m evals.core2.benchmark --domains networking --styles generic --misses --json out.json
QUERYBOT_EVAL_API_KEY=... python -m evals.core2.followup_eval --provider azure_openai --model <deployment> \
    --endpoint https://<resource>.openai.azure.com
python -m evals.core2.followup_eval --replay evals/core2/recorded/followups.<deployment>.json
python -m evals.core2.followup_eval --signals              # only what the words decide, no AI
python -m evals.core2.followup_eval --signals --set heldout
```

Follow-up or new is read from the words first (`core2/plan/followup.py`): a turn that
plainly changes the answer on screen is planned with it, a question of its own is
planned without it, and a short complete question after a narrowed answer is asked
with two buttons. Only what the words leave open is the AI's to read. `--signals`
scores those readings with no AI: on the labelled set 99% of turns are decided and
99% of those are right (4% asked); on the held-out set 95% and 97% (5% asked).

The domains in `domains.DOMAINS` are gated by `learn_eval` in CI. The ones in
`domains.BENCHMARK` (compounding pharmacy, networking, home services) and the public warehouses
plant what the learner is known to miss today, so they are measured, never gated;
a domain moves into `DOMAINS` once the learner passes it. `baselines/phase0.json`
holds the numbers before any of the accuracy work (Phase 0), every miss listed;
`baselines/metrics_and_dates.json` the numbers after the first join fix and the
metric and date work. `baselines/people_and_statuses.json` the numbers after Learn read statuses kept
in lookups, people's data, running numbers and documents' figures (home services added).
`baselines/links_precision.json` the numbers after links stopped being read from values alone (minutes, line
numbers and counts inside small tables' keys, figures named after a table, backup copies) and reached what they
missed (a unique number beside a table's key, a code kept as text against a number key, a document and its line
in two columns): joins 91% accuracy, 99% right among what is not sent to review. Every link, metric and
date of a domain's truth counts once; a link or metric the learner invents counts
against it; what it sends to an admin counts as review, not as right.

Every domain is written once with descriptive names and rendered in each style
(`descriptive`, `warehouse`, `pascal`, `generic`). The ground truth and the
golden questions are written in descriptive (logical) names; the harness maps
them onto each style.

## Writing a domain

Model it on `domains/retail.py`:

* `build(seed=7) -> Domain`, deterministic (one `numpy.random.default_rng(seed)`),
  under a second to build, a few tens of thousands of fact rows at most.
* A calendar from `framework.calendar_frame(...)` when the domain has date keys,
  and its `CALENDAR_TYPES` / `CALENDAR_ATTRIBUTES`.
* Every table a `TableDef` with DuckDB types and its true primary key (which
  must be unique: descriptive tables are created with it declared).
* `Truth` lists everything a perfect learner would conclude: each table's kind,
  primary key, every join (with its role name when a pair of tables joins more
  than one way, `trust="proposed"` when the data supports it only partly), the
  calendar, every date role with its kind and the default per table, every
  measure with its aggregation, additivity, format and unit column, the label
  and code columns of each entity, status columns and what their codes mean,
  the data-quality traps planted, and numeric columns that are *not* measures.
* Plant the traps listed for the domain in the module docstring, and state
  them there: the docstring is the domain's specification.
* No real company, product or person names; no names from any client's data.

## Golden questions (`golden/<domain>.yaml`)

```yaml
domain: retail
today: 2026-06-15
questions:
  - id: retail-001
    question: "What were net sales in 2025?"
    tags: [value]
    plan:
      intent: value
      measures: [{agg: sum, column: order_lines.net_amount}]
      time:
        date: order_lines.order_date_key
        window: {kind: between, start: "2025-01-01", end: "2025-12-31"}
    reference_sql: |
      SELECT SUM(l.net_amount) AS net_sales
      FROM order_lines l
      JOIN calendar c ON c.date_key = l.order_date_key
      WHERE c.full_date >= DATE '2025-01-01' AND c.full_date < DATE '2026-01-01'
    expect: {order_matters: false}
```

### The plan, in logical names
* `intent`: value | breakdown | trend | compare | rank | share | drivers | list |
  count | forecast.
* `measures`: `{agg, column}` with `column` as `table.column`; `{agg: count,
  table: t}` counts rows; `{ratio: [m1, m2], scale: 100}` for a ratio of two
  measures.
* `group_by`: `table.column` attributes, or `time:day_of_week`,
  `time:month_of_year`, `time:is_weekend`.
* `filters`: `{column, op, values}`; ops as in DESIGN §7.1. A filter on a total
  ("warehouses with more than 9,000 on hand") names the measure instead:
  `{measure: {agg, column}, op, values}`.
* `time`: `{date, grain, window, compare}`; `date` is the date-role column
  (`order_lines.ship_date_key`), omitted when the question names none and the
  default applies (write the default explicitly anyway, so the case is
  self-contained); `grain` day | week | month | quarter | year |
  fiscal_month | fiscal_quarter | fiscal_year.
* `via`: `{attribute column: [fk column, fk column, ...]}` only when the
  question asks for a non-default path ("by the customer's home store").
* `sort: [{by, desc}]`, `limit`.

### Conventions the reference SQL must follow
* DuckDB SQL over the descriptive names. Qualify every column with its table
  alias. One output column per measure, named by its business name in
  snake_case; periods as a `DATE` column named `period` holding the period's
  first day (`DATE_TRUNC('month', c.full_date)::DATE`); weeks start on Monday
  (`DATE_TRUNC('week', ...)`).
* Windows are half-open on the date: `>= start AND < day after end`.
* `between a and b`: from the first day of `a` to the last day of `b` at the unit
  the user named ("February to March 2026" = 1 Feb to 31 Mar 2026).
* `since a`: from `a` up to and including today. `until b`: everything up to `b`.
* `last N months|weeks|quarters|years`: the N complete units before the
  current one (on 15 June 2026, "last 6 months" is December 2025 to May 2026).
* `this year`, `year to date`: 1 January of today's year up to today.
* Placeholder date keys (the calendar's Unknown row) are excluded whenever a
  question is windowed or grouped by time.
* Rows with a cancelled status are included unless the question excludes them
  (excluding them by default is a business decision an admin makes once).
* A grouping keeps the rows that do not reach the grouping's table, or have no
  value there, as one empty group (`LEFT JOIN`), so a breakdown adds up to the
  total. A top-n ranking leaves that group out (the answer says so).
* Rankings break ties by the label ascending; `LIMIT` the top n.
* Snapshot (semi-additive) measures take, per period and for a whole window,
  only the rows of the last snapshot date in it.
* A question the data cannot answer, or that is genuinely ambiguous, has no
  `reference_sql` and `expect: {asks: measure|date|member|other}` or
  `expect: {declines: true}` instead.

### Multi-turn cases
`turns:` replaces `question`/`plan`/`reference_sql` with a list of them, each
turn's plan written in full (the plan after the follow-up is applied).

### Tags
The issue ids from the plan's inventory (`A1`…`H5`) where a case is the test
for that issue, plus a shape tag (`value`, `trend`, `rank`, ...).
