# QueryBot v2 — Structural Refactor Plan

**Baseline measured:** 2026-08-04, against `origin/main` @ `6a9028e`
**Suite state at baseline:** 3,672 tests passing, 77 subtests, ~3 min runtime

## Purpose

Two structural problems have accumulated: a handful of files carry most of the
system, and the repository lacks the scaffolding a shipped product normally has
(packaging, linting, CI, test configuration). This document records what was
measured, identifies a blocker that must be cleared first, and sequences the work.

Nothing here changes behaviour. Every phase is a refactor or an addition —
if a phase changes what the application does, that phase has gone wrong.

---

## 1. Measured baseline

### 1.1 The ten files carrying the system

| File | Lines | Top-level fns | Largest single fn | Shape |
|---|---|---|---|---|
| `admin/routes.py` | 9,632 | 226 | 423 (`admin_build_kb`) | wide |
| `gateway/webhooks.py` | 3,820 | 20 | **3,252 (`ws_chat`)** | deep |
| `core/query_pipeline.py` | 3,616 | 11 | **3,140 (`_handle_query_impl`)** | deep |
| `core/semantic_model.py` | 2,813 | 57 | 379 (`build_runtime_semantic_plan`) | wide |
| `store/config_store.py` | 2,765 | 96 | 93 (`get_suggestions`) | wide |
| `core/validator.py` | 2,617 | 55 | 681 (`validate_sql_detailed`) | mixed |
| `core/schema.py` | 2,509 | 50 | 359 (`build_entity_graph_from_schema`) | wide |
| `portal/routes.py` | 2,234 | 65 | 139 (`_refresh_chart`) | wide |
| `store/db.py` | 2,067 | 16 | 242 (`_run_migrations`) | wide (55% module-level DDL) |
| `core/llm.py` | 1,492 | 15 | 745 (`build_sql_system_prompt`) | mixed |

Directory spread (Python files): `tests` 169, `core` 120, `store` 17, `gateway` 9,
`evals` 7, root 3, `portal` 2, `deploy` 2, `admin` 2.

Note `admin` and `portal` have **two Python files each** — all their logic lives in a
single `routes.py`.

### 1.2 Wide vs deep — these need opposite treatment

**Wide files** are many small functions in one module. Splitting is mechanical.
`admin/routes.py` is the clearest case: 184 route decorators clustering cleanly by
URL segment, and it already uses a `router` object, so the split is an `APIRouter`
include away.

Route distribution under `/clients/{account_id}/`:

| Segment | Routes | | Segment | Routes |
|---|---|---|---|---|
| graph | 43 | | date-roles | 4 |
| metrics | 11 | | billing | 4 |
| model-health | 10 | | glossary | 5 |
| setup | 9 | | kb / kb-tables | 5 |
| compliance | 9 | | learning-queue | 5 |
| reports | 6 | | evals | 3 |
| users / groups / pending-users | 10 | | date-contexts | 2 |

Plus non-client routes: `/system` (9), `/databases` (11), `/platforms` (3),
auth (3), `/api/clients` (6).

**Deep files** are one function that *is* the module. `ws_chat` is 85% of
`gateway/webhooks.py` and contains five nested closures — `_run_dashboard_chat` (390),
`_run_analysis_work` (317), `_run_local_result_command` (244),
`_run_report_builder_chat` (111), `_run_main_question` (108). Nested closures capture
enclosing scope, so extracting them requires threading context explicitly. This is
where refactors actually introduce bugs, and it is why these come last.

### 1.3 Missing scaffolding

All absent from `origin/main`:

- Packaging: `pyproject.toml`, `setup.py`, `setup.cfg`
- Quality: `ruff`/`flake8` config, `mypy.ini`, `.pre-commit-config.yaml`
- Test config: `pytest.ini`, `conftest.py` (169 test files, no shared fixture config)
- Build/run: `Makefile`, `Dockerfile`, `docker-compose.yml`
- **CI: `.github/` does not exist — there is no automated build or test run**

Layout issues:

- Docs scattered at root (`ARCHITECTURE.md`, `CHANGES.md`, `README_FIXES.md`,
  `ROLLOUT.md`, `ARCHITECTURE_VISUAL.html`, a `.docx` spec) while `docs/` holds one file
- Ad-hoc `migrate_clean_rels.py`, `migrate_env.py` at repo root
- `clients/Test/` (~20k lines of fixture data) committed to the repo
- `tests/` is 169 files flat, no grouping

---

## 2. Blocker: the test suite asserts on file layout

**72 of 169 test files (43%) read source files and assert on their text** — 205 such
assertions. They are concentrated on exactly the files most needing a split:

| Assertion target | Count |
|---|---|
| `admin/routes.py` | 35 |
| `core/query_pipeline.py` | 17 |
| `portal/routes.py` | 14 |
| `gateway/webhooks.py` | 5 |
| `store/config_store.py` | 4 |

Representative pattern (`tests/test_join_coverage_wiring.py`):

```python
self.source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
start = self.source.index("_confidence_context = {")
end = self.source.index("\n    }", start)
block = self.source[start:end]
self.assertIn('"graph_edges"', block)
```

Move the code, rename the variable, or split the file, and this fails — with behaviour
perfectly intact.

**Consequence: the green suite does not protect a refactor.** In these 72 files it
actively opposes one, while implying that layout equals behaviour. Clearing this is
prerequisite to everything else, not a nice-to-have.

---

## 3. Phased plan

### Phase 0 — Make the suite refactor-safe

**Goal:** no test fails purely because code moved.

Convert the 205 source-scan assertions, prioritising the 66 aimed at
`admin/routes.py`, `core/query_pipeline.py`, and `portal/routes.py`. Two tiers:

1. **Preferred — assert on behaviour.** Call the function, check the result.
2. **Cheap fallback — `inspect.getsource(module.func)`** instead of
   `Path(...).read_text()`. Survives file moves as long as the function stays
   importable. Use where a genuine behavioural test is disproportionate effort.

Delete assertions that only restate that a line of code exists; they carry no
information a type checker or the behavioural test above wouldn't.

**Exit criterion:** `git mv` any of the three priority files to a new path and the
suite still passes.

### Phase 1 — Scaffolding (no code movement)

Add, in this order:

1. `pyproject.toml` — project metadata, dependencies, tool config in one place
2. `ruff` config (lint + format), `pytest.ini`, `conftest.py`
3. **CI workflow** running lint + the full suite on push and PR
4. `Makefile` wrapping the common commands
5. Housekeeping: root docs → `docs/`; `migrate_*.py` → `scripts/`;
   `clients/Test/` → `tests/fixtures/` or out of git

CI must exist **before** Phases 2 and 3, so regressions surface during the risky work
rather than after it.

### Phase 2 — Mechanical splits (low risk, largest structural win)

In order of payoff:

1. **`admin/routes.py` → `admin/routes/` package.** One module per feature router —
   `graph`, `metrics`, `compliance`, `setup`, `model_health`, `users`, `kb`, `glossary`,
   `learning_queue`, `evals`, `billing`, `date_roles`, `reports`, `system`, `databases`,
   `auth` — each exporting an `APIRouter`, composed in `admin/routes/__init__.py`.
   The 43-route `graph` module will still be large; split it again by sub-resource.
2. **`store/config_store.py`** by domain (metrics, relationships, suggestions,
   LLM call logs, KB egress).
3. **`store/db.py`** by table group — the `_ensure_*_tables` functions are already
   the seams; make each its own module and keep `_run_migrations` as the orchestrator.
4. **`portal/routes.py`** by feature (chat, dashboard, pins, export, threads).
5. **`core/schema.py`** — separate per-dialect discovery (`_discover_azure_sql` 275,
   `_discover_oracle` 155, `_discover_snowflake` 137) from graph building.
   **This directly pre-paves the planned Looker/Qlik connector work**, which needs a
   per-source discovery seam that does not exist today.
6. **`core/semantic_model.py`** — split build / runtime-plan / health / patch concerns.

Rule for this phase: **moves and imports only.** No signature changes, no logic edits,
no "while I'm here" fixes. One file per commit, suite green between each.

### Phase 3 — God-function extraction (high risk, sequential)

Only with CI green and Phase 0 complete. One function per PR, never two in flight.

1. **`core/llm.py::build_sql_system_prompt` (745)** — start here; largely prompt text.
   Move prompt bodies to template files. Cheapest win, lowest risk, builds confidence.
2. **`core/validator.py::validate_sql_detailed` (681)** — already delegates to named
   `_*_errors` helpers; extract the remaining inline checks to match.
3. **`core/result_commands.py::execute_result_command` (602)** — dispatch table over
   per-command handlers.
4. **`gateway/webhooks.py::ws_chat` (3,252)** — extract each nested closure to its own
   module with an explicit context object. The hardest item here; budget accordingly.
5. **`core/query_pipeline.py::_handle_query_impl` (3,140)** — extract by pipeline stage
   (resolve → plan → generate → validate → execute → respond).

Do not start items 4 and 5 while feature work is landing on those files — they are the
two hottest files in the repo.

### Phase 4 — Optional

Group `core/`'s 120 flat modules into subpackages (`core/semantic/`, `core/query/`,
`core/analysis/`, `core/dashboard/`). Highest churn-to-benefit ratio of anything in this
document, touching imports across the codebase for organisational benefit only.
Genuinely deferrable; revisit once Phases 0–3 have settled.

---

## 4. Risks

| Risk | Mitigation |
|---|---|
| Source-scan tests give false confidence | Phase 0 is prerequisite; exit criterion is an actual file move |
| Refactor collides with in-flight feature work | Phase 3 items 4–5 need a quiet window on those files |
| Import cycles when splitting `store/` and `core/` | Split leaf-first; add a cycle check to CI in Phase 1 |
| Large mechanical diffs hide a real change | Moves-only rule in Phase 2; review diffs with `--find-renames` |
| `execute_governed_query` has no end-to-end test | Add one before touching the compliance path (noted separately) |

## 5. Success criteria

- No application file exceeds ~800 lines; no function exceeds ~150
- Every source-scan assertion is behavioural or `inspect`-based
- CI runs lint + full suite on every push
- A new contributor can find the code for a given admin page by path alone
- Test count and pass rate never regress across any phase
