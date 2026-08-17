# EMCO-shaped test mart — `EMDW_DMART`

A synthetic Infor M3 data mart that reproduces the **structural** characteristics
of the EMCO warehouse, so every failure we have hit can be tested without
touching the client's data.

Deliberately small (~34k fact rows). The point is to exercise planning logic, not
scan speed — the client's real problem is a starved instance, and that is not
something a test model should imitate.

## Deploy

Run in order against a **dedicated test database**:

```
01_create_model.sql     -- drops and recreates EMDW_DMART
02_seed_dimensions.sql
03_seed_facts.sql
```

`01` drops everything in the schema, so it is safely repeatable.

## What it reproduces, and why

| Characteristic | Where | The defect it exercises |
|---|---|---|
| **YYYYMMDD integer date keys** | `DT_DMS.DT_DMS_KEY = 20260630` | Recorded as `surrogate_fk`, so date windows join `DT_DMS` instead of filtering the fact's own integer key |
| **8 role-playing dates on one fact** | `CUS_ORD_IVC_FCT.*_DT_DMS_KEY` | "revenue for the last 2 days" selecting `LST_MOD_DT_DMS_KEY` instead of `CUS_IVC_DT_DMS_KEY` |
| **Audit dates on a dimension** | `CUS_DMS.CR_LMT_n_CHG_DT_DMS_KEY` | ~20 date entities in the graph, most of which nobody will ever ask about |
| **Two routes to one entity** | invoice → customer, and invoice → profit centre → `PFT_CTR_CUS_DAT` → customer | "more than one equally governed relationship path" |
| **Four facts sharing dimensions** | all reach `PFT_CTR_DMS` | a loose metric dragging rival facts into one plan, leaving no anchor |
| **CAD / local currency pairs** | `SOP_CUS_IVC_LIN_AMT` vs `..._CAD_AMT` | summing the wrong currency column — silently wrong, not an error |
| **`AZ_*` infrastructure columns** | every table | columns that must be excluded from business answers |
| **Stale data** | newest fact row is `2026-06-30` | "what is today's revenue" answering a stale number with no as-of date |
| **No indexes past the PKs** | by design | the client cannot add indexes; the test model should not be secretly faster |

`LST_MOD_DT_DMS_KEY` is set to the load date on **every** row. That is what makes
the wrong-date-role failure loud: if the bot anchors on it, a two-day window
returns the entire table instead of two days.

## Setup after deploy

1. Connect the client to this database, run **Discover schema**, build the KB.
2. In the Semantic Layer, approve Date Roles — the important ones:
   - `CUS_ORD_IVC_FCT.CUS_IVC_DT_DMS_KEY` → **Invoice Date**, mark as **default**
   - `PCH_ORD_RCT_FCT.PCH_ORD_RCT_DT_DMS_KEY` → **Receipt Date**, default
   - `FNN_FCT.ACG_DT_DMS_KEY` → **Accounting Date**, default
   - leave `LST_MOD_DT_DMS_KEY` and the `CR_LMT_n_CHG_*` columns **unapproved**
3. Define exactly two metrics, tightly scoped:
   - **Revenue** → `SUM(SOP_CUS_IVC_LIN_AMT)` on `CUS_ORD_IVC_FCT`, default time column `CUS_IVC_DT_DMS_KEY`
   - **Purchase Order Amount** → `SUM(PCH_ORD_LIN_CAD_AMT)` on `PCH_ORD_RCT_FCT`, default time column `PCH_ORD_RCT_DT_DMS_KEY`

   Keep the synonyms narrow. A Revenue metric that also matches "purchase orders"
   is the exact configuration that broke the live tenant.
4. Do **not** press Suggest and bulk-accept. If you want to test that path, do it
   on a second copy — it mints duplicate entities that cannot be cleanly undone.

## Test catalogue

Each row is a question, what should happen, and the defect it guards.

### A. Date roles

| # | Question | Expected |
|---|---|---|
| A1 | `what is my revenue for the last 2 days` | Answers. Uses **Invoice Date**. Log shows `dropped_dates=[]` and no `Last Modified Date` |
| A2 | `show revenue by last modified date` | Answers on `LST_MOD_DT_DMS_KEY`. Explicit request must still be honoured |
| A3 | `show revenue by invoice date` | Answers. No other date role appears |
| A4 | `what is my revenue per warehouse for the available dates` | Answers. Warehouse + Invoice Date only |
| A5 | `latest 5 orders` | No date role selected from the word "latest" |
| A6 | `what is today's revenue` | **Must disclose** the as-of date: "the most recent business data is 30 Jun 2026…". Never a bare number |
| A7 | `revenue yesterday` | Same disclosure. Never "0" without naming the latest date |

### B. Fact selection

| # | Question | Expected |
|---|---|---|
| B1 | `what is the total amount of confirmed purchase orders by profit center` | Resolves to `PCH_ORD_RCT_FCT`, **not** the invoice fact. No clarification |
| B2 | `what is my revenue by each customer, provide top 5` | Answers, 5 rows. No "Missing governed path" |
| B3 | `show total revenue by profit centre` | Uses `CUS_ORD_IVC_FCT` only. `FNN_FCT` is not dragged in |
| B4 | `show me the inventory value by warehouse` | Uses `ITM_BAL_PRD_FCT`. Semi-additive: not summed across months |

### C. Joins

| # | Question | Expected |
|---|---|---|
| C1 | `show the profit centers along with their customer first invoice dates` | Answers. If the path is near-tied it says *"Using the … relationship"* — it must not stop and ask |
| C2 | `revenue by customer type` | Traverses `CUS_DMS → CUS_TYP_DMS`. No duplicate-path dead end |
| C3 | `revenue by region` | Reaches `PFT_CTR_DMS.RGN_NM` |
| C4 | `show customers with no invoices` | Anti-join. Must not silently become an inner join |

### D. Clarification behaviour

| # | Action | Expected |
|---|---|---|
| D1 | Answer any clarification with `no, don't use this` | Clears it. Does not re-ask. Does not run the rejected option |
| D2 | Answer a clarification with a brand-new question | Old clarification cleared, new question answered |
| D3 | Ask `show customers with no orders` while a clarification is open | Treated as a question, **not** as a rejection |

### E. Result follow-ups

| # | Sequence | Expected |
|---|---|---|
| E1 | `what was my revenue for the past 5 days` → `provide the trend` | Same window, date added to GROUP BY, 5 rows summing to the original total |
| E2 | …then `now by week` | Re-grains again, still the same window |
| E3 | `revenue by customer` → `only the top 3` | Applied to the cached result, no new warehouse query |

### F. Things that must NOT happen

| # | Check |
|---|---|
| F1 | No suggested question in the portal fails when clicked |
| F2 | No answer sums `..._CAD_AMT` and `..._AMT` together |
| F3 | No answer includes an `AZ_*` column as a business field |
| F4 | No question is answered on `LST_MOD_DT_DMS_KEY` unless it asked for it |
| F5 | A timeout is reported as a timeout, never repaired into a different error |

## Moving the staleness

`03_seed_facts.sql` opens with:

```sql
DECLARE @LatestBusinessDate date = '2026-06-30';
```

Set it to `CAST(GETDATE() AS date)` to test the fully-current case, where A6/A7
should answer without a staleness note. Re-run all three scripts after changing it.
