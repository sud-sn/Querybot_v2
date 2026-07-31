# Data Warehouse Naming Convention Reference

This document describes the structural naming grammar used across all tables.
Use it alongside per-table KB documents to understand column roles and correct SQL patterns.

---

## Table Types (by suffix)

| Table Suffix | Type | Meaning | SQL Guidance |
| --- | --- | --- | --- |
| `_FCT` | fact_table | Fact table — transactional records containing measures (amounts, quantities) and FK keys to dimension tables | Contains the numerical measures you aggregate (SUM, COUNT, AVG). Join to _DMS tables to resolve dimension keys to display labels. One row typically represents one business event (invoice line, transaction, receipt). |
| `_FACT` | fact_table | Fact table — transactional records with measures and FK dimension keys | Same as _FCT. Contains additive measures. Join to dimension tables for display labels. |
| `_DMS` | dimension_table | Dimension table — reference/lookup data with descriptive attributes for a business entity | Contains display fields (_DSC, _NM, _CD) and attributes for a single entity (warehouse, customer, item). Join from fact tables via the matching _DMS_KEY. Never aggregate measures from this table alone. |
| `_DIM` | dimension_table | Dimension table — reference/lookup data (alternate suffix for _DMS) | Same as _DMS. Contains display fields. Join from fact tables via the matching FK. |
| `_EXT` | extended_table | Extended table — supplementary or externally-sourced data joined to a core table | Contains additional attributes that extend the main fact or dimension. Join to the corresponding base table to enrich results. |
| `_STG` | staging_table | Staging table — pre-production ETL landing zone, typically not used in business reporting | Avoid using staging tables for business queries — data may be incomplete or un-validated. Prefer the corresponding _FCT or _DMS production table. |
| `_RPT` | report_table | Report/summary table — pre-aggregated data for faster reporting queries | Data is already summarised — do not re-aggregate unless joining at a finer grain. Check the grain documented in the KB before grouping. |
| `_VW` | view | Database view — a virtual table combining multiple underlying tables | Treat like a regular table. Check the KB for what underlying tables this view consolidates. |
| `_AGG` | aggregate_table | Aggregate table — pre-computed roll-up for performance-critical queries | Data is pre-aggregated. Do not double-aggregate. Use only when the required grain matches. |

---

## Column Suffix Rules

Suffixes encode the **role** of a column and the **correct way to use it in SQL**.

| Suffix | Role | Aggregation | SQL Guidance | Anti-Pattern |
| --- | --- | --- | --- | --- |
| `_DT_DMS_KEY` | date_fk | identifier | Use in JOIN to DT_DMS for calendar attributes (month, quarter, year). For date range filters use BETWEEN with YYYYMMDD integers, e.g. BETWEEN 20240101 AND 20241231. Never use FORMAT() or CONVERT() on this column — it is an integer, not a date type. | FORMAT({col}, 'yyyy-MM-dd') — this column is an INT (YYYYMMDD), not a DATE/DATETIME. |
| `_DMS_KEY` | surrogate_fk | identifier | NEVER SELECT directly as a display value — it is a numeric surrogate key. Always JOIN to the corresponding _DMS table and SELECT the _DSC or _NM column instead. Pattern: JOIN {prefix}_DMS d ON fact.{col} = d.{col} → SELECT d.{prefix}_DSC | SELECT {col} AS [entity name] — this returns a meaningless integer like 1000547, not the warehouse/customer/item name the user expects. |
| `_KEY` | surrogate_fk | identifier | Surrogate keys are internal identifiers. Use only in JOIN conditions. Never expose in SELECT as a business label — resolve to a display column. | GROUP BY {col} for user-facing results — use the corresponding display column. |
| `_DSC` | display | dimension | Use in SELECT and GROUP BY for all user-facing results. This is the canonical display field. | — |
| `_DESC` | display | dimension | Use in SELECT and GROUP BY for all user-facing results. | — |
| `_DESCRIPTION` | display | dimension | Use in SELECT and GROUP BY for all user-facing results. | — |
| `_NM` | display | dimension | Use in SELECT and GROUP BY for all user-facing results. | — |
| `_NAME` | display | dimension | Use in SELECT and GROUP BY for all user-facing results. | — |
| `_CD` | code | dimension | Use in WHERE filters (exact match) and GROUP BY. Shorter than _DSC but still a business-meaningful value (e.g. 'MELB01', 'USD', 'A'). | — |
| `_CODE` | code | dimension | Use in WHERE filters (exact match) and GROUP BY. | — |
| `_AMT` | measure | additive | Safe to SUM across all dimensions. Format as currency. | — |
| `_CST` | measure | additive | Safe to SUM across all dimensions. Format as currency. | — |
| `_PFT` | measure | additive | Safe to SUM across all dimensions. Format as currency. | — |
| `_REV` | measure | additive | Safe to SUM across all dimensions. Format as currency. | — |
| `_QTY` | measure | additive | Safe to SUM across all dimensions. Format as integer or decimal. | — |
| `_CNT` | measure | additive | Safe to SUM or COUNT. Format as integer. | — |
| `_VOL` | measure | additive | Safe to SUM across all dimensions. | — |
| `_WGT` | measure | additive | Safe to SUM across all dimensions. | — |
| `_PCT` | ratio | non_additive | NEVER SUM. Always recalculate from component measures: SUM(numerator) / NULLIF(SUM(denominator), 0) * 100. | SUM({col}) — summing percentages produces nonsense (e.g. 847% gross margin). Always recalculate from the underlying additive components. |
| `_RATE` | ratio | non_additive | NEVER SUM. Recalculate from component measures. | SUM({col}) — rates are non-additive and must be recalculated. |
| `_RATIO` | ratio | non_additive | NEVER SUM. Recalculate from component measures. | SUM({col}) — ratios are non-additive. |
| `_PER` | ratio | non_additive | NEVER SUM. Use AVG or recalculate from components. | SUM({col}) — per-unit rates are non-additive. |
| `_BAL` | semi_additive | semi_additive | SUM by entity (product, customer, warehouse) is valid. Do NOT SUM across time periods — use the latest snapshot instead: WHERE date_key = (SELECT MAX(date_key) FROM ...). | SUM({col}) GROUP BY month — sums a balance across months, which double-counts. Use MAX date or a snapshot approach. |
| `_INV` | semi_additive | semi_additive | SUM by entity is valid. Do NOT SUM across time. Use latest snapshot for current stock. | SUM({col}) over a date range — overstates inventory by counting the same stock multiple times. |
| `_DT` | date | none | Use in WHERE for date range filters. If stored as INT (YYYYMMDD), filter with BETWEEN 20240101 AND 20241231. For relative time (last month, YTD), anchor to MAX(date_col) not GETDATE(). | — |
| `_DATE` | date | none | Use in WHERE for date range filters. For relative time queries, anchor to MAX(date_col) not GETDATE()/SYSDATE. | — |
| `_TS` | timestamp | none | Do NOT use for business date filtering. This records when the row was loaded/modified by the ETL pipeline. Use the corresponding _DT or _DMS_KEY business date column instead. | WHERE {col} BETWEEN @start AND @end — this filters by ETL load time, not by the business transaction date. Use the _DT column instead. |
| `_DTM` | timestamp | none | Use _DT or date _DMS_KEY columns for business date filtering, not this system datetime. | Use for business date filters — this is a system/ETL datetime. |
| `_STS` | status | dimension | Use in WHERE to filter by state. Check distinct values in the KB for valid codes. | — |
| `_TYP` | type | dimension | Use in WHERE filters and GROUP BY for type-level analysis. | — |
| `_GRP` | group | dimension | Use in GROUP BY for group-level rollups. Use in WHERE to filter by group. | — |
| `_FLG` | flag | dimension | Use in WHERE filters (= 1 or = 'Y'). Do not SUM unless counting occurrences. | NEVER SUM({col}) as a measure — it is a flag, not a quantity. |
| `_IND` | flag | dimension | Use in WHERE filters. Do not SUM unless counting occurrences. | — |
| `_YN` | flag | dimension | Filter with WHERE {col} = 'Y' or WHERE {col} = 'N'. | — |
| `_NUM` | identifier | identifier | Use in WHERE for exact lookups (WHERE {col} = '12345'). May appear in GROUP BY if reporting at document level. Do not SUM — it is a reference number, not a quantity. | — |
| `_NBR` | identifier | identifier | Use in WHERE for exact lookups. Do not SUM. | — |
| `_NO` | identifier | identifier | Use in WHERE for exact lookups. Do not SUM. | — |
| `_LIN` | grain | none | One row = one line item. To get document-level totals, GROUP BY the header key (order number, invoice number) before aggregating. | — |
| `_HDR` | grain | none | One row = one document header (order, invoice). Measures here are already at header level. | — |
| `_DTL` | grain | none | One row = one detail/line item. To get summary totals, GROUP BY the parent key before aggregating. | — |

---

## Audit / ETL Prefixes — ALWAYS IGNORE IN BUSINESS QUERIES

| Prefix | Meaning | Guidance |
| --- | --- | --- |
| `AZ_` | Azure Data Factory pipeline audit column (load timestamp, batch ID, source tracking) | IGNORE in all business queries. Never use in WHERE for date filtering — this records ETL load time, not transaction date. Never SELECT in results. |
| `ETL_` | ETL pipeline metadata column | IGNORE in all business queries. System-generated ETL tracking column. |
| `DW_` | Data warehouse system metadata column | IGNORE in all business queries. Internal DW management column. |
| `SYS_` | System-generated column — not a business attribute | IGNORE in all business queries. |
| `META_` | Metadata column — pipeline or system tracking | IGNORE in all business queries. |
| `STG_` | Staging metadata column — pre-production ETL state | IGNORE in all business queries. Use the corresponding business column. |
| `CDC_` | Change data capture column — row-level change tracking | IGNORE in all business queries. Used by replication pipelines only. |

---

## Entity Prefix Vocabulary

| Prefix | Business Entity |
| --- | --- |
| `CUS_` | Customer |
| `CUS_IVC_` | Customer Invoice |
| `CUS_ORD_` | Customer Order |
| `DIV_` | Division |
| `DLV_` | Delivery |
| `DLV_MTH_` | Delivery Method |
| `DLV_TER_` | Delivery Territory |
| `DT_` | Date / Calendar |
| `DVN_` | Division |
| `EMCO_RGN_` | Company Region |
| `FCY_` | Facility / Factory |
| `ITM_` | Item / Product |
| `ITM_BUS_ARA_` | Item Business Area |
| `ITM_GRP_` | Item Group |
| `ITM_STS_` | Item Status |
| `IVC_` | Invoice |
| `ORD_` | Order |
| `PC_` | Profit Center |
| `PCH_` | Purchase |
| `PCH_GRP_` | Purchase Group |
| `PCH_ORD_` | Purchase Order |
| `PC_DVN_` | Profit Center Division |
| `PDC_GRP_` | Product Group |
| `PFT_` | Profit Center |
| `RGN_` | Region |
| `SLR_` | Seller / Sales Rep |
| `SOP_` | Sales Order Processing (calculated / derived measure) |
| `WHS_` | Warehouse |

---

## Key Rules Summary

1. **`_DMS_KEY` columns** — NEVER SELECT directly. Always JOIN to the `_DMS` table and use `_DSC` or `_NM`.
2. **`_PCT`, `_RATE`, `_RATIO` columns** — NEVER SUM. Recalculate as `SUM(num)/SUM(denom)*100`.
3. **`_AMT`, `_QTY`, `_CST`, `_PFT`, `_REV` columns** — safe to SUM (additive).
4. **`_BAL`, `_INV` columns** — semi-additive: SUM by entity OK, never SUM across time.
5. **`_TS`, `_DTM` columns** — system/ETL timestamps, NOT business dates. Use `_DT` or `_DT_DMS_KEY` for date filtering.
6. **`AZ_`, `ETL_`, `DW_`, `SYS_` prefixes** — audit/pipeline columns. Never use in business queries.
7. **`_FCT` tables** — contain measures. Always join to `_DMS` tables to resolve FK keys to display labels.
8. **`_DMS` tables** — contain display fields. Use `_DSC`/`_NM` in SELECT, `_KEY` only for JOIN conditions.
